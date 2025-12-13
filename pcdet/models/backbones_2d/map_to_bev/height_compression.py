import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_


        
from ....utils import common_utils
from ....utils.spconv_utils import replace_feature

from pcdet.models.backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
from pcdet.models.fusion_modules.deform_fusion import DeformAttnFusion
from pcdet.models.fusion_modules.adf_block import ADF
from pcdet.models.model_utils.centernet_utils import gaussian2D, sy_draw_gaussian_to_heatmap, draw_gaussian_to_heatmap

from ....utils import common_utils, loss_utils, box_utils
from tools.visual_utils.open3d_vis_utils import Visualizer3D


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, voxel_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        fc_list = []

        self.fuse_rt_feats = self.model_cfg.get('ROT_FUSE', None)
        if self.fuse_rt_feats:
            filters = [self.num_bev_features * self.model_cfg.ROT_FUSE.NUM_ROTS] + self.model_cfg.ROT_FUSE.FUSE_FILTERS
            self.fuse_conv = []
            for i in range(len(filters) - 1):
                self.fuse_conv.append(nn.Conv2d(filters[i], filters[i + 1], 3, stride=1, padding=1, bias=True))
                self.fuse_conv.append(nn.BatchNorm2d(filters[i + 1]))
                self.fuse_conv.append(nn.ReLU())
            self.fuse_conv = nn.Sequential(*self.fuse_conv)
            fc_list.append(self.fuse_conv)

        # self.adf_block_rt = ADF(self.num_bev_features)

        for module_list in fc_list:
            for m in module_list.modules():
                if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def extract_features(self, encoded_spconv_tensor):
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features
    
    def get_pseudo_points(self, pts_range=[0, -40, -3, 70.4, 40, 1], voxel_size=[0.05, 0.05, 0.05], stride=8):

        x_stride = voxel_size[0] * stride
        y_stride = voxel_size[1] * stride

        min_x = pts_range[0] + x_stride / 2
        max_x = pts_range[3] #+ x_stride / 2
        min_y = pts_range[1] + y_stride / 2
        max_y = pts_range[4] + y_stride / 2

        x = np.arange(min_x, max_x, x_stride)
        y = np.arange(min_y, max_y, y_stride)

        x, y = np.meshgrid(x, y)
        zeo = np.zeros(shape=x.shape)

        grids = torch.from_numpy(np.stack([x, y, zeo]).astype(np.float32)).permute(1,2,0).cuda()

        return grids

    def interpolate_from_bev_features(self, points, bev_features, bev_stride):

        cur_batch_points = points

        x_idxs = (cur_batch_points[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (cur_batch_points[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        cur_x_idxs = x_idxs / bev_stride
        cur_y_idxs = y_idxs / bev_stride

        cur_bev_features = bev_features.permute(1, 2, 0)  # (H, W, C)
        point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)

        return point_bev_features

    def bev_align(self, batch_dict, bev_feat, rot_angle, flip_x, flip_y, scale, stride=8):

        batch_size = batch_dict['batch_size']
        w, h = bev_feat.shape[-2], bev_feat.shape[-1]

        all_feat = []
        grid_pts = self.get_pseudo_points(self.point_cloud_range, self.voxel_size, stride).reshape(-1, 3)
        
        for b in range(batch_size):
            
            cur_bev_feat = bev_feat[b]
            grid_pts_b = grid_pts.clone()

            # print(flip_x, rot_angle, scale)
            
            if flip_x:
                grid_pts_b[:, 1] *= -1
            if flip_y:
                grid_pts_b[:, 0] *= -1
            grid_pts_b = common_utils.rotate_points_along_z(grid_pts_b.unsqueeze(0), angle=torch.tensor(rot_angle).to(grid_pts_b).view([1])).squeeze(0)
            if scale != 1.0:
                grid_pts_b = grid_pts_b * scale
            
            points = batch_dict['points']
            points = points[points[..., 0]==b][..., 1:]

            # vis = Visualizer3D()
            # vis.add_points(points)
            # points_trans = points.clone()
            # if flip_x:
            #     points_trans[:, 1] *= -1
            # if flip_y:
            #     points_trans[:, 0] *= -1
            # points_trans = common_utils.rotate_points_along_z(points_trans.unsqueeze(0), angle=torch.tensor(rot_angle).to(points_trans).view([1])).squeeze(0)
            # if scale != 1.0:
            #     points_trans *= scale
            # vis.add_points(points_trans, [1, 1, 0])
            # vis.add_points(grid_pts_b, [1, 1, 0])
            # vis.show()

            aligned_feat = self.interpolate_from_bev_features(grid_pts_b, cur_bev_feat, stride).reshape(w, h, -1)
            aligned_feat=aligned_feat.permute(2,0,1)
            
            # import pdb;pdb.set_trace()
            all_feat.append(aligned_feat)

        return torch.stack(all_feat)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
         
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor'] 
        spatial_features = self.extract_features(encoded_spconv_tensor)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        if 'trans_dict' in batch_dict:
            trans_dict = batch_dict['trans_dict'][0]
            rot_angle_list = trans_dict['rot_angle_list']
            flip_x_list = trans_dict['flip_x_list']
            flip_y_list = trans_dict['flip_y_list']
            scale_list = trans_dict['scale_list']

            spatial_features_list = [spatial_features]

            encoded_spconv_tensor_trans = batch_dict['encoded_spconv_tensor_trans']

            for tensor, rot_angle, flip_x, flip_y, scale in zip(encoded_spconv_tensor_trans, rot_angle_list, flip_x_list, flip_y_list, scale_list):
                spatial_features = self.extract_features(tensor['encoded_spconv_tensor'])
                spatial_features = self.bev_align(batch_dict, spatial_features, rot_angle, flip_x, flip_y, scale, batch_dict['encoded_spconv_tensor_stride'])
                spatial_features_list.append(spatial_features) # (B, C, H, W)

            if self.fuse_rt_feats:
                spatial_features = torch.cat(spatial_features_list, dim=1)
                spatial_features = self.fuse_conv(spatial_features)
            else:
                spatial_features = torch.stack(spatial_features_list, dim=0)
                spatial_features = spatial_features.max(dim=0)[0]

        # spatial_features = self.adf_block_rt(spatial_features)
        batch_dict['spatial_features'] = spatial_features

        return batch_dict
