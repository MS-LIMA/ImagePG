import random
import torch

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from tools.inference.visualizer import Visualizer3D

from .vfe_template import VFETemplate
from ....utils import common_utils, loss_utils, box_utils

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.fusion_modules.deform_fusion import DeformAttnFusion
from pcdet.models.model_utils.attention_utils import FeedForwardPositionalEncoding
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper

class DynamicImageVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
   
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSOLUTE_XYZ
        self.num_point_features_org = num_point_features
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.num_filters = self.model_cfg.NUM_FILTERS
        feat_channels = [num_point_features] + list(self.num_filters)

        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False),
                    nn.BatchNorm1d(out_filters),
                    nn.ReLU()))
        self.vfe_layers = nn.ModuleList(vfe_layers)

        self.srt_conv = self.model_cfg.get('SRT_CONV', None)
        if self.srt_conv:
            self.rot_angles = self.srt_conv.ROT_ANGLES
            self.scales = self.srt_conv.get('SCALES', [1.0] * len(self.rot_angles))
            self.flip_x = self.srt_conv.get('FLIP_X', [True] * len(self.rot_angles))
            self.enable_test = self.srt_conv.get('ENABLE_TEST', False)

        self.pool_method = self.model_cfg.POOL_METHOD
        self.use_img = self.model_cfg.USE_IMG
        self.out_channels = feat_channels[-1]
        if self.use_img:
            self.deform_attn = DeformAttnFusion(mid_channels=self.out_channels, light=True)
            img_feats = self.model_cfg.IMG_FEATURES
            self.ffn = nn.Sequential(
                nn.Linear(num_point_features, img_feats // 2),
                nn.BatchNorm1d(img_feats // 2),
                nn.ReLU(img_feats // 2),
                nn.Linear(img_feats // 2, img_feats, 1),
            )

            self.pts_transform = nn.Sequential(
                nn.Linear(feat_channels[-1], img_feats),
                nn.BatchNorm1d(img_feats, eps=1e-3, momentum=0.01),
                # nn.ReLU()
            )

            self.fuse_conv = nn.Sequential(
                nn.Linear(img_feats, img_feats),
                nn.BatchNorm1d(img_feats, eps=1e-3, momentum=0.01),
                nn.ReLU())

        if self.use_img:
            fc_list = [self.vfe_layers, self.ffn, self.pts_transform, self.fuse_conv]
        else:
            fc_list = [self.vfe_layers]

        init_func = nn.init.xavier_normal_
        for module_list in fc_list:
            for m in module_list.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def get_output_feature_dim(self):
        return self.out_channels

    def fusion_withdeform(self, pts_feat, img_pre_fuse):

        pts_pre_fuse = self.pts_transform(pts_feat)
        fuse_out = pts_pre_fuse + img_pre_fuse
        fuse_out = F.relu(fuse_out)
        fuse_out = self.fuse_conv(fuse_out)

        return fuse_out
    
    def extract_features(self, batch_dict, rot_angle=None, flip_x=False, scale=1.0):

        batch_size = batch_dict['batch_size']
        if rot_angle is not None:
            points = batch_dict['points'].clone() # (batch_idx, x, y, z, i, e)
            rot_angle = torch.tensor(rot_angle).to(points).view(1)
            
            for i in range(batch_size):
                batch_mask = points[..., 0] == i
                if flip_x:
                    points[batch_mask, 2] = -points[batch_mask, 2]
                points[batch_mask, 1:4] = common_utils.rotate_points_along_z(points=points[batch_mask][..., 1:4].unsqueeze(0), angle=rot_angle).squeeze(0)
                if scale != 1.0:
                    points[batch_mask, 1:4] *= scale
                    
                # points_org = batch_dict['points']
                # print(rot_angle.item(), flip_x)
                # vis = Visualizer3D()
                # vis.set_points(points[batch_mask][..., 1:4].cpu().numpy())
                # vis.set_points(points_org[batch_mask][..., 1:4].cpu().numpy(), [0, 1, 0])
                # vis.add_axis()
                # vis.show()
        else:
            points = batch_dict['points']

         # Extract image features 
        if self.use_img:
            if 'points_proj' not in batch_dict:
                batch_size = batch_dict['batch_size']   

                points_org = batch_dict['points']
                points_proj = points_org.new_zeros((points_org.shape[0], 3))
                points_proj[..., 0] = points_org[..., 0]

                for b in range(batch_size):
                    batch_mask = points_org[..., 0] == b
                    pts = points_org[batch_mask][..., 1:4]

                    img = batch_dict['images'][b]
                    calib = batch_dict['calib'][b]    
                    
                    noise_rotation = batch_dict['noise_rotation'][b].view(1) if 'noise_rotation' in batch_dict else torch.tensor(0.0, device=pts.device).view(1)
                    noise_scale = batch_dict['noise_scale'][b] if 'noise_scale' in batch_dict else torch.tensor(1.0, device=pts.device).view(1)
                    flip_x = batch_dict['flip_x'][b] if 'flip_x' in batch_dict else False
                    
                    img_scale = batch_dict['img_scale'][b]
                    pad_size = batch_dict['pad_size'][b]
                    
                    points_proj_batch = common_utils.project_points_to_img_batch(img=img,
                                                                            img_scale=img_scale,
                                                                            pad_size=pad_size,
                                                                            points=pts,
                                                                            calib=calib,
                                                                            noise_rotation=noise_rotation,
                                                                            noise_scale=noise_scale,
                                                                            flip_x=flip_x)

                    points_proj[batch_mask, 1:] = points_proj_batch
                    
                    img = batch_dict['images'][b]
                    img = common_utils.unnormalize_img(img)
                    plt.imshow(img.cpu().numpy().transpose((1,2,0)))
                    plt.scatter(points_proj_batch[:, 0].cpu().numpy(), points_proj_batch[:, 1].cpu().numpy(), c='red', s=0.5)
                    # plt.scatter(batch_dict['points_2d'][b][:, 0].cpu().numpy(), batch_dict['points_2d'][b][:, 1].cpu().numpy(), c='red', s=0.5)
                    plt.show()   

                points_proj = points_proj.view(-1, points_proj.shape[-1]) 
                batch_dict['points_proj'] = points_proj

        points_coords = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / self.voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1,2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)

        f_cluster = points_xyz - points_mean[unq_inv, :]
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        for vfe in self.vfe_layers:
            features = vfe(features)
        
        if self.use_img:
            points_proj = batch_dict['points_proj'][mask]
            features_img = [points[:, 1:], f_cluster, f_center]
            features_img = torch.cat(features_img, dim=-1)
            features_img = self.ffn(features_img)
            features_img = self.deform_attn(batch_dict, features_img, points_proj)
            features = self.fusion_withdeform(features, features_img)

            # for b in range(batch_size):
            #     batch_mask = points_proj[..., 0] == b
            #     points_proj_batch = points_proj[batch_mask][..., 1:]

            #     img = batch_dict['images'][b]
            #     img = common_utils.unnormalize_img(img)
            #     plt.imshow(img.cpu().numpy().transpose((1,2,0)))
            #     plt.scatter(points_proj_batch[:, 0].cpu().numpy(), points_proj_batch[:, 1].cpu().numpy(), c='red', s=0.5)
            #     plt.show()  
            # 

            # for i in range(batch_size):
            #     batch_mask = points[..., 0] == i

            #     points_org = batch_dict['points']
            #     batch_mask_org = points_org[..., 0] == i

            #     vis = Visualizer3D()
            #     vis.set_points(points[batch_mask][..., 1:4].cpu().numpy())
            #     vis.set_points(points_org[batch_mask_org][..., 1:4].cpu().numpy(), [0, 1, 0])
            #     vis.add_axis()
            #     vis.show()
        
        if self.pool_method == 'max_pool':
            features = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        elif self.pool_method == 'mean_pool':
            features = torch_scatter.scatter_mean(features, unq_inv, dim=0)
        else:
            raise NotImplementedError()

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        return features, voxel_coords, rot_angle, points

    def forward(self, batch_dict, **kwargs):

        features, voxel_coords, _, _ = self.extract_features(batch_dict)
        batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords

        trans_dict = {'rotation_angles' : [],
                      'scales' : [],
                      'flip_x' : [],
                      'flip_y' : []}
        
        if self.srt_conv and (self.training or self.enable_test):
            batch_dict['rtspconv'] = True

            points_rotated = []
            voxel_features_rotated = []
            voxel_coords_rotated = []
            for rot_angle, flip_x, scale in zip(self.rot_angles, self.flip_x, self.scales):
                features, voxel_coords, rot_angle_i, points_rot = self.extract_features(batch_dict, rot_angle, flip_x, scale)

                trans_dict['rotation_angles'].append(rot_angle_i)
                trans_dict['flip_x'].append(flip_x)
                trans_dict['scales'].append(scale)

                points_rotated.append(points_rot)
                voxel_features_rotated.append(features)
                voxel_coords_rotated.append(voxel_coords)      

            batch_dict['trans_dict'] = trans_dict
            batch_dict['points_trans'] = points_rotated
            batch_dict['voxel_features_rotated'] = voxel_features_rotated
            batch_dict['voxel_coords_rotated'] = voxel_coords_rotated

        return batch_dict

