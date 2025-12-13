import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.inference.visualizer import Visualizer3D

from ..backbones_3d.vfe import VFETemplate
from ..backbones_3d.vfe.image_pillar_vfe import ImagePillarVFE, PFNLayer
from ..backbones_2d import BaseBEVBackbone

from ...utils import common_utils, loss_utils, box_utils
from ..model_utils.centernet_utils import gaussian2D, sy_draw_gaussian_to_heatmap, draw_gaussian_to_heatmap
from pcdet.models.fusion_modules.deform_fusion import DeformAttnFusion, DeformAttnFusionMultiImage

class BEVHeatmapHead(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        
        self.vis = False
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.use_dense = self.model_cfg.get('USE_DENSE', False)
        self.use_multi_image = self.model_cfg.get('USE_MULTI_IMAGE', False)
        
        self.vfe_cfg = self.model_cfg.VFE
        self.backbone_2d = BaseBEVBackbone(self.model_cfg.BACKBONE_2D, self.num_bev_features)

        pillar_grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(self.vfe_cfg.VOXEL_SIZE)
        self.pillar_grid_size = np.round(pillar_grid_size).astype(np.int64)
        self.nx, self.ny, self.nz = self.pillar_grid_size
        
        self.use_norm = self.vfe_cfg.USE_NORM
        self.with_distance = self.vfe_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.vfe_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.vfe_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = self.vfe_cfg.VOXEL_SIZE[0]
        self.voxel_y = self.vfe_cfg.VOXEL_SIZE[1]
        self.voxel_z = self.vfe_cfg.VOXEL_SIZE[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        
        self.use_img = self.vfe_cfg.USE_IMG
        if self.use_img:
            if not self.use_multi_image:
                self.deform_attn = DeformAttnFusion(mid_channels=out_filters, light=True)
            else:
                # self.deform_attn = DeformAttnFusionMultiImage(mid_channels=out_filters)
                self.deform_attn = DeformAttnFusion(mid_channels=out_filters)

        filters = [sum(self.model_cfg.BACKBONE_2D.NUM_UPSAMPLE_FILTERS)] + self.model_cfg.SHARED_FC
        
        self.shared_conv = []
        for i in range(len(filters) - 1):
            self.shared_conv.append(nn.Conv2d(filters[i], 
                                              filters[i + 1], 3, stride=1, padding=1, bias=True))
            self.shared_conv.append(nn.BatchNorm2d(filters[i + 1]))
            self.shared_conv.append(nn.ReLU())
        self.shared_conv = nn.Sequential(*self.shared_conv)
        
        filters = [filters[-1]] + self.model_cfg.HEAD_FC
        self.head = []
        for i in range(len(filters) - 1):
            self.head.append(nn.Conv2d(filters[i], filters[i + 1], 3, stride=1, padding=1, bias=True))
            self.head.append(nn.BatchNorm2d(filters[i + 1]))
            self.head.append(nn.ReLU())
        self.head.append(nn.Conv2d(filters[-1], 1, 3, 1, 1))
        self.head = nn.Sequential(*self.head)

        self.forward_ret_dict = {}
        self.build_losses()

        from torch.nn.init import kaiming_normal_
        fc_list = [self.head, self.shared_conv]
        for module_list in fc_list:
            for m in module_list.modules():
                if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

        # self.transformer = nn.TransformerEncoderLayer(64, 4, 1024, 0, batch_first=True)
        
    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
                       
    # Heatmap Loss
    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def sy_assign_target_of_single_head(
            self, num_classes, feature_map_size, data_dict,
            bs_idx = None,
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]
        Returns:
        """
        
        heatmap = torch.zeros(num_classes, feature_map_size[1], feature_map_size[0]).cuda()
        coord_c = data_dict['bm_spatial_features'][bs_idx].reshape(feature_map_size[0], feature_map_size[1])
        coord_c = (coord_c == 1).nonzero(as_tuple=False)
        c_coord_x = coord_c[:, 0] 
        c_coord_y = coord_c[:, 1]
        c_center = torch.cat((c_coord_x[:, None], c_coord_y[:, None]), dim=-1)
        radius = int(3)
        
        heatmap = sy_draw_gaussian_to_heatmap(heatmap, c_center, radius, 0)
        return heatmap

    def sy_assign_targets(self, data_dict, feature_map_size=None, **kwargs):
        gt_boxes = data_dict['gt_boxes']
        feature_map_size = feature_map_size[::-1]
        batch_size = gt_boxes.shape[0]
        heatmap_list = []
        for bs_idx in range(batch_size):
            heatmap = self.sy_assign_target_of_single_head(
                num_classes=1,
                feature_map_size=feature_map_size,
                data_dict=data_dict,
                bs_idx=bs_idx,
            )
            heatmap_list.append(heatmap)
        ret_dict = dict(
            heatmaps = torch.stack(heatmap_list)    
        )
        return ret_dict
    
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        loss = 0

        pred_dicts['hm'] = self.sigmoid(pred_dicts['hm'])
        hm_loss = self.hm_loss_func(pred_dicts['hm'], target_dicts['heatmaps']) * self.model_cfg.get('LOSS_WEIGHT', 1.0)

        loss += hm_loss
        tb_dict['hm_bev_loss'] = hm_loss.item()
        
        return loss, tb_dict

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def extract_features(self, batch_dict, **kwargs):
  
        # Extract pillar features.
        if self.use_dense:
            voxel_features, voxel_num_points, coords = batch_dict['pillar_voxels_dense'], \
                batch_dict['pillar_voxel_dense_num_points'], batch_dict['pillar_voxel_dense_coords']
        else:
            voxel_features, voxel_num_points, coords = batch_dict['pillar_voxels'], \
                batch_dict['pillar_voxel_num_points'], batch_dict['pillar_voxel_coords']
            
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        # batch_size = batch_dict['batch_size']
        # pillar_centers = torch.zeros_like(voxel_features[:, :, :3])
        # pillar_centers[:, :, 0] = (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        # pillar_centers[:, :, 1] = (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        # pillar_centers[:, :, 2] = (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # for b in range(batch_size):
        #     points = batch_dict['points']
        #     points = points[points[..., 0] == b][..., 1:]

        #     # import pdb;pdb.set_trace()
        #     vox_centers = pillar_centers[..., :3]
        #     vox_centers = vox_centers[coords[..., 0] == b]
        #     vox_centers = vox_centers.view(-1, 3)

        #     vis = Visualizer3D(4)
        #     vis.set_points(points.cpu().numpy())
        #     vis.set_points(vox_centers.cpu().numpy(), [255, 255, 0])
        #     vis.show()

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        
        # Extract image features 
        if self.use_img:
            batch_size = batch_dict['batch_size']   
            mask = voxel_features.sum(dim=2) != 0
            f_centers = (voxel_features * mask[..., None].float()).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            
            if 'TTA' in batch_dict:
                aug_rot_angle = batch_dict['aug_rot_angle']
                aug_scale = batch_dict['aug_scale'].item()
                aug_flip_x = (batch_dict['aug_flip_x'] == 1.0).item()
                
                f_centers = torch.cat([coords[..., 0:1], f_centers], dim=-1)
                f_centers = common_utils.transform_points(f_centers, aug_rot_angle, aug_flip_x, False, aug_scale, forward=False)
                f_centers = f_centers[..., 1:]

            images = batch_dict['images']
            if images.ndim == 5:
                B, D, C, H, W = images.shape
            else:
                B, C, H, W = images.size()
                D = 1
                images = images.view(B, 1, C, H, W)

            points_proj_cam_list = []
            image_mask_total_list = []
            for img_idx in range(D):

                points_proj = []
                image_mask_total = []
                
                for b in range(batch_size):
                    batch_mask = coords[..., 0] == b
                    pts = f_centers[batch_mask]

                    img = images[b][img_idx]
                    calib = batch_dict['calib'][b]    
                    
                    noise_translation = batch_dict['noise_translation'][b] if 'noise_translation' in batch_dict else torch.tensor([0, 0, 0], device=pts.device)
                    noise_rotation = batch_dict['noise_rotation'][b].view(1) if 'noise_rotation' in batch_dict else torch.tensor(0.0, device=pts.device).view(1)
                    noise_scale = batch_dict['noise_scale'][b] if 'noise_scale' in batch_dict else torch.tensor(1.0, device=pts.device).view(1)
                    flip_x = batch_dict['flip_x'][b] if 'flip_x' in batch_dict else False
                    flip_y = batch_dict['flip_y'][b] if 'flip_y' in batch_dict else False

                    img_scale = batch_dict['img_scale'][b]
                    pad_size = batch_dict['pad_size'][b]
                    
                    C, H, W = img.shape
                    points_proj_batch, depth = common_utils.project_points_to_img_batch(img=img,
                                                                            img_scale=img_scale,
                                                                            pad_size=pad_size,
                                                                            points=pts[..., :3],
                                                                            calib=calib,
                                                                            noise_translation=noise_translation,
                                                                            noise_rotation=noise_rotation,
                                                                            noise_scale=noise_scale,
                                                                            flip_x=flip_x,
                                                                            flip_y=flip_y,
                                                                            only_points=D > 1,
                                                                            cam_id=img_idx if D > 1 else -1)
                    #points_proj_batch = torch.cat([points_proj_batch, depth.view(-1, 1)], dim=-1)
                    
                    image_mask = (points_proj_batch[:, 0] >= 0) & (points_proj_batch[:, 0] < W) & \
                        (points_proj_batch[:, 1] >= 0) & (points_proj_batch[:, 1] < H) & (depth >= 0)
                    image_mask_total.append(image_mask)

                    #if not self.use_multi_image:
                    #    points_proj_batch = points_proj_batch[image_mask]
                    points_proj_batch = points_proj_batch[image_mask]

                    bidx = points_proj_batch.new_zeros(points_proj_batch.shape[0]).fill_(b)
                    points_proj_batch = torch.cat([bidx.unsqueeze(-1), points_proj_batch], dim=-1)
                    points_proj.append(points_proj_batch)

                    # vis = Visualizer3D()
                    # vis.set_points(pts[image_mask].cpu().numpy())
                    # vis.show()

                    # import matplotlib.pyplot as plt
                    # img = common_utils.unnormalize_img(img)
                    # plt.imshow(img.cpu().numpy().transpose((1,2,0)))
                    # plt.scatter(points_proj_batch[:, 1].cpu().numpy(), points_proj_batch[:, 2].cpu().numpy(), c='red', s=0.5)
                    # plt.show()    
                
                points_proj = torch.cat(points_proj)
                image_mask_total = torch.cat(image_mask_total)
                image_mask_total_list.append(image_mask_total)
                points_proj_cam_list.append(points_proj)

                # if not self.use_multi_image:
                features = features.view(-1, features.shape[-1])   
                features[image_mask_total] = self.deform_attn(batch_dict, features[image_mask_total], points_proj, cam_idx=img_idx if D > 1 else -1)

            #if self.use_multi_image:    
            #    features = features.view(-1, features.shape[-1])   
            #    features = self.deform_attn(batch_dict, features, points_proj_cam_list, image_mask_total_list)               

        batch_dict['pillar_features'] = features
        
        return batch_dict
     
    def scatter(self, batch_dict, **kwargs):

        if self.use_dense:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['pillar_voxel_dense_coords']
        else:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['pillar_voxel_coords']
            
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['pillar_spatial_features'] = batch_spatial_features
        
        if self.training == True:
                
            _, _, W, H = batch_dict['spatial_features'].shape # KITTI 200, 176
            
            batch_dict['bm_pillar_features'] = torch.ones(
                batch_dict['bm_voxels'].shape[0], 1,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            c_bm_pillar_features, c_bm_coords = batch_dict['bm_pillar_features'], batch_dict['bm_voxel_coords']
            c_bm_batch_spatial_features = []
            for batch_idx in range(batch_size):
                c_bm_spatial_feature = torch.zeros(
                    1, 1 * H * W,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)

                batch_mask = c_bm_coords[:, 0] == batch_idx
                this_coords = c_bm_coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * H + this_coords[:, 3]
                indices = indices.type(torch.long)
                c_bm_pillars = c_bm_pillar_features[batch_mask, :]
                c_bm_pillars = c_bm_pillars.t()
                c_bm_spatial_feature[:, indices] = c_bm_pillars
                c_bm_batch_spatial_features.append(c_bm_spatial_feature)

            c_bm_batch_spatial_features = torch.stack(c_bm_batch_spatial_features, 0)
            c_bm_batch_spatial_features = c_bm_batch_spatial_features.view(batch_size, 1 * 1, W, H)
            batch_dict['bm_spatial_features'] = c_bm_batch_spatial_features

        return batch_dict
       
    def forward(self, batch_dict, **kwargs):
        
        # Extract pillar-image features.
        # batch_dict = self.vfe(batch_dict)

        # batch_dict = self.extract_features(batch_dict)
        # batch_dict = self.scatter(batch_dict)

        # target_dict = self.sy_assign_targets(
        #         batch_dict, feature_map_size=[200, 176],
        #         feature_map_stride=batch_dict.get('spatial_features_2d_strides', None)
        # )
        # import matplotlib.pyplot as plt
        # hms_gt = target_dict['heatmaps']
        # for i in range(hms_gt.shape[0]):
        #     inst = hms_gt[i]
        #     noise_rotation = batch_dict['noise_rotation'][i] if 'noise_rotation' in batch_dict else torch.tensor(0.0, device=inst.device)
        #     rot = common_utils.rotate_bev_map(inst, torch.tensor(np.pi / 4))
        #     inst = common_utils.bilinear_interpolation_from_bev(rot, torch.tensor(np.pi / 4))

        #     fig, axes = plt.subplots(1, 3)
        #     axes[0].imshow(hms_gt[i].cpu().numpy().transpose(1,2,0))
        #     axes[1].imshow(rot.cpu().numpy().transpose(1,2,0))
        #     axes[2].imshow(inst.cpu().numpy().transpose(1,2,0))
        #     plt.show()

        batch_dict = self.extract_features(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        spatial_features_2d = batch_dict['pillar_spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)
        x_pred = self.head(x)
        
        if self.training == True:
            target_dict = self.sy_assign_targets(
                batch_dict, feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=batch_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict
            self.forward_ret_dict['pred_dicts'] = dict(hm=x_pred)
            batch_dict['gt_hm'] = target_dict['heatmaps']

            # import matplotlib.pyplot as plt
            # hms_gt = target_dict['heatmaps']
            # for i in range(hms_gt.shape[0]):
            #     plt.imshow(hms_gt[i].cpu().numpy().transpose(1,2,0))
            #     plt.show()

        elif self.vis :
            import matplotlib.pyplot as plt
            heatmaps_pred = self.sigmoid(x_pred)
            mask = heatmaps_pred > self.model_cfg.BEV_SHAPE_THRESH
            heatmaps_pred[~mask] = 0
            for i in range(heatmaps_pred.shape[0]):
                plt.imshow(heatmaps_pred[i].cpu().numpy().transpose(1,2,0))
                plt.show()
        
        batch_dict['bev_hm'] = self.sigmoid(x_pred)
        hm_prob = batch_dict['bev_hm']
        mask = hm_prob > self.model_cfg.BEV_SHAPE_THRESH
        hm_prob[~mask] = 0
        batch_dict['hm_prob'] = hm_prob
        
        return batch_dict
