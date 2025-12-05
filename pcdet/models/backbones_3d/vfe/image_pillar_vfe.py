import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vfe import VFETemplate
from ..vfe.pillar_vfe import PFNLayer

from ....utils import common_utils
from pcdet.models.fusion_modules.deform_fusion import DeformAttnFusion

from easydict import EasyDict as edict
 
class ImagePillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
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

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        
        self.use_img = self.model_cfg.USE_IMG
        if self.use_img:
            self.deform_attn = DeformAttnFusion(mid_channels=out_filters, light=True)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
  
        # Extract pillar features.
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
            
            points_proj = coords.new_zeros((coords.shape[0], 3))
            points_proj[..., 0] = coords[..., 0]

            for b in range(batch_size):
                batch_mask = coords[..., 0] == b
                pts = f_centers[batch_mask]

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
                
                # import matplotlib.pyplot as plt
                # img = batch_dict['images'][b]
                # plt.imshow(img.cpu().numpy().transpose((1,2,0)))
                # plt.scatter(points_proj_batch[:, 0].cpu().numpy(), points_proj_batch[:, 1].cpu().numpy(), c='red', s=0.5)
                # plt.show()    
            
            #features_img = [f_cluster.squeeze(1), voxel_num_points.unsqueeze(-1)]
            #features_img = torch.cat(features_img, dim=-1)
            #features_img = self.ffn(features_img)
            features = self.deform_attn(batch_dict, features, points_proj)

        batch_dict['pillar_features'] = features
        
        return batch_dict
