from logging import raiseExceptions
from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from .dynamic_pillar_vfe import PFNLayerV2
from ....utils import common_utils, loss_utils, box_utils
from matplotlib import pyplot as plt

class DynamicVoxelVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        
        self.points_channels = self.model_cfg.PTS_CHANNELS
        self.num_points_features = self.points_channels

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

        self.OUT_CHANNELS = self.model_cfg.OUT_CHANNELS
        self.img_channels = self.model_cfg.IMG_CHANNELS
        self.pool_method = self.model_cfg.POOL_METHOD
        self.num_filters = self.model_cfg.NUM_FILTERS
        self.mid_channels = self.model_cfg.MID_CHANNELS

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

        self.fuse = nn.Sequential(
            nn.Linear(feat_channels[-1] + self.img_channels, self.OUT_CHANNELS, bias=False),
            nn.BatchNorm1d(self.OUT_CHANNELS),
            nn.ReLU()
        )

        # self.pts_transform = nn.Sequential(
        #     nn.Linear(self.num_filters[-1], self.mid_channels, bias=False),
        #     nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
        # )

        self.img_transform = nn.Sequential(
            nn.Linear(self.img_channels, self.img_channels, bias=False),
            nn.BatchNorm1d(self.img_channels),
            nn.ReLU()
        )

        fc_list = [self.vfe_layers, self.img_transform, self.fuse]
        # fc_list = [self.vfe_layers]
        init_func = nn.init.xavier_normal_
        for module_list in fc_list:
            for m in module_list.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def get_output_feature_dim(self):
        return self.OUT_CHANNELS
        return self.num_filters[-1]

    def project_points_img(self, batch_dict, points):
        B = batch_dict['batch_size']
        img_feats = []

        for i in range(B):
            img = batch_dict['images'][i]
            pts = points[points[...,0] == i][...,1:]
            calib = batch_dict['calib'][i]    
            
            noise_rotation = batch_dict['noise_rotation'][i].view(1) if 'noise_rotation' in batch_dict else torch.tensor(0.0, device=points.device).view(1)
            noise_scale = batch_dict['noise_scale'][i] if 'noise_scale' in batch_dict else torch.tensor(1.0, device=points.device).view(1)
            flip_x = batch_dict['flip_x'][i] if 'flip_x' in batch_dict else False
            
            img_scale = batch_dict['img_scale'][i]
            pad_size = batch_dict['pad_size'][i]
            
            points_proj = common_utils.project_points_to_img_batch(img=img,
                                                                   img_scale=img_scale,
                                                                   pad_size=pad_size,
                                                                   points=pts,
                                                                   calib=calib,
                                                                   noise_rotation=noise_rotation,
                                                                   noise_scale=noise_scale,
                                                                   flip_x=flip_x)
            coor_x, coor_y = points_proj[..., 0], points_proj[..., 1]
            C, H, W = img.shape
            norm_coor_y = coor_y / (H - 1) * 2 - 1
            norm_coor_x = coor_x / (W - 1) * 2 - 1

            grid = torch.stack([norm_coor_x.view(-1), norm_coor_y.view(-1)],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2
            
            img_fpn = batch_dict['image_fpn'][0][i]
            x = F.grid_sample(img_fpn.unsqueeze(0), grid)
            x = x.squeeze().t()
            img_feats.append(x)
            
            # for j in range(2):
            #     img_fpn = batch_dict['image_fpn'][j][i]
            #     x = F.grid_sample(img_fpn.unsqueeze(0), grid)
            #     x = x.squeeze().t()
            #     ml_feats.append(x)
            # ml_feats = torch.cat(ml_feats, dim=-1)
            # img_feats.append(ml_feats)    

            if img is not None:
                plt.imshow(img.cpu().numpy().transpose((1,2,0)))
                plt.scatter(coor_x.int().cpu().numpy(), coor_y.int().cpu().numpy(), c='red', s=0.5)
                plt.show()
            
        # points_img = torch.cat((points_img), dim=0) # (N, C)
        # points_img = self.mlp_reduction(points_img)

        img_feats = torch.cat(img_feats, dim=0)
        return img_feats

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

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
        
        img_feats = self.project_points_img(batch_dict, points.clone())
        img_feats = self.img_transform(img_feats)
        # pts_feats = self.pts_transform(features)
        #fuse_out_feats = img_feats + pts_feats
        #fuse_out_feats = F.leaky_relu(fuse_out_feats)
        
        fuse_out_feats = torch.cat((features, img_feats), dim=-1)
        fuse_out_feats = self.fuse(fuse_out_feats)
        
        if self.pool_method == 'max_pool':
            fuse_out_feats = torch_scatter.scatter_max(fuse_out_feats, unq_inv, dim=0)[0]
        elif self.pool_method == 'mean_pool':
            fuse_out_feats = torch_scatter.scatter_mean(fuse_out_feats, unq_inv, dim=0)
        else:
            raise NotImplementedError()

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['voxel_features'] = fuse_out_feats
        batch_dict['voxel_coords'] = voxel_coords

        return batch_dict
