from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from pcdet.models.model_utils.basic_block_1d import BasicBlock1D
from pcdet.ops.deform_attn.ops.modules import MSDeformAttn
from pcdet.ops.deform_attn.ops.modules.ms_deform_attn import MSDeformAttnOnlyExtraction
from pcdet.utils import common_utils

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DeformTransLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=4, n_points=8, 
                 light=True, norm=True):
        super().__init__()
        self.light = light
        self.norm = norm
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        if self.light == False:
            self.dropout1 = nn.Dropout(dropout)
            if self.norm:
                self.norm1 = nn.LayerNorm(d_model)
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.activation = _get_activation_fn(activation)
            self.dropout2 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.dropout3 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)
 
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src_feat, reference_points, key_feat, spatial_shapes, level_start_index, padding_mask=None, query_pos=None):

        query_feat = self.with_pos_embed(src_feat, query_pos)
        src2, loc, weight = self.self_attn(query_feat, reference_points, key_feat, spatial_shapes, level_start_index, padding_mask)
        
        if self.light == False:
            src_feat = src_feat + self.dropout1(src2)
            if self.norm:
                src_feat = self.norm1(src_feat)
            src_feat = self.forward_ffn(src_feat)
        else:
            src_feat = src2

        return src_feat, loc, weight

class DeformAttnFusionMultiImage(nn.Module):
    def __init__(self,
                 light=True,
                 img_feat_interp=True,
                 dropout_ratio=0.0,
                 activate_out=True,
                 fuse_out=True,
                 mid_channels=16):
        
        super().__init__()

        self.img_feat_interp = img_feat_interp
        self.mid_channels = mid_channels
        self.dropout_ratio = dropout_ratio
        self.activate_out = activate_out
        self.fuse_out = fuse_out

        self.pts_key_proj = nn.Sequential(
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
            # nn.ReLU()
        )

        self.pts_transform = nn.Sequential(
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
            # nn.ReLU()
        )

        #self.fuse_blocks = DeformTransLayer(d_model=self.mid_channels, \
        #        n_levels=1, n_heads=4, n_points=4, light=light)
        
        self.fuse_blocks = MSDeformAttnOnlyExtraction(self.mid_channels, 1, 4, 4)
        
        self.fuse_conv = nn.Sequential(
            nn.Linear(self.mid_channels * 2, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
            nn.ReLU())
        
        self.sampling_offsets = nn.Linear(self.mid_channels, 4 * 1 * 4 * 2)
        self.attention_weights = nn.Linear(self.mid_channels, 4 * 1 * 4)
        self.value_proj = nn.Linear(self.mid_channels, self.mid_channels)
        
        self.part = 50000
        
        self.init_weights()
    
    def init_weights(self):
        init_func = nn.init.xavier_normal_
        fc_list = [self.pts_key_proj, self.pts_transform, self.fuse_conv]
        for module_list in fc_list:
            for m in module_list.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def fusion_withdeform(self, img_pre_fuse, pts_pre_fuse):

        # fuse_out = img_pre_fuse + pts_pre_fuse
        fuse_out = torch.cat([pts_pre_fuse, img_pre_fuse], dim=-1)
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def linear_layer(self, layer, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [layer(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
            return x

        return layer(inputs)
    
    def forward(self, batch_dict, point_features, proj_coords_cam_list, image_mask_total_list, img_layer_index=0):

        point_features = point_features.reshape(-1, point_features.shape[-1])
        
        pts_feats_org = self.linear_layer(self.pts_key_proj, point_features)
        
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.linear_layer(self.pts_transform, pts_feats_org)
        
        batch_size = batch_dict['batch_size']

        b, d, c, h, w = batch_dict['images'].shape # B, D, C, H, W 
        
        image_feat_multicam = batch_dict['image_fpn'][img_layer_index]
        if 'image_fpn_interp' in batch_dict:
            image_feat_multicam = batch_dict['image_fpn_interp']
            
        if self.img_feat_interp and 'image_fpn_interp' not in batch_dict:
            for cam_idx in range(d):
                image_feat = image_feat_multicam[:, cam_idx, :, :, :]
                image_feat = nn.functional.interpolate(image_feat, (h, w), mode='bilinear')
                if 'image_fpn_interp' not in batch_dict:
                    batch_dict['image_fpn_interp'] = {}
                batch_dict['image_fpn_interp'][cam_idx] = image_feat
            image_feat_multicam = batch_dict['image_fpn_interp']
            
        image_with_voxelfeatures = point_features.new_zeros((point_features.shape))
        
        sampling_offsets = self.linear_layer(self.sampling_offsets, pts_feats_org)
        attention_weights = self.linear_layer(self.attention_weights, pts_feats_org)
        
        for cam_idx in range(d):
            
            image_feat = image_feat_multicam[cam_idx]
            proj_coords = proj_coords_cam_list[cam_idx]
            image_mask = image_mask_total_list[cam_idx]

            filter_idx_list = []
            
            for b in range(batch_size):
                image_feat_batch = image_feat[b].clone()

                index_mask = proj_coords[:, 0] == b
                voxels_2d = proj_coords[index_mask][:, 1:3]
                depth = proj_coords[index_mask][:, -1]
                
                offset_batch = sampling_offsets[index_mask]
                attention_weight_batch = attention_weights[index_mask]
                
                voxel_features_sparse = pts_feats_org[index_mask]
                voxels_2d_int = voxels_2d.clone().long()
                
                filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w) * (depth >= 0)
                filter_idx_list.append(filter_idx)
                image_grid = voxels_2d_int[filter_idx]
                image_features_batch = image_feat_batch.unsqueeze(0)
                
                _, channel_num, f_h, f_w = image_features_batch.shape
                flatten_img_feat = image_features_batch.permute(0, 2, 3, 1).reshape(1, f_h*f_w, channel_num)
                value_batch = self.value_proj(flatten_img_feat)
                
                if not self.img_feat_interp:
                    raw_shape = tuple(batch_dict['image_shape'][b].cpu().numpy())
                    image_grid = image_grid.float()
                    image_grid[:,0] *= (f_w/raw_shape[1])
                    image_grid[:,1] *= (f_h/raw_shape[0])
                    image_grid = image_grid.long()
                ref_points = image_grid.float()
                ref_points[:, 0] /= f_w
                ref_points[:, 1] /= f_h
                ref_points = ref_points.reshape(1, -1, 1, 2)
                N, Len_in, _ = flatten_img_feat.shape
                pts_feats = voxel_features_sparse[filter_idx].reshape(1, -1, self.mid_channels)
                level_spatial_shapes = pts_feats.new_tensor([(f_h, f_w)], dtype=torch.long)
                level_start_index = pts_feats.new_tensor([0], dtype=torch.long)

                # value_batch = value_batch[filter_idx]
                offset_batch = offset_batch[filter_idx]
                attention_weight_batch = attention_weight_batch[filter_idx]

                if pts_feats.shape[1] > 0:
                    feats, _, _ = self.fuse_blocks(pts_feats, value_batch, offset_batch, attention_weight_batch, ref_points, flatten_img_feat, level_spatial_shapes, level_start_index)
                    voxel_features_sparse[filter_idx] = feats.squeeze(0)
                image_with_voxelfeatures[index_mask] = voxel_features_sparse

        final_voxelimg_feat = self.fusion_withdeform(image_with_voxelfeatures, pts_pre_fuse)

        return final_voxelimg_feat

class DeformAttnFusion(nn.Module):
    def __init__(self,
                 light=True,
                 img_feat_interp=True,
                 dropout_ratio=0.0,
                 activate_out=True,
                 fuse_out=True,
                 mid_channels=16):
        
        super().__init__()

        self.img_feat_interp = img_feat_interp
        self.mid_channels = mid_channels
        self.dropout_ratio = dropout_ratio
        self.activate_out = activate_out
        self.fuse_out = fuse_out

        self.pts_key_proj = nn.Sequential(
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
            # nn.ReLU()
        )

        self.pts_transform = nn.Sequential(
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
            # nn.ReLU()
        )

        self.fuse_blocks = DeformTransLayer(d_model=self.mid_channels, \
                n_levels=1, n_heads=4, n_points=4, light=light)
        
        self.fuse_conv = nn.Sequential(
            nn.Linear(self.mid_channels * 2, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
            nn.ReLU())
        
        self.init_weights()
    
    def init_weights(self):
        init_func = nn.init.xavier_normal_
        fc_list = [self.pts_key_proj, self.pts_transform, self.fuse_conv]
        for module_list in fc_list:
            for m in module_list.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def fusion_withdeform(self, img_pre_fuse, voxel_feat):
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(voxel_feat)

        # fuse_out = img_pre_fuse + pts_pre_fuse
        fuse_out = torch.cat([pts_pre_fuse, img_pre_fuse], dim=-1)
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def forward(self, batch_dict, point_features, proj_coords, img_layer_index=0, cam_idx=-1):

        point_features = point_features.reshape(-1, point_features.shape[-1])
        if point_features.shape[0] <= 1:
            return point_features
        
        pts_feats_org = self.pts_key_proj(point_features)
        batch_size = batch_dict['batch_size']
        
        if cam_idx >= 0:
            h, w = batch_dict['images'].shape[3:] # B, D, C, H, W 
        else:
            h, w = batch_dict['images'].shape[2:] # B, C, H, W

        image_feat = batch_dict['image_fpn'][img_layer_index]
        if cam_idx >= 0:
            # B, D, C, H, W
            if self.img_feat_interp:
                if 'image_fpn_interp' in batch_dict and cam_idx in batch_dict['image_fpn_interp']:
                    image_feat = batch_dict['image_fpn_interp'][cam_idx]
                else:
                    image_feat = nn.functional.interpolate(image_feat[:, cam_idx, :, :, :], (h, w), mode='bilinear')
                    if 'image_fpn_interp' not in batch_dict:
                        batch_dict['image_fpn_interp'] = {}
                    batch_dict['image_fpn_interp'][cam_idx] = image_feat
        else:
            if self.img_feat_interp:
                if 'image_fpn_interp' in batch_dict:
                    image_feat = batch_dict['image_fpn_interp']
                else:
                    image_feat = nn.functional.interpolate(image_feat, (h, w), mode='bilinear')
                    batch_dict['image_fpn_interp'] = image_feat

        image_with_voxelfeatures = point_features.new_zeros((point_features.shape))
        filter_idx_list = []
 
        for b in range(batch_size):
            image_feat_batch = image_feat[b]

            index_mask = proj_coords[:, 0] == b
            voxels_2d = proj_coords[index_mask][:, 1:]
            voxel_features_sparse = pts_feats_org[index_mask]
            voxels_2d_int = voxels_2d.clone().long()

            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)

            filter_idx_list.append(filter_idx)
            image_grid = voxels_2d_int[filter_idx]
            image_features_batch = image_feat_batch.unsqueeze(0)
            
            _, channel_num, f_h, f_w = image_features_batch.shape
            flatten_img_feat = image_features_batch.permute(0, 2, 3, 1).reshape(1, f_h*f_w, channel_num)
            if not self.img_feat_interp:
                raw_shape = tuple(batch_dict['image_shape'][b].cpu().numpy())
                image_grid = image_grid.float()
                image_grid[:,0] *= (f_w/raw_shape[1])
                image_grid[:,1] *= (f_h/raw_shape[0])
                image_grid = image_grid.long()
            ref_points = image_grid.float()
            ref_points[:, 0] /= f_w
            ref_points[:, 1] /= f_h
            ref_points = ref_points.reshape(1, -1, 1, 2)
            N, Len_in, _ = flatten_img_feat.shape
            pts_feats = voxel_features_sparse[filter_idx].reshape(1, -1, self.mid_channels)
            level_spatial_shapes = pts_feats.new_tensor([(f_h, f_w)], dtype=torch.long)
            level_start_index = pts_feats.new_tensor([0], dtype=torch.long)

            if pts_feats.shape[1] > 0:
                feats, sampling_locations, weights = self.fuse_blocks(pts_feats, ref_points, flatten_img_feat, level_spatial_shapes, level_start_index)
                voxel_features_sparse[filter_idx] = feats.squeeze(0)
                sampling_locations = sampling_locations.squeeze(0)
                weights = weights.squeeze(0)

                # img = batch_dict['images'][b]
                # img = common_utils.unnormalize_img(img)

                # weights = weights.squeeze(2)
                # indices = weights.argmax(dim=-1, keepdim=True)
                # indices = indices.unsqueeze(-1).expand(-1, -1, -1, 2)

                # sampling_locations = sampling_locations.squeeze(2)
                # sampling_locations[..., 0] *= f_w
                # sampling_locations[..., 1] *= f_h

                # sampling_locations = torch.gather(sampling_locations, dim=2, index=indices)
                # # sampling_locations = sampling_locations[:, 1, :, :]
                # sampling_locations = sampling_locations.view(-1, 2)

                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # axes[0].imshow(img.cpu().numpy().transpose((1,2,0)))
                # axes[0].scatter(image_grid[:, 0].cpu().numpy(), image_grid[:, 1].cpu().numpy(), c='orange', s=3.5)

                # axes[1].imshow(img.cpu().numpy().transpose((1,2,0)))
                # axes[1].scatter(sampling_locations[:, 0].cpu().numpy(), sampling_locations[:, 1].cpu().numpy(), c='orange', s=3.5)

                # plt.show()

                # import pdb;pdb.set_trace()
                
            image_with_voxelfeatures[index_mask] = voxel_features_sparse

        final_voxelimg_feat = self.fusion_withdeform(image_with_voxelfeatures, point_features)

        return final_voxelimg_feat