from functools import partial

import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone1x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.block_num = self.model_cfg.get('BLOCK_NUM', 2)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.num_point_features = 64
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 64,
        }

        if self.block_num >=3:
            self.conv2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv2', conv_type='spconv'),
                block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
                block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            )
            self.conv3 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(32, 64, 3, norm_fn=norm_fn, padding=1, indice_key='spconv3', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            )
            self.backbone_channels['x_conv2'] = 32
            self.backbone_channels['x_conv3'] = 64
        if self.block_num >=4:
            self.conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(64, 64, 3, norm_fn=norm_fn, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            )
            self.backbone_channels['x_conv4'] = 64


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)


        batch_dict.update({
            'input_spconv_tensor': input_sp_tensor,
        })
        batch_dict.update({
            'encoded_spconv_tensor': x_conv2,
            'encoded_spconv_tensor_stride': 1
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 1,
            }
        })
        if self.block_num >=3:
            x_conv3 = self.conv3(x_conv2)
            batch_dict['encoded_spconv_tensor'] = x_conv3
            batch_dict['multi_scale_3d_features']['x_conv3'] = x_conv3
            batch_dict['multi_scale_3d_strides']['x_conv3'] = 1
        if self.block_num >=4:
            x_conv4 = self.conv4(x_conv3)
            batch_dict['encoded_spconv_tensor'] = x_conv4
            batch_dict['multi_scale_3d_features']['x_conv4'] = x_conv4
            batch_dict['multi_scale_3d_strides']['x_conv4'] = 1

        return batch_dict


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_img = self.model_cfg.get('USE_IMG', False)
        self.use_mm = self.model_cfg.get('USE_MM', False)
        self.channels = self.model_cfg.get('CHANNELS', [16, 32, 64, 64])
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, self.channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(self.channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        # self.conv1 = spconv.SparseSequential(
        #     block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        # )

        # self.conv2 = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        # )

        # self.conv3 = spconv.SparseSequential(
        #     # [800, 704, 21] <- [400, 352, 11]
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        # )

        self.conv1 = spconv.SparseSequential(
            block(self.channels[0], self.channels[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(self.channels[0], self.channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(self.channels[1], self.channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(self.channels[1], self.channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(self.channels[1], self.channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(self.channels[2], self.channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(self.channels[2], self.channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(self.channels[2], self.channels[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(self.channels[3], self.channels[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(self.channels[3], self.channels[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(self.channels[3], 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': self.channels[0],
            'x_conv2': self.channels[1],
            'x_conv3': self.channels[2],
            'x_conv4': self.channels[3]
        }
        
        self.use_dense = self.model_cfg.get("USE_DENSE", False)
        
        if self.use_mm:
            self.conv_input_mm = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 32, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(32),
            nn.ReLU(),
            )
 

            self.conv1_mm = spconv.SparseSequential(
                block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            )

            self.conv2_mm = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            )

            self.conv3_mm = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            )

            self.conv4_mm = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            )

            # last_pad = 0
            # last_pad = self.model_cfg.get('last_pad', last_pad)
            # self.conv_out_mm = spconv.SparseSequential(
            #     # [200, 150, 5] -> [200, 150, 2]
            #     spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
            #                         bias=False, indice_key='spconv_down2'),
            #     norm_fn(128),
            #     nn.ReLU(),
            # )

    def extract_features(self, 
                         batch_dict, 
                         voxel_features, 
                         voxel_coords,
                         is_mm=False):
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        if not is_mm:
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)
        else:
            x = self.conv_input_mm(input_sp_tensor)

            x_conv1 = self.conv1_mm(x)
            x_conv2 = self.conv2_mm(x_conv1)
            x_conv3 = self.conv3_mm(x_conv2)
            x_conv4 = self.conv4_mm(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            # out = self.conv_out_mm(x_conv4)
            out = None
        
        encoded_spconv_tensor = {
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        }
        
        multi_scale_3d_features = {
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        }

        multi_scale_3d_strides = {
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        }

        return encoded_spconv_tensor, multi_scale_3d_features, multi_scale_3d_strides
    
    def decompose_tensor(self, tensor, i, batch_size, num_trans):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // (num_trans+1))
        end_shape_ids = (i + 1) * (input_shape // (num_trans+1))
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // (num_trans+1))
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // (num_trans+1)]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor
    
    def reverse_transformation(self, batch_dict, multi_scale_3d_features, rot_angle, flip_x, flip_y, scale):
        
        for k, src_name in enumerate(['x_conv3', 'x_conv4']):
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = multi_scale_3d_features[src_name]
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
        
        import pdb;pdb.set_trace()
       
    def forward_train(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, voxel_features, voxel_coords)
        batch_dict.update(encoded_spconv_tensor_dict)
        batch_dict.update(multi_scale_3d_features_dict)
        batch_dict.update(multi_scale_3d_strides_dict)
        
        if 'mm_points' in batch_dict:
            voxel_features, voxel_coords = batch_dict['voxel_mm_features'], batch_dict['voxel_mm_coords']
            encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, 
                                                                                                                          voxel_features, 
                                                                                                                          voxel_coords, 
                                                                                                                          True)
            batch_dict.update({'encoded_spconv_tensor_mm' : encoded_spconv_tensor_dict['encoded_spconv_tensor']})
            batch_dict.update({'multi_scale_3d_mm_features' : multi_scale_3d_features_dict['multi_scale_3d_features']})
                        
        if 'trans_dict' in batch_dict:

            trans_dict = batch_dict['trans_dict'][0]
            rot_angle_list = trans_dict['rot_angle_list']
            flip_x_list = trans_dict['flip_x_list']
            scale_list = trans_dict['scale_list']
            num_trans = trans_dict['num_trans']
                
            encoded_spconv_tensor_trans = []
            multi_scale_3d_features_trans = []
            multi_scale_3d_strides_trans = []
            
            if 'mm_points' in batch_dict:
                encoded_spconv_tensor_mm_trans = []
                multi_scale_3d_features_mm_trans = []
                multi_scale_3d_strides_mm_trans = []   
            
            trans_dict = batch_dict['trans_dict'][0]
            num_trans = trans_dict['num_trans']
            for i in range(num_trans):
                voxel_features, voxel_coords = batch_dict['voxel_features_' + str(i+1)], batch_dict['voxel_coords_' + str(i+1)]
                encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, voxel_features, voxel_coords)
                
                encoded_spconv_tensor_trans.append(encoded_spconv_tensor_dict)
                multi_scale_3d_features_trans.append(multi_scale_3d_features_dict)
                multi_scale_3d_strides_trans.append(multi_scale_3d_strides_dict)
                
                if 'mm_points' in batch_dict:
                    voxel_features, voxel_coords = batch_dict['voxel_mm_features_' + str(i+1)], batch_dict['voxel_mm_coords_' + str(i+1)]
                    encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, 
                                                                                                                                  voxel_features, 
                                                                                                                                  voxel_coords,
                                                                                                                                  True)
                    encoded_spconv_tensor_mm_trans.append(encoded_spconv_tensor_dict)
                    multi_scale_3d_features_mm_trans.append(multi_scale_3d_features_dict)

            batch_dict['encoded_spconv_tensor_trans'] = encoded_spconv_tensor_trans
            batch_dict['multi_scale_3d_features_trans'] = multi_scale_3d_features_trans
            batch_dict['multi_scale_3d_strides_trans'] = multi_scale_3d_strides_trans

            if 'mm_points' in batch_dict:      
                multi_scale_3d_strides_mm_trans.append(multi_scale_3d_strides_dict)
                batch_dict['encoded_spconv_tensor_mm_trans'] = encoded_spconv_tensor_mm_trans
                batch_dict['multi_scale_3d_features_mm_trans'] = multi_scale_3d_features_mm_trans
                batch_dict['multi_scale_3d_strides_mm_trans'] = multi_scale_3d_strides_mm_trans
        
        return batch_dict

    def forward_test(self, batch_dict):

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

        all_lidar_feat = []
        all_lidar_coords = []
        
        
        trans_dict = batch_dict['trans_dict'][0]
        num_trans = trans_dict['num_trans']     
        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * (num_trans + 1)]

        all_lidar_feat.append(voxel_features)
        new_coord = voxel_coords.clone()
        new_coord[:, 3] += 0*self.sparse_shape[2]
        all_lidar_coords.append(new_coord)

        if 'trans_dict' in batch_dict:

            encoded_spconv_tensor_trans = []
            multi_scale_3d_features_trans = []
            multi_scale_3d_strides_trans = []

            trans_dict = batch_dict['trans_dict'][0]
            num_trans = trans_dict['num_trans']
            for i in range(num_trans):
                voxel_features, voxel_coords = batch_dict['voxel_features_' + str(i+1)], batch_dict['voxel_coords_' + str(i+1)]
                all_lidar_feat.append(voxel_features)
                new_coord = voxel_coords.clone()
                new_coord[:, 3] += (i+1)*self.sparse_shape[2]
                all_lidar_coords.append(new_coord)

                # encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, voxel_features, voxel_coords)
                
                # encoded_spconv_tensor_trans.append(encoded_spconv_tensor_dict)
                # multi_scale_3d_features_trans.append(multi_scale_3d_features_dict)
                # multi_scale_3d_strides_trans.append(multi_scale_3d_strides_dict)
                
            all_lidar_feat = torch.cat(all_lidar_feat, 0)
            all_lidar_coords = torch.cat(all_lidar_coords)
            batch_size = batch_dict['batch_size']

            input_sp_tensor = spconv.SparseConvTensor(
            features=all_lidar_feat,
            indices=all_lidar_coords.int(),
            spatial_shape=new_shape,
            batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)
            out = self.conv_out(x_conv4)

            for i in range(num_trans + 1):
                this_conv2 = self.decompose_tensor(x_conv2, i, batch_size, num_trans)
                this_conv3 = self.decompose_tensor(x_conv3, i, batch_size, num_trans)
                this_conv4 = self.decompose_tensor(x_conv4, i, batch_size, num_trans)
                this_out = self.decompose_tensor(out, i, batch_size, num_trans)

                encoded_spconv_tensor = {
                        'encoded_spconv_tensor': this_out,
                        'encoded_spconv_tensor_stride': 8
                    }
                    
                multi_scale_3d_features = {
                    'multi_scale_3d_features': {
                        'x_conv1': None,
                        'x_conv2': this_conv2,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    }
                }

                multi_scale_3d_strides = {
                    'multi_scale_3d_strides': {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                }
                
                if i == 0:
                    batch_dict.update(encoded_spconv_tensor)
                    batch_dict.update(multi_scale_3d_features)
                    batch_dict.update(multi_scale_3d_strides)
                else:
                    encoded_spconv_tensor_trans.append(encoded_spconv_tensor)
                    multi_scale_3d_features_trans.append(multi_scale_3d_features)
                    multi_scale_3d_strides_trans.append(multi_scale_3d_strides)

            batch_dict['encoded_spconv_tensor_trans'] = encoded_spconv_tensor_trans
            batch_dict['multi_scale_3d_features_trans'] = multi_scale_3d_features_trans
            batch_dict['multi_scale_3d_strides_trans'] = multi_scale_3d_strides_trans
   
        return batch_dict


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        if self.training:
            batch_dict = self.forward_train(batch_dict)
        else:
            
            if 'trans_dict' in batch_dict:
                batch_dict = self.forward_test(batch_dict)
            else:
                batch_dict = self.forward_train(batch_dict)
        

        return batch_dict

class VoxelBackBone8xTiny(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_img = self.model_cfg.get('USE_IMG', False)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(32, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.num_point_features = 64
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 32,
            'x_conv4': 32
        }
        
        self.use_dense = self.model_cfg.get("USE_DENSE", False)

    def extract_features(self, 
                         batch_dict, 
                         voxel_features, 
                         voxel_coords):
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        encoded_spconv_tensor = {
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        }
        
        multi_scale_3d_features = {
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        }

        multi_scale_3d_strides = {
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        }

        return encoded_spconv_tensor, multi_scale_3d_features, multi_scale_3d_strides

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, voxel_features, voxel_coords)
        batch_dict.update(encoded_spconv_tensor_dict)
        batch_dict.update(multi_scale_3d_features_dict)
        batch_dict.update(multi_scale_3d_strides_dict)        

        if 'trans_dict' in batch_dict:
            encoded_spconv_tensor_rot = []
            multi_scale_3d_features_rot = []
            multi_scale_3d_strides_rot = []

            voxel_features_rotated_list, voxel_coords_rotated_list = batch_dict['voxel_features_rotated'], batch_dict['voxel_coords_rotated']
            for v, c in zip(voxel_features_rotated_list, voxel_coords_rotated_list):
                encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, v, c)
                encoded_spconv_tensor_rot.append(encoded_spconv_tensor_dict)
                multi_scale_3d_features_rot.append(multi_scale_3d_features_dict)
                multi_scale_3d_strides_rot.append(multi_scale_3d_strides_dict)
            
            batch_dict['encoded_spconv_tensor_rot'] = encoded_spconv_tensor_rot
            batch_dict['multi_scale_3d_features_rot'] = multi_scale_3d_features_rot
            batch_dict['multi_scale_3d_strides_rot'] = multi_scale_3d_strides_rot
        
        return batch_dict

class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.channels = self.model_cfg.get('CHANNELS', [16, 32, 64, 64])

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, self.channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(self.channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(self.channels[0], self.channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(self.channels[0], self.channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(self.channels[0], self.channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(self.channels[1], self.channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(self.channels[1], self.channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(self.channels[1], self.channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(self.channels[2], self.channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(self.channels[2], self.channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(self.channels[2], self.channels[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(self.channels[3], self.channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(self.channels[3], self.channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(self.channels[3], 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': self.channels[0],
            'x_conv2': self.channels[1],
            'x_conv3': self.channels[2],
            'x_conv4': self.channels[3]
        }

    def extract_features(self, 
                         batch_dict, 
                         voxel_features, 
                         voxel_coords,
                         is_mm=False):
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        if not is_mm:
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)
        else:
            x = self.conv_input_mm(input_sp_tensor)

            x_conv1 = self.conv1_mm(x)
            x_conv2 = self.conv2_mm(x_conv1)
            x_conv3 = self.conv3_mm(x_conv2)
            x_conv4 = self.conv4_mm(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            # out = self.conv_out_mm(x_conv4)
            out = None
        
        encoded_spconv_tensor = {
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        }
        
        multi_scale_3d_features = {
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        }

        multi_scale_3d_strides = {
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        }

        return encoded_spconv_tensor, multi_scale_3d_features, multi_scale_3d_strides
    
    def decompose_tensor(self, tensor, i, batch_size, num_trans):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // (num_trans+1))
        end_shape_ids = (i + 1) * (input_shape // (num_trans+1))
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // (num_trans+1))
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // (num_trans+1)]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor
    
    def reverse_transformation(self, batch_dict, multi_scale_3d_features, rot_angle, flip_x, flip_y, scale):
        
        for k, src_name in enumerate(['x_conv3', 'x_conv4']):
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = multi_scale_3d_features[src_name]
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
        
        import pdb;pdb.set_trace()
       
    def forward_train(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, voxel_features, voxel_coords)
        batch_dict.update(encoded_spconv_tensor_dict)
        batch_dict.update(multi_scale_3d_features_dict)
        batch_dict.update(multi_scale_3d_strides_dict)
        
        if 'mm_points' in batch_dict:
            voxel_features, voxel_coords = batch_dict['voxel_mm_features'], batch_dict['voxel_mm_coords']
            encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, 
                                                                                                                          voxel_features, 
                                                                                                                          voxel_coords, 
                                                                                                                          True)
            batch_dict.update({'encoded_spconv_tensor_mm' : encoded_spconv_tensor_dict['encoded_spconv_tensor']})
            batch_dict.update({'multi_scale_3d_mm_features' : multi_scale_3d_features_dict['multi_scale_3d_features']})
                        
        if 'trans_dict' in batch_dict:

            trans_dict = batch_dict['trans_dict'][0]
            rot_angle_list = trans_dict['rot_angle_list']
            flip_x_list = trans_dict['flip_x_list']
            scale_list = trans_dict['scale_list']
            num_trans = trans_dict['num_trans']
                
            encoded_spconv_tensor_trans = []
            multi_scale_3d_features_trans = []
            multi_scale_3d_strides_trans = []
            
            if 'mm_points' in batch_dict:
                encoded_spconv_tensor_mm_trans = []
                multi_scale_3d_features_mm_trans = []
                multi_scale_3d_strides_mm_trans = []   
            
            trans_dict = batch_dict['trans_dict'][0]
            num_trans = trans_dict['num_trans']
            for i in range(num_trans):
                voxel_features, voxel_coords = batch_dict['voxel_features_' + str(i+1)], batch_dict['voxel_coords_' + str(i+1)]
                encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, voxel_features, voxel_coords)
                
                encoded_spconv_tensor_trans.append(encoded_spconv_tensor_dict)
                multi_scale_3d_features_trans.append(multi_scale_3d_features_dict)
                multi_scale_3d_strides_trans.append(multi_scale_3d_strides_dict)
                
                if 'mm_points' in batch_dict:
                    voxel_features, voxel_coords = batch_dict['voxel_mm_features_' + str(i+1)], batch_dict['voxel_mm_coords_' + str(i+1)]
                    encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, 
                                                                                                                                  voxel_features, 
                                                                                                                                  voxel_coords,
                                                                                                                                  True)
                    encoded_spconv_tensor_mm_trans.append(encoded_spconv_tensor_dict)
                    multi_scale_3d_features_mm_trans.append(multi_scale_3d_features_dict)

            batch_dict['encoded_spconv_tensor_trans'] = encoded_spconv_tensor_trans
            batch_dict['multi_scale_3d_features_trans'] = multi_scale_3d_features_trans
            batch_dict['multi_scale_3d_strides_trans'] = multi_scale_3d_strides_trans

            if 'mm_points' in batch_dict:      
                multi_scale_3d_strides_mm_trans.append(multi_scale_3d_strides_dict)
                batch_dict['encoded_spconv_tensor_mm_trans'] = encoded_spconv_tensor_mm_trans
                batch_dict['multi_scale_3d_features_mm_trans'] = multi_scale_3d_features_mm_trans
                batch_dict['multi_scale_3d_strides_mm_trans'] = multi_scale_3d_strides_mm_trans
        
        return batch_dict

    def forward_test(self, batch_dict):

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

        all_lidar_feat = []
        all_lidar_coords = []
        trans_dict = batch_dict['trans_dict'][0]
        num_trans = trans_dict['num_trans']     
        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * (num_trans + 1)]

        all_lidar_feat.append(voxel_features)
        new_coord = voxel_coords.clone()
        new_coord[:, 3] += 0*self.sparse_shape[2]
        all_lidar_coords.append(new_coord)

        if 'trans_dict' in batch_dict:

            encoded_spconv_tensor_trans = []
            multi_scale_3d_features_trans = []
            multi_scale_3d_strides_trans = []

            trans_dict = batch_dict['trans_dict'][0]
            num_trans = trans_dict['num_trans']
            for i in range(num_trans):
                voxel_features, voxel_coords = batch_dict['voxel_features_' + str(i+1)], batch_dict['voxel_coords_' + str(i+1)]
                all_lidar_feat.append(voxel_features)
                new_coord = voxel_coords.clone()
                new_coord[:, 3] += (i+1)*self.sparse_shape[2]
                all_lidar_coords.append(new_coord)

                # encoded_spconv_tensor_dict, multi_scale_3d_features_dict, multi_scale_3d_strides_dict = self.extract_features(batch_dict, voxel_features, voxel_coords)
                
                # encoded_spconv_tensor_trans.append(encoded_spconv_tensor_dict)
                # multi_scale_3d_features_trans.append(multi_scale_3d_features_dict)
                # multi_scale_3d_strides_trans.append(multi_scale_3d_strides_dict)
                
            all_lidar_feat = torch.cat(all_lidar_feat, 0)
            all_lidar_coords = torch.cat(all_lidar_coords)
            batch_size = batch_dict['batch_size']

            input_sp_tensor = spconv.SparseConvTensor(
            features=all_lidar_feat,
            indices=all_lidar_coords.int(),
            spatial_shape=new_shape,
            batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)
            out = self.conv_out(x_conv4)

            for i in range(num_trans + 1):
                this_conv2 = self.decompose_tensor(x_conv2, i, batch_size, num_trans)
                this_conv3 = self.decompose_tensor(x_conv3, i, batch_size, num_trans)
                this_conv4 = self.decompose_tensor(x_conv4, i, batch_size, num_trans)
                this_out = self.decompose_tensor(out, i, batch_size, num_trans)

                encoded_spconv_tensor = {
                        'encoded_spconv_tensor': this_out,
                        'encoded_spconv_tensor_stride': 8
                    }
                    
                multi_scale_3d_features = {
                    'multi_scale_3d_features': {
                        'x_conv1': None,
                        'x_conv2': this_conv2,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    }
                }

                multi_scale_3d_strides = {
                    'multi_scale_3d_strides': {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                }
                
                if i == 0:
                    batch_dict.update(encoded_spconv_tensor)
                    batch_dict.update(multi_scale_3d_features)
                    batch_dict.update(multi_scale_3d_strides)
                else:
                    encoded_spconv_tensor_trans.append(encoded_spconv_tensor)
                    multi_scale_3d_features_trans.append(multi_scale_3d_features)
                    multi_scale_3d_strides_trans.append(multi_scale_3d_strides)

            batch_dict['encoded_spconv_tensor_trans'] = encoded_spconv_tensor_trans
            batch_dict['multi_scale_3d_features_trans'] = multi_scale_3d_features_trans
            batch_dict['multi_scale_3d_strides_trans'] = multi_scale_3d_strides_trans
   
        return batch_dict


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        if self.training or 'trans_dict' not in batch_dict:
            batch_dict = self.forward_train(batch_dict)
        else:
            batch_dict = self.forward_test(batch_dict)
        

        return batch_dict