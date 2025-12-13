import torch
import torch.nn as nn

from ....utils import common_utils
from ....utils.spconv_utils import replace_feature
from pcdet.models.fusion_modules.deform_fusion import DeformAttnFusion

from tools.visual_utils.open3d_vis_utils import Visualizer3D

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

        self.hm_cfg = self.model_cfg.get('ENCODE_IMG_FEATURES', None)
        if self.hm_cfg:
            self.voxel_size = kwargs.get('voxel_size')
            self.point_cloud_range = kwargs.get('point_cloud_range')
            self.deformattn_layers = nn.ModuleList()
            #self.proj_layers = nn.ModuleList()
            for src_name in self.hm_cfg.FEATURES_SOURCE:
                #LAYER_CFG = self.hm_cfg.POOL_LAYERS[src_name]
                
                self.deformattn_layers.append(DeformAttnFusion(mid_channels=self.hm_cfg.IMG_CHANNELS))
                #self.proj_layers.append(nn.Linear(LAYER_CFG.IN_CHANNELS, self.hm_cfg.IMG_CHANNELS))

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
        

        if self.hm_cfg:
            
            batch_size = batch_dict['batch_size']
            multi_scale_img_spatial_features = {}

            for i, src_name in enumerate(self.hm_cfg.FEATURES_SOURCE):
                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )

                points_proj = cur_voxel_xyz.new_zeros((cur_voxel_xyz.shape[0], 3))
                points_proj[..., 0] = cur_coords[..., 0]

                for b in range(batch_size):
                    batch_mask = cur_coords[..., 0] == b

                    pts = cur_voxel_xyz[batch_mask]
                    
                    # points = batch_dict['points']
                    # points = points[points[..., 0]==b][..., 1:]
                    # vis = Visualizer3D()
                    # vis.add_points(points.cpu().numpy())
                    # vis.add_points(pts, [255, 255, 0])
                    # vis.show()

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

                    # if img is not None:
                    #     plt.imshow(img.cpu().numpy().transpose((1,2,0)))
                    #     plt.scatter(points_proj_batch[:, 0].cpu().numpy(), points_proj_batch[:, 1].cpu().numpy(), c='red', s=0.5)
                    #     plt.show()     

                x_conv_features = cur_sp_tensors.features
                x_conv_features = self.deformattn_layers[i](batch_dict, x_conv_features, points_proj)

                dense_features = cur_sp_tensors.dense()
                N, C, D, H, W = dense_features.shape
                dense_features = dense_features.view(N, C * D, H, W)
                multi_scale_img_spatial_features[src_name] = dense_features
            batch_dict['multi_scale_img_spatial_features'] = multi_scale_img_spatial_features
            
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        return batch_dict
