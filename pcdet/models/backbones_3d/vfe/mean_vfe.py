import torch
from tools.visual_utils.open3d_vis_utils import Visualizer3D

from .vfe_template import VFETemplate
from ....utils import common_utils

class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.use_dense = model_cfg.get('USE_DENSE', False)

    def get_output_feature_dim(self):
        return self.num_point_features

    def extract_features(self, voxels, num_points):
        voxel_features, voxel_num_points = voxels, num_points      
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        return points_mean.contiguous()
    
    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """

        voxel_features = self.extract_features(batch_dict['voxels'], batch_dict['voxel_num_points'])
        batch_dict['voxel_features'] = voxel_features

        if 'mm_points' in batch_dict:
            voxel_features = self.extract_features(batch_dict['voxels_mm'], batch_dict['voxel_mm_num_points'])
            batch_dict['voxel_mm_features'] = voxel_features
 
        if 'trans_dict' in batch_dict:
            trans_dict = batch_dict['trans_dict'][0]
            num_trans = trans_dict['num_trans']
            for i in range(num_trans):
                voxel_features = self.extract_features(batch_dict['voxels_' + str(i+1)], 
                                                       batch_dict['voxel_num_points_' + str(i+1)])
                batch_dict['voxel_features_' + str(i+1)] = voxel_features
                
                if 'mm_points' in batch_dict:
                    voxel_features = self.extract_features(batch_dict['voxels_mm_' + str(i+1)], 
                                                        batch_dict['voxel_mm_num_points_' + str(i+1)])
                    batch_dict['voxel_mm_features_' + str(i+1)] = voxel_features
              
        # batch_size = batch_dict['batch_size']   
        # voxel_coords = batch_dict['voxel_dense_coords']
        # for b in range(batch_size):
        #     batch_mask = voxel_coords[..., 0] == b
        #     pts = points_mean[batch_mask]
        #     vis = Visualizer3D()
        #     vis.add_points(pts.cpu().numpy())
        #     vis.show()

        # batch_size = batch_dict['batch_size']   
        # points = batch_dict['points']
        # mask = voxel_features.sum(dim=2) != 0
        
        # points = batch_dict['points']   
        # batch_size = batch_dict['batch_size']
        # points_proj = points.new_zeros((points.shape[0], 3))
        # points_proj[..., 0] = points[..., 0]

        # for b in range(batch_size):
        #     batch_mask = points[..., 0] == b
        #     pts = points[batch_mask][..., 1:]

        #     img = batch_dict['images'][b]
        #     calib = batch_dict['calib'][b]    
            
        #     noise_rotation = batch_dict['noise_rotation'][b].view(1) if 'noise_rotation' in batch_dict else torch.tensor(0.0, device=pts.device).view(1)
        #     noise_scale = batch_dict['noise_scale'][b] if 'noise_scale' in batch_dict else torch.tensor(1.0, device=pts.device).view(1)
        #     flip_x = batch_dict['flip_x'][b] if 'flip_x' in batch_dict else False
            
        #     img_scale = batch_dict['img_scale'][b]
        #     pad_size = batch_dict['pad_size'][b]
            
        #     points_proj_batch = common_utils.project_points_to_img_batch(img=img,
        #                                                             img_scale=img_scale,
        #                                                             pad_size=pad_size,
        #                                                             points=pts,
        #                                                             calib=calib,
        #                                                             noise_rotation=noise_rotation,
        #                                                             noise_scale=noise_scale,
        #                                                             flip_x=flip_x)

        #     points_proj[batch_mask, 1:] = points_proj_batch 
            
        #     import matplotlib.pyplot as plt
        #     img = batch_dict['images'][b]
        #     img = common_utils.unnormalize_img(img)
        #     plt.imshow(img.cpu().numpy().transpose((1,2,0)))
        #     plt.scatter(points_proj_batch[:, 0].cpu().numpy(), points_proj_batch[:, 1].cpu().numpy(), c='red', s=0.5)
        #     plt.show()    

        return batch_dict
