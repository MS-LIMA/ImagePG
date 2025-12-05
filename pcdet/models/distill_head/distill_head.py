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
from pcdet.models.fusion_modules.deform_fusion import DeformAttnFusion

class DistillHead(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.bev_channels = self.model_cfg.BEV_CHANNELS
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.batch_dict = {}
        
        self.pseuo_param = nn.Identity(1)
        # self.attention = nn.MultiheadAttention(self.bev_channels, 1, batch_first=True)
   
    def get_loss(self, tb_dict=None):
        
        loss = 0
        
        batch_dict_teacher = self.batch_dict['batch_dict_teacher']
        batch_dict = self.batch_dict
        
        teacher_features = batch_dict_teacher['spatial_features_2d'].detach()
        student_features = batch_dict['spatial_features_2d']
        
        mimic_mode = 'gt'
        if mimic_mode == 'gt':
            rois = batch_dict['gt_boxes']
        elif mimic_mode == 'roi_t':
            rois = batch_dict['rois_teacher'].detach()
        elif mimic_mode == 'roi_s':
            rois = batch_dict['rois_student'].detach()

        min_x = self.point_cloud_range[0]
        min_y = self.point_cloud_range[1]
        voxel_size_x = self.voxel_size[0]
        voxel_size_y = self.voxel_size[1]
        down_sample_ratio = 8

        batch_size, height, width = teacher_features.size(0), teacher_features.size(2), teacher_features.size(3)
        
        # mask = batch_dict['gt_hm'].squeeze(1)
        loss_mimic_bev_total = torch.norm(teacher_features - student_features, p=2, dim=1)
        loss_mimic_bev_total = torch.mean(loss_mimic_bev_total)
        
        loss += loss_mimic_bev_total
        tb_dict['loss_mimic_bev'] = loss_mimic_bev_total
            
        pooled_features_t = batch_dict_teacher['pooled_features'].detach()
        pooled_features_s = batch_dict['pooled_features']
        
        
        
        import pdb;pdb.set_trace()
            
            # loss_mimic_bev = (loss_mimic_bev_total * mask).sum(1).sum(1) / (rois[..., -1] > 0).sum(1) / (mask > 0).sum(1).sum(1)
            # loss_mimic_bev = torch.mean(loss_mimic_bev)
            
            # tb_dict['loss_mimic_bev'] = loss_mimic_bev    
            
            # feat_teacher_roi = teacher_features * mask.unsqueeze(1).repeat(1, teacher_features.shape[1], 1, 1)
            # feat_student_roi = student_features * mask.unsqueeze(1).repeat(1, teacher_features.shape[1], 1, 1)
            
            # feat_teacher_roi = feat_teacher_roi.permute(0, 2, 3, 1).view(feat_teacher_roi.shape[0], -1, self.bev_channels)
            # feat_student_roi = feat_student_roi.permute(0, 2, 3, 1).view(feat_student_roi.shape[0], -1, self.bev_channels)
            
            # # teacher_features = teacher_features.permute(0, 2, 3, 1).view(teacher_features.shape[0], -1, self.bev_channels)
            # student_features = student_features.permute(0, 2, 3, 1).view(student_features.shape[0], -1, self.bev_channels)
            
            # import pdb;pdb.set_trace()
            
            # attn_output_t = self.attention(feat_teacher_roi, student_features, student_features)[0]
            # attn_output_s = self.attention(feat_student_roi, student_features, student_features)[0]
            
            # roi_size = rois.size(1)

            # x1 = (rois[:, :, 0] - rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            # x2 = (rois[:, :, 0] + rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            # y1 = (rois[:, :, 1] - rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            # y2 = (rois[:, :, 1] + rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            # #print(height, width,x1.min(),x2.max(),y1.min(),y2.max())
            # mask = torch.zeros(batch_size, roi_size, height, width).bool().cuda()
            # grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
            # grid_y = grid_y[None, None].repeat(batch_size, roi_size, 1, 1).cuda()
            # grid_x = grid_x[None, None].repeat(batch_size, roi_size, 1, 1).cuda()

            # mask_y = (grid_y >= y1[:, :, None, None]) * (grid_y <= y2[:, :, None, None])
            # mask_x = (grid_x >= x1[:, :, None, None]) * (grid_x <= x2[:, :, None, None])
            # mask = (mask_y * mask_x).float()
            # if mimic_mode == 'gt':
            #     mask[rois[:,:,-1] == 0] = 0
            # weight = mask.sum(-1).sum(-1) #bz * roi
            # weight[weight == 0] = 1
            # mask = mask / weight[:, :, None, None]

            # mimic_loss = torch.norm(teacher_features - student_features, p=2, dim=1)
            # #mimic_loss_pillar = torch.norm(pillar_features_2d_t - pillar_features_2d_s, p=2, dim=1)

            # mask = mask.sum(1)
            # if mimic_mode == 'gt':
                
            #     mask[mask > 0] = 1
            
            #     mask_list = []
            #     for b in range(batch_size):
            #         indices = torch.nonzero(mask[b] == 1)
            #         x_coords = indices[:, 1]
            #         y_coords = indices[:, 0]

            #         c_center = torch.cat((y_coords[:, None], x_coords[:, None]), dim=-1)
            #         radius = int(10)
                
            #         mask_batch = mask[b]
            #         mask_batch = sy_draw_gaussian_to_heatmap(mask_batch.unsqueeze(0), c_center, radius, 0).squeeze(0)
            #         mask_list.append(mask_batch)
                    
            #         # plt.imshow(mask_batch.cpu().numpy())
            #         # plt.show()
                    
            #     mask = torch.stack(mask_list, dim=0)
                
            #     loss = (mimic_loss * mask).sum(dim=1).sum(dim=1) / (mask[:, ...]  > 0).sum(dim=1).sum(dim=1) / (rois[:, :, -1] > 0).sum(dim=1)
            #     mimic_loss = torch.mean(loss)
                
            #     # for b in range(batch_size):
                    
            #     #     f, axes = plt.subplots(2,1) 

            #     #     axes[0].imshow(gt_bev_mask[b].cpu().numpy())
            #     #     axes[1].imshow(mask[b].cpu().numpy())
            #     #     plt.show()
                
            #     disp_dict['loss_mimic'] = mimic_loss    
            #     #mimic_loss_pillar = (mimic_loss_pillar * mask).sum() / (rois[:,:,-1] > 0).sum()
            # else:
            #     mimic_loss = (mimic_loss * mask).sum() / batch_size / roi_size
            #     #mimic_loss_pillar = (mimic_loss_pillar * mask).sum() / batch_size / roi_size

        # elif mimic_mode == 'all':
        #     mimic_loss = torch.mean(torch.norm(teacher_features - student_features, p=2, dim=1))
        # # mimic_loss_pillar = torch.mean(torch.norm(pillar_features_2d_t - pillar_features_2d_s, p=2, dim=1))
        # else:
        #     raise NotImplementedError

        return loss, tb_dict
    
    def forward(self, batch_dict, **kwargs):
        self.batch_dict = batch_dict
        return batch_dict