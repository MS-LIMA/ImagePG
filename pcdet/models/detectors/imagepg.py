from .detector3d_template import Detector3DTemplate
from pcdet.utils import box_utils
from torchvision.ops import roi_pool
from .. import backbones_image
from ..backbones_image import img_neck
from .. import bev_heads
from .. import distill_head
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads

from pcdet.utils import common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import torch
from tools.visual_utils.open3d_vis_utils import Visualizer3D
from ..model_utils import model_nms_utils
from easydict import EasyDict as edict

class ImagePG(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)    
        
        self.module_topology = [
            'image_backbone', 'neck', 'vfe', 'backbone_3d_img', 'backbone_3d', 'map_to_bev_module', 'pfe', 'bev_heatmap_head',
            'backbone_2d', 'dense_head', 'point_head', 'roi_head', 'distill_head'
        ]

        self.module_list = self.build_networks()

    def forward(self, batch_dict, debug=False, return_points=True):
 
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }

            disp_dict['batch_dict'] = batch_dict
            return ret_dict, tb_dict, disp_dict
        else:

            if 'is_teacher' in batch_dict:
                pred_dicts, recall_dicts = None, None
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
            
            if return_points:
                point_dict = {
                    'rebuilt_points': batch_dict['rebuilt_points'],
                    'rois': batch_dict['rois'],
                    'roi_labels': batch_dict['roi_labels']
                }
                point_dict.update(batch_dict)
                return pred_dicts, recall_dicts, point_dict
            
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss = loss + loss_rpn  
        
        if self.roi_head:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss + loss_rcnn
        
        if self.point_head:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
            loss += loss_point          
            
        if hasattr(self.map_to_bev_module, 'get_loss'):
            loss_hm_bev, tb_dict = self.map_to_bev_module.get_loss(tb_dict)
            loss += loss_hm_bev
            
        if hasattr(self.backbone_2d, 'get_loss'):
            loss_hm_bev, tb_dict = self.backbone_2d.get_loss(tb_dict)
            loss += loss_hm_bev            

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
        
        if hasattr(self.bev_heatmap_head, 'get_loss'):
            loss_hm_bev, tb_dict = self.bev_heatmap_head.get_loss(tb_dict)
            loss += loss_hm_bev            
            
        if self.distill_head:
            loss_distill, tb_dict = self.distill_head.get_loss(tb_dict)
            loss += loss_distill
            
        return loss, tb_dict, disp_dict
    
    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = img_neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict
    
    def build_image_backbone(self, model_info_dict):
        if self.model_cfg.get('IMAGE_BACKBONE', None) is None:
            return None, model_info_dict
        image_backbone_module = backbones_image.__all__[self.model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=self.model_cfg.IMAGE_BACKBONE
        )
        image_backbone_module.init_weights()
        model_info_dict['module_list'].append(image_backbone_module)

        return image_backbone_module, model_info_dict
    
    def build_bev_heatmap_head(self,model_info_dict):
        if self.model_cfg.get('BEV_HEATMAP_HEAD', None) is None:
            return None, model_info_dict
        
        bevmodule = bev_heads.__all__[self.model_cfg.BEV_HEATMAP_HEAD.NAME](
            model_cfg=self.model_cfg.BEV_HEATMAP_HEAD,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
    
        model_info_dict['module_list'].append(bevmodule)

        return bevmodule, model_info_dict
    
    def build_backbone_3d_img(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D_IMG', None) is None:
            return None, model_info_dict

        backbone_3d_img_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D_IMG.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D_IMG,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_img_module)
        return backbone_3d_img_module, model_info_dict
    
    def build_distill_head(self,model_info_dict):
        if self.model_cfg.get('DISTILL_HEAD', None) is None:
            return None, model_info_dict
        
        distillmodule = distill_head.__all__[self.model_cfg.DISTILL_HEAD.NAME](
            model_cfg=self.model_cfg.DISTILL_HEAD,
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
        )
    
        model_info_dict['module_list'].append(distillmodule)

        return distillmodule, model_info_dict