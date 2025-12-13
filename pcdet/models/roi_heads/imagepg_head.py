from tty import setraw
import torch
import torch.nn as nn

from pcdet.models.fusion_modules.adf_block import ADF, AttentiveFusion
from pcdet.models.model_utils.attention_utils import FeedForwardPositionalEncoding
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_np_ops
from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils, loss_utils, box_utils, voxel_aggregation_utils
from .roi_head_template_pg import RoIHeadTemplatePG, RoIHeadTemplate
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ..model_utils.attention_utils import TransformerEncoder, TransformerDecoder, get_positional_encoder
import numpy as np
import torch
from torch import Tensor
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from torch.nn import functional as F
from tools.visual_utils.open3d_vis_utils import Visualizer3D
from pcdet.models.fusion_modules.deform_fusion import DeformAttnFusion, DeformTransLayer, DeformAttnFusionMultiImage
from pcdet.models.backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch, sector_fps, sample_points_with_roi
from ..model_utils import model_nms_utils
from easydict import EasyDict as edict
from pcdet.datasets.augmentor.augmentor_utils_mbm import _rotation_matrix_3d_

class PositionalEmbedding(nn.Module):
    def __init__(self, demb=256):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    # pos_seq =  pos_seq = torch.arange(seq_len-1, -1, -1.0)
    def forward(self, pos_seq, batch_size=2):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        else:
            return pos_emb[:, None, :]

class CrossAttention(nn.Module):

    def __init__(self, hidden_dim, pos = True, head = 4):
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.pos_dim = 8
        self.pos = pos

        if self.pos:
            self.pos_en = PositionalEmbedding(self.pos_dim)

            self.Q_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
        else:

            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.att = nn.MultiheadAttention(hidden_dim, head)


    def forward(self, inputs, Q_in): # N,B,C

        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]

        if self.pos:
            pos_input = torch.from_numpy(np.arange(seq_len)+1).cuda()
            pos_input = self.pos_en(pos_input, batch_size)
            inputs_pos = torch.cat([inputs, pos_input], -1)
            pos_Q = torch.from_numpy(np.array([seq_len])).cuda()
            pos_Q = self.pos_en(pos_Q, batch_size)
            Q_in_pos = torch.cat([Q_in, pos_Q], -1)
        else:
            inputs_pos = inputs
            Q_in_pos = Q_in

        Q = self.Q_linear(Q_in_pos)
        K = self.K_linear(inputs_pos)
        V = self.V_linear(inputs_pos)

        out = self.att(Q, K, V)

        return out[0]

class Attention_Layer(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs): # B,K,N


        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs).permute(0, 2, 1)
        V = self.V_linear(inputs)

        alpha = torch.matmul(Q, K)

        alpha = F.softmax(alpha, dim=2)

        out = torch.matmul(alpha, V)

        out = torch.mean(out, -2)

        return out
    
class ImagePGHead(RoIHeadTemplatePG):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        self.attention_cfg = model_cfg.TRANSFORMER
        self.point_cfg = model_cfg.POINT_FEATURE_CONFIG
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        use_bn = self.model_cfg.USE_BN
        self.use_multi_image = self.model_cfg.get('USE_MULTI_IMAGE', False)
        self.SA_modules = nn.ModuleList()
        
        self.use_dense = self.model_cfg.get('USE_DENSE', False)
        self.use_rot = self.model_cfg.get("USE_ROT", False)
        self.use_mm = self.model_cfg.get('USE_MM', False)
        
        # RoI Grid Pooling
        c_out = 0
        self.pool_raw_points = False
        self.keypoint_sampling = False
        self.pool_method = self.pool_cfg.get('POOL_MODEL', 'voxel_query')
        self.roi_grid_pool_layers = nn.ModuleList()
        for i, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            if src_name == 'bev':
                channels = LAYER_cfg[src_name].CHANNELS
                self.bev_conv = []
                for k in range(len(channels) - 1):
                    self.bev_conv.append(nn.Conv2d(channels[k], 
                                                   channels[k+1], kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False))
                    self.bev_conv.append(nn.BatchNorm2d(channels[k+1], eps=0.001, momentum=0.01, affine=True, track_running_stats=True))
                    self.bev_conv.append(nn.ReLU())
                self.bev_conv = nn.Sequential(*self.bev_conv)
                c_out += channels[-1]
            elif src_name == 'raw_points':
                num_points_feats = LAYER_cfg[src_name].NUM_PTS_FEATS
                self.num_pts_feats = num_points_feats
                self.pool_raw_points = True
                
                if self.pool_cfg.KEYPOINTS.ENABLED:
                    self.agg_keypoints_layer, c = pointnet2_stack_modules.build_local_aggregation_module(
                        input_channels=num_points_feats - 3, config=self.pool_cfg.KEYPOINTS)
                    self.keypoint_sampling = True
                else:
                    c = num_points_feats - 3

                self.roi_grid_points_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                    input_channels=c, config=LAYER_cfg[src_name])
                c_out += num_c_out
            else:
                layer_cfg = LAYER_cfg[src_name]
                
                if self.pool_method == 'acc_pointnet':
                    layer_cfg = layer_cfg.PN
                    mlps = layer_cfg.MLPS
                    for k in range(len(mlps)):
                        mlps[k] = [backbone_channels[src_name]] + mlps[k]
                        
                    pool_layer = pointnet2_stack_modules.AcceleratedStackSAModuleMSG(
                        radii=layer_cfg.POOL_RADIUS,
                        nsamples=layer_cfg.NSAMPLE,
                        mlps=mlps,
                        use_xyz=True,
                        pool_method=layer_cfg.POOL_METHOD
                    )
                elif self.pool_method == 'voxel_query':
                    layer_cfg = layer_cfg.VOX
                    mlps = layer_cfg.MLPS
                    for k in range(len(mlps)):
                        mlps[k] = [backbone_channels[src_name]] + mlps[k]
                        
                    pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                        query_ranges=layer_cfg.QUERY_RANGES,
                        nsamples=layer_cfg.NSAMPLE,
                        radii=layer_cfg.POOL_RADIUS,
                        mlps=mlps,
                        pool_method=layer_cfg.POOL_METHOD,
                    )
                
                c_out += sum([x[-1] for x in mlps])
                self.roi_grid_pool_layers.append(pool_layer)
        
        c_out_roi_grid_pooling = c_out
        
        if self.use_mm:
            self.roi_grid_pool_layers_mm = nn.ModuleList()
            for i, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
                if src_name == 'bev':
                    channels = LAYER_cfg[src_name].CHANNELS
                    self.bev_conv = []
                    for k in range(len(channels) - 1):
                        self.bev_conv.append(nn.Conv2d(channels[k], 
                                                    channels[k+1], kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False))
                        self.bev_conv.append(nn.BatchNorm2d(channels[k+1], eps=0.001, momentum=0.01, affine=True, track_running_stats=True))
                        self.bev_conv.append(nn.ReLU())
                    self.bev_conv = nn.Sequential(*self.bev_conv)
                    c_out += channels[-1]
                elif src_name == 'raw_points':
                    num_points_feats = LAYER_cfg[src_name].NUM_PTS_FEATS
                    self.num_pts_feats = num_points_feats
                    self.pool_raw_points = True
                    
                    if self.pool_cfg.KEYPOINTS.ENABLED:
                        self.agg_keypoints_layer, c = pointnet2_stack_modules.build_local_aggregation_module(
                            input_channels=num_points_feats - 3, config=self.pool_cfg.KEYPOINTS)
                        self.keypoint_sampling = True
                    else:
                        c = num_points_feats - 3

                    self.roi_grid_points_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                        input_channels=c, config=LAYER_cfg[src_name])
                    c_out += num_c_out
                else:
                    layer_cfg = LAYER_cfg[src_name]
                    
                    if self.pool_method == 'acc_pointnet':
                        layer_cfg = layer_cfg.PN
                        mlps = layer_cfg.MLPS
                        for k in range(len(mlps)):
                            mlps[k] = [backbone_channels[src_name]] + mlps[k]
                            
                        pool_layer = pointnet2_stack_modules.AcceleratedStackSAModuleMSG(
                            radii=layer_cfg.POOL_RADIUS,
                            nsamples=layer_cfg.NSAMPLE,
                            mlps=mlps,
                            use_xyz=True,
                            pool_method=layer_cfg.POOL_METHOD
                        )
                    elif self.pool_method == 'voxel_query':
                        layer_cfg = layer_cfg.VOX
                        mlps = layer_cfg.MLPS
                        for k in range(len(mlps)):
                            mlps[k] = [backbone_channels[src_name]] + mlps[k]
                            
                        pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                            query_ranges=layer_cfg.QUERY_RANGES,
                            nsamples=layer_cfg.NSAMPLE,
                            radii=layer_cfg.POOL_RADIUS,
                            mlps=mlps,
                            pool_method=layer_cfg.POOL_METHOD,
                        )
                    
                    c_out += sum([x[-1] for x in mlps])
                    self.roi_grid_pool_layers_mm.append(pool_layer)
        
        if self.pool_cfg.FUSE_ALL_FEATS:
            self.fuse_vp_feats_layer = nn.Sequential(
                nn.Linear(c_out_roi_grid_pooling, self.pool_cfg.FUSE_FEATS),
                nn.ReLU()
            )
            c_out_roi_grid_pooling = self.pool_cfg.FUSE_FEATS

        # RoI Voxel-Point Fusion
        # self.deform_attn_point_voxel_layers = nn.ModuleList()
        # for i in range(len(self.pool_cfg.FEATURES_SOURCE)):
        #     self.deform_attn_point_voxel_layers.append(DeformAttnFusion(mid_channels=64))

        # Transformer Encoder
        self.pos_encoder_attn, input_feats = get_positional_encoder(self.attention_cfg)
        
        self.attention_head = TransformerEncoder(self.attention_cfg.ENCODER)
        self.use_img = self.model_cfg.get('USE_IMG', True)
        if self.use_img:
            img_feats = self.pool_cfg.IMG_FEATURES
            self.ffn = nn.Sequential(
                nn.Conv1d(input_feats, img_feats // 2, 1),
                nn.BatchNorm1d(img_feats // 2),
                nn.ReLU(),
                nn.Conv1d(img_feats // 2, img_feats, 1),
            )

            c_out = self.attention_cfg.ENCODER.NUM_FEATURES
            self.up_ffn = nn.Sequential(
                nn.Conv1d(img_feats, c_out // 2, 1),
                nn.BatchNorm1d(c_out // 2),
                nn.ReLU(),
                nn.Conv1d(c_out // 2, c_out, 1),
            )

            if not self.use_multi_image:
                self.deform_attn_grid = DeformAttnFusion(mid_channels=img_feats, light=True)
            else:
                # self.deform_attn_grid = DeformAttnFusionMultiImage(mid_channels=img_feats, light=True)
                self.deform_attn_grid = DeformAttnFusion(mid_channels=img_feats)

        # Point Generation
        c_out = self.attention_cfg.ENCODER.NUM_FEATURES
        gen_fc_list = []
        for k in range(0, self.model_cfg.GEN_FC.__len__()):
            gen_fc_list.extend([
                nn.Linear(c_out, self.model_cfg.GEN_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.GEN_FC[k]),
                nn.ReLU()
            ])
            c_out = self.model_cfg.GEN_FC[k]
            if k != self.model_cfg.GEN_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                gen_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        gen_fc_list.append(nn.Linear(c_out, (3 + self.point_cfg.POINT_FEATURE_NUM) * self.point_cfg.NUM_POINTS, bias=True))
        self.gen_fc_layers= nn.Sequential(*gen_fc_list)
    
        self.point_pred_layer = nn.Linear(self.point_cfg.POINT_FEATURE_NUM, self.num_class, bias=True)
        
        # Generated Points Feature Extraction
        self.num_prefix_channels = 3        # x, y, z
        if self.point_cfg.USE_DEPTH:
            self.num_prefix_channels += 1   # d
        if self.point_cfg.USE_SCORE:
            self.num_prefix_channels += 1   # s
            
        xyz_mlps = [self.num_prefix_channels] + self.point_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        if self.point_cfg.get('MERGE_DOWN', True):
            c_out = self.point_cfg.XYZ_UP_LAYER[-1]
            self.merge_down_layer = nn.Sequential(
                nn.Conv2d(c_out + self.point_cfg.POINT_FEATURE_NUM, c_out, kernel_size=1, bias=not use_bn),
                # nn.Conv2d(c_out + self.point_cfg.POINT_FEATURE_NUM + self.point_cfg.POINT_IMG_FEAT_NUM, c_out, kernel_size=1, bias=not use_bn),
                *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
            )
        else:
            c_out = self.point_cfg.XYZ_UP_LAYER[-1] + self.point_cfg.POINT_FEATURE_NUM
        c_merge_down = c_out
        #if self.use_img:
        #    self.deform_attn_pts = DeformAttnFusion(mid_channels=c_merge_down, light=True)

        for k in range(self.point_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [c_out] + self.point_cfg.SA_CONFIG.MLPS[k]

            npoint = self.point_cfg.SA_CONFIG.NPOINTS[k] if self.point_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.point_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.point_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            c_out = mlps[-1]

        if self.model_cfg.BOX_EMBEDDING:
            self.box_embedding = nn.Sequential(
                nn.Linear(self.box_coder.code_size, c_out // 2, 1),
                nn.BatchNorm1d(c_out // 2),
                nn.ReLU(c_out // 2),
                nn.Linear(c_out // 2, c_out, 1),
            )
             
        # Shard FC
        self.refine_stages = self.model_cfg.get('REFINE_STAGES', 1)
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out_roi_grid_pooling

        if self.refine_stages > 1:
            self.cross_attention_layers = Attention_Layer(c_out)
        else:
            self.cross_attention_layers = None
 
        if self.model_cfg.get("SHARED_FC", None) is not None and self.use_img:
            shared_fc_list = []
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.model_cfg.SHARED_FC[k]

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        else:
            self.shared_fc_layer = None
            self.shared_fc_layer_mm = None        
        # Confidence Head
        pre_channel = c_out
        if self.cross_attention_layers:
            pre_channel *= 2
        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)
   
        # Regression Head
        pre_channel = c_out
        if self.cross_attention_layers:
            pre_channel *= 2
        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)
        
        if self.use_mm:
            self.mm_fuse_conv = nn.Sequential(
                nn.Conv1d(c_out_roi_grid_pooling * 2, c_out_roi_grid_pooling, 1, 1),
                nn.BatchNorm1d(c_out_roi_grid_pooling),
                nn.ReLU(),
                nn.Conv1d(c_out_roi_grid_pooling, c_out_roi_grid_pooling, 1, 1),
                nn.BatchNorm1d(c_out_roi_grid_pooling),
                nn.ReLU(),)
            if self.shared_fc_layer is not None:
                self.mm_fuse_conv_vox = nn.Sequential(
                nn.Conv1d(c_out_roi_grid_pooling * 2, c_out_roi_grid_pooling, 1, 1),
                nn.BatchNorm1d(c_out_roi_grid_pooling),
                nn.ReLU(),
                nn.Conv1d(c_out_roi_grid_pooling, c_out_roi_grid_pooling, 1, 1),
                nn.BatchNorm1d(c_out_roi_grid_pooling),
                nn.ReLU(),)
        
        self.init_weights()
        
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.use_img:
            fc_list = [self.gen_fc_layers, self.cls_fc_layers, self.reg_fc_layers, self.ffn, self.up_ffn]
        else:
            fc_list = [self.gen_fc_layers, self.cls_fc_layers, self.reg_fc_layers]
        if self.model_cfg.BOX_EMBEDDING:
            fc_list.append(self.box_embedding)
        for module_list in fc_list:
            for m in module_list.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    
        nn.init.normal_(self.point_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.point_pred_layer.bias, 0)
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)
                
        for p in self.attention_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def nms(self, boxes, scores):
        pred_scores, cur_pred_labels, pred_boxes, selected = model_nms_utils.multi_classes_nms(
            cls_scores=scores.sigmoid(), box_preds=boxes,
            nms_config=edict(dict(
                MULTI_CLASSES_NMS= False,
                NMS_TYPE= 'nms_gpu',
                NMS_THRESH= 0.1,
                NMS_PRE_MAXSIZE= 4096,
                NMS_POST_MAXSIZE= 500
            )),
            score_thresh=0.1,
            return_indices=True
        )
        
        return pred_boxes, pred_scores, selected
               
    # RoI Pooling #
    # ==================================== #
    def roi_grid_pts_pool(self, batch_dict, roi_grid_xyz):

        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
                    
        xyz = points[:, 1:4]
        xyz_features = points[:, 4:]      
        batch_idx_pts = points[:, 0]              

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx_pts == k).sum()

        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, roi_grid_xyz.shape[-1])
        new_xyz = roi_grid_xyz.view(-1, roi_grid_xyz.shape[-1])[..., 1:4]
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(roi_grid_xyz.shape[1])

        pooled_points, pooled_features = self.roi_grid_points_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz.contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=xyz_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)

        return pooled_features

    def roi_grid_pool(self, batch_dict):
        """ 
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        
        roi_grid_xyz, roi_grid_xyz_rel = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = torch.div((roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]), self.voxel_size[0])
        roi_grid_coords_y = torch.div((roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]), self.voxel_size[1])
        roi_grid_coords_z = torch.div((roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]), self.voxel_size[2])
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = torch.div(roi_grid_coords,  cur_stride, rounding_mode='trunc')
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features, roi_grid_xyz_rel, roi_grid_xyz, None
    
    def rotation_matrices_3d(self, angles, axis=2):
        """
        Vectorized로 3D 회전 행렬들을 계산합니다.
        
        Args:
            angles (np.ndarray): 회전각 배열, shape (N, ), 단위: 라디안.
            axis (int): 회전축. 
                axis==2 또는 -1인 경우, 회전 행렬은 
                [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]] 의 형태입니다.
                axis==1이면 [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]] 형태,
                axis==0이면 [[1, 0, 0], [0, cos, -sin], [0, sin, cos]] 형태로 생성합니다.
        
        Returns:
            np.ndarray: shape (N, 3, 3) 회전 행렬 배열.
        """
        N = angles.shape[0]
        cosines = np.cos(angles)
        sines = np.sin(angles)
        
        # 기본적으로 identity 행렬을 복제하여 생성
        rot_mats = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
        
        if axis == 2 or axis == -1:
            rot_mats[:, 0, 0] = cosines
            rot_mats[:, 0, 1] = -sines
            rot_mats[:, 1, 0] = sines
            rot_mats[:, 1, 1] = cosines
        elif axis == 1:
            rot_mats[:, 0, 0] = cosines
            rot_mats[:, 0, 2] = -sines
            rot_mats[:, 2, 0] = sines
            rot_mats[:, 2, 2] = cosines
        elif axis == 0:
            rot_mats[:, 1, 1] = cosines
            rot_mats[:, 1, 2] = -sines
            rot_mats[:, 2, 1] = sines
            rot_mats[:, 2, 2] = cosines
        else:
            raise ValueError("axis는 0, 1 또는 2여야 합니다.")
        
        return rot_mats

    def roi_pool_img_grid(self, batch_dict, targets_dict, roi_grid_xyz, pooled_features, rot_angle, flip_x, flip_y, scale):

        B = batch_dict['batch_size']

        # if 'roi_proj' in batch_dict:
        #     points_proj = batch_dict['roi_proj']
        # else:
        roi_grid_xyz = roi_grid_xyz.view(-1, roi_grid_xyz.shape[-1])
        roi_grid_xyz = common_utils.transform_points(roi_grid_xyz.clone().detach(), rot_angle, flip_x, flip_y, scale, forward=False)
        rois = common_utils.transform_bboxes(batch_dict['rois'].clone(), rot_angle, flip_x, flip_y, scale, forward=False)
    
        pooled_features_img = pooled_features.new_zeros(rois.shape[0] * rois.shape[1], self.pool_cfg.GRID_SIZE**3, pooled_features.shape[-1])
        pooled_features_img = pooled_features_img.view(-1, pooled_features_img.shape[-1])
        
        if 'TTA' in batch_dict:
            aug_rot_angle = batch_dict['aug_rot_angle']
            aug_scale = batch_dict['aug_scale'].item()
            aug_flip_x = (batch_dict['aug_flip_x'] == 1.0).item()
            
            roi_grid_xyz = common_utils.transform_points(roi_grid_xyz.clone().detach(), aug_rot_angle, aug_flip_x, False, aug_scale, forward=False)
            rois = common_utils.transform_bboxes(batch_dict['rois'].clone(), aug_rot_angle, aug_flip_x, False, aug_scale, forward=False)
        
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
            
            for i in range(B):
                
                img = images[i][img_idx]
                pts = roi_grid_xyz[roi_grid_xyz[..., 0] == i][..., 1:4]
                
                if 'local_noise_trans' in batch_dict or 'local_noise_rotation' in batch_dict:

                    rois_b = rois[i]
                    
                    if 'local_noise_trans' in batch_dict:
                        local_noise_trans = torch.tensor(batch_dict['local_noise_trans'][i]).to(pts)
                        gt_idx_of_rois = targets_dict['gt_idx_of_rois'][i]
                        local_noise_trans = local_noise_trans[gt_idx_of_rois.long()]

                        pts = pts.view(-1, self.pool_cfg.GRID_SIZE**3, 3)
                        pts = pts - local_noise_trans.unsqueeze(1).repeat(1, pts.shape[1], 1)
                        pts = pts.view(-1, 3)

                        rois_b[..., :3] = rois_b[..., :3] - local_noise_trans

                    if 'local_noise_rotation' in batch_dict:
                        local_noise_rotation = torch.tensor(batch_dict['local_noise_rotation'][i]).to(pts)
                        gt_idx_of_rois = targets_dict['gt_idx_of_rois'][i]
                        local_noise_rotation = local_noise_rotation[gt_idx_of_rois.long()]
                        
                        pts = pts.view(-1, self.pool_cfg.GRID_SIZE**3, 3)
                        rois_b = rois_b.unsqueeze(1).repeat(1, pts.shape[1], 1)

                        pts = pts - rois_b[..., :3]   

                        rot_mat_T = self.rotation_matrices_3d(-local_noise_rotation.cpu().numpy())
                        pts = torch.bmm(pts, torch.tensor(rot_mat_T).to(pts))

                        pts = pts + rois_b[..., :3]   
                        pts = pts.view(-1, 3)
                        
                calib = batch_dict['calib'][i]    

                noise_translation = batch_dict['noise_translation'][i] if 'noise_translation' in batch_dict else torch.tensor([0, 0, 0], device=pts.device)
                noise_rotation = batch_dict['noise_rotation'][i].view(1) if 'noise_rotation' in batch_dict else torch.tensor(0.0, device=pts.device).view(1)
                noise_scale = batch_dict['noise_scale'][i] if 'noise_scale' in batch_dict else torch.tensor(1.0, device=pts.device).view(1)
                flip_x = batch_dict['flip_x'][i] if 'flip_x' in batch_dict else False
                flip_y = batch_dict['flip_y'][i] if 'flip_y' in batch_dict else False
                
                img_scale = batch_dict['img_scale'][i]
                pad_size = batch_dict['pad_size'][i]
                
                C, H, W = img.shape
                #if D > 1:
                #    img_scale = img_scale / 2
                        
                points_proj_batch, depth = common_utils.project_points_to_img_batch(img=img,
                                                                        img_scale=img_scale,
                                                                        pad_size=pad_size,
                                                                        points=pts,
                                                                        calib=calib,
                                                                        noise_translation=noise_translation,
                                                                        noise_rotation=noise_rotation,
                                                                        noise_scale=noise_scale,
                                                                        flip_x=flip_x,
                                                                        flip_y=flip_y,
                                                                        only_points=D > 1,
                                                                        cam_id=img_idx if D > 1 else -1)
                # points_proj_batch = torch.cat([points_proj_batch, depth.view(-1, 1)], dim=-1)

                image_mask = (points_proj_batch[:, 0] >= 0) & (points_proj_batch[:, 0] < W) & \
                        (points_proj_batch[:, 1] >= 0) & (points_proj_batch[:, 1] < H) & (depth >= 0)
                image_mask_total.append(image_mask)

                #if not self.use_multi_image:
                #    points_proj_batch = points_proj_batch[image_mask]
                points_proj_batch = points_proj_batch[image_mask]

                bidx = points_proj_batch.new_zeros(points_proj_batch.shape[0]).fill_(i)
                points_proj_batch = torch.cat([bidx.unsqueeze(-1), points_proj_batch], dim=-1)
                points_proj.append(points_proj_batch)
                
                # if img is not None:
                #     img = common_utils.unnormalize_img(img)
                #     plt.imshow(img.cpu().numpy().transpose((1,2,0)))
                #     plt.scatter(points_proj_batch[:, 1].cpu().numpy(), points_proj_batch[:, 2].cpu().numpy(), c='red', s=3.5)
                #     plt.show()
            
            points_proj = torch.cat(points_proj)
            image_mask_total = torch.cat(image_mask_total)   
            image_mask_total_list.append(image_mask_total)
            points_proj_cam_list.append(points_proj)
            
            # if not self.use_multi_image:
            pooled_features = pooled_features.view(-1, pooled_features.shape[-1])   
            pooled_features_img[image_mask_total] = self.deform_attn_grid(batch_dict, pooled_features[image_mask_total], points_proj, cam_idx=img_idx if D > 1 else -1)
        
        #if self.use_multi_image:    
        #    pooled_features = pooled_features.view(-1, pooled_features.shape[-1])   
        #    pooled_features_img = self.deform_attn_grid(batch_dict, pooled_features, points_proj_cam_list, image_mask_total_list)
                
        pooled_features_img = pooled_features_img.view(-1, self.pool_cfg.GRID_SIZE**3, pooled_features_img.shape[-1])
            
        return pooled_features_img

    def roi_pool_img_pts(self, batch_dict, roi_grid_xyz, pooled_features, rot_angle, flip_x, flip_y, scale):

        B = batch_dict['batch_size']

        points_proj = pooled_features.new_zeros((B, (roi_grid_xyz.shape[0] // B) * roi_grid_xyz.shape[1], 3))
        roi_grid_xyz = roi_grid_xyz.view(-1, roi_grid_xyz.shape[-1])
        roi_grid_xyz = common_utils.transform_points(roi_grid_xyz.clone().detach(), rot_angle, flip_x, flip_y, scale, forward=False)
        
        for i in range(B):
            points_proj[i, :, 0] = i

        for i in range(B):
            
            img = batch_dict['images'][i]
            pts = roi_grid_xyz[roi_grid_xyz[..., 0] == i][..., 1:4]
            calib = batch_dict['calib'][i]    
            
            noise_rotation = batch_dict['noise_rotation'][i].view(1) if 'noise_rotation' in batch_dict else torch.tensor(0.0, device=pts.device).view(1)
            noise_scale = batch_dict['noise_scale'][i] if 'noise_scale' in batch_dict else torch.tensor(1.0, device=pts.device).view(1)
            flip_x = batch_dict['flip_x'][i] if 'flip_x' in batch_dict else False
            
            img_scale = batch_dict['img_scale'][i]
            pad_size = batch_dict['pad_size'][i]
            
            points_proj_batch = common_utils.project_points_to_img_batch(img=img,
                                                                    img_scale=img_scale,
                                                                    pad_size=pad_size,
                                                                    points=pts,
                                                                    calib=calib,
                                                                    noise_rotation=noise_rotation,
                                                                    noise_scale=noise_scale,
                                                                    flip_x=flip_x)
            points_proj[i, :, 1:] = points_proj_batch
            
            # if img is not None:
            #     img = common_utils.unnormalize_img(img)
            #     plt.imshow(img.cpu().numpy().transpose((1,2,0)))
            #     plt.scatter(points_proj_batch[:, 0].cpu().numpy(), points_proj_batch[:, 1].cpu().numpy(), c='red', s=0.5)
            #     plt.show()
            
        batch_dict['roi_proj'] = points_proj

        points_proj = points_proj.view(-1, points_proj.shape[-1])

        pooled_features_img = self.deform_attn_pts(batch_dict, pooled_features.permute(0, 2, 1).contiguous(), points_proj)
        pooled_features_img = pooled_features_img.view(-1, self.pool_cfg.GRID_SIZE**3, pooled_features_img.shape[-1])
        return pooled_features_img
    
    def roi_grid_pool_rtspconv(self, batch_dict, refine_idx, points, rois, multi_scale_3d_feats, is_mm=False):

        """ 
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, roi_grid_xyz_rel = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = torch.div((roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]), self.voxel_size[0])
        roi_grid_coords_y = torch.div((roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]), self.voxel_size[1])
        roi_grid_coords_z = torch.div((roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]), self.voxel_size[2])
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):

            if src_name == 'bev':
                bev_feats = self.bev_conv(batch_dict['spatial_features'])
                idx = torch.arange(batch_size, device=roi_grid_xyz.device).view(-1, 1)\
                    .repeat(1, roi_grid_xyz.shape[1]).view(-1, 1)
                keypoints_bev = torch.cat((idx.float(), roi_grid_xyz.view(-1, 3)), dim=1)
                point_bev_features = self.interpolate_from_bev_features(
                    keypoints_bev, bev_feats, batch_dict['batch_size'],
                    bev_stride=batch_dict['spatial_features_stride']
                )
                point_bev_features = point_bev_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    point_bev_features.shape[-1]
                )
                pooled_features_list.append(point_bev_features)
            else:
                if not is_mm:
                    pool_layer = self.roi_grid_pool_layers[k]
                else:
                    pool_layer = self.roi_grid_pool_layers_mm[k]
                    
                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
                cur_sp_tensors = multi_scale_3d_feats[src_name]

                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor
                v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = torch.div(roi_grid_coords, cur_stride, rounding_mode='trunc')
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation

                if src_name in batch_dict:
                    features = batch_dict[src_name]
                    features = features[features[..., 0] == refine_idx][..., 1:]
                    pooled_features = pool_layer.forward_fast(
                        xyz=cur_voxel_xyz.contiguous(),
                        xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                        new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                        new_xyz_batch_cnt=roi_grid_batch_cnt,
                        new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                        features=features.contiguous(),
                        voxel2point_indices=v2p_ind_tensor
                    )
                else:
                    pooled_features = pool_layer(
                        xyz=cur_voxel_xyz.contiguous(),
                        xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                        new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                        new_xyz_batch_cnt=roi_grid_batch_cnt,
                        new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                        features=cur_sp_tensors.features.contiguous(),
                        voxel2point_indices=v2p_ind_tensor
                    )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features, roi_grid_xyz_rel, roi_grid_xyz, None

    def bilinear_interpolate_torch_gridsample(self, image, samples_x, samples_y):
        C, H, W = image.shape
        image = image.unsqueeze(1)  # change to:  C x 1 x H x W        C,K,1,2   C,K,1,1

        samples_x = samples_x.unsqueeze(2)
        samples_x = samples_x.unsqueeze(3)# 49,K,1,1
        samples_y = samples_y.unsqueeze(2)
        samples_y = samples_y.unsqueeze(3)

        samples = torch.cat([samples_x, samples_y], 3)
        samples[:, :, :, 0] = (samples[:, :, :, 0] / W)  # normalize to between  0 and 1

        samples[:, :, :, 1] = (samples[:, :, :, 1] / H)  # normalize to between  0 and 1
        samples = samples * 2 - 1  # normalize to between -1 and 1  # 49,K,1,2

        #B,C,H,W
        #B,H,W,2
        #B,C,H,W

        return torch.nn.functional.grid_sample(image, samples, align_corners=False)
    
    def obtain_conf_preds(self, confi_im, anchors):

        confi = []

        for i, im in enumerate(confi_im):
            boxes = anchors[i]
            im = confi_im[i]
            if len(boxes) == 0:
                confi.append(torch.empty(0).type_as(im))
            else:
                (xs, ys) = self.gen_grid_fn(boxes)
                out = self.bilinear_interpolate_torch_gridsample(im, xs, ys)
                x = torch.mean(out, 0).view(-1, 1)
                confi.append(x)

        confi = torch.cat(confi)

        return confi

    def roi_part_pool(self, batch_dict, parts_feat):
        rois = batch_dict['rois'].clone()
        confi_preds = self.obtain_conf_preds(parts_feat, rois)

        return confi_preds
           
    # ==================================== #

    #region Utils
    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
            """
            Args:
                keypoints: (N1 + N2 + ..., 4)
                bev_features: (B, C, H, W)
                batch_size:
                bev_stride:

            Returns:
                point_bev_features: (N1 + N2 + ..., C)
            """
            x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
            y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]

            x_idxs = x_idxs / bev_stride
            y_idxs = y_idxs / bev_stride

            point_bev_features_list = []
            for k in range(batch_size):
                bs_mask = (keypoints[:, 0] == k)
                cur_x_idxs = x_idxs[bs_mask]
                cur_y_idxs = y_idxs[bs_mask]
                cur_bev_features = bev_features[k].permute(1, 2, 0)
                point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
                point_bev_features_list.append(point_bev_features)

            point_bev_features = torch.cat(point_bev_features_list, dim=0)
            return point_bev_features
                       
    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def get_positional_input(self, points, rois, roi_labels, local_roi_grid_points):

        from pcdet.utils import density_utils
        import math

        if self.attention_cfg.POSITIONAL_ENCODER == 'grid_points':
            positional_input = local_roi_grid_points
        elif self.attention_cfg.POSITIONAL_ENCODER == 'grid_points_corners':
            local_rois = rois.view(-1, rois.shape[-1]).clone()
            local_rois[:, 0:3] = 0
            local_corners = box_utils.boxes_to_corners_3d(local_rois)
            positional_input_corners = (local_corners.unsqueeze(1) - local_roi_grid_points.unsqueeze(2)).reshape(*local_roi_grid_points.shape[:-1], -1)
            positional_input = torch.cat([positional_input_corners, local_roi_grid_points], dim=-1)
        elif self.attention_cfg.POSITIONAL_ENCODER == 'density_grid_points':
            points_per_part = density_utils.find_num_points_per_part_multi(points,
                                                                           rois,
                                                                           self.model_cfg.ROI_GRID_POOL.GRID_SIZE,
                                                                           20,
                                                                           #self.pool_cfg.DENSITYQUERY.MAX_NUM_BOXES,
                                                                           return_centroid=True)
            points_per_part_num_features = 1 if len(points_per_part.shape) <= 5 else points_per_part.shape[-1]
            points_per_part = points_per_part.view(points_per_part.shape[0]*points_per_part.shape[1], -1, points_per_part_num_features).float()
            # First feature is density, other potential features are xyz
            points_per_part[..., 0] = torch.log10(points_per_part[..., 0] + 0.5) - (math.log10(0.5) if self.model_cfg.get('DENSITY_LOG_SHIFT') else 0)
            # positional_input = torch.cat((local_roi_grid_points, points_per_part), dim=-1)
            positional_input = torch.cat((local_roi_grid_points, points_per_part[..., :1]), dim=-1)
        elif self.attention_cfg.POSITIONAL_ENCODER == 'density_grid_points_corners':
            local_rois = rois.view(-1, rois.shape[-1]).clone()
            local_rois[:, 0:3] = 0
            local_corners = box_utils.boxes_to_corners_3d(local_rois)
            positional_input_corners = (local_corners.unsqueeze(1) - local_roi_grid_points.unsqueeze(2)).reshape(*local_roi_grid_points.shape[:-1], -1)
            points_per_part = density_utils.find_num_points_per_part_multi(points,
                                                                           rois,
                                                                           self.model_cfg.ROI_GRID_POOL.GRID_SIZE,
                                                                           20,
                                                                           #self.pool_cfg.DENSITYQUERY.MAX_NUM_BOXES,
                                                                           return_centroid=True)
            points_per_part_num_features = 1 if len(points_per_part.shape) <= 5 else points_per_part.shape[-1]
            points_per_part = points_per_part.view(points_per_part.shape[0]*points_per_part.shape[1], -1, points_per_part_num_features).float()
            # First feature is density, other potential features are xyz
            points_per_part[..., 0] = torch.log10(points_per_part[..., 0] + 0.5) - (math.log10(0.5) if self.model_cfg.get('DENSITY_LOG_SHIFT') else 0)
            positional_input = torch.cat((positional_input_corners, local_roi_grid_points, points_per_part), dim=-1)
        else:
            positional_input = None

        rois = rois.view(-1, rois.shape[-1])
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        bs, box_num = roi_labels.shape
        bidx = torch.arange(bs).repeat_interleave(box_num).to(rois.device)
        bidx = bidx.reshape(-1, 1, 1).expand(-1, positional_input.shape[1], 1)
        
        global_roi_grid_points = torch.cat([bidx, global_roi_grid_points], dim=-1)  # [bidx, x, y, z]

        return positional_input, global_roi_grid_points

    #endregion
    def extract_features(self, batch_dict, refine_idx, points, rot_angle, flip_x, flip_y, scale, multi_scale_3d_feats, transform=True, is_mm=False):
        
        batch_size = batch_dict['batch_size']
        if 'rois_org' not in batch_dict:

            if not self.training:
                batch_dict['rois_org'] = batch_dict['rois'].clone()
                batch_dict['roi_labels_org'] = batch_dict['roi_labels'].clone()
                batch_dict['roi_scores_org'] = batch_dict['roi_scores'].clone()                

        if transform:
            if 'gt_boxes' in batch_dict:
                if 'gt_boxes_org' not in batch_dict:
                    batch_dict['gt_boxes_org'] = batch_dict['gt_boxes'].clone()
                gt_boxes = batch_dict['gt_boxes_org'].clone()
                gt_boxes = common_utils.transform_bboxes(gt_boxes, rot_angle, flip_x, flip_y, scale)
                gt_boxes[gt_boxes[..., -1]==0] = 0.0      
                batch_dict['gt_boxes'] = gt_boxes
            if 'bm_points' in batch_dict:
                if 'bm_points_org' not in batch_dict:
                    batch_dict['bm_points_org'] = batch_dict['bm_points'].clone()
                bm_points = batch_dict['bm_points_org'].clone()
                bm_points = common_utils.transform_points(bm_points, rot_angle, flip_x, flip_y, scale)
                batch_dict['bm_points'] = bm_points

            rois = common_utils.transform_bboxes(batch_dict['rois'], rot_angle, flip_x, flip_y, scale)
            batch_dict['rois'] = rois

        if self.training:
            if transform == True:
                targets_dict = self.assign_targets(batch_dict, refine_idx)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
            else:
                targets_dict = None
        else:
            targets_dict = None

        pooled_features_vox, roi_grid_xyz_rel, _, _ = self.roi_grid_pool_rtspconv(batch_dict, refine_idx, points, batch_dict['rois'], multi_scale_3d_feats, is_mm)  # (BxN, 6x6x6, C)
        positional_input, global_roi_grid_points = self.get_positional_input(points, batch_dict['rois'], batch_dict['roi_labels'], roi_grid_xyz_rel)

        if self.use_img:
            pooled_features_img = self.ffn(positional_input.permute(0, 2, 1)).permute(0, 2, 1).contiguous() # (B, GGG, C)
            pooled_features_img = pooled_features_img.reshape(-1, pooled_features_img.shape[-1])
            pooled_features_img = self.roi_pool_img_grid(batch_dict, targets_dict, global_roi_grid_points, pooled_features_img, rot_angle, flip_x, flip_y, scale)
            pooled_features_img = self.up_ffn(pooled_features_img.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            pooled_features = pooled_features_vox + pooled_features_img
        else:
            pooled_features = pooled_features_vox
            pooled_features_img = None
    
        # positional_input = positional_input[..., 0:4]
        positional_embedding = self.pos_encoder_attn(positional_input)

        positional_embedding = positional_embedding.reshape(-1, self.pool_cfg.GRID_SIZE ** 3, positional_embedding.shape[-1])
        attention_output = self.attention_head(pooled_features + positional_embedding).contiguous() + pooled_features

        ret_dict = {
            'pooled_features_vox' : pooled_features_vox,
            'pooled_features_img' : pooled_features_img,
            'positional_embedding' : positional_embedding,
            'attention_output' : attention_output,
            'global_roi_grid_points' : global_roi_grid_points,
            'points' : points
        }
        
        return ret_dict, targets_dict
        
    def extract_shared_features(self, batch_dict, feats_dict, rot_angle, flip_x, flip_y, scale, is_mm=False):
        
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        
        points = feats_dict['points']
        pooled_features_vox = feats_dict['pooled_features_vox']
        pooled_features_img = feats_dict['pooled_features_img']
        positional_embedding = feats_dict['positional_embedding']
        attention_output = feats_dict['attention_output']        
        global_roi_grid_points = feats_dict['global_roi_grid_points']
        
        # Point Generation
        bs, GGG, _ = pooled_features_vox.shape
        gen_output = self.gen_fc_layers(attention_output.view(-1, attention_output.shape[-1]))

        gen_output = gen_output.reshape(bs, pooled_features_vox.shape[1] * self.point_cfg.NUM_POINTS, -1)   # (BxN, 6x6x6xP, 3 + C')
        gen_points_offset = gen_output[..., :3]        # (BxN, 6x6x6xP, 3)
        if self.point_cfg.get('CANONICAL_OFFSET', False):
            gen_points_offset = common_utils.rotate_points_along_z(
                gen_points_offset, batch_dict['rois'].view(-1, batch_dict['rois'].shape[-1])[:, 6]
            )
        gen_points_features = gen_output[..., 3:]    # (BxN, 6x6x6xP, C')
    
        gen_points_score = self.point_pred_layer(gen_points_features)  # (BxN, 6x6x6xP, 1)

        # global_rebuilt_points
        gen_points_xyz = global_roi_grid_points.repeat(1, self.point_cfg.NUM_POINTS, 1).clone()
        gen_points_xyz[..., 1:4] = gen_points_xyz[..., 1:4] + gen_points_offset # (BxN, 6x6x6xP, bxyz)
   
        # canonical transform
        xyz_local = gen_points_xyz[..., 1:4] - batch_dict['rois'].reshape(bs, 1, -1)[..., :3]
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local, -batch_dict['rois'].view(-1, batch_dict['rois'].shape[-1])[:, 6]
        )
        
        rebuilt_points = torch.cat([gen_points_xyz, gen_points_score], dim=-1)  # (BxN, 6x6x6xP, bxyzs)
        batch_dict['rebuilt_points'] = rebuilt_points
        
        # generated points feature extraction
        xyz_input = [xyz_local]
        if self.point_cfg.USE_DEPTH:
            point_depths = gen_points_xyz[..., 1:4].norm(dim=-1) / self.point_cfg.DEPTH_NORMALIZER - 0.5
            xyz_input.append(point_depths.unsqueeze(-1))
        if self.point_cfg.USE_SCORE:
            xyz_input.append(torch.sigmoid(gen_points_score))
            
        xyz_input = torch.cat(xyz_input, dim=-1).transpose(1, 2).unsqueeze(dim=3).contiguous()
        xyz_features = self.xyz_up_layer(xyz_input.clone().detach())

        if self.point_cfg.SCORE_WEIGHTING:
            xyz_features = xyz_features * torch.sigmoid(gen_points_score).unsqueeze(1)
        
        merged_features = torch.cat((xyz_features, 
                                     gen_points_features.transpose(1, 2).unsqueeze(dim=3)), dim=1)
        
        if self.point_cfg.get('MERGE_DOWN', True):
            merged_features = self.merge_down_layer(merged_features)

        #merged_features = self.roi_pool_img_pts(batch_dict, rebuilt_points, merged_features.squeeze(-1), rot_angle, flip_x, False, scale)
        #merged_features = merged_features.permute(0, 2, 1).unsqueeze(-1)

        threshold = self.point_cfg.get('POINT_THRESHOLD', 0)
        xyz_local[torch.sigmoid(gen_points_score.squeeze(-1)) < threshold] = 0
        l_xyz, l_features = [xyz_local.contiguous()], [merged_features.squeeze(dim=3).contiguous()]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        shared_features = l_features[-1].squeeze(dim=-1)
        if self.model_cfg.BOX_EMBEDDING:
            shared_features += self.box_embedding(batch_dict['rois'].view(bs, -1))
        
        if self.use_img and self.shared_fc_layer is not None:
            shared_features_vp = self.shared_fc_layer(pooled_features_vox.view(bs, -1))
            shared_features = shared_features + shared_features_vp

        return shared_features
    
    def forward(self, batch_dict):
        
        # print('roi')
        """
        :param input_data: input dict
        :return:
        """

        if 'rois' not in batch_dict:
            targets_dict = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            ) # RPN에서 나온 proposal들 중에서 sampling
        
        # RoI aware pooling
        bs = batch_dict['batch_size']

        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_shared_features = []
        all_shared_features_mm = []

        if self.training:
            self.forward_ret_dict = []
            self.forward_ret_dict_mm = []
            self.forward_ret_dict_final = []

        if 'trans_dict' in batch_dict or True:

            if 'trans_dict' in batch_dict:
                trans_dict = batch_dict['trans_dict'][0]
                rot_angle_list = trans_dict['rot_angle_list']
                flip_x_list = trans_dict['flip_x_list']
                flip_y_list = trans_dict['flip_y_list']
                scale_list = trans_dict['scale_list']
                num_trans = trans_dict['num_trans']

                points_list = []
                points_mm_list = []
                multi_scale_3d_features_list = []
                multi_scale_3d_features_mm_list = []
                
                for i in range(num_trans):
                    points_trans = batch_dict['points'].clone()
                    points_trans = common_utils.transform_points(points_trans, 
                                                                torch.tensor(rot_angle_list[i]).to(points_trans).view([1]), 
                                                                flip_x_list[i], 
                                                                flip_y_list[i],
                                                                scale_list[i])
                    points_list.append(points_trans)
                    multi_scale_3d_features_list.append(batch_dict['multi_scale_3d_features_trans'][i]['multi_scale_3d_features'])
                    
                    if 'mm_points' in batch_dict:
                        points_trans = batch_dict['mm_points'].clone()
                        points_trans = common_utils.transform_points(points_trans, 
                                                                    torch.tensor(rot_angle_list[i]).to(points_trans).view([1]), 
                                                                    flip_x_list[i], 
                                                                    False, 
                                                                    scale_list[i])
                        points_mm_list.append(points_trans)
                        multi_scale_3d_features_mm_list.append(batch_dict['multi_scale_3d_features_mm_trans'][i]['multi_scale_3d_features'])
                    else:
                        points_mm_list.append(None)
                        multi_scale_3d_features_mm_list.append(None)    
                
                rot_angle_list = rot_angle_list + [0.0] 
                flip_x_list = flip_x_list + [False] 
                flip_y_list = flip_y_list +  [False]
                scale_list = scale_list + [1.0] 
                points_list =  points_list + [batch_dict['points']]
                multi_scale_3d_features_list =  multi_scale_3d_features_list + [batch_dict['multi_scale_3d_features']]
                points_mm_list.append(batch_dict['mm_points'] if 'mm_points' in batch_dict else None)
                multi_scale_3d_features_mm_list.append(batch_dict['multi_scale_3d_mm_features'] if 'mm_points' in batch_dict else None)

            else:

                rot_angle_list = []
                flip_x_list = []
                flip_y_list= []
                scale_list = []
                points_list = []
                multi_scale_3d_features_list = []
                points_mm_list = []
                multi_scale_3d_features_mm_list = []
                
                for i in range(self.refine_stages):
                    rot_angle_list.append(0.0)
                    flip_x_list.append(False)
                    flip_y_list.append(False)
                    scale_list.append(1.0)
                    points_list.append(batch_dict['points'])
                    multi_scale_3d_features_list.append(batch_dict['multi_scale_3d_features'])
                    points_mm_list.append(batch_dict['mm_points'] if 'mm_points' in batch_dict else None)
                    multi_scale_3d_features_mm_list.append(batch_dict['multi_scale_3d_mm_features'] if 'mm_points' in batch_dict else None)

            for refine_idx, (points, mm_points, rot_angle, flip_x, flip_y, scale, multi_scale_3d_feats, multi_scale_3d_mm_feats) in \
                enumerate(zip(points_list, points_mm_list, rot_angle_list, flip_x_list, flip_y_list, scale_list, multi_scale_3d_features_list, multi_scale_3d_features_mm_list)):
                
                rot_angle = torch.tensor(rot_angle).to(points).view([1])

                feats_dict, targets_dict = self.extract_features(batch_dict, refine_idx, points, rot_angle, flip_x, flip_y, scale, multi_scale_3d_feats, transform=True)
                if self.use_mm:
                    feats_dict_mm, _ = self.extract_features(batch_dict, refine_idx, mm_points, rot_angle, flip_x, flip_y, scale, multi_scale_3d_mm_feats, transform=False, is_mm=True)
                    attn = feats_dict['attention_output']
                    attn_mm = feats_dict_mm['attention_output']
                    attn = torch.cat([attn, attn_mm],dim=-1)
                    attn = self.mm_fuse_conv(attn.permute(0, 2, 1)).permute(0, 2, 1)
                    feats_dict['attention_output'] = attn.contiguous()
                    
                    if self.shared_fc_layer is not None:
                        feats_vox = feats_dict['pooled_features_vox']
                        feats_vox_mm = feats_dict_mm['pooled_features_vox']
                        feats_vox = torch.cat([feats_vox, feats_vox_mm],dim=-1)
                        feats_vox = self.mm_fuse_conv_vox(feats_vox.permute(0, 2, 1)).permute(0, 2, 1)
                        feats_dict['pooled_features_vox'] = feats_vox.contiguous()
                    
                shared_features = self.extract_shared_features(batch_dict, feats_dict, rot_angle, flip_x, flip_y, scale)
                if self.cross_attention_layers:
                    shared_features = shared_features.unsqueeze(0)  # 1,B,C
                    all_shared_features.append(shared_features)
                    pre_feat = torch.cat(all_shared_features, 0)
                    attentive_cur_feat = self.cross_attention_layers(pre_feat.permute(1, 0, 2)).unsqueeze(0)
                    # attentive_cur_feat = self.cross_attention_layers[refine_idx](pre_feat, shared_features)
                    attentive_cur_feat = torch.cat([attentive_cur_feat, shared_features], -1)
                    attentive_cur_feat = attentive_cur_feat.squeeze(0)  # B, C*2
                else:
                    attentive_cur_feat = shared_features
                
                rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(attentive_cur_feat))
                rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(attentive_cur_feat))

                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(batch_size=batch_dict['batch_size'], 
                                                                                    rois=batch_dict['rois'], 
                                                                                    cls_preds=rcnn_cls, 
                                                                                    box_preds=rcnn_reg)
                if self.training:
                    
                    new_forward_dict = {}
                    new_forward_dict.update(targets_dict)

                    new_forward_dict['rcnn_cls'] = rcnn_cls
                    new_forward_dict['rcnn_reg'] = rcnn_reg
                    new_forward_dict['rebuilt_points'] = batch_dict['rebuilt_points']
                    # new_forward_dict['target_points_mask'] = batch_dict['bm_points_mask']
                    new_forward_dict['target_points'] = batch_dict['bm_points'].clone()
                    new_forward_dict['gt_boxes'] = batch_dict['gt_boxes'].clone()
                    new_forward_dict['batch_size'] = batch_dict['batch_size']                    

                    self.forward_ret_dict.append(new_forward_dict)
                    
                batch_box_preds = common_utils.transform_bboxes(batch_box_preds, rot_angle, flip_x, flip_y, scale, forward=False)
                batch_dict['rois'] = batch_box_preds.clone()
                batch_dict['roi_scores'] = batch_cls_preds.squeeze(-1)

                if not self.training:
                    all_pred_boxes.append(batch_box_preds.clone())
                    all_pred_scores.append(batch_cls_preds.clone())
                    all_pred_labels.append(batch_dict['roi_labels'].clone())

            if not self.training:
                
                if 'WBF' not in batch_dict:
                    all_pred_boxes = torch.stack(all_pred_boxes)
                    all_pred_scores = torch.stack(all_pred_scores)
                    final_pred_boxes = torch.mean(all_pred_boxes, 0)
                    final_pred_scores = torch.mean(all_pred_scores, 0)
                    
                    batch_dict['rois'] = batch_dict['rois_org']
                    batch_dict['roi_labels'] = batch_dict['roi_labels_org']
                    batch_dict['roi_scores'] = batch_dict['roi_scores_org']
                    batch_dict['cls_preds_normalized'] = False

                else:
                    final_pred_boxes = torch.cat(all_pred_boxes, dim=1)
                    final_pred_scores = torch.cat(all_pred_scores, dim=1)
                    final_pred_labels = torch.cat(all_pred_labels, dim=1)
                    batch_dict['roi_labels'] = final_pred_labels
                    batch_dict['cls_preds_normalized'] = False

                batch_dict['batch_box_preds'] = final_pred_boxes
                batch_dict['batch_cls_preds'] = final_pred_scores
                batch_dict['cls_preds_normalized'] = False

        return batch_dict
    
    def corners_to_img_bbox(self, calib, corners : Tensor, img_size, bbox_scale=1.25):
        
        proj_corners = torch.stack([torch.tensor(calib.lidar_to_img(c.cpu())[0], device=corners.device) for c in corners])
        
        H, W = img_size
        half_img_area = H * W * 0.5

        x_coords = torch.clamp(proj_corners[..., 0], 0, W - 1)
        y_coords = torch.clamp(proj_corners[..., 1], 0, H - 1)

        # Compute min and max values in batch
        min_x = torch.min(x_coords, dim=1)[0]
        min_y = torch.min(y_coords, dim=1)[0]
        max_x = torch.max(x_coords, dim=1)[0]
        max_y = torch.max(y_coords, dim=1)[0]

        # Compute image size and inside image mask
        img_size = (max_x - min_x) * (max_y - min_y)
        inside_img_mask = (img_size < half_img_area) & (img_size > 0)
        bboxes = torch.cat((min_x.view(-1, 1), 
                            min_y.view(-1, 1), 
                            max_x.view(-1, 1), 
                            max_y.view(-1, 1)), dim=1)
        
        # Compute valid indices
        valid_indices = torch.nonzero(inside_img_mask).squeeze().view(-1)

        # Scale bboxes
        r = bbox_scale  

        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2

        width = bboxes[:, 2] - bboxes[:, 0]
        height = bboxes[:, 3] - bboxes[:, 1]

        new_width = width * r
        new_height = height * r

        new_bboxes = torch.zeros_like(bboxes)
        new_bboxes[:, 0] = torch.clamp(cx - new_width / 2, 0, W - 1)
        new_bboxes[:, 1] = torch.clamp(cy - new_height / 2, 0, H - 1)
        new_bboxes[:, 2] = torch.clamp(cx + new_width / 2, 0, W - 1)
        new_bboxes[:, 3] = torch.clamp(cy + new_height / 2, 0, H - 1)
        
        return new_bboxes, valid_indices