from functools import partial

import torch
from .roi_head_template import RoIHeadTemplate
from pytorch3d.loss import chamfer_distance as chamfer_dist
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as point_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tools.visual_utils.open3d_vis_utils import Visualizer3D

from ...utils.loss_utils import ChamferDistance
from ...ops.iou3d_nms import iou3d_nms_utils


class RoIHeadTemplatePG(RoIHeadTemplate):
    
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.fps_num = self.model_cfg.LOSS_CONFIG.get('SAMPLE_POINTS', False)
        self.min_cd_num = self.model_cfg.LOSS_CONFIG.get("MIN_CD_POINTS", 10)
        self.loss_cos = nn.CosineEmbeddingLoss()
        self.loss_chamfer = ChamferDistance('l2')
        
        self.logit_scale_init_value = self.model_cfg.LOSS_CONFIG.get("LOGIT_SCALE", 2.6592)
        # self.logit_scale = nn.Parameter(torch.tensor(self.logit_scale_init_value))

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        l_2d = self.contrastive_loss(similarity)
        l_pooled = self.contrastive_loss(similarity.t())
        return (l_2d + l_pooled) / 2.0

    def get_clip_loss(self, forward_ret_dict):
        x_2d = forward_ret_dict['x_2d']
        x_pooled = forward_ret_dict['x_pooled']
        
        x_2d = x_2d / x_2d.norm(p=2, dim=-1, keepdim=True)
        x_pooled = x_pooled / x_pooled.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits -> (Batch_size, Batch_size)
        logit_scale = self.logit_scale.exp()
        logits_2d = torch.matmul(x_2d, x_pooled.t()) * logit_scale
        logits_pooled = logits_2d.t()
        
        loss = self.clip_loss(logits_2d) * 0.1
        tb_dict = {
            'cntrsv_loss' : loss.item()
        }
        return loss, tb_dict
    
    def cosine_loss(self, x: torch.Tensor):
        cosine_sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        loss = 1 - cosine_sim.triu(diagonal=1).mean()
        return loss
        
    def get_sim_loss(self, forward_ret_dict):
        
        x_merged = forward_ret_dict['x_merged']
        valid_mask = forward_ret_dict['reg_valid_mask'].bool().view(-1)
        gt_of_rois = forward_ret_dict['gt_of_rois']
        
        x_merged_fg = x_merged[valid_mask]
        x_merged_bg = x_merged[~valid_mask]
        gt_of_rois = gt_of_rois.view(gt_of_rois.shape[0] * gt_of_rois.shape[1], -1)[valid_mask]
        
        loss = 0
        for i in [1, 2, 3]:
            indices = (gt_of_rois[:, -1] == i).nonzero(as_tuple=False)
            x = x_merged_fg[indices]
            
            if x.shape[0] > 0:
                loss += (self.cosine_loss(x) / x.shape[0])
        if x_merged_bg.shape[0] > 0:
            loss += (self.cosine_loss(x_merged_bg) / x_merged_bg.shape[0])

        tb_dict = {
            'sim_loss' : loss.item()
        }
        return loss, tb_dict
    
    def get_point_cls_loss(self, forward_ret_dict, set_ignore_flag=True):
        
        if isinstance(forward_ret_dict, dict):
            forward_ret_dict_list = [forward_ret_dict]
        else:
            forward_ret_dict_list = forward_ret_dict
        
        total_loss = 0
        point_pos_num = 0
        
        for forward_ret_dict in forward_ret_dict_list:
            rebuilt_points = forward_ret_dict['rebuilt_points']
            gt_boxes = forward_ret_dict['gt_boxes']
            target_gt_idx = forward_ret_dict['gt_idx_of_rois']
            batch_size, roi_size = target_gt_idx.shape
                    
            # calculate point cls loss for rebuilt points
            rebuilt_points = rebuilt_points.reshape(batch_size, -1, rebuilt_points.shape[-1])
            if self.fps_num:
                sample_pt_idxs = point_utils.farthest_point_sample(
                    rebuilt_points[..., 1:4].contiguous(), self.fps_num
                ).long()
                sampled_points = sample_pt_idxs.new_zeros((batch_size, self.fps_num, 5), dtype=torch.float)
                for bidx in range(batch_size):
                    sampled_points[bidx] = rebuilt_points[bidx][sample_pt_idxs[bidx]]
            else:
                sampled_points = rebuilt_points
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                sampled_points[..., 1:4], gt_boxes[..., :-1]
            )
            
            box_fg_flag = (box_idxs_of_pts >= 0)
            point_cls_labels = box_fg_flag.new_zeros(box_fg_flag.shape)
            if set_ignore_flag:
                extend_gt_boxes = box_utils.enlarge_box3d(
                    gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
                ).view(batch_size, -1, gt_boxes.shape[-1])
                
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    sampled_points[..., 1:4], extend_gt_boxes[..., :-1]
                )
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels[ignore_flag] = -1
            for bidx in range(batch_size):
                gt_box_of_fg_points = torch.index_select(gt_boxes[bidx], 0, box_idxs_of_pts[bidx][fg_flag[bidx]])    
                point_cls_labels[bidx][fg_flag[bidx]] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels = point_cls_labels.view(-1)
            point_cls_preds = sampled_points[..., -1].view(-1, self.num_class)
            
            positives = (point_cls_labels > 0)
            negative_cls_weights = (point_cls_labels == 0) * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]
            cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
            point_loss_cls = cls_loss_src.sum()

            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss = point_loss_cls * loss_weights_dict['point_cls_weight']
            tb_dict = {
                'point_loss_cls': point_loss_cls.item(),
                'point_pos_num': pos_normalizer.item() / batch_size,
            }
        
            total_loss += point_loss
            point_pos_num += pos_normalizer.item() / batch_size
        
        # total_loss = total_loss / len(forward_ret_dict_list)
        point_pos_num /= len(forward_ret_dict_list)
        
        return total_loss, {'point_loss_cls' : total_loss.item(), 'point_pos_num' : point_pos_num}

    def get_chamfer_distance(self, forward_ret_dict):
        
        if isinstance(forward_ret_dict, dict):
            forward_ret_dict_list = [forward_ret_dict]
        else:
            forward_ret_dict_list = forward_ret_dict
        
        total_loss = 0
        
        for forward_ret_dict in forward_ret_dict_list:
            rebuilt_points = forward_ret_dict['rebuilt_points'].clone()
            target_points = forward_ret_dict['target_points'].clone()

            fg_mask = forward_ret_dict['reg_valid_mask']
            gt_boxes = forward_ret_dict['gt_boxes']
            target_gt_idx = forward_ret_dict['gt_idx_of_rois']
            batch_size, roi_size = target_gt_idx.shape
            tb_dict = {}
            
            cd_loss = 0
            rebuilt_points = rebuilt_points.reshape(batch_size, -1, *rebuilt_points.shape[-2:])
            
            # filter non-foreground
            if fg_mask.sum() == 0:
                tb_dict['cd_loss'] = cd_loss
                return cd_loss, tb_dict
            
            # collate target points
            # target_points_mask = rebuilt_points.new_tensor(forward_ret_dict['target_points_mask'])
            # target_points_mask = []
            # for bidx in range(batch_size):
            #     target_points_in_roi = roiaware_pool3d_utils.points_in_boxes_gpu(target_points[target_points[..., 0]==bidx][..., 1:4].unsqueeze(0), gt_boxes[bidx, ..., :7].unsqueeze(0))[0]
            #     target_points_mask.append(target_points_in_roi)

            # cd_loss = []
            # for (bidx, ridx) in fg_mask.nonzero():
            #     current_target_gt_idx = target_gt_idx[bidx, ridx]
            #     roi_source = rebuilt_points[bidx, ridx, ...]
            #     roi_target = target_points[target_points[..., 0]==bidx][target_points_mask[bidx]==current_target_gt_idx]
            #     if roi_target.shape[0] > self.min_cd_num and roi_target.shape[0] > 0 and roi_source.shape[0] > 0:
            #         #center = gt_boxes[bidx, ridx, :3]
            #         #roi_target[..., 1:4] -= center
            #         #roi_source[..., 1:4] -= center
            #         cd_loss_t, cd_loss_s = self.loss_chamfer(roi_target[..., 1:4].unsqueeze(0), roi_source[..., 1:4].unsqueeze(0))
            #         cd_loss_single = (cd_loss_t + cd_loss_s) * 0.5
            #         # cd_loss_single, _ = chamfer_dist(roi_target[..., 1:4].unsqueeze(0), roi_source[..., 1:4].unsqueeze(0), point_reduction='mean', batch_reduction='sum')
            #         cd_loss.append(cd_loss_single)
                    
            #     # vis = Visualizer3D()
            #     # vis.add_points(roi_target[..., 1:4].view(-1, 3), [1, 1, 0])
            #     # vis.add_points(roi_source[..., 1:4].view(-1, 3))   
            #     # vis.show()
            # if len(cd_loss) > 0:
            #     cd_loss = torch.stack(cd_loss).mean()
            # else:
            #     cd_loss = torch.zeros(1, device='cuda', dtype=torch.float32, requires_grad=True).mean()

            tp_list = []
            for bidx in range(batch_size):
                tp_list.append(target_points[target_points[:, 0] == bidx])
            tp_batch = target_points.new_zeros(batch_size, max([len(x) for x in tp_list]), target_points.shape[-1])
            for bidx, cur_tp in enumerate(tp_list):
                tp_batch[bidx, :len(cur_tp)] = cur_tp
            target_points_in_roi = roiaware_pool3d_utils.points_in_boxes_gpu(tp_batch[..., 1:4], gt_boxes[...,:-1])

            # target_points_in_roi = target_points_mask

            tpr_list = []   # target points in roi
            roi_target_num = [] 
            for (bidx, ridx) in fg_mask.nonzero():
                current_target_gt_idx = target_gt_idx[bidx, ridx]
                tpr = tp_batch[bidx, target_points_in_roi[bidx] == current_target_gt_idx.item()]
                # tpr = target_points[target_points[..., 0]==bidx][target_points_mask[bidx]==current_target_gt_idx.item()]
                if len(tpr) >= self.min_cd_num:
                    tpr_list.append(tpr)
                else:
                    fg_mask[bidx, ridx] = 0
            if len(tpr_list) == 0:
                tb_dict['cd_loss'] = None
                return None, tb_dict
            roi_target = target_points.new_zeros(len(tpr_list), max([len(x) for x in tpr_list]), target_points.shape[-1])
            for ridx, cur_tpr in enumerate(tpr_list):
                roi_target[ridx, :len(cur_tpr)] = cur_tpr
                roi_target_num.append(len(cur_tpr))
            roi_source = rebuilt_points[fg_mask > 0]
            cd_loss, _ = chamfer_dist(roi_target[..., 1:4], roi_source[..., 1:4], x_lengths=torch.tensor(roi_target_num).to(roi_target.device), point_reduction='mean', batch_reduction='sum')

            if fg_mask.sum() > 0:
                cd_loss /= fg_mask.sum()
                loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
                cd_loss = cd_loss * loss_weights_dict['chamfer_dist_weight']
                tb_dict['cd_loss'] = cd_loss.item()

                if self.model_cfg.LOSS_CONFIG.get('POINT_LOSS_REGULARIZATION', False):
                    rois_fg = forward_ret_dict['rois'][fg_mask > 0].unsqueeze(1)
                    offset = roi_source[..., 1:4] - rois_fg[..., :3]
                    if self.model_cfg.LOSS_CONFIG.get('NORMALIZE_POINT_LOSS_REGULARIZATION', False):
                        offset = offset / rois_fg[..., 3:6]
                    mean_dist = -0.5 * offset.norm(dim=-1).mean()
                    tb_dict['pg_regularization'] = mean_dist
                    cd_loss += mean_dist
            
            total_loss += cd_loss
        
        # total_loss = total_loss / len(forward_ret_dict_list)

        return total_loss, {'cd_loss' : total_loss.item()}
        
    def get_point_generation_loss(self, forward_ret_dict):
        
        tb_dict = {}
        point_gen_loss = 0
        cd_loss, cd_tb_dict = self.get_chamfer_distance(forward_ret_dict)
        point_loss, pc_tb_dict = self.get_point_cls_loss(forward_ret_dict)        
        tb_dict.update(cd_tb_dict)
        tb_dict.update(pc_tb_dict)

        if not isinstance(cd_loss, torch.Tensor):
            cd_loss = 0.0
        if not isinstance(point_loss, torch.Tensor):
            point_loss = 0.0
            
        pg_loss = cd_loss + point_loss

        tb_dict['pg_loss'] = pg_loss.item()
        return pg_loss, tb_dict
    
    def get_bbox_2d_loss(self, forward_ret_dict):
        pass

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        
        rcnn_loss_pg, pg_tb_dict =  self.get_point_generation_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_pg
        tb_dict.update(pg_tb_dict)
        
        # rcnn_loss_cntrsv, ct_tb_dict =  self.get_clip_loss(self.forward_ret_dict)
        # rcnn_loss += rcnn_loss_cntrsv
        # tb_dict.update(ct_tb_dict)
        
        # rcnn_loss_sim, sim_tb_dict =  self.get_sim_loss(self.forward_ret_dict)
        # rcnn_loss += rcnn_loss_sim
        # tb_dict.update(sim_tb_dict)
        
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        
        return rcnn_loss, tb_dict