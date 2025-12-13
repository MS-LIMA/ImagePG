from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None
        
        self.epoch = 0
            
    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def set_epoch(self, epoch):
        self.epoch = epoch 
        self.data_augmentor.set_epoch(epoch)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.
        Args:
            index:
        Returns:
        """
        raise NotImplementedError
    
    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...
        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            
            if 'calib' in data_dict:
                calib = data_dict['calib']
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            ) 
            if 'calib' in data_dict:
                data_dict['calib'] = calib
                
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]
            if data_dict.get('obj_ids', None) is not None:
                data_dict['obj_ids'] = data_dict['obj_ids'][selected]

            if data_dict.get('segmentation', None) is not None:
                # print("#selected=", selected, " len of data_dict['segmentation']", len(data_dict['segmentation']), "len of gt_boxes", len(data_dict['gt_boxes']))
                data_dict['segmentation'] = [data_dict['segmentation'][ind] for ind in selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                if key == 'TTA_trans_dict':
                    continue
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'bm_voxels', 'pillar_voxels', 'bm_voxel_num_points', 'pillar_voxel_dense_num_points',
                           'pillar_voxel_num_points', 'voxel_num_points', 'voxels_dense', 'pillar_voxels_dense', 'voxel_dense_num_points', 
                           'voxels_1', 'voxels_2', 'voxels_3', 'voxels_4', 'voxels_5',
                           'voxel_num_points_1', 'voxel_num_points_2', 'voxel_num_points_3', 'voxel_num_points_4', 'voxel_num_points_5',
                           'voxels_mm', 'voxels_mm_1', 'voxels_mm_2', 'voxels_mm_3', 'voxels_mm_4', 'voxels_mm_5',
                           'voxel_mm_num_points', 'voxel_mm_num_points_1', 'voxel_mm_num_points_2', 'voxel_mm_num_points_3', 'voxel_mm_num_points_4', 'voxel_mm_num_points_5'] or 'voxels_' in key or 'voxel_num_points' in key:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['mm_points', 'points', 'points_dense', 'bm_voxel_coords', 
                             'voxel_coords', 'voxel_dense_coords', 'pillar_voxel_coords', 'pillar_voxel_dense_coords', 'bm_points', 
                             'voxel_coords_1', 'voxel_coords_2', 'voxel_coords_3', 'voxel_coords_4', 'voxel_coords_5',
                             'voxel_mm_coords', 'voxel_mm_coords_1', 'voxel_mm_coords_2', 'voxel_mm_coords_3', 'voxel_mm_coords_4', 'voxel_mm_coords_5'] or 'voxel_coords_' in key or ('points_' in key and 'bm' not in key):
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps", "overlap_mask", "depth_mask", 'segmentation_label']:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        if isinstance(image, list):
                            for im in image:
                                max_h = max(max_h, im.shape[0])
                                max_w = max(max_w, im.shape[1])
                        else:
                            max_h = max(max_h, image.shape[0])
                            max_w = max(max_w, image.shape[1])

                    # Change size of images
                    if isinstance(val[0], list):
                        images_list = []
                        pad_shapes = []
                        for images in val:
                            imgs = []
                            ps = []
                            for image in images:
                                pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                                pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                                pad_width = (pad_h, pad_w)
                                pad_value = 0

                                if key == "images":
                                    pad_width = (pad_h, pad_w, (0, 0))
                                    pad_value = 0
                                elif key == "depth_maps":
                                    pad_width = (pad_h, pad_w)
                                elif key == "overlap_mask":                            
                                    pad_width = (pad_h, pad_w)
                                    pad_value = 0
                                elif key == "depth_mask":
                                    pad_width = (pad_h, pad_w, (0, 0))
                                    pad_value = 0

                                image_pad = np.pad(image,
                                                pad_width=pad_width,
                                                mode='constant',
                                                constant_values=pad_value)

                                imgs.append(image_pad)
                                ps.append((pad_h[1], pad_w[1]))
                            imgs = np.stack(imgs, axis=0)
                            ps = np.stack(ps)
                            images_list.append(imgs)
                            pad_shapes.append(ps)
                        ret[key] = np.stack(images_list, axis=0)
                        ret['pad_size'] = np.array(pad_shapes)
                    else:
                        images = []
                        pad_shapes = []
                        for image in val:
                            pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                            pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                            pad_width = (pad_h, pad_w)
                            pad_value = 0

                            if key == "images":
                                pad_width = (pad_h, pad_w, (0, 0))
                                pad_value = 0
                            elif key == "depth_maps":
                                pad_width = (pad_h, pad_w)
                            elif key == "overlap_mask":                            
                                pad_width = (pad_h, pad_w)
                                pad_value = 0
                            elif key == "depth_mask":
                                pad_width = (pad_h, pad_w, (0, 0))
                                pad_value = 0
                            elif key=='segmentation_label':
                                pad_width = (pad_h, pad_w)
                                pad_value = 255
                                
                            image_pad = np.pad(image,
                                            pad_width=pad_width,
                                            mode='constant',
                                            constant_values=pad_value)

                            images.append(image_pad)
                            pad_shapes.append((pad_h[1], pad_w[1]))
                        ret[key] = np.stack(images, axis=0)
                        ret['pad_size'] = np.array(pad_shapes)
                elif key in ['calib', 'gt_dense', 'bm_points_mask', 'epoch']:
                    ret[key] = val
                elif key in ["points_2d", 'points_dense_2d', 'points_trans_1', 'points_trans_2', 'points_trans_3', 'points_trans_4', 'points_trans_5']:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len-len(_points)), (0,0))
                        points_pad = np.pad(_points,
                                            pad_width=pad_width,
                                            mode='constant',
                                            constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                elif key in ["point2img"]:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['image_grid_dict', 'image_depths_dict', 'voxel_grid_dict', 'lidar_grid_dict']:
                    results = {}
                    for _val in val:
                        for _layer in _val:
                            if _layer in results:
                                results[_layer] = results[_layer].append(_val[_layer])
                            else:
                                results[_layer] = [_val[_layer]]
                    for _layer in results:
                        results[_layer] = torch.cat(results[_layer], dim=0)
                    ret[key] = results
                elif key in ['trans_dict', 'local_noise_scale', 'local_noise_rotation', 'local_noise_trans', 'obj_ids']:
                    ret[key] = val
                elif key == 'TTA_data_dict_list':
                    tta_list = []
                    for batch_idx in range(len(val)):
                        temp_list = val[batch_idx]
                        for tta_idx in range(len(temp_list)):
                            tta_dict = DatasetTemplate.collate_batch([temp_list[tta_idx]])
                            tta_list.append(tta_dict)
                    ret[key] = tta_list
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        ret['epoch'] = data_dict['epoch']

        return ret