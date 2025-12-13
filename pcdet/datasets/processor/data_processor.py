from functools import partial
from pathlib import Path

import pickle
import copy

import os
from tkinter import NO
import numpy as np
from skimage import transform
from sklearn.utils import shuffle
import torchvision
import torch
import random
import torch.nn.functional as F
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, box_np_ops, point_box_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        points = points.copy()
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None
        self.voxel_generator_pillar = None
        self.voxel_generator_bm = None
        self.voxel_generator_dense = None
        self.voxel_generator_pillar_dense = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    

    def load_depth_completion_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.load_depth_completion_points, config=config)   
             
        sample_idx = data_dict['frame_id']

        file_path = os.path.join(config.DATA_ROOT,  "{}.npy".format(sample_idx))
        mm_points = np.load(file_path)[..., :3]
        ones_col = np.ones((mm_points.shape[0], 1))  # arr.shape[0]는 N 값
        mm_points = np.concatenate((mm_points, ones_col), axis=1)
        mm_points, is_numpy = common_utils.check_numpy_to_torch(mm_points)

        if 'flip_x' in data_dict:
            if data_dict['flip_x']:
                mm_points[:, 1] = -mm_points[:, 1]
            
        if 'noise_rotation' in data_dict:
            mm_points[..., 0:3] = common_utils.rotate_points_along_z(points=mm_points[..., 0:3].unsqueeze(0), 
                                                                        angle=torch.tensor(data_dict['noise_rotation']).view(1)).squeeze(0)
            
        if 'noise_scale' in data_dict:
            mm_points[:, :3] *= data_dict['noise_scale']
        
        data_dict['mm_points'] = mm_points.numpy()
         
        return data_dict

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        data_dict_list = [data_dict]
            
        for d in data_dict_list:
            if d.get('points', None) is not None:
                mask = common_utils.mask_points_by_range(d['points'], self.point_cloud_range)
                d['points'] = d['points'][mask]
                #if data_dict.get('points_2d', None) is not None:
                #    data_dict['points_2d'] = data_dict['points_2d'][mask]

            if d.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
                mask = box_utils.mask_boxes_outside_range_numpy(
                    d['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
                )
                d['gt_boxes'] = d['gt_boxes'][mask]
                if 'local_noise_scale' in d:
                    d['local_noise_scale'] = d['local_noise_scale'][mask]
                if 'local_noise_rotation' in d:
                    d['local_noise_rotation'] = d['local_noise_rotation'][mask]
                if 'local_noise_trans' in d:
                    d['local_noise_trans'] = d['local_noise_trans'][mask]
                    
            if 'points_trans_list' in d:
                points_trans_list = []
                for points in d['points_trans_list']:
                    mask = common_utils.mask_points_by_range(points, self.point_cloud_range)
                    points_trans_list.append(points[mask])
                d['points_trans_list'] = points_trans_list

            if 'mm_points' in d:
                mask = common_utils.mask_points_by_range(d['mm_points'], self.point_cloud_range)
                d['mm_points'] = d['mm_points'][mask]           
                
            if 'points_mm_trans_list' in d:
                points_mm_trans_list = []
                for points in d['points_mm_trans_list']:
                    mask = common_utils.mask_points_by_range(points, self.point_cloud_range)
                    points_mm_trans_list.append(points[mask])
                d['points_mm_trans_list'] = points_mm_trans_list         
        
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            
            data_dict_list = [data_dict]
            
            for d in data_dict_list:
                points = d['points']
                shuffle_idx = np.random.permutation(points.shape[0])
                points = points[shuffle_idx]
                d['points'] = points
                #if data_dict.get('points_2d', None) is not None:
                #    points_2d = data_dict['points_2d']
                #    points_2d = points_2d[shuffle_idx]
                #    data_dict['points_2d'] = points_2d
                
                if 'points_trans_list' in d:
                    points_trans_list = []
                    for points in d['points_trans_list']:
                        shuffle_idx = np.random.permutation(points.shape[0])
                        points_trans = points[shuffle_idx]
                        points_trans_list.append(points_trans)
                    d['points_trans_list'] = points_trans_list

                if 'points_dense' in d:
                    points_dense = d['points_dense']
                    shuffle_idx = np.random.permutation(points_dense.shape[0])
                    points_dense = points_dense[shuffle_idx]
                    d['points_dense'] = points_dense
                    if d.get('points_dense_2d', None) is not None:
                        points_dense_2d = d['points_dense_2d']
                        points_dense_2d = points_dense_2d[shuffle_idx]
                        d['points_dense_2d'] = points_dense_2d

                if 'mm_points' in d:
                    mm_points = d['mm_points']
                    shuffle_idx = np.random.permutation(mm_points.shape[0])
                    mm_points = mm_points[shuffle_idx]
                    d['mm_points'] = mm_points
                
                if 'points_mm_trans_list' in d:
                    points_mm_trans_list = []
                    for points in d['points_mm_trans_list']:
                        shuffle_idx = np.random.permutation(points.shape[0])
                        points_trans = points[shuffle_idx]
                        points_mm_trans_list.append(points_trans)
                    d['points_mm_trans_list'] = points_mm_trans_list

        return data_dict
    
    def image_resize(self, data_dict=None, config=None):
        
        def resize_img(img, config):
            img = torch.from_numpy(img.copy())
            H, W, C = img.shape
            
            if self.training:
                random_downscale = config.get('RANDOM_DOWNSCALE', None)
                if random_downscale:
                    factors = random_downscale.FACTORS
                    factor = random.uniform(factors[0], factors[1])
                    img = torchvision.transforms.Resize((int(H * factor), int(W * factor)), antialias=False)(img.permute(2, 0, 1))
                    img = torchvision.transforms.Resize((int(H), int(W)), antialias=False)(img).permute(1, 2, 0)  

            factor = config.IMG_SCALE
            # img = torchvision.transforms.Resize((int(H * factor), int(W * factor)), antialias=False)(img.permute(2, 0, 1)).permute(1, 2, 0) # H, W, C
            final_factor = config.IMG_SCALE
            H2, W2 = int(H * final_factor), int(W * final_factor)
            img = F.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=(H2, W2), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)       

            return img
        
        if data_dict is None:
            return partial(self.image_resize, config=config)
        
        if 'images' not in data_dict:
            return data_dict
        
        data_dict_list = [data_dict]
        
        for d in data_dict_list:        
            
            if isinstance(d['images'], list):
                images = d['images']
                for i in range(len(images)):
                    images[i] = resize_img(images[i], config)
                d['images'] = images
            else:
                img = d['images'] # H, W, C
                img = resize_img(img, config)
                d["images"] = img

            d['img_scale'] = config.IMG_SCALE
            # d['img_org_shape'] = np.array([H, W])
        
        return data_dict
    
    def image_normalize(self, data_dict=None, config=None):

        def noramlize(img, mean, std):
            img = img.permute(2, 0, 1) # C, H, W
            img[0] = (img[0] - mean[0]) / std[0]
            img[1] = (img[1] - mean[1]) / std[1]
            img[2] = (img[2] - mean[2]) / std[2]
            
            img = img.permute(1, 2, 0) # H, W, C
            return img

        if data_dict is None:
            return partial(self.image_normalize, config=config)
        
        if 'images' not in data_dict:
            return data_dict
        
        mean = config.get("MEAN", [0, 0, 0])
        std = config.get("STD", [1, 1, 1])

        data_dict_list = [data_dict]
        
        for d in data_dict_list:   

            if isinstance(d['images'], list):
                images = d['images']
                for i in range(len(images)):
                    images[i] = noramlize(images[i], mean, std)
                d['images'] = images
            else:
                img = noramlize(d['images'], mean, std)
                d['images'] = img
        return data_dict
    
    def combine_dense_points(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.combine_dense_points, config=config)
        
        if self.training:
            if 'bm_points' in data_dict:            
                points = data_dict['points'].copy()
                bm_points = data_dict['bm_points'].copy()
                bm_points = box_utils.remove_points_not_in_boxes3d(bm_points, data_dict['gt_boxes_org'][:, 0:7])
                bm_points = box_utils.remove_points_in_boxes3d(bm_points, data_dict['gt_boxes_empty'][:, 0:7])

                ones_col = np.ones((bm_points.shape[0], 1))  # arr.shape[0]는 N 값
                bm_points = np.concatenate((bm_points, ones_col), axis=1)
                points = np.concatenate((points, bm_points), axis=0)

                calib = data_dict['calib']
                image = data_dict['images']
                img_shape = image.shape

                points_2d, _ = calib.lidar_to_img(points[:, 0:3])
                points_2d[:,0] = np.clip(points_2d[:,0], a_min=0, a_max=img_shape[1]-1)
                points_2d[:,1] = np.clip(points_2d[:,1], a_min=0, a_max=img_shape[0]-1)
                points_2d = points_2d.astype(np.int32)

                #data_dict['points'] = points
                #data_dict['points_2d'] = points_2d

                data_dict['bm_points'] = bm_points[:, :3]
                data_dict['points_dense'] = points
                data_dict['points_dense_2d'] = points_2d

                data_dict.pop('gt_boxes_org', None)
                data_dict.pop('gt_boxes_empty', None)
        else:
            return data_dict
        
            root_path = Path(config.DATA_PATH)
            bm_root = {
                'Car': root_path.resolve() / config.PCN.CAR_MLT_BM_ROOT,
                'Cyclist': root_path.resolve() / config.PCN.CYC_MLT_BM_ROOT,
                'Pedestrian': root_path.resolve() / config.PCN.PED_MLT_BM_ROOT,
            }
            bm_root_backup = {
                'Car': root_path.resolve() / config.BACKUP.CAR_MLT_BM_ROOT,
                'Cyclist': root_path.resolve() / config.BACKUP.CYC_MLT_BM_ROOT,
                'Pedestrian': root_path.resolve() / config.BACKUP.PED_MLT_BM_ROOT,
            }
            
            load_point_features = config.get("NUM_POINT_FEATURES", 3)
        
            obj_points_list = []
            gt_boxes_num = data_dict['gt_boxes'].shape[0]
            for idx in range(gt_boxes_num):
                gt_box = data_dict['gt_boxes'][idx]
                gt_name = data_dict['gt_names'][idx]
                
                sample_idx = data_dict['frame_id']
                file_path = bm_root[gt_name] / "{}_{}.pkl".format(int(sample_idx), idx)
                
                if os.path.exists(str(file_path)) == False:
                    file_path = bm_root_backup[gt_name] / "{}_{}.pkl".format(int(sample_idx), idx)
                    
                if os.path.exists(str(file_path)):
                    with open(str(file_path), 'rb') as f:
                        obj_points = pickle.load(f)
                        obj_points = obj_points.reshape(
                            [-1, load_point_features])[:,:3].astype(np.float32)
                        
                        #if not (gt_name == 'Car' and obj_points.shape[0] < 64):
                        gtrotation = point_box_utils.get_yaw_rotation(gt_box[6])
                        obj_points = np.einsum("nj,ij->ni", obj_points, gtrotation) + gt_box[:3]
                        obj_points_list.append(obj_points.astype(np.float32))
                else:
                    pass
                    #print(str(file_path) + "NO EIXSTS")
                    
            if len(obj_points_list) > 0:
                obj_points_list = np.concatenate(obj_points_list, axis=0)
                ones_col = np.ones((obj_points_list.shape[0], 1))  # arr.shape[0]는 N 값
                obj_points_list = np.concatenate((obj_points_list, ones_col), axis=1)
                
                points = data_dict['points'].copy()
                points = np.concatenate((points, obj_points_list), axis=0)

                bm_points = obj_points_list[..., :3]
            else:
                points = data_dict['points']
                bm_points = np.zeros((1, 3), dtype=np.float32)
                
            data_dict['points_dense'] = points
            data_dict['bm_points'] = bm_points
            
            # upsample_points_list = []
            # upsample_points_path = self.sampler_cfg.get("UPSAMPLE_POINTS_PATH")
            # for img_idx, gt_idx, name, box in zip(image_indices, gt_indices, gt_names, gt_boxes):
            #     file_path = os.path.join(upsample_points_path, "{}_{}_{}.bin".format(str(img_idx[0]).zfill(6), name, gt_idx[0]))
            #     obj_points_up = np.fromfile(str(file_path), dtype=np.float32).reshape(
            #         [-1, self.sampler_cfg.get('NUM_UPSAMPLE_POINTS_FEATURES', 3)]).astype(np.float32)
            #     obj_points_up[:, :3] += box[:3]
            #     obj_points_up[:, 2] -= box[5] * 0.5
            #     # obj_points_up[:, 2] += box[2] * 2.0
                
            #     upsample_points_list.append(obj_points_up[:,:3])
            # upsample_points_list = np.concatenate(upsample_points_list, axis=0)

           # data_dict['points_dense'] = points
            # data_dict['points_2d_dense'] = points_2d

        return data_dict
    
    def WBF(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.WBF, config=config)

        if config.ENABLED:
            data_dict['WBF'] = True
        
        return data_dict

    def points_translation(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.points_translation, config=config)
        
        if self.training:
            flip_x_list = config.TRAIN.FLIP_X
            flip_y_list = config.TRAIN.FLIP_Y
            scale_list = config.TRAIN.SCALES
            rot_angle_list = config.TRAIN.ROT_ANGLES

            if isinstance(flip_x_list[0], list):
                flip_x_list = [np.random.choice(k, 1).item() for k in flip_x_list]
            if isinstance(flip_y_list[0], list):
                flip_y_list = [np.random.choice(k, 1).item() for k in flip_y_list]
            if isinstance(scale_list[0], list):
                scale_list = [np.random.choice(k, 1).item() for k in scale_list]
            if isinstance(rot_angle_list[0], list):
                rot_angle_list = [np.random.choice(k, 1).item() for k in rot_angle_list]     
        else:
            flip_x_list = config.TEST.FLIP_X
            flip_y_list = config.TEST.FLIP_Y
            scale_list = config.TEST.SCALES
            rot_angle_list = config.TEST.ROT_ANGLES     

            if isinstance(flip_x_list[0], list):
                flip_x_list = [np.random.choice(k, 1).item() for k in flip_x_list]
            if isinstance(flip_y_list[0], list):
                flip_y_list = [np.random.choice(k, 1).item() for k in flip_y_list]
            if isinstance(scale_list[0], list):
                scale_list = [np.random.choice(k, 1).item() for k in scale_list]
            if isinstance(rot_angle_list[0], list):
                rot_angle_list = [np.random.choice(k, 1).item() for k in rot_angle_list]       

        data_dict_list = [data_dict]
        
        for d in data_dict_list:
            points = d['points']
            points, is_numpy = common_utils.check_numpy_to_torch(points)

            points_trans_list = []
            if 'mm_points' in d:
                mm_points = d['mm_points']
                mm_points, is_numpy = common_utils.check_numpy_to_torch(mm_points)
                points_mm_trans_list = []
                
            for flip_x, flip_y, rot_angle, scale in zip(flip_x_list, flip_y_list, rot_angle_list, scale_list):
                points_trans = points.clone()
                if flip_x:
                    points_trans[..., 1] = -points_trans[..., 1]
                if flip_y:
                    points_trans[..., 0] = -points_trans[..., 0]
                points_trans[..., 0:3] = common_utils.rotate_points_along_z(points=points_trans[..., 0:3].unsqueeze(0), 
                                                                            angle=torch.tensor(rot_angle).view(1)).squeeze(0)
                if scale != 1.0:
                    points_trans[..., 0:3] *= scale
                points_trans_list.append(points_trans)
                
                if 'mm_points' in d:
                    points_mm_trans = mm_points.clone()
                    if flip_x:
                        points_mm_trans[..., 1] = -points_mm_trans[..., 1]
                    points_mm_trans[..., 0:3] = common_utils.rotate_points_along_z(points=points_mm_trans[..., 0:3].unsqueeze(0), 
                                                                                angle=torch.tensor(rot_angle).view(1)).squeeze(0)
                    if scale != 1.0:
                        points_mm_trans[..., 0:3] *= scale
                    points_mm_trans_list.append(points_mm_trans)

            points_trans_list = [x.numpy() for x in points_trans_list]
            d['points_trans_list'] = points_trans_list

            if 'mm_points' in d:
                points_mm_trans_list = [x.numpy() for x in points_mm_trans_list]
                d['points_mm_trans_list'] = points_mm_trans_list
            
            d['trans_dict'] = {
                'flip_x_list' : flip_x_list,
                'flip_y_list' : flip_y_list,
                'scale_list' : scale_list,
                'rot_angle_list' : rot_angle_list,
                'num_trans' : len(rot_angle_list)
            }

        return data_dict
    
    def filter_empty_boxes(self, data_dict=None, config=None):
        
        if data_dict is None:
            return partial(self.filter_empty_boxes, config=config)
        
        if self.training:
            boxes3d = data_dict['gt_boxes']
            points = data_dict['points']

            boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
            points, is_numpy = common_utils.check_numpy_to_torch(points)

            non_empty_indices = []
            empty_indices = []
            for idx, box in enumerate(boxes3d):
                point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], box.unsqueeze(0)[:, 0:7])
                if point_masks.sum() != 0:
                    non_empty_indices.append(idx)
                else:
                    empty_indices.append(idx)

            #data_dict['gt_boxes_org'] = boxes3d.numpy()

            empty_boxes3d = boxes3d[empty_indices].numpy()
            #data_dict['gt_boxes_empty'] = empty_boxes3d

            boxes3d = boxes3d[non_empty_indices].numpy()
            data_dict['gt_boxes'] = boxes3d
            
            if 'local_noise_scale' in data_dict:
                data_dict['local_noise_scale'] = data_dict['local_noise_scale'][non_empty_indices]
            if 'local_noise_rotation' in data_dict:
                data_dict['local_noise_rotation']  = data_dict['local_noise_rotation'][non_empty_indices]
            if 'local_noise_trans' in data_dict:
                data_dict['local_noise_trans'] = data_dict['local_noise_trans'][non_empty_indices]

            # if 'bm_points' in data_dict:
            #     bm_points = data_dict['bm_points']
            #     bm_points = box_utils.remove_points_not_in_boxes3d(bm_points, boxes3d)
            #     data_dict['bm_points'] = bm_points

        return data_dict

    def transform_points_to_pillar_dense(self, data_dict=None, config=None):
        if data_dict is None:
            pillar_grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.pillar_grid_size = np.round(pillar_grid_size).astype(np.int64)
            self.pillar_voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_pillar_dense, config=config)

        if self.voxel_generator_pillar_dense is None:
            self.voxel_generator_pillar_dense = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=4,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points_dense']
        voxel_output = self.voxel_generator_pillar_dense.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['pillar_voxels_dense'] = voxels
        data_dict['pillar_voxel_dense_coords'] = coordinates
        data_dict['pillar_voxel_dense_num_points'] = num_points
        return data_dict
    
    def transform_points_to_pillar(self, data_dict=None, config=None):
        if data_dict is None:
            pillar_grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.pillar_grid_size = np.round(pillar_grid_size).astype(np.int64)
            self.pillar_voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_pillar, config=config)

        if self.voxel_generator_pillar is None:
            self.voxel_generator_pillar = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=config.NUM_POINT_FEATURES,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        data_dict_list = [data_dict]
        
        for d in data_dict_list:
            points = d['points_init'] if 'points_init' in d else d['points']
            
            voxel_output = self.voxel_generator_pillar.generate(points)
            voxels, coordinates, num_points = voxel_output

            if not d['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

            d['pillar_voxels'] = voxels
            d['pillar_voxel_coords'] = coordinates
            d['pillar_voxel_num_points'] = num_points
            d.pop('points_init', None)

        return data_dict
    
    def transform_points_to_voxels_dense(self, data_dict=None, config=None):
        
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_dense, config=config)

        if self.voxel_generator_dense is None:
            self.voxel_generator_dense = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=config.NUM_POINT_FEATURES,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points_dense']
        
        # use point->image index
        use_index = config.get('USE_INDEX', False)

        voxel_output = self.voxel_generator_dense.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        
        if use_index:
            data_dict["point2img"] = voxels[...,-1]
            voxels = voxels[...,:-1] # pop point->image in voxels (N, N', -1)

        data_dict['voxels_dense'] = voxels
        data_dict['voxel_dense_coords'] = coordinates
        data_dict['voxel_dense_num_points'] = num_points
        return data_dict
    
    def bev_shape_mask_transform_points_to_voxels(self, data_dict=None, config=None):
        
        if data_dict is None:
            shape_grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.shape_grid_size = np.round(shape_grid_size).astype(np.int64)
            self.shape_voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.bev_shape_mask_transform_points_to_voxels, config=config)
        
        enable_test = config.get('ENABLE_TEST', False)
        if self.training == False and enable_test == False:
            return data_dict

        if self.voxel_generator_bm is None:
            self.voxel_generator_bm = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=4,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['bm_points_init'] if 'bm_points_init' in data_dict else data_dict['bm_points']
        #angle = torch.tensor((np.pi/ 4)).view(1)
        #points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], angle)[0]

        points = np.pad(points, (0, 1), 'constant')
        voxel_output = self.voxel_generator_bm.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]
        data_dict['bm_voxels'] = voxels
        data_dict['bm_voxel_coords'] = coordinates
        data_dict['bm_voxel_num_points'] = num_points
        data_dict.pop('bm_points_init', None)

        return data_dict
    
    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=config.NUM_POINT_FEATURES,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        data_dict_list = [data_dict]
        
        for d in data_dict_list:
            points = d['points']
            if config.get('RANDOM_POINT_DROP_PROB', False):
                gt_boxes = d['gt_boxes']
                origin = (0.5, 0.5, 0.5)
                gt_box_corners = box_np_ops.center_to_corner_box3d(
                    gt_boxes[:, :3],
                    gt_boxes[:, 3:6],
                    gt_boxes[:, 6],
                    origin=origin,
                    axis=2)

                surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
                if (points is not None):
                    point_masks = box_np_ops.points_in_convex_polygon_3d_jit(
                        points[:, :3], surfaces)
                    
                    num_box = gt_boxes.shape[0]
                    num_points = points.shape[0]

                    has_true = point_masks.any(axis=1)
                    point_masks = point_masks.argmax(axis=1)
                    point_masks = np.where(has_true, point_masks, -1).reshape(-1)

                    points_bg = points[point_masks < 0]
                    points_fg_list = []
                    min_ratio, max_ratio = config.RANDOM_POINT_DROP_RATE[0], config.RANDOM_POINT_DROP_RATE[1]
                    
                    for j in range(num_box):
                        points_fg = points[point_masks == j]
                        N = points_fg.shape[0]
                        if N > 0 and random.randrange(0, 1) <= config.RANDOM_POINT_DROP_PROB:
                            min_count = int(np.floor(min_ratio * N))
                            max_count = int(np.floor(max_ratio * N))
                            all_indices = np.arange(N)
                            np.random.shuffle(all_indices)
                            selected_count = np.random.randint(min_count, max_count + 1)
                            selected_indices = all_indices[:selected_count]
                            selected_points = points[selected_indices]
                            points_fg_list.append(selected_points)
                    points = [points_bg] + points_fg_list
                    points = np.concatenate(points, axis=0)
                    d['points'] = points

            # use point->image index
            use_index = config.get('USE_INDEX', False)

            voxel_output = self.voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

            if not d['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
            
            if use_index:
                d["point2img"] = voxels[...,-1]
                voxels = voxels[...,:-1] # pop point->image in voxels (N, N', -1)

            d['voxels'] = voxels
            d['voxel_coords'] = coordinates
            d['voxel_num_points'] = num_points

            if 'mm_points' in d:
                mm_points = d['mm_points']
                voxel_output = self.voxel_generator.generate(mm_points)
                if isinstance(voxel_output, dict):
                    voxels, coordinates, num_points = \
                        voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
                else:
                    voxels, coordinates, num_points = voxel_output
                d['voxels_mm'] = voxels
                d['voxel_mm_coords'] = coordinates
                d['voxel_mm_num_points'] = num_points  
                    
            if 'trans_dict' in d:
                points_trans_list = d['points_trans_list']
                len(points_trans_list)
                for i, points in enumerate(points_trans_list):
                    voxel_output = self.voxel_generator.generate(points)
                    if isinstance(voxel_output, dict):
                        voxels, coordinates, num_points = \
                            voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
                    else:
                        voxels, coordinates, num_points = voxel_output
                    d['voxels_' + str(i + 1)] = voxels
                    d['voxel_coords_' + str(i + 1)] = coordinates
                    d['voxel_num_points_' + str(i + 1)] = num_points   
                    
                if 'points_mm_trans_list' in d:
                    
                    points_mm_trans_list = d['points_mm_trans_list']
                    for i, points in enumerate(points_mm_trans_list):
                        voxel_output = self.voxel_generator.generate(points)
                        if isinstance(voxel_output, dict):
                            voxels, coordinates, num_points = \
                                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
                        else:
                            voxels, coordinates, num_points = voxel_output
                        d['voxels_mm_' + str(i + 1)] = voxels
                        d['voxel_mm_coords_' + str(i + 1)] = coordinates
                        d['voxel_mm_num_points_' + str(i + 1)] = num_points  
                                
            d.pop('points_trans_list', None)
            d.pop('points_mm_trans_list', None)

        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def mask_points_and_boxes_outside_range_sfd(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range_sfd, config=config)

            
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]

        mask = common_utils.mask_points_by_range(data_dict['points_pseudo'], self.point_cloud_range)
        data_dict['points_pseudo'] = data_dict['points_pseudo'][mask]


        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points_sfd(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points_sfd, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

            points_pseudo = data_dict['points_pseudo']
            shuffle_idx = np.random.permutation(points_pseudo.shape[0])
            points_pseudo = points_pseudo[shuffle_idx]
            data_dict['points_pseudo'] = points_pseudo
            
        if config.get('USE_RAW_FEATURES', False):
            data_dict['points_valid'] = data_dict['points']
            data_dict['points'] = data_dict['points'][:,:4]

        return data_dict
    
    def transform_points_to_voxels_valid(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_valid, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        
        # use point->image index
        use_index = config.get('USE_INDEX', False)

        voxel_output = self.voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        
        if use_index:
            data_dict["point2img"] = voxels[...,-1]
            voxels = voxels[...,:-1] # pop point->image in voxels (N, N', -1)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def grid_sample_points_pseudo(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.grid_sample_points_pseudo, config=config)

        max_distance = config.MAX_DISTANCE
        points = data_dict['points_pseudo']
        dist_mask = points[:,0] < max_distance
        col_mask  = (points[:,6]%2 == 0) & dist_mask
        row_mask  = (points[:,7]%2 == 0) & dist_mask

        ignore_mask = col_mask | row_mask
        sample_mask = ~ignore_mask
        data_dict['points_pseudo'] = points[sample_mask]
        return data_dict

    def set_lidar_aug_matrix(self, data_dict=None, config=None):
        """
            Get lidar augment matrix (4 x 4), which are used to recover orig point coordinates.
        """
        if data_dict is None:
            return partial(self.set_lidar_aug_matrix, config=config)
        
        B = data_dict['images'].shape[0]
        lidar_aug_matrix_list = []
        for i in range(B):
            lidar_aug_matrix = np.eye(4)
            if 'flip_x' in data_dict.keys():
                flip_x = data_dict['flip_x']
                if flip_x:
                    lidar_aug_matrix[:3,:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3,:3]
            if 'noise_rotation' in data_dict.keys():
                noise_rot = data_dict['noise_rotation']
                lidar_aug_matrix[:3,:3] = common_utils.angle2matrix(torch.tensor(noise_rot)) @ lidar_aug_matrix[:3,:3]
            if 'noise_scale' in data_dict.keys():
                noise_scale = data_dict['noise_scale']
                lidar_aug_matrix[:3,:3] *= noise_scale
            if 'noise_translate' in data_dict.keys():
                noise_translate = data_dict['noise_translate']
                lidar_aug_matrix[:3,3:4] = noise_translate.T
            lidar_aug_matrix_list.append(torch.tensor(lidar_aug_matrix))
        lidar_aug_matrix_list = torch.stack(lidar_aug_matrix_list)
        data_dict['lidar_aug_matrix'] = lidar_aug_matrix
        return data_dict

    def image_calibrate(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        
        B = data_dict['images'].shape[0]
        transforms = []
        for i in range(B):
            if 'flip_x' in data_dict.keys():
                flip = data_dict['flip_x'][i]
            else:
                flip = False
                
            resize, crop, flip, rotate = config.SCALE_FACTOR, [0, 0, 0, 0], flip, 0

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...
        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        if 'points_trans_list' in data_dict:
            points_trans_list = data_dict['points_trans_list']
            for i in range(len(points_trans_list)):
                data_dict[f'points_{str(i + 1)}'] = points_trans_list[i]
        data_dict.pop('points_trans_list', None)

        return data_dict