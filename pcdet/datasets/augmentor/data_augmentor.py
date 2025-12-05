from functools import partial

import numpy as np
import cv2
from pathlib import Path
import os

from pcdet.datasets.augmentor import database_sampler_mbm, database_sampler_mbm_fix

from ...utils import common_utils, box_utils, point_box_utils
from . import augmentor_utils, database_sampler, augmentor_utils_mbm, database_sampler_mbm, add_multi_best_match_cd_fix


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.epoch = 0
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST
        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
    
    def set_epoch(self, epoch):
        self.epoch = epoch 
        for aug in self.data_augmentor_queue:
            aug.epoch = epoch

    def gt_sampling(self, config=None):
        if 'depth' == config.get('MODE', None) :
            db_sampler = database_sampler_depth.DataBaseSampler(
                root_path=self.root_path,
                sampler_cfg=config,
                class_names=self.class_names,
                logger=self.logger
            )
        else:
            db_sampler = database_sampler.DataBaseSampler(
                root_path=self.root_path,
                sampler_cfg=config,
                class_names=self.class_names,
                logger=self.logger
            )
        return db_sampler
    
    def gt_sampling_sfd(self, config=None):
        db_sampler = database_sampler_sfd.DataBaseSamplerSFD(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def gt_sampling_mbm(self, config=None):
        db_sampler = database_sampler_mbm.DataBaseSamplerMBM(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler
    
    def gt_sampling_mbm_fix(self, config=None):
        db_sampler = database_sampler_mbm_fix.DataBaseSamplerMBM(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler
        
    def gt_sampling_cmaug(self, config=None):
        db_sampler = database_sampler_cmaug.DataBaseSamplerCMAug(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler 

    def gt_sampling_dense(self, config=None):
        db_sampler = database_sampler_dense.DataBaseSamplerDense(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def add_multi_best_match(self, config=None):
        db_sampler = add_multi_best_match.AddBestMatch(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler
    
    def add_multi_best_match_cd_fix(self, config=None):
        db_sampler = add_multi_best_match_cd_fix.AddBestMatch(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler
    
    def add_multi_best_match_dense(self, config=None):
        db_sampler = add_multi_best_match_dense.AddBestMatchDense(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler
    
    def combine_dense_points(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.combine_dense_points, config=config)
        
        data_path = config.DATA_PATH
        load_point_features = config.get("NUM_POINT_FEATURES", 5)
        
        mask_list = []
        obj_points_list = []
        gt_boxes_num = data_dict['gt_boxes'].shape[0]
           
        for idx in range(gt_boxes_num):
            gt_box = data_dict['gt_boxes'][idx]
            gt_name = data_dict['gt_names'][idx]
            obj_id = data_dict['gt_dense'][idx]
            
            file_path = os.path.join(data_path, '{}.bin'.format(obj_id))
            if os.path.exists(file_path):
                obj_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, load_point_features)[..., :3]
                gtrotation = point_box_utils.get_yaw_rotation(gt_box[6])
                obj_points = np.einsum("nj,ij->ni", obj_points, gtrotation) + gt_box[:3]
                # obj_points[..., 2] -= gt_box[5] * 0.5
                max_num = config.get("MAX_NUM", 1024)
                if obj_points.shape[0] > max_num:
                    indices = np.random.choice(obj_points.shape[0], size=max_num, replace=False)
                    obj_points = obj_points[indices]
                obj_points_list.append(obj_points)
                mask = np.full(obj_points.shape[0], idx, dtype=np.int32)
                mask_list.append(mask)
            else:
                mask = np.full(0, idx, dtype=np.int32)
                mask_list.append(mask)

        if len(obj_points_list) > 0:
            obj_points_list = np.concatenate(obj_points_list, axis=0)
        else:
            # obj_points_list = np.zeros([0, 3], dtype=np.float32)
            obj_points_list = data_dict['points'].copy()[..., :3]
            
        data_dict['bm_points'] = obj_points_list    
        # data_dict['bm_points_mask'] = np.concatenate(mask_list, axis=0)

            # sample_idx = data_dict['frame_id']
            # file_path = bm_root[gt_name] / "{}_{}.pkl".format(int(sample_idx), idx)
            
            # if os.path.exists(str(file_path)) == False:
            #     file_path = bm_root_backup[gt_name] / "{}_{}.pkl".format(int(sample_idx), idx)
                
            # if os.path.exists(str(file_path)):
            #     with open(str(file_path), 'rb') as f:
            #         obj_points = pickle.load(f)
            #         obj_points = obj_points.reshape(
            #             [-1, load_point_features])[:,:3].astype(np.float32)
                    
            #         #if not (gt_name == 'Car' and obj_points.shape[0] < 64):
            #         gtrotation = point_box_utils.get_yaw_rotation(gt_box[6])
            #         obj_points = np.einsum("nj,ij->ni", obj_points, gtrotation) + gt_box[:3]
            #         obj_points_list.append(obj_points.astype(np.float32))
            # else:
            #     pass
                #print(str(file_path) + "NO EIXSTS")
                
        # if len(obj_points_list) > 0:
        #     obj_points_list = np.concatenate(obj_points_list, axis=0)
        #     ones_col = np.ones((obj_points_list.shape[0], 1))  # arr.shape[0]는 N 값
        #     obj_points_list = np.concatenate((obj_points_list, ones_col), axis=1)
            
        #     points = data_dict['points'].copy()
        #     points = np.concatenate((points, obj_points_list), axis=0)

        #     bm_points = obj_points_list[..., :3]
        # else:
        #     points = data_dict['points']
        #     bm_points = np.zeros((1, 3), dtype=np.float32)
            
        # data_dict['points_dense'] = points
        # data_dict['bm_points'] = bm_points
        
        
        # if 'bm_points' in data_dict:            
        #     points = data_dict['points'].copy()
        #     bm_points = data_dict['bm_points'].copy()
        #     bm_points = box_utils.remove_points_not_in_boxes3d(bm_points, data_dict['gt_boxes_org'][:, 0:7])
        #     bm_points = box_utils.remove_points_in_boxes3d(bm_points, data_dict['gt_boxes_empty'][:, 0:7])

        #     ones_col = np.ones((bm_points.shape[0], 1))  # arr.shape[0]는 N 값
        #     bm_points = np.concatenate((bm_points, ones_col), axis=1)
        #     points = np.concatenate((points, bm_points), axis=0)

        #     calib = data_dict['calib']
        #     image = data_dict['images']
        #     img_shape = image.shape

        #     points_2d, _ = calib.lidar_to_img(points[:, 0:3])
        #     points_2d[:,0] = np.clip(points_2d[:,0], a_min=0, a_max=img_shape[1]-1)
        #     points_2d[:,1] = np.clip(points_2d[:,1], a_min=0, a_max=img_shape[0]-1)
        #     points_2d = points_2d.astype(np.int32)

        #     #data_dict['points'] = points
        #     #data_dict['points_2d'] = points_2d

        #     data_dict['bm_points'] = bm_points[:, :3]
        #     data_dict['points_dense'] = points
        #     data_dict['points_dense_2d'] = points_2d

        #     data_dict.pop('gt_boxes_org', None)
        #     data_dict.pop('gt_boxes_empty', None)
            
        # else:
            
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        
        return_noise_flag = True
        if config.get('RETURN_NOISE_TRANSLATION', None) is not None:
            if config['RETURN_NOISE_TRANSLATION']:
                return_noise_flag = True

        if return_noise_flag:
            gt_boxes, points, noise_trans_mat_T = augmentor_utils.global_translation(
                data_dict['gt_boxes'],
                data_dict['points'],
                config['STD'],
                return_std_noise=True
            )
            trans_mat_T_inv = noise_trans_mat_T
            data_dict['noise_translation'] = trans_mat_T_inv
        else:
            gt_boxes, points = augmentor_utils.global_translation(
                data_dict['gt_boxes'], data_dict['points'], config['STD']
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points

        return data_dict

    def random_world_translation_mbm(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation_mbm, config=config)
        
        return_noise_flag = True
        if config.get('RETURN_NOISE_TRANSLATION', None) is not None:
            if config['RETURN_NOISE_TRANSLATION']:
                return_noise_flag = True

        if return_noise_flag:
            gt_boxes, points, bm_points, noise_trans_mat_T = augmentor_utils.global_translation_mbm(
                data_dict['gt_boxes'],
                data_dict['points'],
                config['STD'],
                data_dict['bm_points'],
                return_std_noise=True
            )
            trans_mat_T_inv = noise_trans_mat_T
            data_dict['noise_translation'] = trans_mat_T_inv
        else:
            gt_boxes, points = augmentor_utils.global_translation_mbm(
                data_dict['gt_boxes'], data_dict['points'], config['STD']
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['bm_points'] = bm_points
        
        return data_dict
        
    def random_world_flip_mbm(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip_mbm, config=config)
        
        gt_boxes, points, bm_points = data_dict['gt_boxes'], data_dict['points'], data_dict['bm_points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            
            mm_points = data_dict['mm_points'] if 'mm_points' in data_dict else None
            gt_boxes, points, bm_points, mm_points, flip = getattr(augmentor_utils_mbm, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, bm_points, mm_points
            )
            
            data_dict['flip_{}'.format(cur_axis)] = flip
            
            if flip and 'images' in data_dict and config.get('IMAGE_FLIP', True):
                image = data_dict['images']
                image = np.flip(image, axis=1)
                data_dict['images'] = image

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['bm_points'] = bm_points
        if 'mm_points' in data_dict:
            data_dict['mm_points'] = mm_points
            
        return data_dict

    def random_world_rotation_mbm(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation_mbm, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, bm_points, mm_points, noise_rotation = augmentor_utils_mbm.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], data_dict['bm_points'], data_dict.get('mm_points', None), rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['bm_points'] = bm_points
        data_dict['noise_rotation'] = noise_rotation
        if 'mm_points' in data_dict:
            data_dict['mm_points'] = mm_points
        
        return data_dict

    def random_world_scaling_mbm(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling_mbm, config=config)
        gt_boxes, points, bm_points, mm_points, noise_scale = augmentor_utils_mbm.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], data_dict['bm_points'], data_dict.get('mm_points', None), config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['bm_points'] = bm_points
        data_dict['noise_scale'] = noise_scale
        if 'mm_points' in data_dict:
            data_dict['mm_points'] = mm_points
        
        return data_dict
    
    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points, bm_points,  = getattr(augmentor_utils_mbm, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
        
    def random_local_rotation_mbm(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_local_rotation_mbm, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, bm_points, noise_rotation = augmentor_utils_mbm.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], data_dict['bm_points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['bm_points'] = bm_points
        data_dict['local_noise_rotation'] = noise_rotation
        
        return data_dict

    def random_local_scaling_mbm(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_local_scaling_mbm, config=config)
        gt_boxes, points, bm_points, noise_scale = augmentor_utils_mbm.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], data_dict['bm_points'], config['LOCAL_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['bm_points'] = bm_points
        data_dict['local_noise_scale'] = noise_scale
        
        return data_dict

    def random_local_noise_mbm(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_local_noise_mbm, config=config)
        
        data_dict['points_init'] = data_dict['points'].copy()
        if 'bm_points' in data_dict:
            data_dict['bm_points_init'] = data_dict['bm_points'].copy() 

        data_dict['gt_boxes'][:, 6] = -data_dict['gt_boxes'][:, 6]
        points, loc_noise_list, rot_noise_list = augmentor_utils_mbm.noise_per_object_v3_(data_dict['gt_boxes'], data_dict['points'], data_dict['bm_points'], 
                                        data_dict.get('valid_noise', None),  
                                        config['LOCAL_ROT_RANGE'], config['TRANSLATION_STD'], 
                                        config['GLOBAL_ROT_RANGE'], config['EXTRA_WIDTH'])
        data_dict['gt_boxes'][:, 6] = -data_dict['gt_boxes'][:, 6]
        data_dict['points'] = points
        data_dict['local_noise_trans'] = loc_noise_list
        data_dict['local_noise_rotation'] = rot_noise_list
        
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict 

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
    
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, return_flip=True
            )
            data_dict['flip_%s'%cur_axis] = enable

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot
        return data_dict
    
    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict
    
    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )
        
        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict
    
    # def random_world_translation(self, data_dict=None, config=None):
    #     if data_dict is None:
    #         return partial(self.random_world_translation, config=config)
    #     noise_translate_std = config['NOISE_TRANSLATE_STD']
    #     if noise_translate_std == 0:
    #         return data_dict
    #     gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
    #     for cur_axis in config['ALONG_AXIS_LIST']:
    #         assert cur_axis in ['x', 'y', 'z']
    #         gt_boxes, points = getattr(augmentor_utils, 'random_translation_along_%s' % cur_axis)(
    #             gt_boxes, points, noise_translate_std,
    #         )

    #     data_dict['gt_boxes'] = gt_boxes
    #     data_dict['points'] = points
    #     return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)
        
        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)
        
        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper: 
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)
        
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                                 config['SWAP_PROB'],
                                                                 config['SWAP_MAX_NUM'],
                                                                 pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...
        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        if data_dict['gt_boxes'].shape[0] > 0:
            data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
        # if 'calib' in data_dict:
        #     data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]
            
            data_dict.pop('gt_boxes_mask')
        return data_dict