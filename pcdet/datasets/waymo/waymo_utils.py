# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import pickle
import numpy as np
from ...utils import common_utils
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
from PIL import Image
from io import BytesIO

try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
cam_list = [
    '_FRONT',
    '_FRONT_LEFT',
    '_FRONT_RIGHT',
    '_SIDE_LEFT',
    '_SIDE_RIGHT',
]
type_list = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
selected_waymo_classes = ['Vehicle', 'Pedestrian', 'Sign', 'Cyclist']

def generate_labels(frame):

    # id_to_bbox = dict()
    # id_to_name = dict()
    # for labels in frame.projected_lidar_labels:
    #     name = labels.name
    #     for label in labels.labels:
    #         # TODO: need a workaround as bbox may not belong to front cam
    #         bbox = [
    #             label.box.center_x - label.box.length / 2,
    #             label.box.center_y - label.box.width / 2,
    #             label.box.center_x + label.box.length / 2,
    #             label.box.center_y + label.box.width / 2
    #         ]
    #         id_to_bbox[label.id] = bbox
    #         id_to_name[label.id] = name - 1

    # group_id = 0
    # instance_infos = []
    # for obj in frame.laser_labels:
    #     instance_info = dict()
    #     bounding_box = None
    #     name = None
    #     id = obj.id
    #     for proj_cam in cam_list:
    #         if id + proj_cam in id_to_bbox:
    #             bounding_box = id_to_bbox.get(id + proj_cam)
    #             name = id_to_name.get(id + proj_cam)
    #             break

    #     # if True:
    #     #     if obj.most_visible_camera_name:
    #     #         name = cam_list.index(
    #     #             f'_{obj.most_visible_camera_name}')
    #     #         box3d = obj.camera_synced_box
    #     #     else:
    #     #         continue
        
    #     box3d_lidar = obj.box
    #     box3d = obj.box

    #     if bounding_box is None or name is None:
    #         name = 0
    #         bounding_box = [0.0, 0.0, 0.0, 0.0]

    #     my_type = type_list[obj.type]

    #     if my_type not in selected_waymo_classes:
    #         continue
    #     else:
    #         label = selected_waymo_classes.index(my_type)

    #     if True and obj.num_lidar_points_in_box < 1:
    #         continue

    #     group_id += 1
    #     instance_info['obj_id'] = id
    #     instance_info['group_id'] = group_id
    #     instance_info['camera_id'] = name
    #     instance_info['bbox'] = bounding_box
    #     instance_info['bbox_label'] = label
    #     instance_info['obj_name'] = my_type
    #     instance_info['tracking_difficulty'] = obj.tracking_difficulty_level
    #     instance_info['difficulty'] = obj.detection_difficulty_level

    #     l_height = box3d_lidar.height
    #     l_width = box3d_lidar.width
    #     l_length = box3d_lidar.length
    #     l_x = box3d_lidar.center_x
    #     l_y = box3d_lidar.center_y
    #     l_z = box3d_lidar.center_z
    #     l_rotation_y = box3d_lidar.heading

    #     instance_info['bbox_3d_lidar'] = np.array(
    #         [l_x, l_y, l_z, l_length, l_width, l_height, l_rotation_y]).astype(np.float32).tolist()

    #     height = box3d.height
    #     width = box3d.width
    #     length = box3d.length

    #     # NOTE: We save the bottom center of 3D bboxes.
    #     x = box3d.center_x
    #     y = box3d.center_y
    #     z = box3d.center_z - height * 0.5

    #     rotation_y = box3d.heading

    #     instance_info['bbox_3d'] = np.array(
    #         [x, y, z, length, width, height, rotation_y]).astype(np.float32).tolist()
    #     instance_info['bbox_label_3d'] = label
    #     instance_info['num_points_in_gt'] = obj.num_lidar_points_in_box

    #     if True:
    #         instance_info['track_id'] = obj.id
    #     instance_infos.append(instance_info)
  
    # annotations = {}
    # annotations['name'] = np.array([x['obj_name'] for x in instance_infos])
    # annotations['difficulty'] = np.array([x['difficulty'] for x in instance_infos])
    # annotations['dimensions'] = np.array([x['bbox_3d'][3:6] for x in instance_infos])
    # annotations['location'] = np.array([x['bbox_3d'][:3] for x in instance_infos])
    # annotations['heading_angles'] = np.array([x['bbox_3d'][-1] for x in instance_infos])

    # annotations['obj_ids'] = np.array([x['obj_id'] for x in instance_infos])
    # annotations['tracking_difficulty'] = np.array([x['tracking_difficulty'] for x in instance_infos])
    # annotations['num_points_in_gt'] = np.array([x['num_points_in_gt'] for x in instance_infos])
    # # annotations['gt_boxes_lidar'] = np.array([x['bbox_3d'] for x in instance_infos])
    # annotations['camera_id'] = np.array([x['camera_id'] for x in instance_infos])
    # annotations['bbox'] = np.array([x['bbox'] for x in instance_infos])
    # annotations['bbox_3d_lidar'] = np.array([x['bbox_3d_lidar'] for x in instance_infos])
    # annotations['gt_boxes_lidar'] = annotations['bbox_3d_lidar'].copy()

    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels
    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)

    annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis]],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 7))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    
    return annotations


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def save_lidar_points(frame, cur_save_path, use_two_returns=True):
    #range_images, camera_projections, seg_labels, range_image_top_pose = \
    #    frame_utils.parse_range_image_and_camera_projection(frame)
    range_images, camera_projections, range_image_top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    
    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1) if use_two_returns else (0,)
    )

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)

    np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)
    return num_points_of_each_lidar

def cart_to_homo( mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret
    
def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True, use_two_returns=True,  update_info_only=False):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    sequence_infos = []
    # if pkl_file.exists():
    #     sequence_infos = pickle.load(open(pkl_file, 'rb'))
    #     print('Skip sequence since it has been processed before: %s' % pkl_file)
    #     return sequence_infos

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }

        image_info = dict()
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_SIDE_LEFT',
            'CAM_SIDE_RIGHT',
        ]

        # for j in range(5):
        #     width = frame.context.camera_calibrations[j].width
        #     height = frame.context.camera_calibrations[j].height
        #     image_info.update({'image_shape_%d' % j: (height, width)})
        # info['image'] = image_info

        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        Tr_velo_to_cams = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            Tr_velo_to_cams.append(Tr_velo_to_cam)

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calibs.append(camera_calib)

        for i, (cam_key, camera_calib, Tr_velo_to_cam) in enumerate(
                zip(camera_types, camera_calibs, Tr_velo_to_cams)):
            
            cam_infos = dict()
            cam_infos['img_path'] = str(cam_key) + '.jpg'
            
            # NOTE: frames.images order is different

            for img in frame.images:
                if img.name == i + 1:
                    img = Image.open(BytesIO(img.image))
                    width, height = img.size
                    img_save_path = os.path.join(cur_save_dir, str(cnt))
                    if os.path.exists(img_save_path) == False:
                        os.mkdir(img_save_path)
                    img_size = img.size
                    # img = img.resize((int(img_size[0] * 0.5), int(img_size[1] * 0.5)), Image.Resampling.BILINEAR)
                    img.save(os.path.join(cur_save_dir, str(cnt), str(cam_key) + '.jpg'),"JPEG")
                    
            cam_infos['height'] = height
            cam_infos['width'] = width
            cam_infos['lidar2cam'] = Tr_velo_to_cam.astype(np.float32).tolist()
            cam_infos['cam2img'] = camera_calib.astype(np.float32).tolist()
            cam_infos['lidar2img'] = (camera_calib @ Tr_velo_to_cam).astype(
                np.float32).tolist()
            
            image_info[cam_key] = cam_infos
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame)
            info['annos'] = annotations

        num_points_of_each_lidar = save_lidar_points(
            frame, cur_save_dir / ('%04d.npy' % cnt), use_two_returns=use_two_returns
        )
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos


