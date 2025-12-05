import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from skimage import io

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from visualizer import Visualizer3D, create_video, create_gif
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from tqdm import tqdm

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        
        self.root_path = root_path
        
        self.lidar_root = os.path.join(self.root_path, 'velodyne_points/data')
        self.lidar_files = os.listdir(self.lidar_root)
        self.lidar_files.sort()
        
        self.img_root = os.path.join(self.root_path, 'image_02/data')
        self.img_files = os.listdir(self.img_root)
        self.img_files.sort()
        
        self.ext = '.bin'
        
    def __len__(self):
        return len(self.lidar_files)

    def __getitem__(self, index):
        
        points = self.get_lidar(index)
        images = self.get_image(index)
        
        calib_file = '../data/kitti/rawdata/calib.txt'
        calib = calibration_kitti.Calibration(calib_file)
        
        input_dict = {
            'points': points,
            'images' : images,
            'frame_id': index,
            'calib': calib
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
    def get_lidar(self, idx):
        
        if self.ext == '.bin':
            points = np.fromfile(os.path.join(self.lidar_root, self.lidar_files[idx]), dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[idx])
        else:
            raise NotImplementedError

        return points
    
    def get_image(self, idx):
        
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        
        img_file = os.path.join(self.img_root, self.img_files[idx])
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    root_path = args.data_path
    vis_results_path = os.path.join(root_path, 'vis_results')
    if not os.path.exists(vis_results_path):
        os.makedirs(vis_results_path)
         
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    conf_score_thres = 0.3
    
    with torch.no_grad():
        for idx, data_dict in tqdm(enumerate(demo_dataset)):
            
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _, points_dict = model.forward(data_dict)
            
            pred_boxes = pred_dicts[0]['pred_boxes']
            pred_scores = pred_dicts[0]['pred_scores']
            pred_labels = pred_dicts[0]['pred_labels']
            
            mask = pred_scores >= conf_score_thres
            pred_boxes = pred_boxes[mask]
            
            points = data_dict['points'][...,1:]
            rebuilt_points = points_dict['rebuilt_points'][pred_dicts[0]['selected']]
            rebuilt_points = rebuilt_points[mask]
            rebuilt_points = rebuilt_points[...,1:]
            rebuilt_points = rebuilt_points.view(-1, rebuilt_points.shape[-1])
            
            # points_indices = roiaware_pool3d_utils.points_in_boxes_gpu(rebuilt_points[...,:3].unsqueeze(0), pred_boxes[:,:7].unsqueeze(0)).long()
            # points_indices = points_indices.view(-1)
            # # points_indices = torch.unique(points_indices)[1:]
            # rebuilt_points = rebuilt_points[points_indices >= 0]
            
            #non_empty_boxes = pred_boxes[points_indices]
            #gt_boxes[i][...] = 0
            #gt_boxes[i][:non_empty_boxes.shape[0]] = non_empty_boxes
            
            #import pdb;pdb.set_trace()
            vis = Visualizer3D(width=1920, height=1080)
            vis.set_points(points.cpu().numpy())
            # vis.set_points(rebuilt_points.cpu().numpy(), [0, 1, 0])
            # vis.draw_box(pred_boxes.cpu().numpy())
            vis.set_camera_params('../data/kitti/rawdata/cam_params.json')
            # vis.show()
            vis.save_to_image(vis_results_path, idx)
            
            #import pdb;pdb.set_trace()

    create_gif(vis_results_path, root_path)
    
    logger.info('Demo done.')

if __name__ == '__main__':
    main()
