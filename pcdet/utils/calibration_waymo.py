import numpy as np
import torch.nn.functional as F
from logging import raiseExceptions
from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from . import common_utils, loss_utils, box_utils
from matplotlib import pyplot as plt

class Calibration(object):
    def __init__(self, cam_infos):
        infos = []
        for idx, key in enumerate(cam_infos.keys()):
            lidar2img = cam_infos[key]['lidar2img']
            lidar2img = np.array(lidar2img, dtype=np.float32)
            infos.append(lidar2img)
        self.infos = infos
        
    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_img(self, pts_lidar, cam_id):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        lidar2img = self.infos[cam_id]

        pts_img = pts_lidar.clone()
        ones = pts_img.new_ones((pts_img.shape[0], 1))
        pts_img = torch.cat((pts_img, ones), dim=-1).cpu().numpy()
        
        pts_img = pts_img @ lidar2img.T
        pts_img[:, 0] /= (pts_img[:, -1] + 1e-07)
        pts_img[:, 1] /= (pts_img[:, -1] + 1e-07)
        
        return pts_img[:, :2], pts_img[:, -1]
    
    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner