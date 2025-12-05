import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import cv2
import imageio

v3d = o3d.utility.Vector3dVector
p2v = o3d.geometry.VoxelGrid.create_from_point_cloud

class Visualizer3D():
    
    def __init__(self,
                 point_size:float=2.0,
                 window_name:str='Default Visualizer',
                 width=1920,
                 height=1080):
        vis = o3d.visualization.Visualizer()
        
        self.vis = vis
        vis.create_window(window_name=window_name,
                          width=width,
                          height=height)
        vis.get_render_option().point_size = point_size
        vis.get_render_option().background_color = [0, 0, 0]
        
    def show(self):
        vis = self.vis
        
        vis.run()
        vis.destroy_window()
    
    def points_to_pcd(self,
                      points:np.ndarray,
                      point_color:List[float]=[1,1,1]):
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = v3d(points)
        pcd.colors= v3d(np.array([point_color for _ in range(len(pcd.points))]))
        
        return pcd

    def set_points(self,
                   points:np.ndarray,
                   point_color:List[float]=[1,1,1]):
        vis = self.vis

        pcd = self.points_to_pcd(points[...,:3], point_color)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
            
        vis.poll_events()
        vis.update_renderer()
    
    def add_axis(self, 
                 origin:List[float]=[0, 0, 0]):
        vis = self.vis
        
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=origin)
        vis.add_geometry(axis)
    
    def translate_boxes_to_open3d_instance(self, gt_boxes):
        """
                4-------- 6
            /|         /|
            5 -------- 3 .
            | |        | |
            . 7 -------- 1
            |/         |/
            2 -------- 0
        """
        center = gt_boxes[0:3]
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = o3d.utility.Vector2iVector(lines)

        return line_set, box3d

    def draw_box(self, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
        vis = self.vis
        for i in range(gt_boxes.shape[0]):
            line_set, box3d = self.translate_boxes_to_open3d_instance(gt_boxes[i])
            if ref_labels is None:
                line_set.paint_uniform_color(color)
            #else:
            #    line_set.paint_uniform_color(box_colormap[ref_labels[i]])

            vis.add_geometry(line_set)
            vis.update_geometry(line_set)
            
        vis.poll_events()
        vis.update_renderer()
            # if score is not None:
            #     corners = box3d.get_box_points()
            #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    
    def add_bboxes_3d(self,
                      bboxes):
        vis = self.vis
        for bbox in bboxes:
            
            vis.add_geometry(bbox)
            vis.update_geometry(bbox)
            
        vis.poll_events()
        vis.update_renderer()
    
    def set_camera_params(self,
                          param_path:str):
        vis = self.vis
        view_control = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters(param_path)
        view_control.convert_from_pinhole_camera_parameters(parameters, 
                                                            allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

    def save_to_image(self,
                      output_folder:str,
                      index:int):
        vis = self.vis
        image = vis.capture_screen_float_buffer(True)
        plt_image = np.asarray(image) * 255
        plt_image = plt_image.astype(np.uint8)
        plt_image = plt_image[:, :, ::-1]
        cv2.imwrite(f"{output_folder}/frame_{index:07d}.png", plt_image)
        
        vis.destroy_window()

def create_video(input_folder, output_file, fps=10):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort()

    # 첫 번째 이미지로부터 프레임 크기 추출
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    # 비디오 라이터 설정
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
def create_gif(input_folder, save_path, interval=0.1):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort()
    
    frames = [cv2.imread(os.path.join(input_folder, x)) for x in images]
    
    gif_config = {
        'loop':0, 
        'duration':interval 
    }
    
    imageio.mimwrite(os.path.join(save_path, 'result.gif'), ## 저장 경로
                 frames, 
                 format='gif', 
                 **gif_config 
                )