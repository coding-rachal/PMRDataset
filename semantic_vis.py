import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
import pickle

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# import carla

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

'''
def semantic_lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T

    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    # int_color *= np.array(data['CosAngle'])[:, None]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='semantic_vis')

    parser.add_argument('--annots', type=str, default='')
    parser.add_argument('--lidar_no', type=str, default='lidar1')
    args = parser.parse_args()

    show_axis = False
    f=open(args.annots,'rb')
    lidar_dict = pickle.load(f)
    # print(len(lidar_dict['semantic_p_cloud']))
    semantic_p_cloud = lidar_dict['lidar'][args.lidar_no]['semantic_lidar_points']

    point_list = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Carla Lidar',
        width=960,
        height=540,
        left=480,
        top=270)
    vis.get_render_option().background_color = [0.9, 0.9, 0.9]
    vis.get_render_option().point_size = 1.3
    vis.get_render_option().show_coordinate_frame = True

    if show_axis:
        add_open3d_axis(vis)

    frame = 0
    dt0 = datetime.now()
    
    # for i in range(12,20):
    #     print(i)
    for data in semantic_p_cloud:
        # print(data)
        # data = np.reshape(data, (int(data.shape[0] / 6), 6))
        # print(np.unique(data[:,5]))
        points = np.array([data[:,0], -data[:,1], data[:,2]]).astype(np.float32).T
        labels = np.array(data[:,3])
        difflabels = set(list(labels))
        print(difflabels)
        Pedestrian_index = np.where(labels==0)[0]
        int_color = LABEL_COLORS[labels%23]
        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)
        if frame == 2:
            vis.add_geometry(point_list)
        vis.update_geometry(point_list)

        vis.poll_events()
        vis.update_renderer()
        # # This can fix Open3D jittering issues:
        time.sleep(0.12)
        frame += 1

  
        
        