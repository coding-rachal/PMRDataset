import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
# import torch
import pickle
# from torch.autograd.variable import Variable
import os
from PIL import Image
import math
from scipy.spatial.transform import Rotation as R
# from camera_tools import *
import matplotlib.pyplot as plt
import copy
import torch
import cv2
import copy

def vis_kpts(cam_poses, points, save_dir, bg_path=None):
    width = 1280
    height = 640
    fov = 60
    kpts = points['kpts_3d']
    if 'lidar' in points.keys():
      ped_points = points['lidar']
      foot_center = points['foot_center']
    views = ['view1']
    steps = 30
    bg_ind = 0
    connect = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5,6),
            (6, 7), (3, 8), (8, 9), (9, 10),
            (9, 11), (3, 12), (12, 13), (13, 14), (14, 15),
            (1, 16), (16, 17), (17, 18), (18, 19), (1, 20),
            (20, 21), (21, 22), (22, 23)]    #,  (24, 25)(19, 20)]  #change
    cmap = plt.cm.get_cmap('viridis')
    K = build_projection_matrix(width, height, fov)
    T_w2c = torch.linalg.inv(cam_poses).numpy()
    n = kpts.shape[1]
    normalize = plt.Normalize(vmin=0, vmax=n)
    cmap = plt.cm.get_cmap('viridis')
    # save_dir = os.path.join(save_root, 'kpts3d')
    os.makedirs(save_dir, exist_ok=True)
    bg_img = cv2.imread(bg_path)[:,:,::-1]
    for i in range(kpts.shape[0]):
        # joints_cam = np.dot(T_w2c[i], kpts[i])
        # joints_2d = np.dot(K, joints_cam)
        if T_w2c.shape[0]==1:
          T_w2c_i = T_w2c[0]
        else:
          T_w2c_i = T_w2c[i]
        joints_2d = get_screen_points_v2(T_w2c_i, K, kpts[i])
        # buffer = np.ones((height, width, 3), np.uint8)*255
        buffer = bg_img.copy()
        # color = cmap(normalize(i))
        for ic, c in enumerate(connect):
            cv2.line(buffer, joints_2d[c[0]-1][:2].astype(int), joints_2d[c[1]-1][:2].astype(int), (0,0,0), 3)
            # draw_line_on_buffer(buffer, width, height, (joints_2d[c[0]-1], joints_2d[c[1]-1]), color=[255,0,0])
        
        if 'lidar' in points.keys():
          ped_points_i = get_screen_points_v2(T_w2c_i, K, ped_points[i]-foot_center[None])
          for p in ped_points_i:
            cv2.circle(buffer, p[:2].astype(int), 3, (255, 0, 0), -1)

        img = Image.fromarray(buffer)
        # savepath = '' ##
        savepath = os.path.join(save_dir, '{:04d}.jpg'.format(i))
        img.save(savepath)
    print(f'saving kpts in {save_dir}')
    return

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def vis_lidar(cam_poses, lidar, kpts):
    width = 1280
    height = 640
    fov = 60
    views = ['view1']
    steps = 30
    bg_ind = 0
    connect = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5,6),
            (6, 7), (4, 8), (8, 9), (9, 10),
            (9, 11), (3, 12), (12, 13), (13, 14), (14, 15),
            (1, 16), (16, 17), (17, 18), (18, 19), (19, 20),
            (1, 21), (21, 22), (22, 23), (23, 24), (24, 25)
    ]  #change
    cmap = plt.cm.get_cmap('viridis')
    K = build_projection_matrix(width, height, fov)
    T_w2c = torch.linalg.inv(cam_poses)
    for i in range(len(lidar)):
        lidar_points = lidar[i] #(n,3)
        lidar_points = lidar_points[np.where(lidar_points[0])] ##
        lidar2d = get_screen_points_v2(T_w2c[i], lidar_points[i])

        buffer = np.ones((height, width, 3), np.uint8)*255
        for p in lidar2d:
            x, y = p[0], p[1]
            cv2.circle(buffer, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        img = Image.fromarray(buffer)
        savepath = '' ##
        img.save(savepath)
    return

def draw_points_on_buffer(buffer, image_w, image_h, points_2d, color, size=4):
    half = int(size / 2)
    # draw each point
    for p in points_2d:
        x = int(p[0])
        y = int(p[1])
        # print('x:{}, y:{}'.format(x, y))
        for j in range(y - half, y + half):
            if (j >=0 and j <image_h):
                for i in range(x - half, x + half):
                    if (i >=0 and i <image_w):
                        buffer[j][i][0] = color[0]*255.0
                        buffer[j][i][1] = color[1]*255.0
                        buffer[j][i][2] = color[2]*255.0



def get_screen_points_v2(world_2_camera, K, points3d):
    

    # build the points array in numpy format as (x, y, z, 1) to be operable with a 4x4 matrix
    points_temp = []
    for p in points3d:
        points_temp += [p[0], p[1], p[2], 1]
    points = np.array(points_temp).reshape(-1, 4).T
    camera_2_word = torch.linalg.inv(torch.from_numpy(world_2_camera[None]))[0]
    # convert world points to camera space
    points_camera = np.dot(world_2_camera, points)
    points = points_camera
    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # # and we remove the fourth component also
    # points = np.array([
    #     points_camera[1],
    #     points_camera[2] * -1,
    #     points_camera[0]])
    points = np.array([
        points_camera[0],
        points_camera[1],
        points_camera[2]])
    
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, points)

    # normalize the values and transpose
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]]).T
    

    return points_2d

def draw_line_on_buffer(buffer, image_w, image_h, points_2d, color, size=4):
  x0 = int(points_2d[0][0])
  y0 = int(points_2d[0][1])
  x1 = int(points_2d[1][0])
  y1 = int(points_2d[1][1])
  dx = abs(x1 - x0)
  if x0 < x1:
    sx = 1
  else:
    sx = -1
  dy = -abs(y1 - y0)
  if y0 < y1:
    sy = 1
  else:
    sy = -1
  err = dx + dy
  while True:
    draw_points_on_buffer(buffer, image_w, image_h, ((x0,y0),), color, size)
    if (x0 == x1 and y0 == y1):
      break
    e2 = 2 * err
    if (e2 >= dy):
      err += dy
      x0 += sx
    if (e2 <= dx):
      err += dx
      y0 += sy

