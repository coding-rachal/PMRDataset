import sys
import os
import cv2
import numpy as np
# import carla
import torch
from constants import joint_mapping, carla_pmr_23, SMPL_ALL_44, COCO_17
from camera import make_rotation_batch, make_4x4_pose
import math
from kpts_projecting import *



def convert_wc_to_cc(joint_world, camera_intrinsic):
    """
    世界坐标系 -> 相机坐标系: R * (pt - T):
    joint_cam = np.dot(R, (joint_world - T).T).T
    :return:
    """
    joint_world = np.asarray(joint_world)
    R = np.asarray(camera_intrinsic["R"])
    T = np.asarray(camera_intrinsic["T"])
    joint_num = len(joint_world)
    # 世界坐标系 -> 相机坐标系
    # [R|t] world coords -> camera coords
    # joint_cam = np.zeros((joint_num, 3))  # joint camera
    # for i in range(joint_num):  # joint i
    #     joint_cam[i] = np.dot(R, joint_world[i] - T)  # R * (pt - T)
    # .T is 转置, T is translation mat
    joint_cam = np.dot(R, (joint_world - T).T).T  # R * (pt - T)
    return joint_cam

def convert_cc_to_wc(joint_world, camera_intrinsic):
    """
    相机坐标系 -> 世界坐标系: inv(R) * pt +T 
    joint_cam = np.dot(inv(R), joint_world.T)+T
    :return:
    """
    joint_world = np.asarray(joint_world)
    R = np.asarray(camera_intrinsic["R"])
    T = np.asarray(camera_intrinsic["T"])
    # 相机坐标系 -> 世界坐标系
    joint_cam = np.dot(np.linalg.inv(R), joint_world.T).T + T
    return joint_cam


def cam2pixel(cam_coord, f, c):
    """
    相机坐标系 -> 像素坐标系: (f / dx) * (X / Z) = f * (X / Z) / dx
    cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
    将从3D(X,Y,Z)映射到2D像素坐标P(u,v)计算公式为：
    u = X * fx / Z + cx
    v = Y * fy / Z + cy
    D(v,u) = Z / Alpha
    =====================================================
    camera_matrix = [[428.30114, 0.,   316.41648],
                    [   0.,    427.00564, 218.34591],
                    [   0.,      0.,    1.]])
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]
    =====================================================
    :param cam_coord:
    :param f: [fx,fy]
    :param c: [cx,cy]
    :return:
    """
    # 等价于：(f / dx) * (X / Z) = f * (X / Z) / dx
    # 三角变换， / dx, + center_x
    u = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
    v = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
    d = cam_coord[..., 2]
    return u, v, d


def convert_cc_to_ic(joint_cam, camera_intrinsic):
    """
    相机坐标系 -> 像素坐标系
    :param joint_cam:
    :return:
    """
    # 相机坐标系 -> 像素坐标系，并 get relative depth
    # Subtract center depth
    # 选择 Pelvis骨盆 所在位置作为相机中心，后面用之求relative depth
    root_idx = 0
    center_cam = joint_cam[root_idx]  # (x,y,z) mm
    joint_num = len(joint_cam)
    f = camera_intrinsic["f"]
    c = camera_intrinsic["c"]
    # joint image_dict，像素坐标系，Depth 为相对深度 mm
    joint_img = np.zeros((joint_num, 3))
    joint_img[:, 0], joint_img[:, 1], joint_img[:, 2] = cam2pixel(joint_cam, f, c)  # x,y
    joint_img[:, 2] = joint_img[:, 2] - center_cam[2]  # z
    return joint_img

#---------------------------------------------------------------------

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def projection_carla(world_2_camera, K, points3d):

    # build the points array in numpy format as (x, y, z, 1) to be operable with a 4x4 matrix
    points_temp = []
    for p in points3d:
        points_temp += [p[0], p[1], p[2], 1]
    points = np.array(points_temp).reshape(-1, 4).T
    
    # convert world points to camera space
    points_camera = np.dot(world_2_camera, points)
    
    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth component also
    points = np.array([
        points_camera[1],
        points_camera[2] * -1,
        points_camera[0]])
    # points = np.array([
    #     points_camera[0],
    #     points_camera[1],
    #     points_camera[2]])
    
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, points)

    # normalize the values and transpose
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]]).T

    return points_2d

def get_screen_points(trans_cam, K, image_w, image_h, points3d):
    camera_transform = carla.Transform(carla.Location(x=trans_cam[0][0], y=trans_cam[0][1], z=trans_cam[0][2]))
    # print('111:', camera_transform.get_inverse_matrix())
    camera_transform.rotation.pitch = trans_cam[1][0]
    camera_transform.rotation.yaw = trans_cam[1][1]
    camera_transform.rotation.roll = trans_cam[1][2]
    # print('222:', camera_transform.get_inverse_matrix())
    # get 4x4 matrix to transform points from world to camera coordinates
    world_2_camera = np.array(camera_transform.get_inverse_matrix())

    # build the points array in numpy format as (x, y, z, 1) to be operable with a 4x4 matrix
    points_temp = []
    for p in points3d:
        points_temp += [p[0], p[1], p[2], 1]
    points = np.array(points_temp).reshape(-1, 4).T
    
    # convert world points to camera space
    points_camera = np.dot(world_2_camera, points)
    
    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth component also
    points = np.array([
        points_camera[1],
        points_camera[2] * -1,
        points_camera[0]])
    
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, points)

    # normalize the values and transpose
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]]).T
    return points_2d


###############################################

def get_w2c_carlacoord(trans_cam, org=None):
    def convert_coord_carla_withr(trans, rot):
        rot = torch.from_numpy(rot).double()
        rot_inverse = torch.linalg.inv(rot)
        trans = torch.from_numpy(trans)
        trans_afterrot = torch.einsum("...ij,...j->...i", -rot, trans)
        return trans_afterrot.numpy()
    T = trans_cam.shape[0]
    cam_euler = torch.from_numpy(trans_cam[:, 1, :]) * math.pi / 180.0
    cam_euler_array = list(cam_euler.numpy())
    cam_R = make_rotation_batch(cam_euler[:,2], cam_euler[:,0], cam_euler[:,1]).numpy().reshape(1, T, 3, 3)
    cam_T = trans_cam[:, 0, :].reshape(1, T, -1)- org.reshape(1,T,-1) if org is not None else trans_cam[:, 0, :].reshape(1, T, -1) ######
    c2w = make_4x4_pose(torch.from_numpy(cam_R[0]), torch.from_numpy(cam_T[0]))
    w2c = torch.linalg.inv(c2w)
    # cam_T = convert_coord_carla_withr(cam_T , cam_R)
    # w2c = make_4x4_pose(torch.from_numpy(cam_R[0]), torch.from_numpy(cam_T[0])) #*,4,4
    return w2c

def get_w2c_slahmrcoord(trans_cam, org=None):
    def convert_coord_carla_withr(trans, rot):
        rot = torch.from_numpy(rot).double()
        rot_inverse = torch.linalg.inv(rot)
        trans = torch.from_numpy(trans)
        # trans_afterrot = torch.einsum("...ij,...j->...i", rot, trans)
        trans_afterrot = torch.einsum("...ij,...j->...i", -rot, trans)
        return trans_afterrot.numpy()
    T = trans_cam.shape[0]
    cam_euler = torch.from_numpy(trans_cam[:, 1, :]) * math.pi / 180.0
    cam_euler_array = list(cam_euler.numpy())
    cam_R = make_rotation_batch(-cam_euler[:,0], - cam_euler[:,1], -cam_euler[:,2], 'zyx').numpy().reshape(1, T, 3, 3)
    
    cam_T = trans_cam[:, 0, :].reshape(1, T, -1)- org.reshape(1,T,-1) if org is not None else trans_cam[:, 0, :].reshape(1, T, -1) ######
    index = [1,2,0]
    cam_T[:,:,] = cam_T[:,:,index]
    cam_T[:,:,1] = -cam_T[:,:,1]
    cam_T = convert_coord_carla_withr(cam_T , cam_R)
    w2c = make_4x4_pose(torch.from_numpy(cam_R[0]), torch.from_numpy(cam_T[0])) #*,4,4
    return w2c
    # return cam_R, cam_T

def get_trans_slahmrcoord(trans, crl_root=None): # trans:T,n,3
    trans = trans - crl_root[:,None,:] if crl_root is not None else trans
    index = [1,2,0]
    trans[:,:,] = trans[:,:,index]
    trans[:,:,1] = -trans[:,:,1] #+0.3
    return trans

def get_trans_carlacoord(trans, crl_root=None): # trans:T,n,3
    trans = trans - crl_root[:,None,:] if crl_root is not None else trans
    return trans

def transform_trans_w2c(w2c, trans_world):#T,4,4  transworld:T,n,3
    w2cs = w2c.numpy()[:,None]
    T = trans_world.shape[0]
    n = trans_world.shape[1]
    w2cs = np.repeat(w2cs, n, axis=1) #T,n,4,4
    trans_world = np.concatenate((trans_world, np.ones((T,n,1))), axis=2)
    trans_cam = torch.einsum("...ij,...j->...i", torch.from_numpy(w2cs), torch.from_numpy(trans_world))
    return trans_cam.numpy()[:,:,:3]

def get_screen_points_coco(sklt_choices, trans_cam):  #sklt_choices:T,20,
    T_sklt = sklt_choices.shape[0]
    kpts3d = (sklt_choices[:, :, 1:]/100.0).reshape(T_sklt,20,-1,3)
    selected_kpts = list(range(1,20)) #  去掉toe_end和crlroot
    selected_kpts.extend(list(range(21, 25)))
    crl_root = kpts3d[:,10,0].reshape(-1,3)
    crl_root_repeat = crl_root[:,None].repeat(20, axis=1).reshape(-1,3)
    trans = kpts3d[:,:, selected_kpts].reshape(-1, 23, 3)


    trans_cam = np.array(trans_cam)
    T = min(trans.shape[0], trans_cam.shape[0])
    trans = get_trans_slahmrcoord(trans, crl_root_repeat)
    w2c = get_w2c_slahmrcoord(trans_cam, crl_root)[:,None]
    w2c = np.repeat(w2c,20,axis=1)
    w2c = w2c.reshape(-1,4,4)
    trans_cam = transform_trans_w2c(w2c, trans)  #T*20, 23, 3
    width = 1280
    height = 640
    fov = 60
    K = torch.from_numpy(build_projection_matrix(width, height, fov))
    points_2d = torch.einsum("...i,ij->...j", torch.from_numpy(trans_cam), K)
    # points_2d = torch.einsum("ij, ...j->...i", K, torch.from_numpy(trans_cam))
    points_2d[:,:,0]/=points_2d[:,:,2]
    points_2d[:,:,1]/=points_2d[:,:,2]
    points_2d[:,:,2]/=points_2d[:,:,2]
    points_2d = np.concatenate((points_2d, np.zeros((points_2d.shape[0], 1, 3))), axis=1)
    kp3d_mapper = joint_mapping(carla_pmr_23, COCO_17)
    kpts_converted = points_2d[:, kp3d_mapper]  #也许会存在一些对应不上的点 T*20, 17, 3 无效点为-1
    kpts_converted = kpts_converted.reshape(T_sklt, 20, 17, 3)   #[:,:,:,:2]
    
    print('111')
    return kpts_converted

def get_screen_points_coco_v2(sklt_choices, trans_cam, choice_n=20):  #sklt_choices:T,20,
    T_sklt = sklt_choices.shape[0]
    kpts3d = (sklt_choices[:, :, 1:]/100.).reshape(T_sklt,choice_n,-1,3)
    selected_kpts = list(range(1,20)) #  去掉toe_end和crlroot
    selected_kpts.extend(list(range(21, 25)))
    trans = kpts3d[:,:, selected_kpts].reshape(T_sklt, -1, 3)

    trans_cam = np.array(trans_cam)
    T = min(trans.shape[0], trans_cam.shape[0])
    
    width = 1280
    height = 640
    fov = 60
    K = torch.from_numpy(build_projection_matrix(width, height, fov))
    trans_total = np.ones((T, trans.shape[1],3))
    for t in range(T):
        camera_pos = trans_cam[t][0]
        # camera_pos *= (100, 100, 100)  
        camera_rot = trans_cam[t][1]
        camera_rot[[0, 1]] = camera_rot[[1, 0]] 
        walker_points = camera_transform(trans[t], camera_pos, camera_rot)
        points_2d, visible = project_viewport_transform(walker_points, width=1280, height=640, fov=60)
        points_2d_v2 = project_viewport_transform_v2(walker_points, K)
        points_2d_v3 = project_viewport_transform_v3(trans[t], camera_pos, camera_rot, K)
        trans_total[t,:,:2]=points_2d
    trans_total = trans_total.reshape(T, choice_n, -1, 3)
    trans_total = trans_total.reshape(T*choice_n,-1,3)
    
    trans_total = np.concatenate((trans_total, np.zeros((trans_total.shape[0], 1, 3))), axis=1)
    kp3d_mapper = joint_mapping(carla_pmr_23, COCO_17)
    kpts_converted = trans_total[:, kp3d_mapper]  #也许会存在一些对应不上的点 T*choice_n, 17, 3 无效点为-1
    kpts_converted = kpts_converted.reshape(T, choice_n, 17, 3)   #[:,:,:,:2]

    return kpts_converted




