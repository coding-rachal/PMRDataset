import os
import glob

import imageio
import numpy as np
import torch
import pickle

from util.loaders import resolve_cfg_paths, load_smpl_body_model
from util.tensor import get_device, move_to,  to_torch
from vis.output import prep_result_vis, animate_scene
from vis.viewer import init_viewer
import math
import cv2

def run_vis(
    cfg,
    out_dir,
    dev_id,
    phases=["mosh"],
    # render_views=["src_cam", "above", "side"],
    render_views=["src_cam"],
    make_grid=False,
    overwrite=False,
    save_dir=None,
    render_kps=True,
    render_layers=False,
    save_frames=False,
    **kwargs,
):
    save_dir = out_dir if save_dir is None else save_dir



    print("OUT_DIR", out_dir)
    print("SAVE_DIR", save_dir)
    print("VISUALIZING PHASES", phases)
    print("RENDERING VIEWS", render_views)
    print("OVERWRITE", overwrite)

    total_seq_name = cfg.data.seq
    date = total_seq_name.split('/')[0]
    seq_name = total_seq_name.split('/')[1]
    view_name = total_seq_name.split('/')[2]  ####
    id = seq_name.split('_')[0].replace('id', '')
    annotspkl_root = cfg.data.annots_root
    save_root = cfg.data.save_root
    annotspkl_dir = f'{annotspkl_root}/{date}/{seq_name}.pkl'
    save_dir = f'{save_root}/{seq_name}-{view_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_seq_name = total_seq_name.replace('/', '-')

    # save input frames
    seq_dir = os.path.join(cfg.data.root, cfg.data.seq)
    sel_img_paths = [os.path.join(seq_dir, name) for name in os.listdir(seq_dir)]
    sel_img_paths = sorted(sel_img_paths)
    inp_vid_path = save_input_frames(
        sel_img_paths,
        f"{save_dir}/{save_seq_name}_input.mp4",
        fps=cfg.fps,
        overwrite=True,
    )
    img_size = imageio.imread(sel_img_paths[0]).shape[:2]
    
    cfg.data.img_size = img_size[::-1]
    cfg.data.sel_img_paths = sel_img_paths

    if len(render_views) < 1:
        return

    out_ext = "/" if render_layers or save_frames else ".mp4"
    phase_results = {}
    phase_max_iters = {}
    for phase in phases:
        out_name = f"{save_dir}/{save_seq_name}_{phase}_final"

        out_paths = [f"{out_name}_{view}{out_ext}" for view in render_views]
        if not overwrite and all(os.path.exists(p) for p in out_paths):
            print("FOUND OUT PATHS", out_paths)
            continue

        res = loadannots(annotspkl_dir , view_name)
        phase_results[phase] = out_name, res

    if len(phase_results) > 0:
        out_names, res_dicts = zip(*phase_results.values())
        save_paths = render_results(
            cfg,
            dev_id,
            res_dicts,
            out_names,
            render_views=render_views,
            render_layers=render_layers,
            save_frames=save_frames,
            **kwargs,
        )
    if make_grid:
        save_grid_path = os.path.join(save_dir, 'grid_result.mp4')
        # if 'view1' in seq_dir:
            # save_paths['view1']=seq
        save_paths['view1'] = seq_dir
        seq_oppo_dir = os.path.join(cfg.data.root, cfg.data.seq_oppo)
        save_paths['view2'] = seq_oppo_dir
        save_paths['egoview'] = os.path.join(cfg.data.root, 'egoviews_aligned', '/'.join(cfg.data.seq.split('/')[:-1]))
        make_gridvideo(save_paths, save_grid_path, cfg.fps)


def make_gridvideo(save_dirs, save_path, fps):
    height = 240 #480
    width = 320 #640
    dim = (width, height)
    def crop(img):
        original_dim = (1280, 640)
        cropped_width = int(3*original_dim[0]/4)
        need_crop = (original_dim[0]-cropped_width)//2
        cropped_img = img[:,need_crop:-need_crop]
        return cropped_img
    def concat_h(img_path1, img_path2, img_path3):
        img1 = cv2.resize(crop(cv2.imread(img_path1)[:,:,::-1]), dim)
        img2 = cv2.resize(crop(cv2.imread(img_path2)[:,:,::-1]), dim)
        if img_path3:
            img3 = cv2.resize(crop(cv2.imread(img_path3)[:,:,::-1]), dim)
        else:
            img3 = np.ones(img1.shape, np.uint8)*255
        result = cv2.hconcat([img1, img2, img3])
        return result
    def concat_v(img1, img2):
        result = cv2.vconcat([img1, img2])
        return result
    def write_video(imgs):
        imageio.mimwrite(save_path, imgs, fps=fps)
        print("wrote video to", save_path)        
    len_img = 200
    save_paths = {}
    save_lens = []
    for k,v in save_dirs.items():
        names = os.listdir(v)
        names.sort()
        save_paths[k] = [os.path.join(v, n) for n in names]
        save_lens.append(len(save_paths[k]))
        len_img = min(len_img, len(save_paths[k]))
    concat_imgs = []
    for i in range(len_img):
        img1 = concat_h(save_paths['view1'][i], save_paths['view2'][i], save_paths['egoview'][i])
        img2 = concat_h(save_paths['mesh_above'][i], save_paths['kpts_above'][i], None) # save_paths['semantic_lidar']
        img = concat_v(img1, img2)
        concat_imgs.append(img)
    write_video(concat_imgs)


def render_results(cfg, dev_id, res_dicts, out_names, **kwargs):
    """
    render results for all selected phases
    """
    assert len(res_dicts) == len(out_names)
    if len(res_dicts) < 1:
        print("no results to render, skipping")
        return

    B = 1
    T = res_dicts[0]['vis_mask'].shape[-1]

    device = get_device(dev_id)

    ###instincs
    width = 1280
    height = 640
    fov = 60
    fx = width / (2.0 * math.tan(fov * math.pi / 360.0))
    instrns = torch.from_numpy(np.array([fx, fx, width/2, height/2]))

    # load models
    cfg = resolve_cfg_paths(cfg)
    body_model, _ = load_smpl_body_model(cfg.paths.smpl, B * T, device=device)
    vis = init_viewer(
        cfg.data.img_size,
        instrns,
        # cam_data["intrins"][0],
        vis_scale=1.0,
        bg_paths=cfg.data.sel_img_paths,
        fps=cfg.fps,
    )

    save_paths_all = {}
    save_seq_name = cfg.data.seq.replace('/', '-')
    for res_dict, out_name in zip(res_dicts, out_names):
        res_dict = move_to(res_dict, device)
        scene_dict = prep_result_vis(
            res_dict,
            res_dict["vis_mask"],
            res_dict["track_ids"],
            body_model,
        )
        print(kwargs)
        save_paths = animate_scene(
            vis, scene_dict, out_name, seq_name=save_seq_name, **kwargs
        )
        save_paths_all.update(save_paths)

    vis.close()

    return save_paths_all


def loadannots(pkl_path, view_name):
    # pkl_path = '/data/carla/annots_0611/output_1006/id75_m0_w1_s2.pkl'
    pkl_data = pickle.load(open(pkl_path, 'rb'))
    res = {}
    smplx = pkl_data['smplx']
    extrinsics =  pkl_data['extrinsics'][view_name]
    kpts_3d = pkl_data['kpts3d']
    # res['trans'] = kpts_3d[:,0][None]
    foot_center = (kpts_3d[0,18]+kpts_3d[0,22])/2.0
    res['trans'] = (kpts_3d[:,15]+kpts_3d[:,19])/2.0 - foot_center[None]
    res['trans'] = res['trans'][None]
    res['pose_body'] = smplx['pose_body'][None]
    T = smplx['pose_body'].shape[0]
    res['root_orient'] = smplx['root_orient'][None]
    # res['T_w2c'] = extrinsics['cam_w2c'].numpy()
    res['T_w2c'] = extrinsics

    track_ids = np.array([0])
    res['vis_mask'] = np.ones((1, T))
    res = to_torch(res)
    res['track_ids'] = torch.from_numpy(track_ids)
    
    res['kpts_3d'] = kpts_3d - foot_center[None,None]
    res['lidar'] = pkl_data['lidar']['lidar1']['ped_lidar_points']
    res['foot_center'] = foot_center

    return res

def save_input_frames(sel_img_paths, vid_path, fps=30, overwrite=False):
    if not overwrite and os.path.isfile(vid_path):
        return

    writer = imageio.get_writer(vid_path, fps=fps)
    for path in sel_img_paths:
        writer.append_data(imageio.imread(path))
    writer.close()
    print(f"SAVED {len(sel_img_paths)} INPUT FRAMES TO {vid_path}")
    return vid_path
