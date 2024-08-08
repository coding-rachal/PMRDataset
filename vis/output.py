import os
import imageio
import numpy as np
import torch
import subprocess

from body_model import run_smpl
from geometry import camera as cam_util
from geometry.mesh import make_batch_mesh
from geometry.plane import parse_floor_plane, get_plane_transform

from util.tensor import detach_all, to_torch, move_to

from .fig_specs import get_seq_figure_skip, get_seq_static_lookat_points
from .tools import smpl_to_geometry
from .vis_points import vis_kpts, vis_lidar


def prep_result_vis(res, vis_mask, track_ids, body_model):
    """
    :param res (dict) with (B, T, *) tensor elements, B tracks and T frames
    :param vis_mask (B, T) with visibility of each track in each frame
    :param track_ids (B,) index of each track
    """
    print("RESULT FIELDS", res.keys())
    device = res['trans'].device
    res = detach_all(res)
    with torch.no_grad():
        world_smpl = run_smpl(
            body_model,
            res["trans"],
            res["root_orient"],
            res["pose_body"],
            res.get("betas", None),
        )
    T_w2c = None
    floor_plane = None
    T_w2c_others = None
    T_w2c_oppo = None

    if 'T_w2c' in res:
        T_w2c = res['T_w2c']
    elif "cam_R" in res and "cam_t" in res:
        T_w2c = cam_util.make_4x4_pose(res["cam_R"][0], res["cam_t"][0])
    if 'other_cams' in res:
        other_cams = res['other_cams']
        T_w2c_others = {k: cam_util.make_4x4_pose(other_cams[k]["cam_R"][0], other_cams[k]["cam_t"][0]) for k in other_cams.keys()}
    # if 'cam_R_oppo' in res and 'cam_t_oppo' in res:
    #     T_w2c_oppo = cam_util.make_4x4_pose(res['cam_R_oppo'][0], res['cam_t_oppo'][0])
    if "floor_plane" in res:
        floor_plane = res["floor_plane"][0]
    points = {}
    if 'kpts_3d' in res.keys():
        points['kpts_3d'] = res['kpts_3d']
    if 'lidar' in res.keys():
        points['lidar'] = res['lidar']
        points['foot_center'] = res['foot_center']
    return build_scene_dict(
        world_smpl,
        vis_mask,
        track_ids,
        T_w2c=T_w2c,
        T_w2c_oppo=T_w2c_oppo,
        floor_plane=floor_plane,
        T_w2c_others=T_w2c_others,
        points=points
    )


def build_scene_dict(
    world_smpl, vis_mask, track_ids, T_w2c=None, T_w2c_oppo=None, floor_plane=None, T_w2c_others=None, points=None, **kwargs
):
    scene_dict = {}

    # first get the geometry of the people
    # lists of length T with (B, V, 3), (F, 3), (B, 3)
    scene_dict["geometry"] = smpl_to_geometry(
        world_smpl["vertices"], world_smpl["faces"], vis_mask, track_ids
    )

    if T_w2c is None:
        T_w2c = torch.eye(4)[None]

    T_c2w = torch.linalg.inv(T_w2c)
    # rotate the camera slightly down and translate back and up
    T = cam_util.make_4x4_pose(
        cam_util.rotx(-np.pi / 10), torch.tensor([0, -1, -2])
    ).to(T_c2w.device)

    # T_c2w_oppo = torch.linalg.inv(T_w2c_oppo) if T_w2c_oppo is not None else None
    # # rotate the camera slightly down and translate back and up
    # T = cam_util.make_4x4_pose(
    #     cam_util.rotx(-np.pi / 10), torch.tensor([0, -1, -2])
    # ).to(T_c2w_oppo.device)

    scene_dict["cameras"] = {
        "src_cam": T_c2w,
        "front": torch.einsum("ij,...jk->...ik", T, T_c2w),
    }

    # if T_c2w_oppo is not None:
        # scene_dict["cameras"]["src_cam_oppo"] = T_c2w_oppo

    # if T_w2c_others is not None:
    #     scene_dict['cameras_others'] = {k: torch.linalg.inv(T_w2c_others[k]) for k in T_w2c_others.keys()}

    if floor_plane is not None:
        # compute the ground transform
        # use the first appearance of a track as the reference point
        tid, sid = torch.where(vis_mask > 0)
        idx = tid[torch.argmin(sid)]
        root = world_smpl["joints"][idx, 0, 0].detach().cpu()
        floor = parse_floor_plane(floor_plane.detach().cpu())
        R, t = get_plane_transform(torch.tensor([0.0, 1.0, 0.0]), floor, root)
        scene_dict["ground"] = cam_util.make_4x4_pose(R, t)
    
    if points is not None:
        scene_dict["points"] = points

    return scene_dict


def render_scene_dict(renderer, scene_dict, out_name, fps=30, **kwargs):
    # lists of T (B, V, 3), (B, 3), (F, 3)
    verts, colors, faces, bounds = scene_dict["geometry"]
    print("NUM VERTS", len(verts))

    # add a top view
    scene_dict["cameras"]["above"] = cam_util.make_4x4_pose(
        torch.eye(3), torch.tensor([0, 0, -10])
    )[None]

    for cam_name, cam_poses in scene_dict["cameras"].items():
        print("rendering scene for", cam_name)
        # cam_poses are (T, 4, 4)
        render_bg = cam_name == "src_cam"
        ground_pose = scene_dict.get("ground", None)
        frames = renderer.render_video(
            cam_poses[None], verts, faces, colors, render_bg, ground_pose=ground_pose
        )
        imageio.mimwrite(f"{out_name}_{cam_name}.mp4", frames, fps=fps)


def animate_scene(
    vis,
    scene,
    out_name,
    seq_name=None,
    accumulate=False,
    render_views=["src_cam", "front", "above", "side"],
    render_bg=True,
    render_cam=True,
    render_ground=True,
    debug=False,
    **kwargs,
):
    if len(render_views) < 1:
        return
 
    scene = build_pyrender_scene(
        vis,
        scene,
        seq_name,
        render_views=render_views,
        render_cam=render_cam,
        accumulate=accumulate,
        debug=debug,
    )

    print("RENDERING VIEWS", scene["cameras"].keys())
    render_ground = render_ground #and "ground" in scene
    save_paths = {}
    for cam_name, cam_poses in scene["cameras"].items():
        is_src = cam_name == "src_cam"
        show_bg = is_src and render_bg
        show_ground = render_ground and not is_src
        show_cam = render_cam and not is_src
        vis_name = f"{out_name}_{cam_name}"
        print(f"{cam_name} has {len(cam_poses)} poses")
        skip = 1 if debug else 1
        vis.set_camera_seq(cam_poses[::skip])

        ######
        # kwargs['save_frames'] = False
        save_path = vis.animate(
            vis_name,
            render_bg=show_bg,
            # render_bg_oppo=show_bg_oppo,
            render_ground=show_ground,
            render_cam=show_cam,
            # save_frames=True,
            **kwargs,
        )
        save_layers_name = f'{vis_name}_layers'
        vis.render_layers(save_layers_name, composite=True)

        # save_paths.append(save_path)
        save_paths[f'mesh_{cam_name}'] = vis_name
        bg_path = os.path.join(save_layers_name, 'background.png')

        points = scene['points']
        # points_save_dir = '/'.join(vis_name.split('/')[:-1])
        points_save_dir = vis_name+'_kpts3d'
        if kwargs['vis_kpts2d'] and 'kpts_3d' in points.keys():
                vis_kpts(cam_poses, points, points_save_dir, bg_path=bg_path)  #c2w
                # vis_kpts(cam_poses, points['kpts_3d'], points_save_dir, bg_path=bg_path)  #c2w
        save_paths[f'kpts_{cam_name}'] = points_save_dir
        # if kwargs['vis_lidar'] and 'lidar' in points.keys():
        #         vis_lidar(cam_poses, points['lidar'], points_save_dir)

    return save_paths


def build_pyrender_scene(
    vis,
    scene,
    seq_name,
    render_views=["src_cam", "src_cam_oppo", "front", "above", "side"],
    render_cam=True,
    accumulate=False,
    debug=False,
):
    """
    :param vis (viewer object)
    :param scene (scene dict with geometry, cameras, etc)
    :param accumulate (optional bool, default False) whether to render entire trajectory together
    :param render_views (list str) camera views to render
    """
    if len(render_views) < 1:
        return

    assert all(view in ["src_cam", "src_cam_oppo", "front", "above", "side"] for view in render_views)

    scene = move_to(detach_all(scene), "cpu")
    src_cams = scene["cameras"]["src_cam"]
    verts, colors, faces, bounds = scene["geometry"]
    T = len(verts)
    print(f"{T} mesh frames")

    # set camera views
    if not "cameras" in scene:
        scene["cameras"] = {}

    # remove default views from source camera perspective if desired
    if "src_cam" not in render_views:
        scene["cameras"].pop("src_cam", None)
    if "src_cam_oppo" not in render_views:
        scene["cameras"].pop("src_cam_oppo", None)
    if "front" not in render_views:
        scene["cameras"].pop("front", None)

    # add static viewpoints if desired
    top_pose, side_pose, _skip = get_static_views(seq_name, bounds)
    if "above" in render_views:
        scene["cameras"]["above"] = top_pose[None]
    if "side" in render_views:
        scene["cameras"]["side"] = side_pose[None]

    # accumulate meshes if possible (can only accumulate for static camera)
    moving_cam = "src_cam" in render_views or "front" in render_views
    accumulate = accumulate and not moving_cam
    skip = _skip if accumulate else 1
    skip = 1

    vis.clear_meshes()

    if "ground" in scene:
        vis.set_ground(scene["ground"])

    if debug:
        skip = 10
    # skip = 5 ####
    times = list(range(0, T, skip))
    for t in times:
        meshes = make_batch_mesh(verts[t], faces[t], colors[t])
        if accumulate:
            vis.add_static_meshes(meshes)
        else:
            vis.add_mesh_frame(meshes, debug=debug)

    # add camera markers
    if render_cam:
        if accumulate:
            vis.add_camera_markers_static(src_cams[::skip])
            if 'cameras_others' in scene.keys():
                for cam_key, cam_value in scene['cameras_others'].items():
                    vis.add_camera_markers_static(cam_value[::skip])
        else:
            vis.add_camera_markers(src_cams[::skip])
            ###############
            if 'cameras_others' in scene.keys():
                for cam_key, cam_value in scene['cameras_others'].items():
                    vis.add_camera_markers_oppo(cam_value[::skip])

    return scene


def get_static_views(seq_name=None, bounds=None):
    print("STATIC VIEWS FOR SEQ NAME", seq_name)
    up = torch.tensor([0.0, 1.0, 0.0])

    skip = get_seq_figure_skip(seq_name)
    top_vp, side_vp = get_seq_static_lookat_points(seq_name, bounds)
    top_source, top_target = top_vp
    side_source, side_target = side_vp
    top_pose = cam_util.lookat_matrix(top_source, top_target, up)
    side_pose = cam_util.lookat_matrix(side_source, side_target, up)
    return top_pose, side_pose, skip


def make_video_grid_2x2(out_path, vid_paths, overwrite=False):
    if os.path.isfile(out_path) and not overwrite:
        print(f"{out_path} already exists, skipping.")
        return

    if any(not os.path.isfile(v) for v in vid_paths):
        print("not all inputs exist!", vid_paths)
        return

    # resize each input by half and then tile
    # so the output video is the same resolution
    v1, v2, v3, v4 = vid_paths
    cmd = (
        f"ffmpeg -i {v1} -i {v2} -i {v3} -i {v4} "
        f"-filter_complex '[0:v]scale=iw/2:ih/2[v0];"
        f"[1:v]scale=iw/2:ih/2[v1];"
        f"[2:v]scale=iw/2:ih/2[v2];"
        f"[3:v]scale=iw/2:ih/2[v3];"
        f"[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]' "
        f"-map '[v]' {out_path} -y"
    )

    print(cmd)
    subprocess.call(cmd, shell=True, stdin=subprocess.PIPE)
