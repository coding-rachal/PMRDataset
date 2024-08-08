import os
import glob
import pandas as pd
from copy import deepcopy


def update_args(args):
    print('update')
    args = deepcopy(args)
    if args.type == "custom":
        if args.seqs is None:
            args.seqs = get_custom_seqs(args.root)
        return args
    if args.type == "egobody":
        if args.root is None:
            args.root = "/path/to/egobody"
        if args.seqs is None:
            args.seqs = get_egobody_seqs(args.root, args.split)
        return args
    if args.type == "3dpw":
        if args.root is None:
            args.root = "/path/to/3DPW"
        if args.seqs is None:
            args.seqs = get_3dpw_seqs(args.root, args.split)
        return args
    elif args.type == "posetrack":
        if args.root is None:
            args.root = "/path/to/posetrack/posetrack2018/posetrack_data"
        if args.seqs is None:
            args.seqs = get_posetrack_seqs(args.root, args.split)
        return args
    elif args.type == "davis":
        if args.root is None:
            args.root = "/path/to/DAVIS"
            if args.seqs is None:
                args.seqs = get_davis_seqs(args.root)
        return args
    elif args.type == 'carla':
        if args.root is None:
            args.root = "/data/carla"
            if args.carla_jfiles is not None:
                with open(args.carla_jfiles, "r") as f:
                    args.seqs = [strs.strip() for strs in f.readlines()]
            elif args.seqs is None:
                print('getting...')
                args.seqs = get_carla_seqs(args.root, args.carla_dates, args.carla_views)
        return args
    raise NotImplementedError


def get_custom_seqs(data_root):
    img_dir = f"{data_root}/images"
    if not os.path.isdir(img_dir):
        return []
    return sorted(os.listdir(img_dir))


def get_egobody_seqs(data_root, split):
    split_file = f"{data_root}/data_splits.csv"
    df = pd.read_csv(split_file)
    if split not in df.columns:
        print(f"{split} not in {split_file}")
        return []
    return sorted(df[split].dropna().tolist())


def get_3dpw_seqs(data_root, split):
    split_dir = f"{data_root}/sequenceFiles/{split}"
    if not os.path.isdir(split_dir):
        return []
    seq_files = sorted(os.listdir(split_dir))
    return [os.path.splitext(f)[0] for f in seq_files]


def get_posetrack_seqs(data_root, split):
    split_dir = f"{data_root}/images/{split}"
    if not os.path.isdir(split_dir):
        return []
    return sorted(os.listdir(split_dir))


def get_davis_seqs(data_root):
    img_root = f"{data_root}/JPEGImages/Full-Resolution"
    if not os.path.isdir(img_root):
        return []
    return sorted(os.listdir(img_root))

def get_carla_seqs(data_root, carla_dates=[], carla_views=[]):
    img_root = data_root
    carla_seqs = []
    if len(carla_dates)==0:
        carla_dates = os.listdir(data_root)
    print(f'{carla_dates}, root:{data_root}')
    for date in carla_dates:
        if not date.startswith("output") or not os.path.isdir(os.path.join(data_root, date)):
            continue
        carla_dates_root = os.path.join(data_root, date)
        carla_scenes = os.listdir(carla_dates_root)
        for scene in carla_scenes:
            if scene.startswith("id") and os.path.isdir(os.path.join(carla_dates_root, scene)):
                carla_scene_root = os.path.join(carla_dates_root, scene)
                if len(carla_views) == 0:
                    carla_views = os.listdir(carla_scene_root)
                carla_seqs.extend([f'{date}/{scene}/{view}' for view in carla_views if (os.path.isdir(os.path.join(carla_scene_root, view)) and len(os.listdir(os.path.join(carla_scene_root, view)))>0)])
    return carla_seqs


def get_img_dir(data_type, data_root, seq, split):
    if data_type == "posetrack":
        return f"{data_root}/images/{split}/{seq}"
    if data_type == "egobody":
        return glob.glob(f"{data_root}/egocentric_color/{seq}/**/PV")[0]
    if data_type == "3dpw":
        return f"{data_root}/imageFiles/{seq}"
    if data_type == "davis":
        return f"{data_root}/JPEGImages/Full-Resolution/{seq}"
    if data_type == "carla":
        return f"{data_root}/{seq}"
    return f"{data_root}/images/{seq}"  # custom sequence
