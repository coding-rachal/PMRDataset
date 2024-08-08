import os
import glob
# import typing

# import imageio
import numpy as np

"""
Define data-related constants
"""
DEFAULT_GROUND = np.array([0.0, -1.0, 0.0, -0.5])

# XXX: TEMPORARY CONSTANTS
SHOT_PAD = 0
MIN_SEQ_LEN = 20
MAX_NUM_TRACKS = 12
MIN_TRACK_LEN = 20
MIN_KEYP_CONF = 0.4


def expand_source_paths(data_sources):
    return {k: get_data_source(v) for k, v in data_sources.items()}

def get_data_source(source):
    matches = glob.glob(source)
    if len(matches) < 1:
        print(f"{source} does not exist")
        return source  # return anyway for default values
    if len(matches) > 1:
        raise ValueError(f"{source} is not unique")
    return matches[0]


def is_image(x):
    return (x.endswith(".png") or x.endswith(".jpg")) and not x.startswith(".")


def get_name(x):
    return os.path.splitext(os.path.basename(x))[0]


def split_name(x, suffix):
    return os.path.basename(x).split(suffix)[0]


def get_names_in_dir(d, suffix):
    files = [split_name(x, suffix) for x in glob.glob(f"{d}/*{suffix}")]
    return sorted(files)


def batch_join(parent, names, suffix=""):
    return [os.path.join(parent, f"{n}{suffix}") for n in names]
