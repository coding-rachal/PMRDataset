import os

from data import expand_source_paths
from util.logger import Logger
from vis_gt_vis import run_vis

import hydra
from omegaconf import DictConfig, OmegaConf

N_STAGES = 3


@hydra.main(version_base=None, config_path="confs", config_name="config.yaml")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)

    out_dir = os.getcwd()

    save_dir = out_dir
    print("out_dir", out_dir)
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    Logger.init(f"{save_dir}/opt_log.txt")

    # make sure we get all necessary inputs
    cfg.data.sources = expand_source_paths(cfg.data.sources)
    print("SOURCES", cfg.data.sources)

  
    run_vis(
        cfg, out_dir, 0, save_dir=save_dir, **cfg.get("vis", dict())
    )


if __name__ == "__main__":
    print('out_dir0:',os.getcwd())
    main()
