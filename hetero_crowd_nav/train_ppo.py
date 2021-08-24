import sys
import shutil
import logging
import argparse
from pathlib import Path
from copy import deepcopy

import torch
import gym
import git
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


from crowd_sim.envs.utils.config import Config
from crowd_sim.envs.utils.logging import logging_init
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import VNRLTrainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument("--policy", type=str, default="cadrl")
    parser.add_argument("--tag", type=str)
    parser.add_argument(
        "--config", type=str, default="configs/configs_hetero.toml"
    )
    parser.add_argument("--output_dir", type=str, default="data/output")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--weights", type=str)
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "--test_after_every_eval", default=False, action="store_true"
    )
    parser.add_argument("--randomseed", type=int, default=2021)
    args = parser.parse_args()

    set_random_seeds(args.randomseed)

     configure paths
    if args.debug:  # overwrite debug output by default
        if args.tag is None:
            args.tag = "debug"
            args.overwrite = True
    policy_path = (
        args.policy if args.tag is None else args.policy + "_" + args.tag
    )
    output_dir = Path(
        args.output_dir, policy_path
    )  # make a new output dir for each policy
    config_dir = Path(args.config)
    config_output = Path(
        output_dir, "configs.toml"  # config_dir.name
    )  # location to store config in output dir
    if args.resume:
        if config_output.exists():
            config = Config(config_output)
        else:
            parser.error("Nothing to resume.")
    else:
        if Path(output_dir).exists():
            if args.overwrite:
                shutil.rmtree(output_dir)
            else:
                key = input(
                    f"Output directory for {args.policy} already exists! Overwrite the folder? (y/[n])"
                )
                if key == "y":
                    shutil.rmtree(output_dir)
        Path(output_dir).mkdir(exist_ok=True)  # make new dir

        # load config
        config = Config(config_dir)
        config["policy"]["name"] = args.policy
        config["policy"]["tag"] = args.tag

        # store current config to output
        config.dump(config_output)

    if args.debug:
        config.merge(config("debug"))

    log_file = Path(output_dir, "output.log")
    il_weight_file = Path(output_dir, "il_model.pth")
    rl_weight_file = Path(output_dir, "rl_model.pth")

    # configure logging
    level = logging.INFO if not args.debug else logging.DEBUG
    logging_init(level, log_file, args.resume)
    logger = logging.getLogger(__name__)
    repo = git.Repo(search_parent_directories=True)
    logger.info("-" * 80)
    logger.info(f"Git head hash code: {repo.head.object.hexsha}")
    logger.info(f"Policy: {args.policy}")
    logger.info(f"Output dir: {output_dir}")
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
    )
    logger.info("Device: %s", device)
    logger.info("-" * 80)
    writer = SummaryWriter(log_dir=output_dir)

    