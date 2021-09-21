import sys
import shutil
import logging
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch as th
import gym
import git
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy

from crowd_sim.envs.utils.config import Config
from crowd_sim.envs.crowd_env import CrowdEnv
from hetero_crowd_nav.utils.custom_policy import (
    PairwiseAttentionFeaturesExtractor,
)
from hetero_crowd_nav.utils.callbacks import (
    SaveOnBestTrainingRewardCallback,
    TensorboardCallback,
)

# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/utils.py
def linear_schedule(
    initial_value: Union[float, str]
) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument(
        "--config", type=str, default="crowd_nav/configs/configs_hetero.toml"
    )
    parser.add_argument("-o", "--output_dir", type=str, default="data/PPO")
    parser.add_argument("-n", "--num_timestep", type=int, default=1_000_000)
    parser.add_argument("-v", "--verbose", type=int, default=1)
    parser.add_argument("-r", "--resume", action="store_true")
    args = parser.parse_args()
    # config = Config(args.config)

    LOG_DIR = Path(args.output_dir)
    VERBOSE = args.verbose

    logger = configure(str(LOG_DIR), ["log", "json", "tensorboard"])

    eval_env = CrowdEnv(args.config)
    eval_env = Monitor(
        eval_env,
        str(Path(LOG_DIR, "eval")),
        allow_early_resets=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=10000,
        log_path=str(Path(LOG_DIR, "eval")),
        best_model_save_path=str(Path(LOG_DIR, "eval")),
        verbose=VERBOSE,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(Path(LOG_DIR, "ckpt")),
        name_prefix="ckpt_",
    )
    save_best_callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, log_dir=str(LOG_DIR), verbose=VERBOSE
    )
    ts_callback = TensorboardCallback(VERBOSE)
    callback = CallbackList(
        [ts_callback, checkpoint_callback, eval_callback, save_best_callback]
    )

    env = CrowdEnv(args.config)
    env = Monitor(env, str(Path(LOG_DIR, "train")), allow_early_resets=True)
    env = DummyVecEnv([lambda: env] * 8)
    # env = Monitor(CrowdEnv(config, "test"))

    network_dims = dict(
        mlp1_dims=[150, 100, 0],
        mlp2_dims=[100, 50],
        attention_dims=[100, 100, 1],
    )
    policy_kwargs = dict(
        features_extractor_class=PairwiseAttentionFeaturesExtractor,
        features_extractor_kwargs=dict(dims=network_dims),
        activation_fn=th.nn.ReLU,
        net_arch=[150, 100, 100, dict(pi=[env.action_space.n], vf=[1])],
    )
    if not args.resume:

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(2.5e-4),
            gamma=0.99,
            gae_lambda=0.95,
            n_steps=64,  # 16
            n_epochs=4,
            batch_size=64,
            clip_range=linear_schedule(0.1),
            vf_coef=0.5,
            ent_coef=0.01,
            verbose=0,
            seed=2021,
            tensorboard_log=LOG_DIR,
        )
    else:
        model = PPO.load(
            Path(LOG_DIR, "model"),
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(2.5e-4),
            gamma=0.99,
            gae_lambda=0.95,
            n_steps=64,
            n_epochs=4,
            batch_size=64,
            clip_range=linear_schedule(0.1),
            vf_coef=0.5,
            ent_coef=0.01,
            verbose=0,
            seed=2021,
        )

    model.set_logger(logger)

    model.learn(
        total_timesteps=args.num_timestep,
        callback=callback,
    )
    model.save(Path(LOG_DIR, "model"))
