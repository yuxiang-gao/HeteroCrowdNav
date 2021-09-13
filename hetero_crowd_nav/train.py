import sys
import shutil
import logging
import argparse
from pathlib import Path
from copy import deepcopy

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument(
        "--config", type=str, default="crowd_nav/configs/configs_hetero.toml"
    )
    parser.add_argument("-o", "--output_dir", type=str, default="data/PPO")
    parser.add_argument("-n", "--num_timestep", type=int, default=1_000_000)
    parser.add_argument("-v", "--verbose", type=int, default=1)
    args = parser.parse_args()
    config = Config(args.config)

    LOG_DIR = args.output_dir
    VERBOSE = args.verbose

    logger = configure(LOG_DIR, ["log", "json", "tensorboard"])

    eval_env = CrowdEnv(config, "val")
    eval_env = Monitor(eval_env, LOG_DIR + "/eval", allow_early_resets=True)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=10000,
        log_path=LOG_DIR + "/eval/",
        best_model_save_path=LOG_DIR + "/eval/",
        verbose=VERBOSE,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=LOG_DIR + "/ckpt"
    )
    save_best_callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, log_dir=LOG_DIR, verbose=VERBOSE
    )
    ts_callback = TensorboardCallback(VERBOSE)
    callback = CallbackList(
        [ts_callback, checkpoint_callback, eval_callback, save_best_callback]
    )

    env = CrowdEnv(config, "train")
    env = Monitor(env, LOG_DIR + "/train", allow_early_resets=True)
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
        net_arch=[150, 100, 100, dict(pi=[81], vf=[1])],
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=16,
        n_epochs=4,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=LOG_DIR,
    )

    model.set_logger(logger)

    model.learn(
        total_timesteps=args.num_timestep,
        callback=callback,
    )
    model.save(LOG_DIR + "model")
