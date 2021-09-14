import logging
import argparse

import numpy as np

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from crowd_sim.envs.utils.config import Config
from crowd_sim.envs.crowd_env import CrowdEnv
from hetero_crowd_nav.utils.callbacks import (
    SaveOnBestTrainingRewardCallback,
    TensorboardCallback,
)


def run_test(policy, env, num_episodes, render=False):
    success_times = []
    collision_times = []
    collision_obstacles_times = []
    timeout_times = []
    success = 0
    collision = 0
    collision_obstacles = 0
    timeout = 0
    discomfort = 0
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    collision_obstacles_cases = []
    timeout_cases = []
    pbar = tqdm(range(num_episodes), total=num_episodes)

    # with logging_redirect_tqdm():
    for i in pbar:
        pbar.set_description("Case {}".format(i))
        ob = env.reset()
        done = False
        env_time = 0
        while not done:
            action, _states = policy.predict(ob)
            ob, reward, done, info = env.step(action)
            events = info["events"]
            if render:
                env.render("human")
            env_time += env.sim.time_step
            if events["discomfort"] != 0:
                discomfort += 1
                min_dist.append(min(events["min_dist"]))
            cumulative_rewards.append(reward)
        if events["succeed"] != 0:
            success += 1
            success_times.append(env_time)
        elif events["collision"] != 0:
            collision += 1
            collision_cases.append(i)
            collision_times.append(env_time)
        elif events["obstacle_collision"] != 0:
            collision_obstacles += 1
            collision_obstacles_cases.append(i)
            collision_obstacles_times.append(env_time)
        elif events["timeout"] != 0:
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(env_time)
        else:
            raise ValueError("Invalid end signal from environment")
    success_rate = success / float(num_episodes)
    collision_rate = collision / float(num_episodes)
    collision_obstacles_rate = collision_obstacles / float(num_episodes)

    assert success + collision + timeout + collision_obstacles == num_episodes
    avg_nav_time = (
        sum(success_times) / float(len(success_times))
        if success_times
        else np.nan
    )

    print(
        """success rate: {:.2f}, collision rate: {:.2f},
    collision from other agents rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}""".format(
            success_rate,
            collision_rate,
            collision_obstacles_rate,
            avg_nav_time,
            np.mean(cumulative_rewards),
        )
    )

    total_time = sum(
        success_times
        + collision_times
        + collision_obstacles_times
        + timeout_times
    )
    print(
        f"Frequency of being in danger: {discomfort / float(total_time):.2f} and average min separate distance in danger: {np.mean(min_dist):.2f}"
    )

    return success_rate, avg_nav_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument(
        "--config", type=str, default="crowd_nav/configs/configs_hetero.toml"
    )
    parser.add_argument("--model", type=str, default="data/PPO/best_model.zip")
    parser.add_argument("-n", "--num_episodes", type=int, default=500)
    parser.add_argument("-v", "--verbose", type=int, default=1)
    parser.add_argument("-r", "--render", action="store_true")
    args = parser.parse_args()

    config = Config(args.config)
    model = PPO.load(args.model)
    eval_env = CrowdEnv(config, "test")
    # eval_env = Monitor(eval_env, LOG_DIR + "/test", allow_early_resets=True)

    run_test(model, eval_env, args.num_episodes, args.render)
