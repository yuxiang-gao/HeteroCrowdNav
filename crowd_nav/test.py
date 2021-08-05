import logging
import argparse
from pathlib import Path

import gym
import toml
import torch
import numpy as np
import matplotlib.pyplot as plt

from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA


def main():
    parser = argparse.ArgumentParser("Parse configuration file")
    # parser.add_argument("--env_config", type=str, default="configs/env.config")
    # parser.add_argument(
    #     "--policy_config", type=str, default="configs/policy.config"
    # )
    parser.add_argument("--config", type=str, default="configs/configs.toml")
    parser.add_argument("--policy", type=str, default="orca")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--il", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--test_case", type=int, default=None)
    parser.add_argument("--square", default=False, action="store_true")
    parser.add_argument("--circle", default=False, action="store_true")
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--traj", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--human_num", type=int, default=None)
    parser.add_argument("--safety_space", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--test_scenario", type=str, default=None)
    args = parser.parse_args()

    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
    )
    logging.info("Using device: %s", device)

    if args.model_dir is not None:
        config_file = Path(args.model_dir, Path(args.config).name)
        if args.il:
            model_weights = Path(args.model_dir, "il_model.pth")
            logging.info("Loaded IL weights")
        else:
            if Path(args.model_dir, "resumed_rl_model.pth").exists():
                model_weights = Path(args.model_dir, "resumed_rl_model.pth")
            else:
                model_weights = Path(args.model_dir, "rl_model.pth")
            logging.info("Loaded RL weights")
    else:
        config_file = Path(args.config)

    config = toml.load(config_file)
    policy_config = config["policy"]
    env_config = config["env"]
    agent_config = env_config["agents"]

    # configure policy
    policy = policy_factory[args.policy]()
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error(
                "Trainable policy must be specified with a model weights directory"
            )
        policy.load_model(model_weights)

    # configure environment
    if args.human_num is not None:
        env_config["sim"]["human_num"] = args.human_num
    env = gym.make("CrowdSim-v0")
    env.configure(env_config)

    if args.square:
        env.test_scenario = "square_crossing"
    if args.circle:
        env.test_scenario = "circle_crossing"
    if args.test_scenario is not None:
        env.test_scenario = args.test_scenario

    robot = Robot(agent_config, "robot")
    env.set_robot(robot)
    robot.time_step = env.time_step
    robot.set_policy(policy)
    explorer = Explorer(env, robot, device, gamma=0.9)

    epsilon_end = config["train"]["epsilon_end"]
    if not isinstance(robot.policy, ORCA):
        robot.policy.set_epsilon(epsilon_end)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = args.safety_space
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info("ORCA agent buffer: %f", robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    if args.visualize:
        rewards = []
        done = False
        ob = env.reset(args.phase, args.test_case)
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob)
            ob, reward, done, info = env.step(action)
            rewards.append(reward)
            current_pos = np.array(robot.get_position())
            logging.debug(
                "Speed: %.2f",
                np.linalg.norm(current_pos - last_pos) / robot.time_step,
            )
            last_pos = current_pos

        cumulative_reward = sum(
            [
                pow(args.gamma, t * robot.time_step * robot.v_pref) * reward
                for t, reward in enumerate(rewards)
            ]
        )

        if args.traj:
            env.render("traj", args.video_file)
        else:
            env.render("video", args.video_file)

        logging.info(
            "It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f",
            env.global_time,
            info,
            cumulative_reward,
        )
        if robot.visible and info == "reach goal":
            human_times = env.get_human_times()
            logging.info(
                "Average time for humans to reach goal: %.2f",
                sum(human_times) / len(human_times),
            )
    else:
        explorer.run_k_episodes(
            env.case_size[args.phase], args.phase, print_failure=True
        )

        if args.plot_test_scenarios_hist:
            test_angle_seeds = np.array(env.test_scene_seeds)
            b = [i * 0.01 for i in range(101)]
            n, bins, patches = plt.hist(test_angle_seeds, b, facecolor="g")
            plt.savefig(Path(args.model_dir, "test_scene_hist.png"))
            plt.close()


if __name__ == "__main__":
    main()
