import sys
import shutil
import logging
import argparse
import collections
from pathlib import Path
from copy import deepcopy

import torch
import toml
import gym
import git
from tensorboardX import SummaryWriter

from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import VNRLTrainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory


def dict_merge(dct, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (
            k in dct
            and isinstance(dct[k], dict)
            and isinstance(merge_dct[k], collections.Mapping)
        ):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def main():
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument("--policy", type=str, default="cadrl")
    parser.add_argument("--config", type=str, default="configs/configs.toml")
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

    # configure paths
    output_dir = Path(
        args.output_dir, args.policy
    )  # make a new output dir for each policy
    config_dir = Path(args.config)
    config_output = Path(
        output_dir, config_dir.name
    )  # location to store config in output dir
    if args.resume:
        if config_output.exists():
            config = toml.load(config_output)
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
        config = toml.load(config_dir)

        # store current config to output
        with open(config_output, "w") as f:
            toml.dump(config, f)

    if args.debug:
        dict_merge(config, config["debug"])

    log_file = Path(output_dir, "output.log")
    il_weight_file = Path(output_dir, "il_model.pth")
    rl_weight_file = Path(output_dir, "rl_model.pth")

    # configure logging
    mode = "a" if args.resume else "w"
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(
        level=level,
        handlers=[stdout_handler, file_handler],
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    repo = git.Repo(search_parent_directories=True)
    logging.info("-" * 80)
    logging.info(f"Git head hash code: {repo.head.object.hexsha}")
    logging.info(f"Policy: {args.policy}")
    logging.info(f"Output dir: {output_dir}")
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
    )
    logging.info("Device: %s", device)
    logging.info("-" * 80)
    writer = SummaryWriter(log_dir=output_dir)

    # configure policy
    logging.info("Configuring policy...")
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        parser.error("Policy has to be trainable")
    policy.configure(config["policy"])
    policy.set_device(device)

    # configure environment
    logging.info("Configuring environment...")
    env = gym.make("CrowdSim-v0")
    env.configure(config["env"])
    robot = Robot(config["env"]["agents"], "robot")
    env.set_robot(robot)

    # read training parameters
    train_config = config["train"]
    rl_learning_rate = train_config.get("rl_learning_rate")
    train_batches = train_config.get("train_batches")
    train_episodes = train_config.get("train_episodes")
    sample_episodes = train_config.get("sample_episodes")
    target_update_interval = train_config.get("target_update_interval")
    evaluation_interval = train_config.get("evaluation_interval")
    capacity = train_config.get("capacity")
    epsilon_start = train_config.get("epsilon_start")
    epsilon_end = train_config.get("epsilon_end")
    epsilon_decay = train_config.get("epsilon_decay")
    checkpoint_interval = train_config.get("checkpoint_interval")

    # configure trainer and explorer
    logging.info("Configuring trainer and explorer...")
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config["trainer"]["batch_size"]
    trainer = VNRLTrainer(
        model,
        memory,
        device,
        batch_size,
        writer,
        train_config["trainer"]["optimizer"],
    )
    explorer = Explorer(
        env,
        robot,
        device,
        memory,
        policy.gamma,
        target_policy=policy,
        writer=writer,
    )

    # imitation learning
    logging.info("-" * 80)
    logging.info("Start imitation learning...")
    if args.resume:
        if not Path(rl_weight_file).exists():
            logging.error("RL weights does not exist")
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = Path(output_dir, "resumed_rl_model.pth")
        logging.info(
            f"Load reinforcement learning trained weights from {rl_weight_file}. Resume training..."
        )
    elif Path(il_weight_file).exists():
        model.load_state_dict(torch.load(il_weight_file))
        logging.info("Load imitation learning trained weights...")
    else:
        il_episodes = train_config["imitation_learning"]["il_episodes"]
        il_policy = train_config["imitation_learning"]["il_policy"]
        il_epochs = train_config["imitation_learning"]["il_epochs"]
        il_learning_rate = train_config["imitation_learning"][
            "il_learning_rate"
        ]
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config["imitation_learning"]["safety_space"]
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.set_policy(il_policy)
        explorer.run_k_episodes(
            il_episodes, "train", update_memory=True, imitation_learning=True
        )
        trainer.optimize_epoch(il_epochs)
        torch.save(model.state_dict(), il_weight_file)
        logging.info(
            f"Finish imitation learning. Weights saved to {il_weight_file}."
        )
        logging.info("Experience set size: %d/%d", len(memory), memory.capacity)
    trainer.update_target_model(model)

    # reinforcement learning
    logging.info("-" * 80)
    logging.info("Start reinforcement learning...")
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_learning_rate(rl_learning_rate)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, "train", update_memory=True, episode=0)
        logging.info("Experience set size: %d/%d", len(memory), memory.capacity)
    episode = 0
    best_val_reward = -1
    best_val_model = None
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = (
                    epsilon_start
                    + (epsilon_end - epsilon_start) / epsilon_decay * episode
                )
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(
            sample_episodes, "train", update_memory=True, episode=episode
        )
        explorer.log("train", episode)

        trainer.optimize_batch(train_batches, episode)
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            _, _, _, reward, _ = explorer.run_k_episodes(
                env.case_size["val"], "val", episode=episode
            )
            explorer.log("val", episode // evaluation_interval)

            if reward > best_val_reward:
                best_val_reward = reward
                best_val_model = deepcopy(policy.get_state_dict())
                torch.save(model.state_dict(), rl_weight_file)

            if args.test_after_every_eval:
                explorer.run_k_episodes(
                    env.case_size["test"],
                    "test",
                    episode=episode,
                    print_failure=True,
                )
                explorer.log("test", episode // evaluation_interval)

        if episode != 0 and episode % checkpoint_interval == 0:
            current_checkpoint = episode // checkpoint_interval - 1
            save_every_checkpoint_rl_weight_file = Path(
                rl_weight_file.parent,
                str(rl_weight_file.stem)
                + "_"
                + str(current_checkpoint)
                + ".pth",
            )

            policy.save_model(save_every_checkpoint_rl_weight_file)

    # # test with the best val model
    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, Path(output_dir, "best_val.pth"))
        logging.info(
            "Save the best val model with the reward: {}".format(
                best_val_reward
            )
        )
    explorer.run_k_episodes(
        env.case_size["test"], "test", episode=episode, print_failure=True
    )

    # final test
    explorer.run_k_episodes(env.case_size["test"], "test", episode=episode)


if __name__ == "__main__":
    main()
