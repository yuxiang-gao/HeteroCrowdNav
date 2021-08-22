import copy
import logging
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from crowd_sim.envs.utils.info import *

logger = logging.getLogger(__name__)


class Explorer(object):
    def __init__(
        self,
        env,
        robot,
        device,
        memory=None,
        gamma=None,
        target_policy=None,
        writer=None,
    ):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        self.writer = writer
        self.statistics = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(
        self,
        k,
        phase,
        update_memory=False,
        imitation_learning=False,
        episode=None,
        epoch=None,
        print_failure=False,
    ):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        obs_collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        timeout_cases = []

        episode_info_record = []

        if k != 1:
            pbar = tqdm(
                range(k),
                total=k,
                desc=f"Explore [{phase.upper()}]",
                leave=False,
            )
        else:
            pbar = range(k)

        with logging_redirect_tqdm():
            for i in pbar:
                ob = self.env.reset(phase)
                done = False
                states = []
                actions = []
                rewards = []
                while not done:
                    action = self.robot.act(ob)
                    ob, reward, done, info = self.env.step(action)
                    states.append(self.robot.policy.last_state)
                    actions.append(action)
                    rewards.append(reward)

                    if isinstance(info, Discomfort):
                        discomfort += 1
                        min_dist.append(info.min_dist)

                if isinstance(info, ReachGoal):
                    success += 1
                    success_times.append(self.env.global_time)
                elif isinstance(info, Collision):
                    collision += 1
                    collision_cases.append(i)
                    collision_times.append(self.env.global_time)
                elif isinstance(info, Timeout):
                    timeout += 1
                    timeout_cases.append(i)
                    timeout_times.append(self.env.time_limit)

                if isinstance(info, dict):
                    episode_info_record.append(info)
                    if info["events"]["succeed"]:
                        success += 1
                        success_times.append(self.env.global_time)
                    elif info["events"]["collision"]:
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(self.env.global_time)
                    elif info["events"]["obstacle_collision"]:
                        obs_collision += 1
                    elif info["events"]["timeout"]:
                        timeout += 1
                        timeout_cases.append(i)
                        timeout_times.append(self.env.time_limit)
                    # success += info["events"]["succeed"]
                    # collision += info["events"]["collision"]
                    # obs_collision += info["events"]["obstacle_collision"]
                    # timeout += info["events"]["timeout"]

                else:
                    raise ValueError("Invalid end signal from environment")

                if update_memory:
                    if isinstance(info, ReachGoal) or isinstance(
                        info, Collision
                    ):
                        # only add positive(success) or negative(collision) experience in experience set
                        self.update_memory(
                            states, actions, rewards, imitation_learning
                        )
                    elif isinstance(info, dict):
                        if (
                            info["events"]["succeed"]
                            or info["events"]["collision"]
                            or info["events"]["obstacle_collision"]
                        ):
                            self.update_memory(
                                states, actions, rewards, imitation_learning
                            )

                cumulative_rewards.append(self.calc_value(0, rewards))

                returns = []
                for step in range(len(rewards)):
                    step_return = sum(
                        [
                            pow(
                                self.gamma,
                                t * self.robot.time_step * self.robot.v_pref,
                            )
                            * reward
                            for t, reward in enumerate(rewards[step:])
                        ]
                    )
                    returns.append(step_return)
                average_returns.append(average(returns))

                if k > 1:
                    pbar.set_postfix(
                        reward=cumulative_rewards[-1],
                        avg_return=average_returns[-1],
                    )

        success_rate = success / k
        collision_rate = (collision + obs_collision) / k
        assert success + collision + obs_collision + timeout == k
        avg_nav_time = (
            sum(success_times) / len(success_times)
            if success_times
            else self.env.time_limit
        )

        extra_info = "" if episode is None else "in episode {} ".format(episode)
        extra_info = (
            extra_info + ""
            if epoch is None
            else extra_info + " in epoch {} ".format(epoch)
        )
        if k != 1:
            logger.info(
                "{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f},"
                " average return: {:.4f}{}".format(
                    phase.upper(),
                    extra_info,
                    success_rate,
                    collision_rate,
                    avg_nav_time,
                    average(cumulative_rewards),
                    average(average_returns),
                    k,
                )
            )
        if phase in ["val", "test"]:
            num_step = (
                sum(success_times + collision_times + timeout_times)
                / self.robot.time_step
            )
            logger.info(
                "Frequency of being in danger: %.2f and average min separate distance in danger: %.2f",
                discomfort / num_step,
                average(min_dist),
            )

        if print_failure:
            logger.info(
                "Collision cases: "
                + " ".join([str(x) for x in collision_cases])
            )
            logger.info(
                "Timeout cases: " + " ".join([str(x) for x in timeout_cases])
            )

        self.statistics = (
            success_rate,
            collision_rate,
            avg_nav_time,
            average(cumulative_rewards),
            average(average_returns),
        )

        return self.statistics

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError("Memory or gamma value is not set!")

        for i, (state, reward) in enumerate(zip(states[:-1], rewards[:-1])):

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                next_state = self.target_policy.transform(states[i + 1])
                value = self.calc_value(i, rewards)
            else:
                next_state = states[i + 1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
                # if i == len(states) - 1:
                #     # terminal state
                #     value = reward
                # else:
                #     gamma_bar = pow(
                #         self.gamma, self.robot.time_step * self.robot.v_pref
                #     )
                #     value = (
                #         reward
                #         + gamma_bar
                #         * self.target_model(next_state.unsqueeze(0)).data.item()
                #     )
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value, reward, next_state))

    def calc_value(self, idx, rewards):
        # optimal value function
        value = sum(
            [
                pow(
                    self.gamma,
                    max(t - idx, 0) * self.robot.time_step * self.robot.v_pref,
                )
                * reward
                * (1 if t >= idx else 0)
                for t, reward in enumerate(rewards)
            ]
        )
        return value

    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return = self.statistics
        self.writer.add_scalar(tag_prefix + "/time", time, global_step)
        self.writer.add_scalars(
            tag_prefix + "/performance",
            {"success_rate": sr, "collision_rate": cr},
            global_step,
        )
        self.writer.add_scalars(
            tag_prefix + "/rewards",
            {"rewards": reward, "returns": avg_return},
            global_step,
        )


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
