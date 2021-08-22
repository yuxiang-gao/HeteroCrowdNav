import logging
import random
import math

import gym
import toml
import numpy as np
from copy import deepcopy
from numpy.linalg import norm

from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import patches

from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import tensor_to_joint_state, JointState
from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import *
from crowd_sim.envs.utils.scenarios import (
    Scenario,
    ScenarioManager,
    SceneManager,
)

EPISODE_INFO_TEMPLATE = """
title = "Episode info"
time = 0
collisions = 0
obstacle_collisions = 0
progress = 0
goal = 0

    [events]
    timeout = 0
    collision = 0
    obstacle_collision = 0
    succeed = 0
    discomfort = 0
    min_dist = []   

    [robot]
    distance_traversed = []
    velocity = []
    
    [[pedestrians]]
    id = -1
    goal = []
    velocity = []
    distance_traversed = []
"""
logger = logging.getLogger(__name__)


class CrowdSim(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.robot = None
        self.humans = []
        self.global_time = None
        self.robot_sensor_range = None
        # simulation configuration
        self.config = None
        self.agent_config = None
        self.time_limit = None
        self.time_step = None
        self.goal_radius = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = []
        self.perpetual = None
        self.centralized_planning = None
        self.centralized_planner = None
        # reward function
        self.progress_reward = None
        self.success_reward = None
        self.collision_penalty = None
        self.static_obstacle_collision_penalty = None
        self.time_penalty = None
        self.discomfort_penalty = None
        self.discomfort_scale = None
        self.discomfort_dist = None

        # Internal environment configuration
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        # self.parallel = None
        # self.max_tries = None
        self.phase_scenario = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        self.scene_manager = None
        self.obstacles = []  # xmin,xmax,ymin,ymax

        self.end_on_collision = True
        self.discomfort_scale = 1.0
        # self.progress_reward = 1.25
        self.initial_distance = None
        self.previous_distance = None
        self.use_groups = False
        self.use_types = None
        self.time_penalty = 0.0
        self.human_times = None

        self.episode_info = None

    def configure(self, config):
        self.config = config
        self.agent_config = config("agents")

        self.use_groups = self.agent_config("use_groups")
        self.use_types = self.agent_config("use_types")

        self.time_limit = config("time_limit")
        self.time_step = config("time_step")
        self.randomize_attributes = config("randomize_attributes")
        self.robot_sensor_range = config("robot_sensor_range")
        self.goal_radius = config("goal_radius")

        # rewards
        self.__dict__.update(config["reward"])
        # progress_reward = 0.1
        # success_reward = 1.0
        # collision_penalty = 0.25
        # static_obstacle_collision_penalty = 0.25
        # time_penalty = 0.0
        # discomfort_penalty = 0.5
        # discomfort_scale = 1.0
        # discomfort_dist = 0.2

        self.perpetual = config("scenarios", "perpetual")
        self.human_num = config("scenarios", "human_num")

        human_policy = self.agent_config("humans")[0]["policy"]
        if self.centralized_planning:
            self.centralized_planner = policy_factory[
                "centralized_" + human_policy
            ]()
        if (
            human_policy == "orca"
            or human_policy == "socialforce"
            or human_policy == "centralized_socialforce"
        ):
            self.case_capacity = {
                "train": np.iinfo(np.uint32).max - 2000,
                "val": 1000,
                "test": 1000,
            }
            self.case_size = config["case_size"]
            self.case_size["train"] = np.iinfo(np.uint32).max - 2000
            # self.case_size = {
            #     "train": np.iinfo(np.uint32).max - 2000,
            #     "val": config("val_size"),
            #     "test": config("test_size"),
            # }
            self.phase_scenario = config("scenarios", "phase")
            # self.human_num = config("scenarios", "human_num")
            # self.nonstop_human = self.agent_config("scenarios", "nonstop_human")
            # self.centralized_planning = config(
            #     "scenarios", "centralized_planning"
            # )

            if self.centralized_planning:
                self.centralized_planner.force_vectors = np.zeros(
                    (sum(self.human_num) + 1, 6, 2)
                )
        else:
            raise NotImplementedError
        self.case_counter = {"train": 0, "test": 0, "val": 0}
        logger.info("Human number: {}".format(self.human_num))
        if self.randomize_attributes:
            logger.info("Randomize human's radius and preferred speed")
        else:
            logger.info("NOT randomize human's radius and preferred speed")
        logger.info(
            "Training simulation: {}, test simulation: {}".format(
                self.phase_scenario["train"], self.phase_scenario["test"]
            )
        )

        # set robot
        self.robot = Robot(self.agent_config, "robot")
        self.robot.time_step = self.time_step

    def set_obstacles(self, obs):
        self.obstacles = obs

    def set_scene(self, scenario=None, seed=None):
        if self.scene_manager is None:
            self.scene_manager = SceneManager(
                scenario, self.robot, self.config, seed
            )
        else:
            self.scene_manager.set_scenario(scenario, seed)

    def reset(self, phase="test", test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError("Robot has to be set!")
        assert phase in ["train", "val", "test"]
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        counter_offset = {
            "train": self.case_capacity["val"] + self.case_capacity["test"],
            "val": 0,
            "test": self.case_capacity["val"],
        }
        if self.case_counter[phase] >= 0:
            human_num = (
                self.human_num if self.robot.policy.multiagent_training else [1]
            )
            seed = counter_offset[phase] + self.case_counter[phase]
            self.set_scene(self.phase_scenario[phase], seed)
            self.scene_manager.spawn(
                num_human=human_num,
                set_robot=True,
                use_groups=self.use_groups,
                use_types=self.use_types,
                group_sizes=None,
            )
            (
                self.humans,
                self.obstacles,
                self.group_membership,
                self.individual_membership,
            ) = self.scene_manager.get_scene()
            self.num_groups = len(self.group_membership)

            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (
                self.case_counter[phase] + 1
            ) % self.case_size[phase]
        else:
            assert phase == "test"
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = [3]
                self.humans = [
                    Human(self.config, "humans")
                    for _ in range(sum(self.human_num))
                ]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, "action_values"):
            self.action_values = list()
        if hasattr(self.robot.policy, "get_attention_weights"):
            self.attention_weights = [np.zeros(sum(self.human_num))]

        # get current observation
        if self.robot.sensor == "coordinates":
            ob = self.compute_observation_for(self.robot)
        elif (
            self.robot.sensor.lower() == "rgb"
            or self.robot.sensor.lower() == "gray"
        ):
            raise NotImplementedError
        else:
            raise ValueError("Unknown robot sensor type.")
        self.initial_distance = np.linalg.norm(
            [
                (self.robot.px - self.robot.get_goal_position()[0]),
                (self.robot.py - self.robot.get_goal_position()[1]),
            ]
        )
        self.previous_distance = self.initial_distance
        self.states.append(
            [
                self.robot.get_full_state(),
                [human.get_full_state() for human in self.humans],
                # self.centralized_planner.get_force_vectors(),
            ]
        )
        # info contains the various contributions to the reward:
        # self.episode_info = {
        #     "collisions": 0.0,
        #     "obstacle_collisions": 0.0,
        #     "time": 0.0,
        #     "discomfort": 0.0,
        #     "progress": 0.0,
        #     "goal": 0.0,
        #     "group_discomfort": 0.0,
        #     "global_time": 0.0,
        #     "did_timeout": 0.0,
        #     "did_collide": 0.0,
        #     "did_collide_static_obstacle": 0.0,
        #     "did_succeed": 0.0,
        #     "group_intersection_violations": {},
        #     "pedestrian_distance_traversed": {},
        #     "pedestrian_goal": {},
        #     "robot_distance_traversed": list(),
        #     "pedestrian_velocity": {},
        #     "robot_velocity": list(),
        # }
        self.episode_info = toml.loads(EPISODE_INFO_TEMPLATE)

        # init episode_info
        human_num = len(self.humans)
        for i in range(human_num - 1):
            self.episode_info["pedestrians"].append(
                deepcopy(self.episode_info["pedestrians"][0])
            )

        # Initiate forces log
        # self.episode_info.update(
        #     {"avg_" + force: [] for force in self.force_list}
        # )
        # self.episode_info.update({"robot_social_force": []})

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """

        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(
                    agent_states, self.group_membership, self.obstacles
                )[:-1]
            else:
                human_actions = self.centralized_planner.predict(
                    agent_states, self.group_membership, self.obstacles
                )
        else:
            human_actions = []
            for human in self.humans:
                # Choose new target if human has reached goal and in perpetual mode:
                if human.reached_destination(0.3) and self.perpetual:
                    human.go_back()
                human_ob = self.compute_observation_for(human)
                human_actions.append(human.act(human_ob, self.group_membership))
        # collision detection
        collisions, human_distances = self.detect_collisions_with_human(action)
        self.episode_info["collisions"] -= self.collision_penalty * collisions

        # collision detection between robot and static obstacle
        (
            obstacle_collisions,
            obstacle_distances,
        ) = self.detect_collisions_with_obstacles(action)
        self.episode_info["obstacle_collisions"] -= (
            self.static_obstacle_collision_penalty * obstacle_collisions
        )

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (
                    (dx ** 2 + dy ** 2) ** (1 / 2)
                    - self.humans[i].radius
                    - self.humans[j].radius
                )
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logger.debug("Collision happens between humans in step()")
        # check if reaching the goal
        end_position = np.array(
            self.robot.compute_position(action, self.time_step)
        )
        reaching_goal = (
            norm(end_position - np.array(self.robot.get_goal_position()))
            < self.robot.radius + self.goal_radius
        )

        done = False
        info = Nothing()
        reward = -self.time_penalty
        goal_distance = np.linalg.norm(
            [
                (end_position[0] - self.robot.get_goal_position()[0]),
                (end_position[1] - self.robot.get_goal_position()[1]),
            ]
        )
        progress = self.previous_distance - goal_distance
        self.previous_distance = goal_distance
        reward += self.progress_reward * progress
        self.episode_info["progress"] += self.progress_reward * progress

        if self.global_time >= self.time_limit:
            done = True
            info = Timeout()
            self.episode_info["events"]["timeout"] = 1.0
        elif collisions > 0:
            reward -= self.collision_penalty * collisions
            if self.end_on_collision:
                done = True
            info = Collision()
            self.episode_info["events"]["collision"] = 1.0
        elif obstacle_collisions > 0:
            reward -= (
                self.static_obstacle_collision_penalty * obstacle_collisions
            )
            if self.end_on_collision:
                done = True
            info = Collision()
            self.episode_info["events"]["obstacle_collision"] = 1.0
        elif reaching_goal:
            reward += self.success_reward
            done = True
            info = ReachGoal()
            self.episode_info["goal"] = self.success_reward
            self.episode_info["events"]["succeed"] = 1.0
        elif (
            len(human_distances) > 0
            and 0
            <= min(human_distances)
            < self.discomfort_dist * self.discomfort_scale
        ):
            info = Discomfort(min(human_distances))
            self.episode_info["events"]["discomfort"] += 1
            self.episode_info["events"]["min_dist"].append(min(human_distances))

        else:
            info = Nothing()

        for human_dist in human_distances:
            if 0 <= human_dist < self.discomfort_dist * self.discomfort_scale:
                discomfort = (
                    (human_dist - self.discomfort_dist * self.discomfort_scale)
                    * self.discomfort_penalty
                    * self.time_step
                )
                reward += discomfort
                self.episode_info["events"]["discomfort"] += discomfort

        # Record episode info
        human_num = len(self.humans)
        for i in range(human_num):
            human_pos = [self.humans[i].px, self.humans[i].py]
            human_goal = [self.humans[i].gx, self.humans[i].gy]
            self.episode_info["pedestrians"][i]["id"] = self.humans[i].id
            self.episode_info["pedestrians"][i]["distance_traversed"].append(
                human_pos
            )
            self.episode_info["pedestrians"][i]["goal"].append(human_goal)

            self.episode_info["pedestrians"][i]["velocity"].append(
                [action.vx, action.vy]
            )  # holonomic

        robot_pos = [self.robot.px, self.robot.py]
        robot_vel = [self.robot.vx, self.robot.vy]
        self.episode_info["robot"]["distance_traversed"].append(robot_pos)
        self.episode_info["robot"]["velocity"].append(robot_vel)

        if update:
            # update all agents            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                human.step(action)
            self.global_time += self.time_step
            # for i, human in enumerate(self.humans):
            #     # only record the first time the human reaches the goal
            #     if self.human_times[i] == 0 and human.reached_destination():
            #         self.human_times[i] = self.global_time
            # compute the observation
            if self.robot.sensor == "coordinates":
                ob = self.compute_observation_for(self.robot)
            elif (
                self.robot.sensor.lower() == "rgb"
                or self.robot.sensor.lower() == "gray"
            ):
                raise NotImplementedError
            else:
                raise ValueError("Unknown robot sensor type")
            # store state, action value and attention weights
            self.states.append(
                [
                    self.robot.get_full_state(),
                    [human.get_full_state() for human in self.humans],
                    [human.id for human in self.humans],
                ]
            )
            # self.robot_actions.append(action)
            # self.rewards.append(reward)
            if hasattr(self.robot.policy, "action_values"):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, "get_attention_weights"):
                self.attention_weights.append(
                    self.robot.policy.get_attention_weights()
                )
        else:
            if self.robot.sensor == "coordinates":
                ob = [
                    human.get_next_observable_state(action)
                    for human, action in zip(self.humans, human_actions)
                ]
            elif self.robot.sensor == "RGB":
                raise NotImplementedError

        if done:
            self.episode_info["time"] = (
                -self.global_time * self.time_penalty / self.time_step
            )
            self.episode_info["global_time"] = self.global_time
            info = (
                self.episode_info
            )  # Return full episode information at the end
        return ob, reward, done, info

    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                ob.append(human.get_observable_state())
        else:
            ob = [
                other_human.get_observable_state()
                for other_human in self.humans
                if other_human != agent
            ]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def detect_collisions_with_human(self, action):
        dmin = float("inf")
        collisions = 0
        human_distances = list()
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == "holonomic":
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = (
                point_to_segment_dist(px, py, ex, ey, 0, 0)
                - human.radius
                - self.robot.radius
            )
            if closest_dist < 0:
                collisions += 1
                logger.debug(
                    "Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(
                        human.id, closest_dist, self.global_time
                    )
                )
                break
            elif closest_dist < dmin:
                dmin = closest_dist
            human_distances.append(closest_dist)

        return collisions, human_distances

    def detect_collisions_with_obstacles(self, action):
        # static_obstacle_dmin = float("inf")
        static_obstacle_collision = 0
        obstacle_distances = list()
        min_dist = self.robot.radius
        px = self.robot.px
        py = self.robot.py

        if self.robot.kinematics == "holonomic":
            vx = action.vx
            vy = action.vy
        else:
            vx = action.v * np.cos(action.r + self.robot.theta)
            vy = action.v * np.sin(action.r + self.robot.theta)
        ex = px + vx * self.time_step
        ey = py + vy * self.time_step
        for i, obstacle in enumerate(self.obstacles):
            robot_position = ex, ey
            obst_dist = line_distance(obstacle, robot_position)
            if obst_dist < min_dist:
                static_obstacle_collision += 1
                break
        return static_obstacle_collision, obstacle_distances

    def render(self, mode="video", output_file=None):
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.2
        y_offset = 0.4
        # cmap = plt.cm.get_cmap("hsv", 10)
        cmap = plt.cm.get_cmap("tab20")
        robot_color = "black"
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True
        display_roles = True

        xlim, ylim = self.scene_manager.get_map_size()

        if mode == "traj":
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            ax.set_xlabel("x(m)", fontsize=16)
            ax.set_ylabel("y(m)", fontsize=16)

            # draw static obstacles
            for ob in self.obstacles:
                ax.plot(ob[:2], ob[2:4], "-o", color="black", markersize=2.5)

            # add human start positions and goals
            human_colors = [cmap(i) for i, _ in enumerate(self.humans)]
            for i, human in enumerate(self.humans):
                human = self.humans[i]
                human_goal = mlines.Line2D(
                    [human.get_goal_position()[0]],
                    [human.get_goal_position()[1]],
                    color=human_colors[i],
                    marker="*",
                    linestyle="None",
                    markersize=15,
                )
                ax.add_artist(human_goal)
                human_start = mlines.Line2D(
                    [human.get_start_position()[0]],
                    [human.get_start_position()[1]],
                    color=human_colors[i],
                    marker="o",
                    linestyle="None",
                    markersize=15,
                )
                ax.add_artist(human_start)

            robot_positions = [
                self.states[i][0].position for i in range(len(self.states))
            ]
            human_positions = [
                [self.states[i][1][j].position for j in range(len(self.humans))]
                for i in range(len(self.states))
            ]

            for k, _ in enumerate(self.states):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(
                        robot_positions[k],
                        self.robot.radius,
                        fill=False,
                        color=robot_color,
                    )
                    humans = [
                        plt.Circle(
                            human_positions[k][i],
                            self.humans[i].radius,
                            fill=False,
                            color=cmap(i),
                        )
                        for i in range(len(self.humans))
                    ]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [
                        plt.text(
                            agents[i].center[0] - x_offset,
                            agents[i].center[1] - y_offset,
                            "{:.1f}".format(global_time),
                            color="black",
                            fontsize=14,
                        )
                        for i in range(sum(self.human_num) + 1)
                    ]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D(
                        (self.states[k - 1][0].px, self.states[k][0].px),
                        (self.states[k - 1][0].py, self.states[k][0].py),
                        color=robot_color,
                        ls="solid",
                    )
                    human_directions = [
                        plt.Line2D(
                            (
                                self.states[k - 1][1][i].px,
                                self.states[k][1][i].px,
                            ),
                            (
                                self.states[k - 1][1][i].py,
                                self.states[k][1][i].py,
                            ),
                            color=cmap(i),
                            ls="solid",
                        )
                        for i in range(sum(self.human_num))
                    ]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ["Robot"], fontsize=16)
            plt.show()
        elif mode == "video":
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            ax.set_xlabel("x(m)", fontsize=14)
            ax.set_ylabel("y(m)", fontsize=14)
            show_human_start_goal = True

            # draw static obstacles
            for ob in self.obstacles:
                ax.plot(ob[:2], ob[2:4], "-o", color="black", markersize=2.5)

            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            if show_human_start_goal:
                for i, human in enumerate(self.humans):
                    human_goal = mlines.Line2D(
                        [human.get_goal_position()[0]],
                        [human.get_goal_position()[1]],
                        color=human_colors[i],
                        marker="*",
                        linestyle="None",
                        markersize=8,
                    )
                    ax.add_artist(human_goal)
                    human_start = mlines.Line2D(
                        [human.get_start_position()[0]],
                        [human.get_start_position()[1]],
                        color=human_colors[i],
                        marker="o",
                        linestyle="None",
                        markersize=8,
                    )
                    ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D(
                [self.robot.get_start_position()[0]],
                [self.robot.get_start_position()[1]],
                color=robot_color,
                marker="o",
                linestyle="None",
                markersize=8,
            )
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D(
                [self.robot.get_goal_position()[0]],
                [self.robot.get_goal_position()[1]],
                color=robot_color,
                marker="*",
                linestyle="None",
                markersize=15,
                label="Goal",
            )
            robot = plt.Circle(
                robot_positions[0],
                self.robot.radius,
                fill=False,
                color=robot_color,
            )
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(robot_start)
            ax.add_artist(goal)
            plt.legend([robot, goal], ["Robot", "Goal"], fontsize=14)

            # add humans and their numbers
            human_positions = [
                [state[1][j].position for j in range(len(self.humans))]
                for state in self.states
            ]
            humans = [
                plt.Circle(
                    human_positions[0][i],
                    self.humans[i].radius,
                    fill=False,
                    color=cmap(i),
                )
                for i in range(len(self.humans))
            ]

            # disable showing human numbers
            if display_numbers:
                human_numbers = [
                    plt.text(
                        humans[i].center[0] - x_offset,
                        humans[i].center[1] + y_offset,
                        str(i),
                        color="black",
                    )
                    for i in range(len(self.humans))
                ]
            if display_roles:
                human_roles = [
                    plt.text(
                        humans[i].center[0] - 3 * x_offset,
                        humans[i].center[1] - 2 * y_offset,
                        self.humans[i].role,
                        color="black",
                    )
                    for i in range(len(self.humans))
                ]

            for i, human in enumerate(humans):
                ax.add_artist(human)
                if display_numbers:
                    ax.add_artist(human_numbers[i])
                if display_roles:
                    ax.add_artist(human_roles[i])

            # add time annotation
            time = plt.text(
                0.4,
                0.9,
                "Time: {}".format(0),
                fontsize=16,
                transform=ax.transAxes,
            )
            ax.add_artist(time)

            # visualize attention scores
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            orientations = []
            for i in range(sum(self.human_num) + 1):
                orientation = []
                for state in self.states:
                    agent_state = state[0] if i == 0 else state[1][i - 1]
                    if self.robot.kinematics == "unicycle" and i == 0:
                        direction = (
                            (agent_state.px, agent_state.py),
                            (
                                agent_state.px
                                + radius * np.cos(agent_state.theta),
                                agent_state.py
                                + radius * np.sin(agent_state.theta),
                            ),
                        )
                    else:
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        direction = (
                            (agent_state.px, agent_state.py),
                            (
                                agent_state.px + radius * np.cos(theta),
                                agent_state.py + radius * np.sin(theta),
                            ),
                        )
                    orientation.append(direction)
                orientations.append(orientation)
                if i == 0:
                    arrow_color = "black"
                    arrows = [
                        patches.FancyArrowPatch(
                            *orientation[0],
                            color=arrow_color,
                            arrowstyle=arrow_style,
                        )
                    ]
                else:
                    arrows.extend(
                        [
                            patches.FancyArrowPatch(
                                *orientation[0],
                                color=human_colors[i - 1],
                                arrowstyle=arrow_style,
                            )
                        ]
                    )

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            if len(self.trajs) != 0:
                human_future_positions = []
                human_future_circles = []
                for traj in self.trajs:
                    human_future_position = [
                        [
                            tensor_to_joint_state(traj[step + 1][0])
                            .human_states[i]
                            .position
                            for step in range(self.robot.policy.planning_depth)
                        ]
                        for i in range(sum(self.human_num))
                    ]
                    human_future_positions.append(human_future_position)

                for i in range(sum(self.human_num)):
                    circles = []
                    for j in range(self.robot.policy.planning_depth):
                        circle = plt.Circle(
                            human_future_positions[0][i][j],
                            self.humans[0].radius / (1.7 + j),
                            fill=False,
                            color=cmap(i),
                        )
                        ax.add_artist(circle)
                        circles.append(circle)
                    human_future_circles.append(circles)

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]

                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    if display_numbers:
                        human_numbers[i].set_position(
                            (
                                human.center[0] - x_offset,
                                human.center[1] + y_offset,
                            )
                        )
                    if display_roles:
                        human_roles[i].set_position(
                            (
                                human.center[0] - 3 * x_offset,
                                human.center[1] - 2 * y_offset,
                            )
                        )
                for arrow in arrows:
                    arrow.remove()

                for i in range(sum(self.human_num) + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [
                            patches.FancyArrowPatch(
                                *orientation[frame_num],
                                color="black",
                                arrowstyle=arrow_style,
                            )
                        ]
                    else:
                        arrows.extend(
                            [
                                patches.FancyArrowPatch(
                                    *orientation[frame_num],
                                    color=cmap(i - 1),
                                    arrowstyle=arrow_style,
                                )
                            ]
                        )

                for arrow in arrows:
                    ax.add_artist(arrow)
                    # if hasattr(self.robot.policy, 'get_attention_weights'):
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text("Time: {:.2f}".format(frame_num * self.time_step))

                if len(self.trajs) != 0:
                    for i, circles in enumerate(human_future_circles):
                        for j, circle in enumerate(circles):
                            circle.center = human_future_positions[global_step][
                                i
                            ][j]

            def plot_value_heatmap():
                if self.robot.kinematics != "holonomic":
                    print("Kinematics is not holonomic")
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(
                    self.action_values[global_step % len(self.states)][1:]
                )
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(
                    z,
                    (
                        self.robot.policy.rotation_samples,
                        self.robot.policy.speed_samples,
                    ),
                )
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color="k", ls="none")
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print(
                    "   " + " ".join(["{:>5}".format(i - 1) for i in range(w)])
                )
                for i in range(h):
                    print(
                        "{:<3}".format(i - 1)
                        + " ".join(
                            [
                                "{:.3f}".format(self.As[global_step][i][j])
                                for j in range(w)
                            ]
                        )
                    )
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print("feat is: ")
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print("X is: ")
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == "a":
                        if hasattr(self.robot.policy, "get_matrix_A"):
                            print_matrix_A()
                        if hasattr(self.robot.policy, "get_feat"):
                            print_feat()
                        if hasattr(self.robot.policy, "get_X"):
                            print_X()
                        if hasattr(self.robot.policy, "action_values"):
                            plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect("key_press_event", on_click)
            anim = animation.FuncAnimation(
                fig,
                update,
                frames=len(self.states),
                interval=self.time_step * 500,
            )
            anim.running = True

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(
                    fps=10, metadata=dict(artist="Me"), bitrate=1800
                )
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()
        else:
            raise NotImplementedError
