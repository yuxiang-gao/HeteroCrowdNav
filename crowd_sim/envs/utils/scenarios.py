from crowd_sim.envs.utils.logging import logging_info, logging_debug
from enum import Enum
from abc import ABC

import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.utils import line_distance


class Scenario(str, Enum):
    CIRCLE_CROSSING = "circle_crossing"
    CORRIDOR = "corridor"
    LONG_CORRIDOR = "long_corridor"
    CORNER = "corner"
    T_INTERSECTION = "t_intersection"
    COCKTAIL_PARTY = "cocktail_party"

    @classmethod
    def find(cls, name):
        for i in cls:
            if name in i.value:
                return i


def convert_polygon(square_corners):
    """
    Convert corners of a polygon to lines: x1,x2,y1,y2
    """
    lines = []
    length = len(square_corners)
    for i in range(length):
        c1 = square_corners[i]
        c2 = square_corners[(i + 1) % length]
        lines.append([c1[0], c2[0], c1[1], c2[1]])
    return lines


class ScenarioManager(object):
    def __init__(self, scenario, config, seed):
        """
        scenario: scenario name or Enum object
        config: env config
        """
        self.config = None
        self.scenario_config = None
        self.map_size = None
        self.obstacles = []  # x_min,x_max, y_min, y_max
        self.polygon_obstacles = None
        self.spawn_positions = []

        self.rng = np.random.default_rng(seed)
        if isinstance(scenario, Scenario):
            self.scenario = scenario
        else:
            self.scenario = Scenario.find(scenario)
        self.configure(config)

    def configure(self, config):
        self.config = config
        self.scenario_config = config("scenarios", self.scenario.value)
        self.map_size = config(
            "scenarios", "map_size"
        )  # half width and height, assumed to be symmetric
        # self.v_pref = config("agents", "humans")[0].get("v_pref")
        # for human in self.agent_config("humans"):
        #     self.human_radius.append(human["radius"])
        self.discomfort_dist = config("reward", "discomfort_dist")

        if self.scenario == Scenario.CORNER:
            width = self.scenario_config("width")
            self.obstacles = np.array(
                [
                    [-width, width, width, width],
                    [-width, 0, 0, 0],
                    [width, width, -width, width],
                    [0, 0, -width, 0],
                ]
            )
            self.spawn_positions = [
                [-width, width / 2],
                [width / 2, -width],
            ]
            self.robot_spawn_positions = self.spawn_positions
        elif self.scenario == Scenario.CORRIDOR:
            width = self.scenario_config("width")
            length = self.scenario_config("length")
            self.obstacles = np.array(
                [
                    [-length - 2, length + 2, width / 2, width / 2],
                    [-length - 2, length + 2, -width / 2, -width / 2],
                ]
            )
            self.spawn_positions = [[length, 0], [-length, 0]]
            self.robot_spawn_positions = [[length + 1, -1], [-length - 1, +1]]
        elif self.scenario == Scenario.T_INTERSECTION:
            width = self.scenario_config("width")
            length = self.scenario_config("length")
            self.obstacles = [
                [-length, length, width, width],
                [-length, -width / 2, 0, 0],
                [width / 2, length, 0, 0],
                [width / 2, width / 2, 0, -length],
                [-width / 2, -width / 2, 0, -length],
            ]
            self.spawn_positions = [
                [-length, width / 2],
                [length, width / 2],
                [0, -length],
            ]
            self.robot_spawn_positions = [
                [0, -length - 1],
                [length + 1, width / 2 + 1],
            ]

        elif self.scenario == Scenario.CIRCLE_CROSSING:
            self.circle_radius = self.scenario_config("circle_radius")
            self.robot_spawn_positions = [
                [0, -self.circle_radius],
                [0, self.circle_radius],
            ]
        elif self.scenario == Scenario.COCKTAIL_PARTY:
            table_placement = np.array(
                self.scenario_config("table_placement")
            ).astype("float64")
            table_shape = np.array(self.scenario_config("table_shape")).astype(
                "float64"
            )
            self.table_radius = self.scenario_config("table_radius")
            table_scale = self.scenario_config("table_scale")
            self.circle_radius = self.scenario_config("circle_radius")
            table_centers = table_placement * table_scale

            self.polygon_obstacles = []
            for center in table_centers:
                self.polygon_obstacles.append(
                    center + table_shape * self.table_radius / np.sqrt(2)
                )
                self.obstacles += convert_polygon(self.polygon_obstacles[-1])

            self.guest_spawn_positions = table_centers

    def get_robot_spawn_position(self):
        if self.scenario == Scenario.COCKTAIL_PARTY:
            return self.sample_pos_goal_on_circle(
                self.circle_radius, np.pi, noise=np.pi / 36
            )  # 5deg noise
        else:
            return self.robot_spawn_positions[0], self.robot_spawn_positions[1]

    def get_spawn_position(
        self, type_idx=None
    ):  # return (center, goal), no noise
        if self.scenario == Scenario.CIRCLE_CROSSING:
            return self.sample_pos_goal_on_circle(
                self.circle_radius, np.pi * 2, noise=np.pi / 36
            )
        if self.scenario == Scenario.COCKTAIL_PARTY:
            if type_idx == 1:  # "guest"
                start_center, goal_center = self.rng.choice(
                    self.guest_spawn_positions,
                    size=2,
                    replace=False,
                )
                offset_margin = 0.4
                start = start_center + np.array(
                    self.sample_pos_goal_on_circle(
                        self.table_radius + offset_margin, np.pi * 2
                    )[0]
                )
                vec = np.array(goal_center - start_center)
                # vec = vec / norm(vec)
                # goal += vec * (self.table_radius + offset_margin)
                goal = start + vec
                return start, goal

            elif type_idx == 0:  # "waiter"
                # generate waiter on circle
                return self.sample_pos_goal_on_circle(
                    self.circle_radius, np.pi * 2, noise=np.pi / 36
                )
        # elif self.scenario == Scenario.LONG_CORRIDOR:
        #     return self.spawn_positions[0], self.spawn_positions[1]
        else:
            start_idx, goal_idx = np.random.choice(
                range(len(self.spawn_positions)), 2, replace=False
            )
            return (
                self.spawn_positions[start_idx],
                self.spawn_positions[goal_idx],
            )

    def get_spawn_positions(self, groups=None, type_idx=None):
        if (
            self.scenario == Scenario.CIRCLE_CROSSING
            or self.scenario == Scenario.COCKTAIL_PARTY
        ):
            return self.get_spawn_position(type_idx=type_idx)
        else:
            # num_human = sum(groups)
            # average_human = num_human / len(self.spawn_positions)  # avg human per spawn pos
            # sample_radius = average_human * (self.human_radius * 2 + self.discomfort_dist)
            sample_radius = 1
            noise = self.rng.uniform(-sample_radius, sample_radius, 2)

        center, goal = self.get_spawn_position()
        return center + noise, goal + noise

    def calc_closest_obstacle_map(self, threshold=3):
        x = np.arange(-self.map_size[0], self.map_size[0], 0.1)
        y = np.arange(-self.map_size[1], self.map_size[1], 0.1)
        xx, yy = np.meshgrid(x, y, sparse=True)
        z = None
        for ob in self.obstacles:
            dist = line_distance(ob, (xx, yy))
            if z is None:
                z = dist
            else:
                z = np.minimum(z, dist)
        z[z > threshold] = 0
        return z

    def sample_pos_goal_on_circle(self, radius, angular_range, noise=None):
        angle = self.rng.random() * angular_range
        pos = radius * np.array([np.cos(angle), np.sin(angle)])
        if noise is not None:
            angle += (self.rng.random() - 0.5) * noise
            goal = -radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            goal = -pos
        return pos, goal


class SceneManager(object):
    def __init__(self, scenario, robot, config, seed):
        self.config = None
        self.agent_config = None
        self.scenario_manager = None

        self.humans = []
        self.membership = []
        # self.obstacles_sampled = None
        self.rng = None

        self.configure(config)
        self.set_scenario(scenario, seed)
        self.robot = robot

    def configure(self, config):
        self.config = config
        self.agent_config = config("agents")
        self.human_radius = []
        for human in self.agent_config("humans"):
            self.human_radius.append(human["radius"])
        self.robot_radius = self.agent_config("robot", "radius")
        self.discomfort_dist = self.config("reward", "discomfort_dist")
        self.randomize_attributes = self.config("randomize_attributes")

        # group related
        self.max_groups = None

    def set_scenario(self, scenario, seed):
        self.humans = []
        self.membership = []
        self.scenario_manager = ScenarioManager(scenario, self.config, seed)
        # pass obstacles to Agent class
        JointState.set_obstacles(self.get_obstacles())
        JointState.polygon_obstacles = self.scenario_manager.polygon_obstacles
        self.rng = np.random.default_rng(seed=seed)
        # self.obstacles_sampled = self.sample_obstalces(self.get_obstacles())

    def get_current_scenario(self):
        return self.scenario_manager.scenario.value

    def get_scene(self):
        group_membership = []
        individual_membership = []
        for group in self.membership:
            if len(group) == 1:
                individual_membership.append(*group)
            else:
                group_membership.append(group)
        return (
            self.humans,
            self.scenario_manager.obstacles,
            group_membership,
            individual_membership,
        )

    def get_obstacles(self):
        return self.scenario_manager.obstacles

    def get_map_size(self):
        return self.scenario_manager.map_size

    def spawn(
        self,
        num_human=[5],
        group_size_lambda=1.2,
        use_groups=False,
        use_types=False,
        set_robot=True,
        group_sizes=None,
    ):
        # group_sizes = self.rng.poisson(group_size_lambda, num_groups) + 1  # no zeros
        # human_idx = np.array(range(sum(group_sizes)))
        # Spawn robot
        if set_robot:
            start, goal = self.scenario_manager.get_robot_spawn_position()
            logging_debug(
                f"[{self.scenario_manager.scenario.value}] Spawn robot: {start} -> {goal}"
            )
            self.spawn_robot(start, goal)

        if use_groups:
            if group_sizes is None:
                group_sizes = []
                if self.max_groups == 1:
                    group_sizes.append(num_human)
                else:
                    while True:
                        size = self.rng.poisson(group_size_lambda) + 1
                        if sum(group_sizes) + size > num_human:
                            if num_human > sum(group_sizes):
                                group_sizes.append(num_human - sum(group_sizes))
                            break
                        else:
                            group_sizes.append(size)
                logging_info(f"Generating groups of size: {group_sizes}")
            # else:
            #     group_sizes = np.ones(num_human)
            human_idx = np.arange(sum(num_human))
            self.membership = self.split_array(human_idx, group_sizes)

            for i, size in enumerate(group_sizes):
                center, goal = self.scenario_manager.get_spawn_positions(
                    group_sizes
                )
                if size > 1:
                    logging_debug(
                        f"Spawn group {i} of size {size}, center: {center}, goal: {goal}"
                    )
                else:
                    logging_debug(f"Spawn p{i}: {center} -> {goal}")
                self.humans += self.spawn_group(size, center, goal)

            for i, human in enumerate(self.humans):
                human.id = i
        elif use_types:
            if not isinstance(num_human, list):
                raise TypeError(
                    "num_human must be a list of num for each type."
                )
            for type_idx, type_size in enumerate(num_human):
                for i, size in enumerate(range(type_size)):
                    center, goal = self.scenario_manager.get_spawn_positions(
                        type_idx=type_idx
                    )
                    self.humans += self.spawn_group(
                        1, center, goal, type_idx=type_idx
                    )
                    logging_debug(
                        f"[{self.scenario_manager.scenario.value}] Spawn p{i} of type {type_idx}: {center} -> {goal}"
                    )

            for i, human in enumerate(self.humans):
                human.id = i

    def spawn_robot(self, start, goal):
        # noise = self.random_vector(length=(self.robot_radius * 2 + self.discomfort_dist))
        # spawn_pos = center + noise  # spawn noise based on group size
        # agent_radius = self.robot_radius
        # while True:
        #     if not self.check_collision(spawn_pos, robot=True):  # break if there is no collision
        #         break
        #     else:
        #         spawn_pos += self.random_vector(
        #             length=agent_radius
        #         )  # gentlely nudge the new ped to avoid collision
        self.robot.set(*start, *goal, 0, 0)

    # def spawn_group_with_form(
    #     self, size, center, goal, form=[[1, 0], [0, 0.25], [0, -0.25]]
    # ):
    #     humans = []
    #     while True:
    #         spawn_pos = np.asarray(center) + np.asarray(form)
    #         while True:
    #             collision = False
    #             for pos in spawn_pos:
    #                 if self.check_collision(
    #                     pos, others=humans
    #                 ):  # break if there is no collision
    #                     collision = True
    #             if collision:
    #                 spawn_pos += self.random_vector(
    #                     length=self.human_radius[0]
    #                 )  # gentlely nudge the new ped to avoid collision
    #             else:
    #                 break
    #         for pos in spawn_pos:
    #             human = Human(self.config, "humans")
    #             if self.randomize_attributes:
    #                 human.sample_random_attributes()
    #             human.set(*pos, *goal, 0, 0)
    #             humans.append(human)
    #         # logging_info(f"spawn humans at {spawn_pos}, {len(humans)},{size}")
    #         if len(humans) == size:
    #             return humans

    def spawn_group(self, size, center, goal, type_idx=None):
        humans = []
        while True:
            spawn_pos = center
            while True:
                if not self.check_collision(
                    spawn_pos, radius=self.human_radius[type_idx], others=humans
                ):  # break if there is no collision
                    break
                else:
                    spawn_pos += self.random_vector(
                        length=self.human_radius[type_idx]
                    )  # gentlely nudge the new ped to avoid collision

            human = Human(self.agent_config, "humans", type_idx)
            if self.randomize_attributes:
                human.sample_random_attributes()
            human.set(*spawn_pos, *goal, 0, 0)
            humans.append(human)
            if len(humans) == size:
                return humans

    def check_collision(self, position, radius=0.3, others=[], robot=False):
        """Check collision, return true if there is collsion, false otherwise"""

        agents = (
            [self.robot] + self.humans + others
            if not robot
            else self.humans + others
        )
        # radius = radius if not robot else self.robot_radius
        # Check collision with agents
        for agent in agents:
            min_dist = radius + agent.radius + self.discomfort_dist
            if norm((position - agent.get_position()), axis=-1) < min_dist:
                return True

        # Check with obstacles
        for ob in self.get_obstacles():
            min_dist = radius + self.discomfort_dist
            if line_distance(ob, position) < min_dist:
                return True
        return False

    def get_closest_obstacle(self, position):
        min_dist = np.inf
        for ob in self.get_obstacles():
            dist = line_distance(ob, position)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def random_vector(self, length=1.0):
        vec = self.rng.random(2) - 0.5
        return vec / norm(vec) * length

    @staticmethod
    def split_array(array, split):
        # split array into subarrays of variable lengths
        assert len(array) == sum(
            split
        ), "sum of splits must equal to array length!"
        assert isinstance(array, np.ndarray), "Input must be an ndarray"
        output_array = []
        idx = 0
        for s in split:
            s = int(s)
            output_array.append(array[idx : idx + s])
            idx += s
        return np.array(output_array)

    @staticmethod
    def sample_obstalces(ob, resolution=10):
        """Sample from line obstacles"""
        ob_sampled = None
        for startx, endx, starty, endy in ob:
            samples = int(norm((startx - endx, starty - endy)) * resolution)
            line = np.array(
                list(
                    zip(
                        np.linspace(startx, endx, samples),
                        np.linspace(starty, endy, samples),
                    )
                )
            )
            ob_sampled = (
                line if ob_sampled is None else np.vstack((ob_sampled, line))
            )
        return ob_sampled
