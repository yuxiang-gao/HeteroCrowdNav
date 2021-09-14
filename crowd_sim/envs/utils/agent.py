import abc
import logging
import numpy as np
from numpy.linalg import norm
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState

logger = logging.getLogger(__name__)


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = config[section]["visible"]
        self.v_pref = config[section]["v_pref"]
        self.radius = config[section]["radius"]
        self.policy = policy_factory[config[section]["policy"]]()
        self.sensor = config[section]["sensor"]
        self.kinematics = (
            self.policy.kinematics if self.policy is not None else None
        )
        self.px = None
        self.py = None
        self.sx = None
        self.sy = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

    def print_info(self):
        logger.info(
            "Agent is {} and has {} kinematic constraint".format(
                "visible" if self.visible else "invisible", self.kinematics
            )
        )

    def set_policy(self, policy):
        if self.time_step is None:
            raise ValueError("Time step is None")
        policy.set_time_step(self.time_step)
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def go_back(self):
        # reverse start and goal
        self.set(self.gx, self.gy, self.sx, self.sy, 0, 0)

    def set(self, px, py, gx, gy, vx, vy, theta=None, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        if theta is None:
            # auto set theta
            vec = np.array([gx - px, gy - py])
            self.theta = np.arctan2(*(vec / np.linalg.norm(vec)))
        else:
            self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == "holonomic":
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta,
        )

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_start_position(self):
        return self.sx, self.sy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob, groups=None):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == "holonomic":
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if isinstance(action, ActionXY):
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        elif isinstance(action, ActionRot):
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if isinstance(action, ActionXY):
            self.vx = action.vx
            self.vy = action.vy
        elif isinstance(action, ActionRot):
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    def reached_destination(self, margin=0.0):
        return (
            norm(
                np.array(self.get_position())
                - np.array(self.get_goal_position())
            )
            < self.radius + margin
        )
