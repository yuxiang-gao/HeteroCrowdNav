import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class Linear(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = "holonomic"
        self.multiagent_training = True

    def configure(self, config):
        assert True

    def predict(self, state):
        robot_state = state.robot_state
        theta = np.arctan2(
            robot_state.gy - robot_state.py, robot_state.gx - robot_state.px
        )
        vx = np.cos(theta) * robot_state.v_pref
        vy = np.sin(theta) * robot_state.v_pref
        action = ActionXY(vx, vy)

        return action
