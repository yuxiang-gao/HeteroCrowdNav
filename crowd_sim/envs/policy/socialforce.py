import numpy as np
from pysocialforce import Simulator
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class SocialForce(Policy):
    def __init__(self):
        super().__init__()
        self.name = "SocialForce"
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = "holonomic"
        self.sim = None
        self.force_vectors = None

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state, groups=None, obstacles=None):
        """
        :param state:
        :param groups: group membership
        :param obs: obstacles
        :return:
        """
        sf_state = []
        self_state = state.self_state
        velocity = np.array(
            (self_state.gx - self_state.px, self_state.gy - self_state.py)
        )
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        sf_state.append(
            (
                self_state.px,
                self_state.py,
                pref_vel[0],
                pref_vel[1],
                self_state.gx,
                self_state.gy,
            )
        )
        for human_state in state.human_states:
            # approximate desired direction with current velocity
            if human_state.vx == 0 and human_state.vy == 0:
                gx = np.random.random()
                gy = np.random.random()
            else:
                gx = human_state.px + human_state.vx
                gy = human_state.py + human_state.vy
            sf_state.append(
                (
                    human_state.px,
                    human_state.py,
                    human_state.vx,
                    human_state.vy,
                    gx,
                    gy,
                )
            )

        self.sim = Simulator(
            np.array(sf_state), groups=groups, obstacles=obstacles
        )
        self.sim.step()
        action = ActionXY(self.sim.peds.state[0, 2], self.sim.peds.state[0, 3])

        self.last_state = state
        self.force_vectors = self.sim.force_vectors.transpose(
            (1, 0, 2)
        )  # (num_ped, num_forces, 2)

        return action

    def get_force_vectors(self, coeff=[1] * 6):
        return self.force_vectors * np.array(coeff).reshape(1, 6, 1)


class CentralizedSocialForce(SocialForce):
    """
    Centralized socialforce, a bit different from decentralized socialforce, where the goal position of other agents is
    set to be (0, 0)
    """

    def __init__(self):
        super().__init__()

        self.forces = None

    def predict(self, state, groups=None, obstacles=None):
        sf_state = []
        for agent_state in state:
            # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
            velocity = np.array(
                (
                    agent_state.gx - agent_state.px,
                    agent_state.gy - agent_state.py,
                )
            )
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed if speed > 1 else velocity

            sf_state.append(
                (
                    agent_state.px,
                    agent_state.py,
                    pref_vel[0],
                    pref_vel[1],
                    agent_state.gx,
                    agent_state.gy,
                )
            )
        self.sim = Simulator(
            np.array(sf_state), groups=groups, obstacles=obstacles
        )
        self.sim.step()
        self.forces = self.sim.forces
        actions = [
            ActionXY(self.sim.peds.state[i, 2], self.sim.peds.state[i, 3])
            for i in range(len(state))
        ]
        self.force_vectors = self.sim.force_vectors.transpose(
            (1, 0, 2)
        )  # (num_ped, num_forces, 2)
        # del self.sim

        return actions

    def get_forces(self):
        return self.forces
