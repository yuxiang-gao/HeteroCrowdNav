from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section, type_idx=0):
        super().__init__(config[section], type_idx)  # hack
        self.type_name = config[section][type_idx].get("role")
        self.type_idx = type_idx

    def act(self, ob, groups=None):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state, groups, state.obstacles)
        return action
