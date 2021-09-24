from collections import UserDict, Mapping
from pathlib import Path

import toml

DEFAULT_CONFIG = """
title = "Config file"

[debug]
    [debug.env]
    case_size.val = 1
    case_size.test = 1

    [debug.train]
    train_episodes = 1
    checkpoint_interval = 1
    evaluation_interval = 1
    target_update_interval = 1

    imitation_learning.il_episodes = 100
    imitation_learning.il_epochs = 5
    
[env]
time_limit = 25
time_step = 0.25
randomize_attributes = false
robot_sensor_range = 5
goal_radius = 0.3

    [env.case_size]
    val = 100
    test = 500
    train = 100000

    [env.reward]
    progress_reward = 0.1
    success_reward = 1.0
    collision_penalty = 0.25
    static_obstacle_collision_penalty = 0.25
    time_penalty = 0.0
    discomfort_penalty = 0.5
    discomfort_scale = 1.0
    discomfort_dist = 0.2

    [env.scenarios]
    phase.train = "cocktail_party"
    phase.val = "cocktail_party"
    phase.test = "cocktail_party"
    map_size = [10, 10]

        [env.scenarios.cocktail_party]
        table_radius = 1.0
        table_scale = 4.0
        table_shape = [[-1.0,1.0],[-1.0,-1.0],[1.0,-1.0],[1.0,1.0]]
        table_placement = [[-1.0,1.0],[-1.0,-1.0],[1.0,-1.0],[1.0,1.0]]
        circle_radius = 8.0

        [env.scenarios.hospital_hallway]
        width = 7
        length = 5

        [env.scenarios.circle_crossing]
        circle_radius = 6

        [env.scenarios.square_crossing]
        square_width = 10

        [env.scenarios.corner]
        width = 7

        [env.scenarios.corridor]
        width = 7
        length = 5

        [env.scenarios.t_intersection]
        width = 7
        length = 5


    [env.agents]
    human_num = [4, 8]
    perpetual = false
    centralized_planning = false
    use_groups = false
    use_types = true

        [[env.agents.humans]]
        role = "waiter"
        visible = true
        policy = "orca"
        radius = 0.35
        v_pref = 1.2
        sensor = "coordinates"

        [[env.agents.humans]]
        role = "guest"
        visible = true
        policy = "orca"
        radius = 0.3
        v_pref = 1
        sensor = "coordinates"


        [env.agents.robot]
        visible = false
        policy = "none"
        radius = 0.35
        v_pref = 1
        sensor = "coordinates"


[train]
rl_learning_rate = 0.001
# number of batches to train at the end of training episode
train_batches = 100
# training episodes in outer loop
train_episodes = 10000
# number of episodes sampled in one training episode
sample_episodes = 1
target_update_interval = 50
evaluation_interval = 1000
# the memory pool can roughly store 2K episodes, total size = episodes * 50
capacity = 100000
epsilon_start = 0.5
epsilon_end = 0.1
epsilon_decay = 4000
checkpoint_interval = 1000

    [train.trainer]
    batch_size = 100
    optimizer = "Adam"

    [train.imitation_learning]
    il_episodes = 3000
    il_policy = "orca"
    il_epochs = 50
    il_learning_rate = 0.01
    # increase the safety space in ORCA demonstration for robot
    safety_space = 0.15


[policy]
randomize_attributes = false # orca config

    [policy.rl]
    gamma = 0.9


    [policy.om]
    cell_num = 4
    cell_size = 1
    om_channel_size = 3


    [policy.action_space]
    kinematics = "differential"
    # action space size is speed_samples * rotation_samples + 1
    speed_samples = 5
    rotation_samples = 6
    sampling = "exponential"
    query_env = true


    [policy.cadrl]
    mlp_dims = [150, 100, 100, 1]
    multiagent_training = false


    [policy.lstm_rl]
    global_state_dim = 50
    mlp1_dims = [150, 100, 100, 50]
    mlp2_dims = [150, 100, 100, 1]
    multiagent_training = true
    with_om = false
    with_interaction_module = false


    [policy.srl]
    mlp1_dims = [150, 100, 100, 50]
    mlp2_dims = [150, 100, 100, 1]
    multiagent_training = true
    with_om = false


    [policy.sarl]
    mlp1_dims = [150, 100]
    mlp2_dims = [100, 50]
    attention_dims = [100, 100, 1]
    mlp3_dims = [150, 100, 100, 1]
    multiagent_training = true
    with_om = false
    with_global_state = true

    [policy.harl] 
    # action space size is speed_samples * rotation_samples + 1
    mlp1_dims = [150, 100, 0]
    mlp2_dims = [100, 50]
    attention_dims = [100, 100, 1]
    mlp3_dims = [150, 100, 100, 0]
    multiagent_training = true
    with_om = false
    with_global_state = true


"""
DEFAULT_CONFIG = toml.loads(DEFAULT_CONFIG)


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
            and isinstance(merge_dct[k], Mapping)
        ):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


class Config(UserDict):
    """
    Config class enables easy loading and getting config entries
    """

    def __init__(self, config) -> None:
        if isinstance(config, str) or isinstance(config, Path):
            print(f"Loading config from {config}")
            config = toml.load(config)
        assert isinstance(
            config, dict
        ), "Config class takes a dict or toml file"
        super().__init__(config)

    # def __getitem__(self, key):
    #     """Return Config rather than dict"""
    #     val = self.data[key]
    #     if isinstance(val, dict):
    #         val = Config(val)
    #     return val

    def __call__(self, *argv):
        """Make class callable to easily get entries"""
        try:
            output = self.data
            for arg in argv:
                if not isinstance(output, dict):
                    print(
                        f"Too many keys, return the closest entry: {argv} - {arg}"
                    )
                    return output
                output = output[arg]
            if isinstance(output, dict):
                output = Config(output)
            return output
        except KeyError as err:
            print(f"Config error {err}")

    def dump(self, config_output):
        """Dump dict to file"""
        with open(config_output, "w") as f:
            toml.dump(self.data, f)

    def merge(self, merge_dct):
        """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
        updating only top-level keys, dict_merge recurses down into dicts nested
        to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
        ``dct``.
        :param dct: dict onto which the merge is executed
        :param merge_dct: dct merged into dct
        :return: None"""
        for k, v in merge_dct.items():
            if (
                k in self
                and isinstance(self.data[k], dict)
                and isinstance(merge_dct[k], Mapping)
            ):
                dict_merge(self.data[k], merge_dct[k])
            else:
                self[k] = merge_dct[k]
