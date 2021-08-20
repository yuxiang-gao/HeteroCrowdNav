import logging
import numpy as np

import torch
import torch.nn as nn

from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.action import ActionRot, ActionXY


class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        robot_state_dim,
        mlp1_dims,
        mlp2_dims,
        mlp3_dims,
        attention_dims,
        action_space_size,
        with_global_state,
        cell_size,
        cell_num,
    ):
        super().__init__()
        self.robot_state_dim = robot_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.last_mpl1 = mlp1_dims[-1]
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.attention_weights = None
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.robot_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims, last_relu=True)
        self.lin_value = nn.Linear(mlp3_dims[-1], 1)
        self.lin_policy = nn.Linear(mlp3_dims[-1], action_space_size)

    def forward(self, state_input):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        if isinstance(state_input, tuple):
            state, lengths = state_input
        else:
            state = state_input
            lengths = torch.IntTensor([state.size()[1]])

        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)

        size = state.shape
        robot_state = state[:, 0, : self.robot_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        num_people = np.sum(state.detach().numpy()[:, :, 3] > 0, axis=1)
        scalar_num_people = num_people[0]
        if self.with_global_state:
            # compute attention scores
            if size[0] > 1 or scalar_num_people != size[1]:  # num_people[0]
                num_people = np.expand_dims(num_people, 1)
                num_people = np.expand_dims(num_people, 2)
                num_people = torch.tensor(
                    num_people, dtype=torch.float32
                ).repeat((1, size[1], self.last_mpl1))
                global_state = (
                    mlp1_output.view(size[0], size[1], -1) / num_people
                )
            else:
                global_state = torch.mean(
                    mlp1_output.view(size[0], size[1], -1), 1, keepdim=True
                )
            global_state = (
                global_state.expand((size[0], size[1], self.global_state_dim))
                .contiguous()
                .view(-1, self.global_state_dim)
            )
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = (
            self.attention(attention_input)
            .view(size[0], size[1], 1)
            .squeeze(dim=2)
        )
        if size[0] > 1 or scalar_num_people != size[1]:
            mask = (state[:, :, 3] > 0).float()
            scores *= mask
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (
            scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)
        ).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([robot_state, weighted_feature], dim=1)
        joint_state = self.mlp3(joint_state)
        policies = torch.nn.functional.softmax(
            self.lin_policy(joint_state), dim=-1
        )
        values = self.lin_value(joint_state)
        return policies, values

    def forward_with_processing(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to policy-value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)
        """
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)
        # POLICY, VALUE UPDATE (Look-ahead not appropriate here):
        occupancy_maps = None
        self.action_values = list()
        batch_state = torch.cat(
            [
                torch.Tensor([state.robot_state + human_state]).to(self.device)
                for human_state in state.human_states
            ],
            dim=0,
        )
        rotated_batch_input = self.rotate(batch_state).unsqueeze(0)
        if self.with_om:
            if occupancy_maps is None:
                occupancy_maps = self.build_occupancy_maps(
                    state.human_states
                ).unsqueeze(0)
            rotated_batch_input = torch.cat(
                [rotated_batch_input, occupancy_maps.to(self.device)], dim=2
            )
        policy, value = self.forward(rotated_batch_input)
        self.action_values = policy.detach().numpy()[0].tolist()
        if self.phase == "train":
            self.last_state = self.transform(state)
        ob = np.expand_dims(self.last_state.detach().numpy(), axis=0)
        if self.env.max_humans > 0:
            ob = np.concatenate(
                (
                    ob,
                    np.zeros(
                        (
                            1,
                            self.env.max_humans - sum(self.env.human_num),
                            ob.shape[2],
                        )
                    ),
                ),
                axis=1,
            )
        return policy, value, ob


class HARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = "HARL"
        self.attention_weights = None

    def configure(self, config):
        self.set_common_parameters(config)
        sarl_config = config["harl"]
        mlp1_dims = sarl_config["mlp1_dims"]
        mlp2_dims = sarl_config["mlp2_dims"]
        attention_dims = sarl_config["attention_dims"]
        mlp3_dims = sarl_config["mlp3_dims"]
        action_space_size = (
            config["action_space"]["speed_samples"]
            * config["action_space"]["rotation_samples"]
            + 1
        )  # action space size is speed_samples * rotation_samples + 1
        self.with_om = sarl_config["with_om"]
        self.multiagent_training = sarl_config["multiagent_training"]
        with_global_state = sarl_config["with_global_state"]
        self.model = ValueNetwork(
            self.input_dim(),
            self.robot_state_dim,
            mlp1_dims,
            mlp2_dims,
            mlp3_dims,
            attention_dims,
            action_space_size,
            with_global_state,
            self.cell_size,
            self.cell_num,
        )
        if self.with_om:
            self.name = "OM-HARL"
        logging.info(
            "Policy: {} {} global state".format(
                self.name, "w/" if with_global_state else "w/o"
            )
        )

    def get_attention_weights(self):
        return self.model.attention_weights

    def predict(self, state, obstacles=None):
        if self.phase is None or self.device is None:
            raise AttributeError("Phase, device attributes have to be set!")
        if self.phase == "train" and self.epsilon is None:
            raise AttributeError(
                "Epsilon attribute has to be set in training phase"
            )

        if self.reach_destination(state):
            return (
                ActionXY(0, 0)
                if self.kinematics == "holonomic"
                else ActionRot(0, 0)
            )
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)

        if not state.human_states:
            assert self.phase != "train"
            if hasattr(self, "attention_weights"):
                self.attention_weights = list()
            return self.select_greedy_action(state.robot_state)

        # POLICY, VALUE UPDATE (Look-ahead not appropriate here):
        occupancy_maps = None
        self.action_values = list()
        batch_state = torch.cat(
            [
                torch.Tensor([state.robot_state + human_state]).to(self.device)
                for human_state in state.human_states
            ],
            dim=0,
        )
        rotated_batch_input = self.rotate(batch_state).unsqueeze(0)
        if self.with_om:
            if occupancy_maps is None:
                occupancy_maps = self.build_occupancy_maps(
                    state.human_states
                ).unsqueeze(0)
            rotated_batch_input = torch.cat(
                [rotated_batch_input, occupancy_maps.to(self.device)], dim=2
            )
        policy, value = self.model(rotated_batch_input)
        self.action_values = policy.detach().numpy()[0].tolist()
        if self.phase == "train":
            self.last_state = self.transform(state)
        ob = np.expand_dims(self.last_state.detach().numpy(), axis=0)
        if self.env.max_humans > 0:
            ob = np.concatenate(
                (
                    ob,
                    np.zeros(
                        (
                            1,
                            self.env.max_humans - self.env.human_num,
                            ob.shape[2],
                        )
                    ),
                ),
                axis=1,
            )

        # batch_states = torch.cat(
        #     [
        #         torch.Tensor([state.robot_state + human_state]).to(self.device)
        #         for human_state in state.human_states
        #     ],
        #     dim=0,
        # )
        # rotated_batch_input = self.rotate(batch_states).unsqueeze(0)

        # pi, val = self.model(rotated_batch_input, getPI=True)
        # val = val.data.item()
        # a = pi.sample()
        # log_pi = pi.log_prob(a).data.item()  # log_prob needs a Tensor
        # a = a.data.item()
        # next_action = self.action_space[a]

        # if self.phase == "train":
        #     self.last_state = self.transform(state)

        return a, next_action, pi, val, log_pi
