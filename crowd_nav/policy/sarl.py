import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.nn.utils.rnn as rnn_utils
from crowd_sim.envs.utils.logging import logging_info, logging_debug
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        robot_state_dim,
        mlp1_dims,
        mlp2_dims,
        mlp3_dims,
        attention_dims,
        with_global_state,
        cell_size,
        cell_num,
    ):
        super().__init__()
        self.robot_state_dim = robot_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.robot_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

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

        size = state.shape
        robot_state = state[:, 0, : self.robot_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
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

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (
            scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)
        ).unsqueeze(2)
        # mask = rnn_utils.pad_sequence(
        #     [torch.ones(length.item()) for length in lengths], batch_first=True
        # )
        # masked_scores = scores * mask.float()
        # max_scores = torch.max(masked_scores, dim=1, keepdim=True)[0]
        # exps = torch.exp(masked_scores - max_scores)
        # masked_exps = exps * mask.float()
        # masked_sums = masked_exps.sum(1, keepdim=True)
        # weights = (masked_exps / masked_sums).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([robot_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = "SARL"
        self.attention_weights = None

    def configure(self, config):
        self.set_common_parameters(config)
        sarl_config = config["sarl"]
        mlp1_dims = sarl_config["mlp1_dims"]
        mlp2_dims = sarl_config["mlp2_dims"]
        attention_dims = sarl_config["attention_dims"]
        mlp3_dims = sarl_config["mlp3_dims"]

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
            with_global_state,
            self.cell_size,
            self.cell_num,
        )
        if self.with_om:
            self.name = "OM-SARL"
        logging_info(
            "Policy: {} {} global state".format(
                self.name, "w/" if with_global_state else "w/o"
            )
        )

    def get_attention_weights(self):
        return self.model.attention_weights
