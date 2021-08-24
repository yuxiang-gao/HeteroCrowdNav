import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


import gym
import torch as th
import torch.nn as nn
from torch.nn.functional import softmax
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    create_mlp,
    BaseFeaturesExtractor,
)

logger = logging.getLogger(__name__)


def mlp(input_dim, mlp_dims):
    layers = create_mlp(input_dim, mlp_dims[-1], mlp_dims[:-1])
    net = nn.Sequential(*layers)
    return net


def out_fts(net):
    if isinstance(net[-1], nn.ReLU):
        return net[-2].out_features
    else:
        return net[-1].out_features


class PairwiseAttentionFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        dims: Dict[str, List[int]],
    ) -> None:
        input_dim = observation_space.shape[1]  # (#humans, 13)
        mlp1_dims = dims["mlp1_dims"]
        mlp2_dims = dims["mlp2_dims"]
        attention_dims = dims["attention_dims"]
        # mlp3_dims = dims["mlp3_dims"]
        features_dim = mlp2_dims[-1]  # feature_dim is the same as mlp2 output
        super(PairwiseAttentionFeaturesExtractor, self).__init__(
            observation_space, features_dim
        )

        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp2 = mlp(out_fts(self.mlp1), mlp2_dims)
        self.attention = mlp(out_fts(self.mlp1) * 2, attention_dims)
        # self.mlp3 = mlp(out_fts(self.mlp2), mlp3_dims)

        self.global_state_dim = out_fts(self.mlp1)

    def forward(self, state: th.Tensor) -> th.Tensor:
        size = state.shape  # (batch_size, # humans, input_dim)å
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        global_state = th.mean(
            mlp1_output.view(size[0], size[1], -1), 1, keepdim=True
        )  # (batch_size, 1, mlp1)
        global_state = (
            global_state.expand((size[0], size[1], self.global_state_dim))
            .contiguous()
            .view(-1, self.global_state_dim)
        )  # (batch_size * # humans, mlp1)
        attention_input = th.cat(
            [mlp1_output, global_state], dim=1
        )  # (batch_size * # humans, 2 * mlp1)
        scores = (
            self.attention(attention_input)
            .view(size[0], size[1], 1)
            .squeeze(dim=2)
        )  # (batch_size, # humans)
        weights = softmax(scores, dim=1).unsqueeze(
            2
        )  # (batch_size, # humans, 1)å
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = th.sum(
            th.mul(weights, features), dim=1
        )  # (batch_size, mlp2)
        # mlp3_output = self.mlp3(weighted_feature)  # (batch_size, mlp3)
        # return mlp3_output
        return weighted_feature
