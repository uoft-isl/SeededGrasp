# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implementation of a Multi-Layer Perceptron."""

import copy

import torch.nn.functional as F
from torch import nn


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MLP(nn.Module):
    """MLP class with optional Layer Normalization."""

    def __init__(
        self, in_features, out_features, num_hidden, hidden_dim, use_layer_norm=False
    ) -> None:
        super().__init__()

        self.use_layer_norm = use_layer_norm

        self.layer0 = nn.Linear(in_features, hidden_dim)
        self.layers = clones(nn.Linear(hidden_dim, hidden_dim), num_hidden)
        self.out = nn.Linear(hidden_dim, out_features)

        # Only add layer norms if requested
        if self.use_layer_norm:
            self.layer_norms = clones(nn.LayerNorm(hidden_dim), num_hidden + 1)

    def forward(self, x):
        x = self.layer0(x)
        if self.use_layer_norm:
            x = self.layer_norms[0](x)
        x = F.mish(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.layer_norms[i + 1](x)
            x = F.mish(x)

        return self.out(x)
