from dataclasses import dataclass

import ml.api as ml
import torch
from omegaconf import MISSING
from torch import Tensor, nn
from torch.distributions.normal import Normal
from torch.nn import functional as F


@dataclass
class A2CModelConfig(ml.BaseModelConfig):
    state_dims: int = ml.conf_field(MISSING, help="The number of state dimensions")
    action_dims: int = ml.conf_field(MISSING, help="The number of action dimensions")
    hidden_dims: int = ml.conf_field(MISSING, help="The number of hidden dimensions")
    num_layers: int = ml.conf_field(MISSING, help="The number of hidden layers")
    activation: str = ml.conf_field("leaky_relu", help="The activation function to use")
    fixed_std: bool = ml.conf_field(False, help="Whether to use a fixed standard deviation")


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: ml.ActivationType = "leaky_relu",
        norm: ml.NormType = "layer_affine",
    ) -> None:
        super().__init__()

        # Saves the model parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.act = act

        # Instantiates the model layers.
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                self.layers.append(ml.get_norm_linear(norm, dim=out_dim))
                self.layers.append(ml.get_activation(act))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Makes the last layer very small.
        last_layer = self.layers[-1]
        assert isinstance(last_layer, nn.Linear)
        nn.init.xavier_uniform_(last_layer.weight, gain=0.01)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@ml.register_model("a2c", A2CModelConfig)
class A2CModel(ml.BaseModel[A2CModelConfig]):
    fixed_std: Tensor | None

    def __init__(self, config: A2CModelConfig) -> None:
        super().__init__(config)

        self.policy_net = FeedForwardNet(
            config.state_dims,
            config.hidden_dims,
            config.action_dims if config.fixed_std else config.action_dims * 2,
            config.num_layers,
            act=ml.cast_activation_type(config.activation),
        )

        if config.fixed_std:
            self.register_parameter("fixed_std", nn.Parameter(torch.ones(1, config.action_dims)))
        else:
            self.register_parameter("fixed_std", None)

        self.value_net = FeedForwardNet(
            config.state_dims,
            config.hidden_dims,
            1,
            config.num_layers,
            act=ml.cast_activation_type(config.activation),
        )

    def forward_policy_net(self, state: Tensor) -> Normal:
        outputs = self.policy_net(state)

        if self.fixed_std is None:
            mean, std = outputs.tensor_split(2, dim=-1)
            std = F.softplus(std)
        else:
            mean, std = outputs, self.fixed_std

        mean, std = mean.clamp(-1e4, 1e4), std.clamp(1e-3, 1e4)
        return Normal(mean, std)

    def forward_value_net(self, state: Tensor) -> Tensor:
        return self.value_net(state)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self.policy_net(state), self.value_net(state)
