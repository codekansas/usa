from dataclasses import dataclass

import ml.api as ml
import torch
from omegaconf import MISSING
from torch import Tensor, nn

POSITION_INPUT_DIMS = 3


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, num_embs: int, learned: bool = False) -> None:
        super().__init__()

        self.num_embs = num_embs
        self.freqs = nn.Parameter(2.0 ** torch.arange(0, num_embs), requires_grad=learned)

    @property
    def out_dims(self) -> int:
        return 2 * self.num_embs + 1

    def forward(self, pos: Tensor) -> Tensor:
        """Applies sinusoidal embeddings to input positions.

        Input:
            pos: Tensor with shape (..., N)

        Output:
            Embedded positions, with shape (..., N * (num_embs * 2) + 1)
        """

        pos = pos.unsqueeze(-1)
        freq_pos = self.freqs * pos
        sin_embs, cos_embs = torch.sin(freq_pos), torch.cos(freq_pos)
        return torch.cat([pos, sin_embs, cos_embs], dim=-1).flatten(-2)


def init_weights(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


@dataclass
class Point2EmbModelConfig(ml.BaseModelConfig):
    num_layers: int = ml.conf_field(MISSING, help="Number of MLP layers for encoding position")
    hidden_dims: int = ml.conf_field(MISSING, help="Number of hidden layer dimensions")
    num_pos_embs: int = ml.conf_field(6, help="Number of positional embedding frequencies")
    output_dims: int = ml.conf_field(MISSING, help="Number of output dimensions")


@ml.register_model("point2emb", Point2EmbModelConfig)
class Point2EmbModel(ml.BaseModel[Point2EmbModelConfig]):
    def __init__(self, config: Point2EmbModelConfig) -> None:
        super().__init__(config)

        # Gets the position embedding MLP.
        self.pos_embs = SinusoidalPositionalEmbeddings(config.num_pos_embs)
        pos_mlp_in_dims = POSITION_INPUT_DIMS * self.pos_embs.out_dims
        layers: list[nn.Module] = []
        layers += [nn.Sequential(nn.Linear(pos_mlp_in_dims, config.hidden_dims), nn.ReLU())]
        layers += [
            nn.Sequential(nn.Linear(config.hidden_dims, config.hidden_dims), nn.ReLU())
            for _ in range(config.num_layers - 1)
        ]
        layers += [nn.Linear(config.hidden_dims, config.output_dims)]
        self.position_mlp = nn.Sequential(*layers)

        self.apply(init_weights)

    def forward(self, points: Tensor) -> Tensor:
        """Simple model mapping a viewing angle to an embedding vector.

        Args:
            points: The point cloud, with shape (B, N, 3)

        Returns:
            The output embedding for the viden views, with shape (B, N, E)
        """

        # Embeds the (X, Y, Z) coordinates.
        pos_embs = self.pos_embs(points)
        preds = self.position_mlp(pos_embs)

        return preds
