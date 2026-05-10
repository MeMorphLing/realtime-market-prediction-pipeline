"""GRU model for binary direction prediction."""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class GRUModel(nn.Module):
    """Stacked GRU with a linear classification head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Initialize layers.

        Args:
            input_size: Number of features per timestep.
            hidden_size: GRU hidden state width.
            num_layers: Number of stacked GRU layers.
            output_size: Output dimension (1 for binary classification logits).
            dropout: Dropout probability between GRU layers and before the head.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Tensor of shape ``(batch, output_size)`` with raw logits.
        """
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


def build_gru(input_size: int, **kwargs) -> GRUModel:
    """Factory that returns a :class:`GRUModel` instance."""
    return GRUModel(input_size=input_size, **kwargs)
