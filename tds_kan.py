"""TDS_KAN: a simple feature-gated KAN model for Google Colab.

Requirements:
    pip install torch efficient-kan
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from efficient_kan import KAN
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "efficient_kan is required. Install it with: pip install efficient-kan"
    ) from exc


class TDS_KAN(nn.Module):
    """Feature-gated KAN model.

    Structure:
    - Input gating vector `gate` (learnable nn.Parameter), initialized as 1 / num_features.
    - Backbone KAN with layers [input_dim, 64, 1].

    Forward:
    1) Normalize gate with softmax.
    2) Element-wise multiply normalized gate with input X.
    3) Feed weighted features into KAN backbone.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")

        self.input_dim = input_dim
        init_value = 1.0 / input_dim
        self.gate = nn.Parameter(torch.full((input_dim,), init_value, dtype=torch.float32))
        self.backbone = KAN([input_dim, 64, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D [batch, features], got shape: {tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected x.shape[1] == {self.input_dim}, got {x.shape[1]}"
            )

        weights = torch.softmax(self.gate, dim=0)
        weighted_x = x * weights
        return self.backbone(weighted_x)

    def freeze_backbone(self) -> None:
        """Freeze all KAN backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_gate_weights(self) -> torch.Tensor:
        """Return current normalized feature weights (softmax(gate))."""
        return torch.softmax(self.gate, dim=0).detach().clone()


if __name__ == "__main__":
    # Minimal Colab-friendly demo
    model = TDS_KAN(input_dim=8)
    x = torch.randn(4, 8)
    y = model(x)
    print("Output shape:", y.shape)
    print("Gate weights:", model.get_gate_weights())
