import torch
import torch.nn as nn


class TemporalFusionMLP(nn.Module):
    """Fuse an ordered pair of per-frame spatial token grids."""

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.dim * 3),
            nn.Linear(self.dim * 3, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(
        self, s_prev: torch.Tensor, s_next: torch.Tensor
    ) -> torch.Tensor:
        if s_prev.shape != s_next.shape:
            raise ValueError(
                "Temporal fusion inputs must have identical shapes, got "
                f"{tuple(s_prev.shape)} and {tuple(s_next.shape)}"
            )
        if s_prev.shape[-1] != self.dim:
            raise ValueError(
                f"Expected feature dim {self.dim}, got {s_prev.shape[-1]}"
            )
        ordered_pair = torch.cat(
            (s_prev, s_next, s_next - s_prev), dim=-1
        )
        return self.net(ordered_pair)
