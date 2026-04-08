"""
RegionPooler — extract a single feature vector per region by mask-pooling
visual tokens. Used to fill <mask> placeholder positions in the LLM input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionPooler(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,   # [B, N, C] — patch tokens (e.g. DINO output)
        masks: torch.Tensor,           # [B, R, H, W] — binary masks per region (R = max regions)
        mask_valid: torch.Tensor,      # [B, R] — bool, True if region exists
        spatial_size: tuple[int, int], # (H_p, W_p) — visual token spatial layout
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            region_tokens: [B, R, out_dim]   — one feature per region
            region_valid:  [B, R]             — same as input mask_valid
        """
        B, N, C = visual_tokens.shape
        H_p, W_p = spatial_size
        assert H_p * W_p == N, f"spatial_size {spatial_size} != N={N}"

        # Downsample masks to patch grid
        R = masks.shape[1]
        masks_down = F.interpolate(
            masks.float().reshape(B * R, 1, masks.shape[-2], masks.shape[-1]),
            size=(H_p, W_p),
            mode="area",
        ).reshape(B, R, H_p * W_p)  # [B, R, N]

        # Mask-weighted pooling: sum(token * mask) / sum(mask)
        weights = masks_down.clamp(min=0)
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, R, 1]
        normed = weights / denom  # [B, R, N]

        # [B, R, N] x [B, N, C] → [B, R, C]
        pooled = torch.bmm(normed, visual_tokens)

        out = self.norm(self.proj(pooled))  # [B, R, out_dim]
        return out, mask_valid
