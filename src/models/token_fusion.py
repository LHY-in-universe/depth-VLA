"""
Conv1d-based fusion of two visual token streams:
  - DINOv3 patch tokens  [B, N_d, D]
  - Qwen visual tokens   [B, N_q, D]
Output: fused tokens     [B, N_q, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenFusion(nn.Module):
    def __init__(self, dim: int):
        """
        Args:
            dim: hidden dimension of both token streams (must match)
        """
        super().__init__()
        # Channel-wise 1×1 conv to mix the two streams
        self.conv = nn.Conv1d(dim * 2, dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(dim)
        # Learnable gate: how much DINOv3 contributes relative to Qwen
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        dino_tokens: torch.Tensor,
        qwen_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            dino_tokens: [B, N_d, D]
            qwen_tokens: [B, N_q, D]
        Returns:
            fused:       [B, N_q, D]
        """
        B, N_q, D = qwen_tokens.shape
        N_d = dino_tokens.shape[1]

        # Align DINOv3 sequence length to Qwen's
        if N_d != N_q:
            # Linear interpolation along the sequence dimension
            dino_tokens = F.interpolate(
                dino_tokens.permute(0, 2, 1),  # [B, D, N_d]
                size=N_q,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)  # [B, N_q, D]

        # Concat along channel dim, conv-fuse, then gated residual
        cat = torch.cat([dino_tokens, qwen_tokens], dim=-1)  # [B, N_q, 2D]
        fused = self.conv(cat.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N_q, D]

        gate = torch.sigmoid(self.gate)
        fused = gate * fused + (1 - gate) * qwen_tokens  # residual blend

        return self.norm(fused)
