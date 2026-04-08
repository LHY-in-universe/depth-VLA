"""
DINOv3 ViT-Base encoder with optional LoRA fine-tuning.
Extracts patch-level features and projects them to the target hidden dimension.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.num_register_tokens = self.backbone.config.num_register_tokens
        hidden_size = self.backbone.config.hidden_size

        if use_lora:
            # Import here to make LoRA optional at import time
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

        self.proj = nn.Linear(hidden_size, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W]
        Returns:
            patch_tokens: [B, N_patches, output_dim]
            cls_token:    [B, 1, hidden_size]  (raw, before proj)
        """
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state  # [B, 1 + num_reg + N, hidden]

        cls_token = last_hidden[:, 0:1, :]  # [B, 1, hidden]
        patch_tokens = last_hidden[:, 1 + self.num_register_tokens:, :]  # [B, N, hidden]

        patch_tokens = self.norm(self.proj(patch_tokens))  # [B, N, output_dim]
        return patch_tokens, cls_token

    def get_spatial_size(self, image_size: int, patch_size: int) -> tuple[int, int]:
        """Returns (H_patches, W_patches) for a square image."""
        n = image_size // patch_size
        return n, n
