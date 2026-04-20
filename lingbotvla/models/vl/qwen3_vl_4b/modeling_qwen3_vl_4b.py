from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lingbotvla.models.vla.vision_models.align_heads.depth_head import DepthHead
from lingbotvla.models.vla.vision_models.module_utils import build_depth_model


try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except ImportError as exc:  # pragma: no cover - depends on local transformers version
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_qwen3_vl() -> None:
    if Qwen3VLForConditionalGeneration is None or AutoProcessor is None:
        raise ImportError(
            "Qwen3-VL support is not available in the installed transformers package. "
            "Install a recent transformers build that provides `Qwen3VLForConditionalGeneration`."
        ) from _IMPORT_ERROR


def _resolve_torch_dtype(torch_dtype: str | None):
    if torch_dtype in (None, "auto"):
        return torch_dtype

    if not hasattr(torch, torch_dtype):
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

    return getattr(torch, torch_dtype)


def build_qwen3_vl_4b_model(
    model_name_or_path: str = "Qwen/Qwen3-VL-4B-Instruct",
    *,
    attn_implementation: str = "flash_attention_2",
    torch_dtype: str | None = "bfloat16",
    gradient_checkpointing: bool = False,
    trust_remote_code: bool = True,
) -> Tuple["Qwen3VLForConditionalGeneration", "AutoProcessor"]:
    _require_qwen3_vl()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=_resolve_torch_dtype(torch_dtype),
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, processor


@dataclass
class Qwen3VLDepthConfig:
    moge_path: str
    morgbd_path: str
    depth_loss_weight: float = 0.1
    model_type: str = "MoRGBD"
    num_layers: int = 1
    num_heads: int = 4
    dim_head: int = 32
    ff_mult: int = 1
    num_backbone_tokens: int = 256
    token_size: int = 16
    dim_out: int = 1024
    input_size: int = 224


class DepthAwareQwen3VL4B(nn.Module):
    def __init__(
        self,
        base_model: "Qwen3VLForConditionalGeneration",
        depth_config: Qwen3VLDepthConfig,
    ) -> None:
        super().__init__()
        self.model = base_model
        self.depth_config = depth_config
        self.depth_loss_weight = depth_config.depth_loss_weight
        self.depth_model = build_depth_model(
            {
                "depth": {
                    "moge_path": depth_config.moge_path,
                    "morgbd_path": depth_config.morgbd_path,
                }
            }
        )
        self.depth_head = DepthHead(
            proj_config={
                "dim_head": depth_config.dim_head,
                "dim_out": depth_config.dim_out,
                "num_layers": depth_config.num_layers,
                "num_heads": depth_config.num_heads,
                "num_backbone_tokens": depth_config.num_backbone_tokens,
                "ff_mult": depth_config.ff_mult,
            },
            llm_hidden_size=self.model.config.hidden_size,
        ).to(dtype=self.model.dtype)
        self._freeze_base_model()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _freeze_base_model(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def trainable_parameter_summary(self) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return total, trainable

    def _image_token_id(self) -> int:
        for attr in ("image_token_id", "vision_token_id"):
            value = getattr(self.model.config, attr, None)
            if value is not None:
                return value
        raise AttributeError("Cannot find `image_token_id` or `vision_token_id` on the Qwen3-VL config.")

    @torch.no_grad()
    def _compute_depth_target(self, depth_images: torch.Tensor) -> torch.Tensor:
        if depth_images.ndim != 5 or depth_images.shape[1] != 1:
            raise ValueError(f"`depth_images` must have shape [batch, 1, 3, H, W], got {tuple(depth_images.shape)}")

        images = depth_images.to(device=self.device, dtype=torch.float32) / 255.0
        flat_images = images[:, 0]
        moge_model, morgbd_model = self.depth_model
        output_moge = moge_model.infer(flat_images, resolution_level=3, num_tokens=256, apply_mask=False)
        depth_pred = output_moge["depth"].squeeze().detach().clone()
        depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)
        depth_target, _ = morgbd_model.infer_feat(
            flat_images,
            depth_pred,
            depth_down_scale=1,
            resolution_level=3,
            num_tokens=256,
            enable_depth_mask=False,
        )
        depth_target = depth_target.permute(0, 2, 3, 1).contiguous()
        depth_target = depth_target.view(depth_target.shape[0], -1, depth_target.shape[-1])
        return depth_target.to(dtype=self.model.dtype)

    def _compute_depth_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        depth_images: torch.Tensor | None,
    ) -> torch.Tensor:
        if depth_images is None:
            return hidden_states.new_zeros(())

        image_token_id = self._image_token_id()
        depth_targets = self._compute_depth_target(depth_images)
        losses = []

        for batch_idx in range(hidden_states.shape[0]):
            image_mask = input_ids[batch_idx] == image_token_id
            image_embs = hidden_states[batch_idx][image_mask]
            if image_embs.numel() == 0:
                continue

            depth_preds = self.depth_head(image_embs.unsqueeze(0).to(dtype=self.model.dtype)).float()
            depth_target = depth_targets[batch_idx : batch_idx + 1].to(dtype=depth_preds.dtype)
            losses.append(F.smooth_l1_loss(depth_preds, depth_target, reduction="mean"))

        if not losses:
            return hidden_states.new_zeros(())

        return torch.stack(losses).mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        depth_images: torch.Tensor | None = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        text_loss = outputs.loss
        depth_loss = self._compute_depth_loss(outputs.hidden_states[-1], input_ids, depth_images)
        loss = self.depth_loss_weight * depth_loss

        outputs.loss = loss
        outputs.text_loss = text_loss.detach()
        outputs.depth_loss = depth_loss.detach()
        return outputs


def build_depth_aware_qwen3_vl_4b_model(
    model_name_or_path: str = "Qwen/Qwen3-VL-4B-Instruct",
    *,
    attn_implementation: str = "flash_attention_2",
    torch_dtype: str | None = "bfloat16",
    gradient_checkpointing: bool = False,
    trust_remote_code: bool = True,
    depth_config: Qwen3VLDepthConfig,
) -> Tuple["DepthAwareQwen3VL4B", "AutoProcessor"]:
    base_model, processor = build_qwen3_vl_4b_model(
        model_name_or_path=model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        gradient_checkpointing=gradient_checkpointing,
        trust_remote_code=trust_remote_code,
    )
    model = DepthAwareQwen3VL4B(base_model=base_model, depth_config=depth_config)
    return model, processor
