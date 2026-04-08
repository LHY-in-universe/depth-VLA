"""
DepthVLA — SpatialRGPT-style architecture using DINOv3 + Qwen3.5-2B.

Inputs per sample:
    rgb_image       — original RGB
    depth_image     — 3-channel pseudo-RGB depth (precomputed offline by DINOv3 Depther)
    masks           — binary masks for each <mask> token in the conversation
    text            — conversation with <image> <depth> <mask> tokens

Forward pipeline:
    rgb   → DINOv3 (LoRA) → dino_rgb_tokens
    depth → DINOv3 (LoRA, shared) → dino_depth_tokens
    rgb   → Qwen visual encoder (hooked) → qwen_visual_tokens

    fusion(dino_rgb, dino_depth, qwen_visual) → fused_visual_tokens

    For each <mask> token in input_ids:
        region_token = RegionPooler(fused_visual_tokens, mask_i)
        replace embedding at that position with region_token

    For each <image>/<depth> placeholder:
        replace with the corresponding token sequence

    Run Qwen LLM (LoRA) → output logits
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .dino_encoder import DINOv3Encoder
from .region_pooler import RegionPooler
from .token_fusion import TokenFusion


class DepthVLA(nn.Module):
    def __init__(
        self,
        # DINOv3
        dino_model_name: str,
        dino_output_dim: int,
        dino_use_lora: bool,
        dino_lora_rank: int,
        dino_lora_alpha: int,
        dino_lora_dropout: float,
        # Qwen
        qwen_model_name: str,
        qwen_use_lora: bool,
        qwen_lora_rank: int,
        qwen_lora_alpha: int,
        qwen_lora_dropout: float,
        qwen_lora_target_modules: list[str],
        # Region pooling
        max_regions: int = 32,
    ):
        super().__init__()
        self.max_regions = max_regions

        # ── Shared DINOv3 encoder for both RGB and depth ──────────────────
        self.dino = DINOv3Encoder(
            model_name=dino_model_name,
            output_dim=dino_output_dim,
            use_lora=dino_use_lora,
            lora_rank=dino_lora_rank,
            lora_alpha=dino_lora_alpha,
            lora_dropout=dino_lora_dropout,
        )
        self._dino_patch_size = self.dino.backbone.config.patch_size

        # ── Three-way fusion: qwen + dino_rgb + dino_depth ────────────────
        # Implemented as two cascaded TokenFusions
        self.fusion_rgb_depth = TokenFusion(dim=dino_output_dim)   # dino_rgb + dino_depth
        self.fusion_with_qwen = TokenFusion(dim=dino_output_dim)   # + qwen_visual

        # ── Region pooler ─────────────────────────────────────────────────
        self.region_pooler = RegionPooler(
            in_dim=dino_output_dim,
            out_dim=dino_output_dim,
        )

        # ── Qwen3.5-2B base ───────────────────────────────────────────────
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if qwen_use_lora:
            self._inject_qwen_lora(
                qwen_lora_rank, qwen_lora_alpha,
                qwen_lora_dropout, qwen_lora_target_modules,
            )

        # Token IDs (resolved at first forward)
        self._image_token_id: Optional[int] = None
        self._depth_token_id: Optional[int] = None
        self._mask_token_id: Optional[int] = None

        # Hook state
        self._hook_handles: list = []
        self._captured_visual_tokens: Optional[torch.Tensor] = None

    # ──────────────────────────────────────────────────────────────────────
    # LoRA helpers
    # ──────────────────────────────────────────────────────────────────────

    def _inject_qwen_lora(self, rank, alpha, dropout, target_modules):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        )
        self.qwen = get_peft_model(self.qwen, cfg)

    def enable_qwen_lora(self, rank=16, alpha=32, dropout=0.05, target_modules=None):
        self._inject_qwen_lora(
            rank, alpha, dropout, target_modules or ["q_proj", "v_proj"]
        )

    # ──────────────────────────────────────────────────────────────────────
    # Visual token interception
    # ──────────────────────────────────────────────────────────────────────

    def _find_visual_encoder(self) -> Optional[nn.Module]:
        for name in ("visual", "vision_model", "vision_encoder", "img_encoder"):
            mod = getattr(self.qwen, name, None)
            if mod is not None:
                return mod
        for name, mod in self.qwen.named_children():
            if "visual" in name.lower() or "vision" in name.lower():
                return mod
        return None

    def _register_visual_hook(self):
        ve = self._find_visual_encoder()
        if ve is None:
            raise RuntimeError("Could not locate Qwen visual encoder.")

        def hook(_m, _inp, out):
            self._captured_visual_tokens = out[0] if isinstance(out, (tuple, list)) else out

        self._hook_handles.append(ve.register_forward_hook(hook))

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,                 # Qwen processed (RGB; depth handled via dino path)
        dino_rgb_pixel_values: torch.Tensor,
        dino_depth_pixel_values: torch.Tensor,
        masks: torch.Tensor,                        # [B, R, H, W] uint8
        mask_valid: torch.Tensor,                   # [B, R] bool
        labels: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # Resolve special token IDs lazily (cached)
        self._resolve_token_ids()

        # ── Step 1: DINOv3 encode RGB and depth ───────────────────────────
        dino_rgb_tokens, _   = self.dino(dino_rgb_pixel_values)    # [B, N_d, D]
        dino_depth_tokens, _ = self.dino(dino_depth_pixel_values)  # [B, N_d, D]

        # ── Step 2: Capture Qwen's visual tokens via hook ────────────────
        self._captured_visual_tokens = None
        self._register_visual_hook()
        with torch.no_grad():
            qwen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            if image_grid_thw is not None:
                qwen_kwargs["image_grid_thw"] = image_grid_thw
            _ = self.qwen(**qwen_kwargs)
        self._remove_hooks()
        qwen_visual = self._captured_visual_tokens  # [B, N_q, D] or None

        # ── Step 3: Three-way fusion ──────────────────────────────────────
        rd = self.fusion_rgb_depth(dino_rgb_tokens, dino_depth_tokens)  # [B, N_d, D]
        if qwen_visual is not None:
            fused = self.fusion_with_qwen(rd, qwen_visual)              # [B, N_q, D]
        else:
            fused = rd

        # ── Step 4: Region pooling for <mask> tokens ──────────────────────
        # Use the unfused dino_rgb_tokens (with depth blended) for region pooling
        H_p = W_p = dino_rgb_pixel_values.shape[-1] // self._dino_patch_size
        region_tokens, _ = self.region_pooler(
            visual_tokens=rd,             # [B, N_d, D]
            masks=masks,
            mask_valid=mask_valid,
            spatial_size=(H_p, W_p),
        )  # [B, R, D]

        # ── Step 5: Inject fused visual + region tokens into Qwen embeds ──
        outputs = self._qwen_forward_with_injection(
            fused_visual=fused,
            region_tokens=region_tokens,
            mask_valid=mask_valid,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            image_grid_thw=image_grid_thw,
        )

        return {
            "loss":   outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Embedding-level injection
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_token_ids(self):
        if self._image_token_id is not None:
            return
        # Pull from config; fall back to None
        cfg = self.qwen.config
        self._image_token_id = getattr(cfg, "image_token_id", None)
        self._depth_token_id = getattr(cfg, "depth_token_id", None)
        self._mask_token_id  = getattr(cfg, "mask_token_id",  None)

    def _qwen_forward_with_injection(
        self,
        fused_visual: torch.Tensor,
        region_tokens: torch.Tensor,
        mask_valid: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
    ):
        embed_layer = self.qwen.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # [B, L, D]

        # Replace <image>/<depth> token positions with fused visual tokens
        if self._image_token_id is not None:
            self._splice_at_token(
                inputs_embeds, input_ids,
                token_id=self._image_token_id,
                replacement=fused_visual,
            )

        # Replace <mask> token positions with region tokens (one per <mask>)
        if self._mask_token_id is not None:
            self._splice_regions(
                inputs_embeds, input_ids,
                token_id=self._mask_token_id,
                region_tokens=region_tokens,
                mask_valid=mask_valid,
            )

        kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw

        return self.qwen(**kwargs)

    @staticmethod
    def _splice_at_token(
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        replacement: torch.Tensor,
    ):
        """In-place: replace embeds at every input_ids==token_id position
        with consecutive entries from `replacement` ([B, N_repl, D])."""
        is_tok = input_ids == token_id
        for b in range(inputs_embeds.shape[0]):
            positions = is_tok[b].nonzero(as_tuple=False).squeeze(-1)
            n = min(positions.shape[0], replacement.shape[1])
            if n > 0:
                inputs_embeds[b, positions[:n], :] = replacement[b, :n].to(inputs_embeds.dtype)

    @staticmethod
    def _splice_regions(
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        region_tokens: torch.Tensor,   # [B, R_max, D]
        mask_valid: torch.Tensor,      # [B, R_max]
    ):
        """Replace each consecutive <mask> token with the next valid region token."""
        is_tok = input_ids == token_id
        for b in range(inputs_embeds.shape[0]):
            positions = is_tok[b].nonzero(as_tuple=False).squeeze(-1)
            valid_idx = mask_valid[b].nonzero(as_tuple=False).squeeze(-1)
            n = min(positions.shape[0], valid_idx.shape[0])
            for k in range(n):
                pos = positions[k]
                ridx = valid_idx[k]
                inputs_embeds[b, pos, :] = region_tokens[b, ridx].to(inputs_embeds.dtype)

    # ──────────────────────────────────────────────────────────────────────
    # Staged training
    # ──────────────────────────────────────────────────────────────────────

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def set_trainable_stage(self, stage: int):
        """
        stage 1: fusion + region_pooler + dino proj only
        stage 2: + DINOv3 LoRA
        stage 3: + Qwen LoRA
        """
        self.freeze_all()
        for module in (self.fusion_rgb_depth, self.fusion_with_qwen, self.region_pooler):
            for p in module.parameters():
                p.requires_grad_(True)
        for p in self.dino.proj.parameters():
            p.requires_grad_(True)
        for p in self.dino.norm.parameters():
            p.requires_grad_(True)

        if stage >= 2:
            for name, p in self.dino.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)

        if stage >= 3:
            for name, p in self.qwen.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)

    @classmethod
    def from_config(cls, model_cfg: dict) -> "DepthVLA":
        return cls(
            dino_model_name=model_cfg["dino"]["model_name"],
            dino_output_dim=model_cfg["dino"]["output_dim"],
            dino_use_lora=model_cfg["dino"]["use_lora"],
            dino_lora_rank=model_cfg["dino"]["lora_rank"],
            dino_lora_alpha=model_cfg["dino"]["lora_alpha"],
            dino_lora_dropout=model_cfg["dino"]["lora_dropout"],
            qwen_model_name=model_cfg["qwen"]["model_name"],
            qwen_use_lora=model_cfg["qwen"]["use_lora"],
            qwen_lora_rank=model_cfg["qwen"]["lora_rank"],
            qwen_lora_alpha=model_cfg["qwen"]["lora_alpha"],
            qwen_lora_dropout=model_cfg["qwen"]["lora_dropout"],
            qwen_lora_target_modules=model_cfg["qwen"]["lora_target_modules"],
            max_regions=model_cfg.get("max_regions", 32),
        )
