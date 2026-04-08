"""
DepthVLA — Qwen3.5-2B-only architecture with dual-pass visual encoding.

Pipeline (depth maps are precomputed offline by DINOv3 Depther in
scripts/preprocess_depth.py):

    RGB image   ──► Qwen visual encoder (LoRA) ──► rgb_tokens
    Depth PNG   ──► Qwen visual encoder (LoRA, shared weights) ──► depth_tokens
                                                      │
                                            Conv1d fusion
                                                      │
                                              fused_tokens
                                                      │
                  for each <mask> in prompt:
                      RegionPooler(fused_tokens, mask) → region_token
                                                      │
              fused_tokens replace <image> embeddings,
              region_tokens replace <mask> embeddings
                                                      │
                                          Qwen3.5-2B LLM (LoRA)
                                                      │
                                                   logits
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .region_pooler import RegionPooler
from .token_fusion import TokenFusion


class DepthVLA(nn.Module):
    def __init__(
        self,
        # Qwen base
        qwen_model_name: str,
        # Qwen visual LoRA (applied to vision encoder)
        visual_lora_rank: int,
        visual_lora_alpha: int,
        visual_lora_dropout: float,
        visual_lora_target_modules: list[str],
        # Qwen LLM LoRA (enabled at stage 3)
        llm_use_lora: bool,
        llm_lora_rank: int,
        llm_lora_alpha: int,
        llm_lora_dropout: float,
        llm_lora_target_modules: list[str],
        # Region pooling
        max_regions: int = 32,
    ):
        super().__init__()
        self.max_regions = max_regions

        # ── Qwen3.5-2B (loads visual encoder + LLM together) ──────────────
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # ── Inject LoRA into Qwen's visual encoder ────────────────────────
        # The same backbone is run twice per forward: RGB with LoRA disabled
        # (preserves pretrained semantics), depth with LoRA enabled (adapts
        # to the pseudo-depth modality). Toggling is done in forward() via
        # the _visual_lora_disabled() context manager.
        self._inject_visual_lora(
            visual_lora_rank, visual_lora_alpha,
            visual_lora_dropout, visual_lora_target_modules,
        )

        # Sanity check: confirm LoRA actually matched something
        from peft.tuners.lora import LoraLayer
        n_lora = sum(
            1 for m in self._find_visual_encoder().modules()
            if isinstance(m, LoraLayer)
        )
        print(f"[DepthVLA] injected {n_lora} LoRA layers into visual encoder")
        if n_lora == 0:
            raise RuntimeError(
                "Visual LoRA injection matched 0 layers. "
                "Check visual_lora.target_modules in configs/model.yaml."
            )

        # ── Resolve hidden dim from Qwen's visual encoder output ──────────
        self.hidden_dim = self._infer_visual_hidden_dim()

        # ── Token fusion (RGB visual + depth visual) ──────────────────────
        self.fusion = TokenFusion(dim=self.hidden_dim)

        # ── Region pooler ─────────────────────────────────────────────────
        self.region_pooler = RegionPooler(
            in_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
        )

        # ── Optional LLM LoRA (stage 3) ───────────────────────────────────
        if llm_use_lora:
            self._inject_llm_lora(
                llm_lora_rank, llm_lora_alpha,
                llm_lora_dropout, llm_lora_target_modules,
            )
        # Cache LLM LoRA params for enable_llm_lora()
        self._llm_lora_cfg = dict(
            rank=llm_lora_rank, alpha=llm_lora_alpha,
            dropout=llm_lora_dropout, target_modules=llm_lora_target_modules,
        )

        # Special token IDs (resolved lazily)
        self._image_token_id: Optional[int] = None
        self._mask_token_id:  Optional[int] = None

    # ──────────────────────────────────────────────────────────────────────
    # LoRA injection
    # ──────────────────────────────────────────────────────────────────────

    def _inject_visual_lora(self, rank, alpha, dropout, target_modules):
        """Apply LoRA to Qwen's visual encoder sub-module only."""
        from peft import LoraConfig, inject_adapter_in_model

        visual_enc = self._find_visual_encoder()
        if visual_enc is None:
            raise RuntimeError(
                "Could not locate Qwen visual encoder for LoRA injection. "
                "Inspect self.qwen children and update _find_visual_encoder()."
            )

        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        # In-place adapter injection (does not wrap with PeftModel)
        inject_adapter_in_model(cfg, visual_enc)

    def _inject_llm_lora(self, rank, alpha, dropout, target_modules):
        from peft import LoraConfig, inject_adapter_in_model
        cfg = LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none",
        )
        # Inject only into the LLM body (skip visual to avoid double-inject)
        llm_body = self._find_llm_body()
        inject_adapter_in_model(cfg, llm_body if llm_body is not None else self.qwen)

    def enable_llm_lora(self):
        """Call before stage 3 to add LoRA to the LLM body."""
        self._inject_llm_lora(**self._llm_lora_cfg)

    # ──────────────────────────────────────────────────────────────────────
    # LoRA toggle (RGB pass bypasses LoRA, depth pass uses it)
    # ──────────────────────────────────────────────────────────────────────

    @contextmanager
    def _visual_lora_disabled(self):
        """
        Context manager: disable all LoRA adapters in the visual encoder
        for the duration of the with-block. Used to run the RGB pass with
        Qwen's pretrained weights only, while the depth pass (outside the
        block) sees the LoRA-adapted backbone.
        """
        from peft.tuners.lora import LoraLayer
        ve = self._find_visual_encoder()
        toggled: list = []
        for m in ve.modules():
            if isinstance(m, LoraLayer):
                # Prefer the public API if available (newer PEFT)
                if hasattr(m, "enable_adapters"):
                    m.enable_adapters(False)
                else:
                    m._disable_adapters = True
                toggled.append(m)
        try:
            yield
        finally:
            for m in toggled:
                if hasattr(m, "enable_adapters"):
                    m.enable_adapters(True)
                else:
                    m._disable_adapters = False

    # ──────────────────────────────────────────────────────────────────────
    # Module discovery
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

    def _find_llm_body(self) -> Optional[nn.Module]:
        for name in ("model", "language_model", "transformer", "llm"):
            mod = getattr(self.qwen, name, None)
            if mod is not None:
                return mod
        return None

    def _infer_visual_hidden_dim(self) -> int:
        ve = self._find_visual_encoder()
        # Try common config attribute names
        for attr in ("hidden_size", "embed_dim", "out_hidden_size", "output_dim"):
            cfg = getattr(ve, "config", None)
            if cfg is not None and hasattr(cfg, attr):
                return getattr(cfg, attr)
        # Fall back to LLM hidden size (works because Qwen visual projects into it)
        return self.qwen.config.hidden_size

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def _encode_visual(self, pixel_values: torch.Tensor, **extras) -> torch.Tensor:
        """Run Qwen's visual encoder once. Returns [B, N, D]."""
        ve = self._find_visual_encoder()
        out = ve(pixel_values, **extras)
        # Output may be a tensor or have .last_hidden_state
        if isinstance(out, torch.Tensor):
            tokens = out
        elif hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            tokens = out[0]
        else:
            raise TypeError(f"Unexpected visual encoder output type: {type(out)}")

        # Ensure shape [B, N, D]
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        return tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rgb_pixel_values: torch.Tensor,            # Qwen-preprocessed RGB
        depth_pixel_values: torch.Tensor,          # Qwen-preprocessed depth (3-ch)
        masks: torch.Tensor,                       # [B, R, H, W] uint8
        mask_valid: torch.Tensor,                  # [B, R] bool
        labels: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        self._resolve_token_ids()

        # ── Step 1: Visual encoding ───────────────────────────────────────
        # RGB runs with LoRA DISABLED — uses Qwen's untouched pretrained
        # weights to preserve original semantic representation.
        # Depth runs with LoRA ENABLED — the adapter specialises the same
        # backbone for the 3-channel pseudo-depth modality.
        extras = {}
        if image_grid_thw is not None:
            extras["grid_thw"] = image_grid_thw

        with self._visual_lora_disabled():
            rgb_tokens = self._encode_visual(rgb_pixel_values, **extras)  # [B, N, D]
        depth_tokens = self._encode_visual(depth_pixel_values, **extras)  # [B, N, D]

        # ── Step 2: Conv fuse the two streams ─────────────────────────────
        fused = self.fusion(depth_tokens, rgb_tokens)  # [B, N, D]

        # ── Step 3: Region pooling for <mask> placeholders ────────────────
        # Infer spatial size from N (assume square)
        N = fused.shape[1]
        side = int(round(N ** 0.5))
        spatial_size = (side, side)
        region_tokens, _ = self.region_pooler(
            visual_tokens=fused,
            masks=masks,
            mask_valid=mask_valid,
            spatial_size=spatial_size,
        )

        # ── Step 4: Build inputs_embeds and inject ────────────────────────
        embed_layer = self.qwen.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        if self._image_token_id is not None:
            self._splice_at_token(
                inputs_embeds, input_ids,
                token_id=self._image_token_id,
                replacement=fused,
            )
        if self._mask_token_id is not None:
            self._splice_regions(
                inputs_embeds, input_ids,
                token_id=self._mask_token_id,
                region_tokens=region_tokens,
                mask_valid=mask_valid,
            )

        # ── Step 5: LLM forward ───────────────────────────────────────────
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {
            "loss":   outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Token-level injection helpers
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_token_ids(self):
        if self._image_token_id is not None:
            return
        cfg = self.qwen.config
        self._image_token_id = getattr(cfg, "image_token_id", None)
        self._mask_token_id  = getattr(cfg, "mask_token_id",  None)

    @staticmethod
    def _splice_at_token(inputs_embeds, input_ids, token_id, replacement):
        is_tok = input_ids == token_id
        for b in range(inputs_embeds.shape[0]):
            positions = is_tok[b].nonzero(as_tuple=False).squeeze(-1)
            n = min(positions.shape[0], replacement.shape[1])
            if n > 0:
                inputs_embeds[b, positions[:n], :] = replacement[b, :n].to(inputs_embeds.dtype)

    @staticmethod
    def _splice_regions(inputs_embeds, input_ids, token_id, region_tokens, mask_valid):
        is_tok = input_ids == token_id
        for b in range(inputs_embeds.shape[0]):
            positions = is_tok[b].nonzero(as_tuple=False).squeeze(-1)
            valid_idx = mask_valid[b].nonzero(as_tuple=False).squeeze(-1)
            n = min(positions.shape[0], valid_idx.shape[0])
            for k in range(n):
                inputs_embeds[b, positions[k], :] = (
                    region_tokens[b, valid_idx[k]].to(inputs_embeds.dtype)
                )

    # ──────────────────────────────────────────────────────────────────────
    # Staged training
    # ──────────────────────────────────────────────────────────────────────

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def set_trainable_stage(self, stage: int):
        """
        stage 1: fusion + region_pooler only (visual & LLM frozen)
        stage 2: + Qwen visual LoRA
        stage 3: + Qwen LLM LoRA
        """
        self.freeze_all()
        for p in self.fusion.parameters():
            p.requires_grad_(True)
        for p in self.region_pooler.parameters():
            p.requires_grad_(True)

        if stage >= 2:
            visual_enc = self._find_visual_encoder()
            for name, p in visual_enc.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)

        if stage >= 3:
            for name, p in self.qwen.named_parameters():
                if "lora_" in name and not self._param_in_visual(name):
                    p.requires_grad_(True)

    def _param_in_visual(self, qualified_name: str) -> bool:
        for prefix in ("visual.", "vision_model.", "vision_encoder.", "img_encoder."):
            if qualified_name.startswith(prefix):
                return True
        return False

    @classmethod
    def from_config(cls, model_cfg: dict) -> "DepthVLA":
        q = model_cfg["qwen"]
        v = model_cfg["visual_lora"]
        l = model_cfg["llm_lora"]
        return cls(
            qwen_model_name=q["model_name"],
            visual_lora_rank=v["rank"],
            visual_lora_alpha=v["alpha"],
            visual_lora_dropout=v["dropout"],
            visual_lora_target_modules=v["target_modules"],
            llm_use_lora=l["enabled_at_init"],
            llm_lora_rank=l["rank"],
            llm_lora_alpha=l["alpha"],
            llm_lora_dropout=l["dropout"],
            llm_lora_target_modules=l["target_modules"],
            max_regions=model_cfg.get("max_regions", 32),
        )
