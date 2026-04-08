"""
Batch collator for SpatialRGPTDataset.
Handles fixed-length text tensors + variable-shape masks.
"""

from __future__ import annotations

from typing import Any

import torch


class DepthVLACollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = batch[0].keys()

        for key in keys:
            values = [item[key] for item in batch]

            if key == "masks":
                # Pad masks to the largest H×W in the batch
                out[key] = self._pad_masks(values)
            elif isinstance(values[0], torch.Tensor):
                try:
                    out[key] = torch.stack(values, dim=0)
                except RuntimeError:
                    # Variable size — fall back to a list
                    out[key] = values
            else:
                out[key] = values

        return out

    @staticmethod
    def _pad_masks(masks_list: list[torch.Tensor]) -> torch.Tensor:
        """masks: list of [R, H_i, W_i] uint8 → [B, R, H_max, W_max]"""
        B = len(masks_list)
        R = masks_list[0].shape[0]
        H_max = max(m.shape[-2] for m in masks_list)
        W_max = max(m.shape[-1] for m in masks_list)

        out = torch.zeros(B, R, H_max, W_max, dtype=torch.uint8)
        for i, m in enumerate(masks_list):
            h, w = m.shape[-2], m.shape[-1]
            out[i, :, :h, :w] = m
        return out
