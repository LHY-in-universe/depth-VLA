from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor, resize


def _load_json_records(data_path: str) -> List[Dict[str, Any]]:
    extension = os.path.splitext(data_path)[1].lower()
    if extension == ".json":
        with open(data_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return payload
        raise ValueError("JSON training file must contain a list of records.")

    if extension == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    raise ValueError(f"Unsupported dataset file extension: {extension}")


@dataclass
class Qwen3VL4BSample:
    image_path: str
    prompt: str
    response: str


class Qwen3VL4BTrainDataset(Dataset):
    """Minimal image-text SFT dataset for Qwen3-VL-4B.

    Supported record shapes:
    1. {"image": "...", "prompt": "...", "response": "..."}
    2. {"images": ["..."], "prompt": "...", "response": "..."}  # first image is used
    """

    def __init__(
        self,
        data_path: str,
        *,
        image_root: str | None = None,
        image_column: str = "image",
        images_column: str = "images",
        prompt_column: str = "prompt",
        response_column: str = "response",
    ) -> None:
        self.records = _load_json_records(data_path)
        self.image_root = image_root
        self.image_column = image_column
        self.images_column = images_column
        self.prompt_column = prompt_column
        self.response_column = response_column

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, record: Dict[str, Any]) -> str:
        image_path = record.get(self.image_column)
        if image_path is None:
            images = record.get(self.images_column)
            if isinstance(images, Sequence) and images:
                image_path = images[0]

        if not image_path:
            raise ValueError(f"Record does not contain `{self.image_column}` or a non-empty `{self.images_column}`.")

        if self.image_root and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)

        return image_path

    def __getitem__(self, index: int) -> Qwen3VL4BSample:
        record = self.records[index]
        prompt = record.get(self.prompt_column)
        response = record.get(self.response_column)

        if prompt is None or response is None:
            raise ValueError(
                f"Each record must contain `{self.prompt_column}` and `{self.response_column}`. Offending index: {index}"
            )

        return Qwen3VL4BSample(
            image_path=self._resolve_image_path(record),
            prompt=prompt,
            response=response,
        )


class Qwen3VL4BTrainCollator:
    def __init__(self, processor, *, max_length: int = 2048, depth_image_size: int = 224) -> None:
        self.processor = processor
        self.max_length = max_length
        self.depth_image_size = depth_image_size

    @staticmethod
    def _load_image(image_path: str) -> Image.Image:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def _prepare_depth_image(self, image: Image.Image) -> torch.Tensor:
        resized = resize(image, [self.depth_image_size, self.depth_image_size])
        return pil_to_tensor(resized).float()

    @staticmethod
    def _build_user_messages(prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _build_full_messages(self, prompt: str, response: str) -> List[Dict[str, Any]]:
        return self._build_user_messages(prompt) + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            }
        ]

    def __call__(self, features: Sequence[Qwen3VL4BSample]) -> Dict[str, torch.Tensor]:
        images = [self._load_image(feature.image_path) for feature in features]
        depth_images = torch.stack([self._prepare_depth_image(image) for image in images], dim=0).unsqueeze(1)

        prompt_texts = [
            self.processor.apply_chat_template(
                self._build_user_messages(feature.prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for feature in features
        ]
        full_texts = [
            self.processor.apply_chat_template(
                self._build_full_messages(feature.prompt, feature.response),
                tokenize=False,
                add_generation_prompt=False,
            )
            for feature in features
        ]

        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        for idx, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[idx, :prompt_length] = -100

        batch["labels"] = labels
        batch["depth_images"] = depth_images
        return batch
