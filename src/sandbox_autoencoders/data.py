from __future__ import annotations

import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


DEFAULT_DATASET = "tsilva/zx-spectrum-worldoflongplays-te8t6fzk-2i"


@dataclass(frozen=True)
class ImageSpec:
    width: int = 256
    height: int = 192


def resize_image_with_padding(image: Image.Image, spec: ImageSpec) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(spec.width / width, spec.height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    image = TF.resize(image, [resized_height, resized_width], antialias=True)
    pad_left = (spec.width - resized_width) // 2
    pad_top = (spec.height - resized_height) // 2
    canvas = Image.new("RGB", (spec.width, spec.height), color=(0, 0, 0))
    canvas.paste(image, (pad_left, pad_top))
    return canvas


def resize_with_padding(image: Image.Image, spec: ImageSpec) -> torch.Tensor:
    return TF.to_tensor(resize_image_with_padding(image, spec))


def _default_cache_dir(dataset_name: str) -> Path:
    return Path(".cache") / dataset_name.replace("/", "__")


def _dataset_metadata_url(dataset_name: str) -> str:
    return f"https://huggingface.co/api/datasets/{dataset_name}"


def _dataset_file_url(dataset_name: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{filename}?download=1"


def _resolve_parquet_filename(dataset_name: str) -> str:
    response = requests.get(_dataset_metadata_url(dataset_name), timeout=30)
    response.raise_for_status()
    payload = response.json()
    for sibling in payload.get("siblings", []):
        filename = sibling["rfilename"]
        if filename.startswith("data/") and filename.endswith(".parquet"):
            return filename
    raise RuntimeError(f"no parquet file found for dataset {dataset_name}")


def _download_if_missing(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    temp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp_path.replace(destination)
    return destination


def load_frame_records(dataset_name: str, cache_dir: str | None = None) -> list[dict[str, object]]:
    cache_root = Path(cache_dir) if cache_dir else _default_cache_dir(dataset_name)
    parquet_name = _resolve_parquet_filename(dataset_name)
    local_path = cache_root / Path(parquet_name).name
    _download_if_missing(_dataset_file_url(dataset_name, parquet_name), local_path)
    frame_table = pd.read_parquet(local_path)
    return frame_table.to_dict(orient="records")


class HuggingFaceFrameDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, object]],
        image_spec: ImageSpec,
    ) -> None:
        self.rows = rows
        self.image_spec = image_spec

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image_data = row["image"]
        image = Image.open(BytesIO(image_data["bytes"])).convert("RGB")
        image = resize_with_padding(image, self.image_spec)
        return {
            "image": image,
            "frame_id": row["frame_id"],
            "timestamp": row["timestamp"],
            "video_title": row["video_title"],
            "index": index,
        }


def load_frame_splits(
    dataset_name: str,
    image_spec: ImageSpec,
    val_fraction: float = 0.1,
    seed: int = 42,
    cache_dir: str | None = None,
) -> tuple[HuggingFaceFrameDataset, HuggingFaceFrameDataset]:
    rows = load_frame_records(dataset_name=dataset_name, cache_dir=cache_dir)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    val_size = min(max(1, round(len(rows) * val_fraction)), len(rows) - 1)
    val_indices = set(indices[:val_size])
    train_rows = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val_rows = [row for idx, row in enumerate(rows) if idx in val_indices]
    return (
        HuggingFaceFrameDataset(train_rows, image_spec=image_spec),
        HuggingFaceFrameDataset(val_rows, image_spec=image_spec),
    )


def load_full_dataset(
    dataset_name: str,
    image_spec: ImageSpec,
    cache_dir: str | None = None,
) -> HuggingFaceFrameDataset:
    rows = load_frame_records(dataset_name=dataset_name, cache_dir=cache_dir)
    return HuggingFaceFrameDataset(rows, image_spec=image_spec)


def collate_frames(batch: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(batch)
    images = torch.stack([row["image"] for row in rows])
    return {
        "image": images,
        "frame_id": [row["frame_id"] for row in rows],
        "timestamp": [row["timestamp"] for row in rows],
        "video_title": [row["video_title"] for row in rows],
        "index": [row["index"] for row in rows],
    }
