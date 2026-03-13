from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

from sandbox_autoencoders.data import ImageSpec
from sandbox_autoencoders.local_video_data import SampledVideoFrameDataset, load_manifest
from sandbox_autoencoders.model import ConvVAE
from sandbox_autoencoders.utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render latent interpolations from a trained ZX Spectrum VAE checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint.")
    parser.add_argument("--manifest", help="Override manifest path. Defaults to the one saved in the checkpoint args.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Manifest split to sample from.")
    parser.add_argument("--rows", type=int, default=4, help="Number of interpolation rows.")
    parser.add_argument("--steps", type=int, default=8, help="Number of decoded interpolation steps, including endpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for pair sampling.")
    parser.add_argument("--device", help="Torch device override.")
    return parser.parse_args()


def _build_dataset(checkpoint: dict[str, object], manifest_path: str, split: str) -> SampledVideoFrameDataset:
    image_spec = ImageSpec(**checkpoint["image_spec"])
    train_args = checkpoint["args"]
    records = load_manifest(manifest_path, split=split)
    samples_per_epoch = max(int(train_args.get("val_samples", 4096)), 512)
    if split == "train":
        samples_per_epoch = max(int(train_args.get("train_samples_per_epoch", 100000)), 512)
    video_burst_size = int(train_args.get("val_video_burst_size", 1))
    burst_span_seconds = float(train_args.get("val_burst_span_seconds", 0.0))
    if split == "train":
        video_burst_size = int(train_args.get("video_burst_size", 1))
        burst_span_seconds = float(train_args.get("burst_span_seconds", 0.0))
    dataset = SampledVideoFrameDataset(
        records=records,
        image_spec=image_spec,
        samples_per_epoch=samples_per_epoch,
        sampling_weight=str(train_args.get("sampling_weight", "sqrt_duration")),
        seed=int(train_args.get("seed", 42)),
        video_burst_size=video_burst_size,
        burst_span_seconds=burst_span_seconds,
        max_open_captures=int(train_args.get("max_open_captures", 8)),
        max_decode_attempts=int(train_args.get("max_decode_attempts", 4)),
        max_sequential_gap_frames=int(train_args.get("max_sequential_gap_frames", 120)),
    )
    return dataset


def _sample_pairs(length: int, rows: int, seed: int) -> list[tuple[int, int]]:
    if length < 2:
        raise ValueError("dataset must have at least two samples to interpolate")
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    for _ in range(rows):
        a = rng.randrange(length)
        b = rng.randrange(length - 1)
        if b >= a:
            b += 1
        pairs.append((a, b))
    return pairs


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest_path = args.manifest or checkpoint["args"]["manifest"]
    image_spec = ImageSpec(**checkpoint["image_spec"])

    device = choose_device(args.device)
    model = ConvVAE(image_spec=image_spec, latent_dim=int(checkpoint["latent_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = _build_dataset(checkpoint, manifest_path, split=args.split)
    pairs = _sample_pairs(len(dataset), args.rows, args.seed)

    grid_rows: list[torch.Tensor] = []
    alphas = torch.linspace(0.0, 1.0, steps=max(2, args.steps), device=device)
    for first_index, second_index in pairs:
        first = dataset[first_index]["image"].unsqueeze(0).to(device)
        second = dataset[second_index]["image"].unsqueeze(0).to(device)
        mu_a, _ = model.encode(first)
        mu_b, _ = model.encode(second)

        decoded_steps = []
        for alpha in alphas:
            latent = torch.lerp(mu_a, mu_b, alpha)
            decoded_steps.append(model.decode(latent).squeeze(0).cpu())

        row = torch.stack(
            [first.squeeze(0).cpu(), *decoded_steps, second.squeeze(0).cpu()],
            dim=0,
        )
        grid_rows.append(row)

    grid = torch.cat(grid_rows, dim=0)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path, nrow=args.steps + 2)
    dataset.close()
    print(output_path)


if __name__ == "__main__":
    main()
