from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from sandbox_autoencoders.data import DEFAULT_DATASET, ImageSpec, collate_frames, load_full_dataset
from sandbox_autoencoders.model import ConvVAE
from sandbox_autoencoders.utils import choose_device, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate between two dataset frames in VAE latent space.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir")
    parser.add_argument("--index-a", type=int, required=True)
    parser.add_argument("--index-b", type=int, required=True)
    parser.add_argument("--steps", type=int, default=9, help="Number of decoded interpolation frames.")
    parser.add_argument("--output", default="outputs/interpolation.png")
    parser.add_argument("--device")
    return parser.parse_args()


def _load_model(checkpoint_path: str, device: torch.device) -> tuple[ConvVAE, ImageSpec]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_spec = ImageSpec(**checkpoint["image_spec"])
    model = ConvVAE(image_spec=image_spec, latent_dim=checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, image_spec


@torch.inference_mode()
def interpolate_tensors(
    model: ConvVAE,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    mu_a, _ = model.encode(image_a.to(device))
    mu_b, _ = model.encode(image_b.to(device))

    alphas = torch.linspace(0.0, 1.0, steps + 2, device=device)
    decoded = []
    for alpha in alphas:
        latent = torch.lerp(mu_a, mu_b, alpha)
        decoded.append(model.decode(latent).cpu())
    return torch.cat(decoded, dim=0)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, image_spec = _load_model(args.checkpoint, device)
    dataset = load_full_dataset(args.dataset, image_spec=image_spec, cache_dir=args.cache_dir)

    frame_a = collate_frames([dataset[args.index_a]])
    frame_b = collate_frames([dataset[args.index_b]])
    image_a = frame_a["image"].to(device)
    image_b = frame_b["image"].to(device)
    strip = interpolate_tensors(model=model, image_a=image_a, image_b=image_b, steps=args.steps, device=device)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    save_image(strip, output_path, nrow=strip.size(0))


if __name__ == "__main__":
    main()
