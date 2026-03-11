from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from sandbox_autoencoders.data import (
    DEFAULT_DATASET,
    ImageSpec,
    collate_frames,
    load_frame_splits,
)
from sandbox_autoencoders.model import ConvVAE, vae_loss
from sandbox_autoencoders.utils import choose_device, ensure_dir, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a convolutional VAE on ZX Spectrum frames.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir", default="outputs/default-run")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--device")
    return parser.parse_args()


def run_epoch(
    model: ConvVAE,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    beta: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
    batch_count = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        with torch.set_grad_enabled(is_training):
            output = model(images)
            loss, metrics = vae_loss(output.reconstruction, images, output.mu, output.logvar, beta)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        for key, value in metrics.items():
            totals[key] += value
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    return {key: value / max(1, batch_count) for key, value in totals.items()}


@torch.inference_mode()
def save_reconstruction_preview(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    destination: Path,
) -> None:
    batch = next(iter(loader))
    images = batch["image"][:8].to(device)
    reconstruction = model(images).reconstruction
    preview = torch.cat([images, reconstruction], dim=0)
    save_image(preview.cpu(), destination, nrow=8)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    image_spec = ImageSpec(width=args.width, height=args.height)

    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    recon_dir = ensure_dir(output_dir / "reconstructions")

    train_dataset, val_dataset = load_frame_splits(
        dataset_name=args.dataset,
        image_spec=image_spec,
        val_fraction=args.val_fraction,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_frames,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_frames,
    )

    model = ConvVAE(image_spec=image_spec, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.beta,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            args.beta,
            max_batches=args.max_val_batches,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_recon_loss": train_metrics["recon_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "val_loss": val_metrics["loss"],
            "val_recon_loss": val_metrics["recon_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
        }
        history.append(record)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "image_spec": {"width": image_spec.width, "height": image_spec.height},
            "latent_dim": args.latent_dim,
            "epoch": epoch,
            "args": vars(args),
            "metrics": record,
        }
        epoch_path = checkpoints_dir / f"epoch-{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, checkpoints_dir / "best.pt")

        save_reconstruction_preview(model, val_loader, device, recon_dir / f"epoch-{epoch:03d}.png")
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_recon={val_metrics['recon_loss']:.4f} "
            f"val_kl={val_metrics['kl_loss']:.4f}"
        )

    write_json(output_dir / "history.json", history)


if __name__ == "__main__":
    main()
