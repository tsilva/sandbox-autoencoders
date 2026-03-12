from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from sandbox_autoencoders.data import (
    ImageSpec,
    collate_frames,
)
from sandbox_autoencoders.frame_shards import (
    DEFAULT_CACHE_SIZE_BYTES,
    FrameShardDataset,
    load_frame_shard_manifest,
    resolve_frame_shard_urls,
    summarize_frame_shards,
)
from sandbox_autoencoders.local_video_data import SampledVideoFrameDataset, load_manifest, summarize_records
from sandbox_autoencoders.model import ConvVAE, vae_loss
from sandbox_autoencoders.utils import choose_device, ensure_dir, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a convolutional VAE on ZX Spectrum frames.")
    parser.add_argument("--manifest", help="Raw video manifest for on-demand OpenCV sampling.")
    parser.add_argument("--shard-manifest", help="Frame shard manifest written by build_frame_shards.")
    parser.add_argument("--shard-root", help="Optional local path or object-store root that contains the shards.")
    parser.add_argument("--train-shards", nargs="+", help="Explicit train shard paths or URLs.")
    parser.add_argument("--val-shards", nargs="+", help="Explicit val shard paths or URLs.")
    parser.add_argument("--output-dir", default="outputs/default-run")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--beta-warmup-epochs", type=int, default=10)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=2.0)
    parser.add_argument("--train-samples-per-epoch", type=int, default=100_000)
    parser.add_argument("--val-samples", type=int, default=4_096)
    parser.add_argument("--sampling-weight", default="sqrt_duration", choices=["uniform", "duration", "sqrt_duration"])
    parser.add_argument("--video-burst-size", type=int, default=16)
    parser.add_argument("--burst-span-seconds", type=float, default=1.5)
    parser.add_argument("--val-video-burst-size", type=int, default=1)
    parser.add_argument("--val-burst-span-seconds", type=float, default=0.0)
    parser.add_argument("--max-open-captures", type=int, default=8)
    parser.add_argument("--max-decode-attempts", type=int, default=4)
    parser.add_argument("--max-sequential-gap-frames", type=int, default=120)
    parser.add_argument("--cache-dir", help="Local cache directory for remote frame shards.")
    parser.add_argument("--cache-size-gb", type=float, default=DEFAULT_CACHE_SIZE_BYTES / (1024**3))
    parser.add_argument("--train-shuffle-buffer", type=int, default=2048)
    parser.add_argument("--val-shuffle-buffer", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--device")
    parser.add_argument("--resume-from")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-upload-every", type=int, default=1)
    parser.add_argument("--wandb-log-every", type=int, default=25)
    args = parser.parse_args()
    using_raw_manifest = bool(args.manifest)
    using_shard_manifest = bool(args.shard_manifest)
    using_explicit_shards = bool(args.train_shards or args.val_shards)
    if sum((using_raw_manifest, using_shard_manifest, using_explicit_shards)) != 1:
        parser.error("choose exactly one data source: --manifest, --shard-manifest, or --train-shards/--val-shards")
    if using_explicit_shards and (not args.train_shards or not args.val_shards):
        parser.error("--train-shards and --val-shards must be provided together")
    return args


def run_epoch(
    model: ConvVAE,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    beta: float,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
    wandb_run=None,
    wandb_log_every: int | None = None,
    step_offset: int = 0,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "recon_l1": 0.0,
        "recon_mse": 0.0,
        "kl_loss": 0.0,
        "mu_abs": 0.0,
        "logvar_abs": 0.0,
    }
    batch_count = 0

    for batch in tqdm(loader, leave=False):
        batch_start = time.perf_counter()
        images = batch["image"].to(device)
        with torch.set_grad_enabled(is_training):
            output = model(images)
            loss, metrics = vae_loss(output.reconstruction, images, output.mu, output.logvar, beta)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite VAE loss encountered")
            metrics["mu_abs"] = output.mu.abs().mean().item()
            metrics["logvar_abs"] = output.logvar.abs().mean().item()
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
        for key, value in metrics.items():
            totals[key] += value
        batch_count += 1
        if is_training and wandb_run is not None and wandb_log_every and batch_count % wandb_log_every == 0:
            batch_seconds = time.perf_counter() - batch_start
            wandb_run.log(
                {
                    "train_batch/loss": metrics["loss"],
                    "train_batch/recon_loss": metrics["recon_loss"],
                    "train_batch/recon_l1": metrics["recon_l1"],
                    "train_batch/recon_mse": metrics["recon_mse"],
                    "train_batch/kl_loss": metrics["kl_loss"],
                    "train_batch/mu_abs": metrics["mu_abs"],
                    "train_batch/logvar_abs": metrics["logvar_abs"],
                    "train_batch/beta": beta,
                    "train_batch/samples_per_second": images.size(0) / max(batch_seconds, 1e-6),
                },
                step=step_offset + batch_count,
            )
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


def _collect_existing_records(checkpoints_dir: Path) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    for checkpoint_path in sorted(checkpoints_dir.glob("epoch-*.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics")
        if metrics:
            records.append(metrics)
    records.sort(key=lambda row: int(row["epoch"]))
    return records


def _compute_early_stop_state(
    records: list[dict[str, float | int]],
    min_delta: float,
) -> tuple[float, int]:
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for record in records:
        val_loss = float(record["val_loss"])
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    return best_val_loss, epochs_without_improvement


def _init_wandb(args: argparse.Namespace):
    if not args.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
    )


def _init_video_worker(_: int) -> None:
    import cv2

    cv2.setNumThreads(1)


def _build_datasets(args: argparse.Namespace, image_spec: ImageSpec):
    if args.manifest:
        train_records = load_manifest(args.manifest, split="train")
        val_records = load_manifest(args.manifest, split="val")
        summary = summarize_records(train_records + val_records)
        train_dataset = SampledVideoFrameDataset(
            records=train_records,
            image_spec=image_spec,
            samples_per_epoch=args.train_samples_per_epoch,
            sampling_weight=args.sampling_weight,
            seed=args.seed,
            video_burst_size=args.video_burst_size,
            burst_span_seconds=args.burst_span_seconds,
            max_open_captures=args.max_open_captures,
            max_decode_attempts=args.max_decode_attempts,
            max_sequential_gap_frames=args.max_sequential_gap_frames,
        )
        val_dataset = SampledVideoFrameDataset(
            records=val_records,
            image_spec=image_spec,
            samples_per_epoch=args.val_samples,
            sampling_weight=args.sampling_weight,
            seed=args.seed + 1,
            video_burst_size=args.val_video_burst_size,
            burst_span_seconds=args.val_burst_span_seconds,
            max_open_captures=args.max_open_captures,
            max_decode_attempts=args.max_decode_attempts,
            max_sequential_gap_frames=args.max_sequential_gap_frames,
        )
        return train_dataset, val_dataset, summary, _init_video_worker

    cache_size_bytes = max(1, round(args.cache_size_gb * 1024**3))
    if args.shard_manifest:
        shard_records = load_frame_shard_manifest(args.shard_manifest)
        summary = summarize_frame_shards(shard_records)
        train_urls = resolve_frame_shard_urls(
            shard_manifest=args.shard_manifest,
            shard_root=args.shard_root,
            split="train",
        )
        val_urls = resolve_frame_shard_urls(
            shard_manifest=args.shard_manifest,
            shard_root=args.shard_root,
            split="val",
        )
    else:
        train_urls = resolve_frame_shard_urls(shards=args.train_shards)
        val_urls = resolve_frame_shard_urls(shards=args.val_shards)
        summary = {
            "train": {"shards": len(train_urls), "samples": 0, "size_bytes": 0.0},
            "val": {"shards": len(val_urls), "samples": 0, "size_bytes": 0.0},
        }

    train_dataset = FrameShardDataset(
        shard_urls=train_urls,
        image_spec=image_spec,
        samples_per_epoch=args.train_samples_per_epoch,
        seed=args.seed,
        cache_dir=args.cache_dir,
        cache_size_bytes=cache_size_bytes,
        shuffle_shards=True,
        shuffle_buffer_size=args.train_shuffle_buffer,
    )
    val_dataset = FrameShardDataset(
        shard_urls=val_urls,
        image_spec=image_spec,
        samples_per_epoch=args.val_samples,
        seed=args.seed + 1,
        cache_dir=args.cache_dir,
        cache_size_bytes=cache_size_bytes,
        shuffle_shards=False,
        shuffle_buffer_size=args.val_shuffle_buffer,
    )
    return train_dataset, val_dataset, summary, None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    image_spec = ImageSpec(width=args.width, height=args.height)

    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    recon_dir = ensure_dir(output_dir / "reconstructions")
    wandb_run = _init_wandb(args)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_frames,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers

    train_dataset, val_dataset, dataset_summary, worker_init_fn = _build_datasets(args, image_spec)
    print(json.dumps(dataset_summary, indent=2, sort_keys=True))
    if args.num_workers > 0 and worker_init_fn is not None:
        loader_kwargs["worker_init_fn"] = worker_init_fn
    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    model = ConvVAE(image_spec=image_spec, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    history = _collect_existing_records(checkpoints_dir)
    if history:
        write_json(output_dir / "history.json", history)
    best_val_loss, epochs_without_improvement = _compute_early_stop_state(
        history,
        min_delta=args.early_stopping_min_delta,
    )
    start_epoch = 1

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(checkpoint["epoch"]) + 1
        print(
            f"resuming from epoch={checkpoint['epoch']} "
            f"best_val_loss={best_val_loss:.4f} "
            f"early_stop_wait={epochs_without_improvement}/{args.early_stopping_patience}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch - 1)
        beta_scale = min(1.0, epoch / max(1, args.beta_warmup_epochs))
        current_beta = args.beta * beta_scale
        train_step_offset = (epoch - 1) * (args.max_train_batches or len(train_loader))
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            current_beta,
            grad_clip_norm=args.grad_clip_norm,
            max_batches=args.max_train_batches,
            wandb_run=wandb_run,
            wandb_log_every=args.wandb_log_every,
            step_offset=train_step_offset,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            current_beta,
            max_batches=args.max_val_batches,
        )
        record = {
            "epoch": epoch,
            "beta": current_beta,
            "train_loss": train_metrics["loss"],
            "train_recon_loss": train_metrics["recon_loss"],
            "train_recon_l1": train_metrics["recon_l1"],
            "train_recon_mse": train_metrics["recon_mse"],
            "train_kl_loss": train_metrics["kl_loss"],
            "train_mu_abs": train_metrics["mu_abs"],
            "train_logvar_abs": train_metrics["logvar_abs"],
            "val_loss": val_metrics["loss"],
            "val_recon_loss": val_metrics["recon_loss"],
            "val_recon_l1": val_metrics["recon_l1"],
            "val_recon_mse": val_metrics["recon_mse"],
            "val_kl_loss": val_metrics["kl_loss"],
            "val_mu_abs": val_metrics["mu_abs"],
            "val_logvar_abs": val_metrics["logvar_abs"],
        }
        history.append(record)
        write_json(output_dir / "history.json", history)
        if wandb_run is not None:
            wandb_run.log(record, step=epoch)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "image_spec": {"width": image_spec.width, "height": image_spec.height},
            "latent_dim": args.latent_dim,
            "epoch": epoch,
            "args": vars(args),
            "metrics": record,
        }
        epoch_path = checkpoints_dir / f"epoch-{epoch:03d}.pt"
        last_path = checkpoints_dir / "last.pt"
        torch.save(checkpoint, epoch_path)
        torch.save(checkpoint, last_path)
        best_updated = False
        if val_metrics["loss"] < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(checkpoint, checkpoints_dir / "best.pt")
            best_updated = True
        else:
            epochs_without_improvement += 1

        preview_path = recon_dir / f"epoch-{epoch:03d}.png"
        save_reconstruction_preview(model, val_loader, device, preview_path)
        if wandb_run is not None:
            import wandb

            wandb_run.log({"reconstruction_preview": wandb.Image(str(preview_path))}, step=epoch)
            if epoch % max(1, args.wandb_upload_every) == 0:
                checkpoint_artifact = wandb.Artifact(
                    name=f"{wandb_run.id}-checkpoints",
                    type="model",
                    metadata={"epoch": epoch},
                )
                checkpoint_artifact.add_file(str(epoch_path), name=f"checkpoints/{epoch_path.name}")
                checkpoint_artifact.add_file(str(last_path), name="checkpoints/last.pt")
                if best_updated:
                    checkpoint_artifact.add_file(str(checkpoints_dir / "best.pt"), name="checkpoints/best.pt")
                wandb_run.log_artifact(checkpoint_artifact)
        print(
            f"epoch={epoch} "
            f"beta={current_beta:.6f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_recon_l1={val_metrics['recon_l1']:.6f} "
            f"val_recon_mse={val_metrics['recon_mse']:.6f} "
            f"val_recon={val_metrics['recon_loss']:.4f} "
            f"val_kl={val_metrics['kl_loss']:.4f} "
            f"val_mu_abs={val_metrics['mu_abs']:.4f} "
            f"val_logvar_abs={val_metrics['logvar_abs']:.4f} "
            f"early_stop_wait={epochs_without_improvement}/{args.early_stopping_patience}"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"early stopping at epoch={epoch} "
                f"best_val_loss={best_val_loss:.4f} "
                f"patience={args.early_stopping_patience}"
            )
            break
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
