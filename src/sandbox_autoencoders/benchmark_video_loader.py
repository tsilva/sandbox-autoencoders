from __future__ import annotations

import argparse
import json
import time

import cv2
from torch.utils.data import DataLoader

from sandbox_autoencoders.data import ImageSpec, collate_frames
from sandbox_autoencoders.frame_shards import (
    DEFAULT_CACHE_SIZE_BYTES,
    FrameShardDataset,
    load_frame_shard_manifest,
    resolve_frame_shard_urls,
    summarize_frame_shards,
)
from sandbox_autoencoders.local_video_data import SampledVideoFrameDataset, load_manifest, summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dataloader throughput for raw-video or frame-shard datasets.")
    parser.add_argument("--manifest", help="Raw video manifest for OpenCV sampling.")
    parser.add_argument("--shard-manifest", help="Frame shard manifest written by build_frame_shards.")
    parser.add_argument("--shard-root", help="Optional local path or object-store root that contains the shards.")
    parser.add_argument("--shards", nargs="+", help="Explicit shard paths or URLs.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-weight", default="duration", choices=("uniform", "duration", "sqrt_duration"))
    parser.add_argument("--video-burst-size", type=int, default=1)
    parser.add_argument("--burst-span-seconds", type=float, default=0.0)
    parser.add_argument("--max-open-captures", type=int, default=8)
    parser.add_argument("--max-decode-attempts", type=int, default=3)
    parser.add_argument("--max-sequential-gap-frames", type=int, default=120)
    parser.add_argument("--cache-dir", help="Local cache directory for remote frame shards.")
    parser.add_argument("--cache-size-gb", type=float, default=DEFAULT_CACHE_SIZE_BYTES / (1024**3))
    parser.add_argument("--sample-shuffle-buffer", type=int, default=2048)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    args = parser.parse_args()
    if sum((bool(args.manifest), bool(args.shard_manifest), bool(args.shards))) != 1:
        parser.error("choose exactly one data source: --manifest, --shard-manifest, or --shards")
    return args


def _init_worker(_: int) -> None:
    cv2.setNumThreads(1)


def _build_dataset(args: argparse.Namespace):
    image_spec = ImageSpec(width=args.width, height=args.height)
    if args.manifest:
        records = load_manifest(args.manifest, split=args.split)
        if not records:
            raise RuntimeError(f"no records found in split={args.split!r} for manifest={args.manifest}")
        summary = summarize_records(records)
        dataset = SampledVideoFrameDataset(
            records=records,
            image_spec=image_spec,
            samples_per_epoch=args.samples,
            sampling_weight=args.sampling_weight,
            seed=args.seed,
            video_burst_size=args.video_burst_size,
            burst_span_seconds=args.burst_span_seconds,
            max_open_captures=args.max_open_captures,
            max_decode_attempts=args.max_decode_attempts,
            max_sequential_gap_frames=args.max_sequential_gap_frames,
        )
        result_meta = {
            "backend": "raw_video",
            "videos": len(records),
            "video_burst_size": args.video_burst_size,
            "burst_span_seconds": args.burst_span_seconds,
        }
        return dataset, summary, result_meta, _init_worker

    if args.shard_manifest:
        shard_records = load_frame_shard_manifest(args.shard_manifest, split=args.split)
        if not shard_records:
            raise RuntimeError(f"no shard records found in split={args.split!r} for manifest={args.shard_manifest}")
        summary = summarize_frame_shards(load_frame_shard_manifest(args.shard_manifest))
        shard_urls = resolve_frame_shard_urls(
            shard_manifest=args.shard_manifest,
            shard_root=args.shard_root,
            split=args.split,
        )
    else:
        shard_urls = resolve_frame_shard_urls(shards=args.shards)
        summary = {args.split: {"shards": len(shard_urls), "samples": 0, "size_bytes": 0.0}}
    dataset = FrameShardDataset(
        shard_urls=shard_urls,
        image_spec=image_spec,
        samples_per_epoch=args.samples,
        seed=args.seed,
        cache_dir=args.cache_dir,
        cache_size_bytes=max(1, round(args.cache_size_gb * 1024**3)),
        shuffle_shards=True,
        shuffle_buffer_size=args.sample_shuffle_buffer,
    )
    result_meta = {
        "backend": "frame_shards",
        "shards": len(shard_urls),
        "cache_dir": args.cache_dir or "",
        "sample_shuffle_buffer": args.sample_shuffle_buffer,
    }
    return dataset, summary, result_meta, None


def main() -> None:
    args = parse_args()
    dataset, summary, result_meta, worker_init_fn = _build_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_frames,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    total_batches = 0
    total_samples = 0
    measured_batches = 0
    measured_samples = 0
    measured_seconds = 0.0

    last_time = time.perf_counter()
    try:
        for batch in loader:
            now = time.perf_counter()
            batch_seconds = now - last_time
            batch_samples = int(batch["image"].shape[0])
            total_batches += 1
            total_samples += batch_samples

            if total_batches > args.warmup_batches:
                measured_batches += 1
                measured_samples += batch_samples
                measured_seconds += batch_seconds

            if total_batches % args.report_every == 0:
                instantaneous = batch_samples / batch_seconds if batch_seconds > 0 else 0.0
                average = measured_samples / measured_seconds if measured_seconds > 0 else 0.0
                print(
                    f"batch={total_batches} "
                    f"samples={total_samples} "
                    f"batch_time={batch_seconds:.4f}s "
                    f"instant_samples_per_sec={instantaneous:.2f} "
                    f"avg_samples_per_sec={average:.2f}"
                )
            last_time = time.perf_counter()
    finally:
        dataset.close()

    results = {
        "split": args.split,
        "samples": total_samples,
        "batches": total_batches,
        "measured_samples": measured_samples,
        "measured_batches": measured_batches,
        "measured_seconds": measured_seconds,
        "samples_per_second": measured_samples / measured_seconds if measured_seconds > 0 else 0.0,
        "batches_per_second": measured_batches / measured_seconds if measured_seconds > 0 else 0.0,
    }
    results.update(result_meta)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
