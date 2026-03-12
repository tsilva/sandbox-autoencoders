from __future__ import annotations

import argparse
import json
import time

import cv2
from torch.utils.data import DataLoader

from sandbox_autoencoders.data import ImageSpec, collate_frames
from sandbox_autoencoders.local_video_data import SampledVideoFrameDataset, load_manifest, summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dataloader throughput against a local video manifest.")
    parser.add_argument("--manifest", required=True)
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
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    return parser.parse_args()


def _init_worker(_: int) -> None:
    cv2.setNumThreads(1)


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest, split=args.split)
    if not records:
        raise RuntimeError(f"no records found in split={args.split!r} for manifest={args.manifest}")
    summary = summarize_records(records)
    print(json.dumps(summary, indent=2, sort_keys=True))

    dataset = SampledVideoFrameDataset(
        records=records,
        image_spec=ImageSpec(width=args.width, height=args.height),
        samples_per_epoch=args.samples,
        sampling_weight=args.sampling_weight,
        seed=args.seed,
        video_burst_size=args.video_burst_size,
        burst_span_seconds=args.burst_span_seconds,
        max_open_captures=args.max_open_captures,
        max_decode_attempts=args.max_decode_attempts,
        max_sequential_gap_frames=args.max_sequential_gap_frames,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_frames,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        worker_init_fn=_init_worker if args.num_workers > 0 else None,
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
        "videos": len(records),
        "video_burst_size": args.video_burst_size,
        "burst_span_seconds": args.burst_span_seconds,
        "samples": total_samples,
        "batches": total_batches,
        "measured_samples": measured_samples,
        "measured_batches": measured_batches,
        "measured_seconds": measured_seconds,
        "samples_per_second": measured_samples / measured_seconds if measured_seconds > 0 else 0.0,
        "batches_per_second": measured_batches / measured_seconds if measured_seconds > 0 else 0.0,
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
