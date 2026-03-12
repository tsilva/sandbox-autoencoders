from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from sandbox_autoencoders.data import ImageSpec
from sandbox_autoencoders.local_video_data import (
    SampledVideoFrameDataset,
    load_manifest,
    summarize_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the local video sampler without decoding frames.")
    parser.add_argument("--manifest", required=True, help="Path to the JSONL manifest created by build_video_manifest.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to audit.")
    parser.add_argument("--samples", type=int, default=100_000, help="Number of samples to draw from the sampler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by the sampler.")
    parser.add_argument(
        "--sampling-weight",
        default="sqrt_duration",
        choices=["uniform", "duration", "sqrt_duration"],
        help="Video-level weighting policy.",
    )
    parser.add_argument("--video-burst-size", type=int, default=1, help="Number of consecutive samples drawn from one video.")
    parser.add_argument("--burst-span-seconds", type=float, default=0.0, help="Temporal span covered within each burst.")
    parser.add_argument(
        "--max-sequential-gap-frames",
        type=int,
        default=120,
        help="Unused by this audit, but accepted so the config matches the training loader.",
    )
    parser.add_argument("--bins", type=int, default=20, help="Number of normalized timestamp bins for coverage histograms.")
    parser.add_argument(
        "--top-videos",
        type=int,
        default=10,
        help="Number of largest per-video weighting deviations to print.",
    )
    parser.add_argument(
        "--min-video-samples",
        type=int,
        default=50,
        help="Minimum samples per video before including that video in per-video coverage stats.",
    )
    return parser.parse_args()


def _bin_index(value: float, bins: int) -> int:
    if bins <= 1:
        return 0
    clamped = min(max(value, 0.0), 1.0)
    if clamped >= 1.0:
        return bins - 1
    return int(clamped * bins)


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest, split=args.split)
    dataset = SampledVideoFrameDataset(
        records=records,
        image_spec=ImageSpec(),
        samples_per_epoch=args.samples,
        sampling_weight=args.sampling_weight,
        seed=args.seed,
        video_burst_size=args.video_burst_size,
        burst_span_seconds=args.burst_span_seconds,
        max_sequential_gap_frames=args.max_sequential_gap_frames,
    )

    sampled_video_counts: Counter[str] = Counter()
    sampled_split_counts: Counter[str] = Counter()
    normalized_hist = [0 for _ in range(args.bins)]
    per_video_hist: dict[str, list[int]] = defaultdict(lambda: [0 for _ in range(args.bins)])
    per_video_timestamp_counts: Counter[str] = Counter()

    burst_video_ids: list[str] = []
    burst_timestamps: list[float] = []
    burst_mismatch_count = 0
    burst_span_violation_count = 0

    for index in range(args.samples):
        record, timestamp = dataset._sample_candidate(index=index, attempt=0)
        sampled_video_counts[record.video_id] += 1
        sampled_split_counts[record.split] += 1

        if record.duration_seconds > 0:
            normalized_timestamp = timestamp / record.duration_seconds
        else:
            normalized_timestamp = 0.0
        bin_index = _bin_index(normalized_timestamp, args.bins)
        normalized_hist[bin_index] += 1
        per_video_hist[record.video_id][bin_index] += 1
        per_video_timestamp_counts[record.video_id] += 1

        burst_video_ids.append(record.video_id)
        burst_timestamps.append(timestamp)
        if len(burst_video_ids) == args.video_burst_size:
            if len(set(burst_video_ids)) != 1:
                burst_mismatch_count += 1
            if args.video_burst_size > 1 and args.burst_span_seconds > 0:
                observed_span = max(burst_timestamps) - min(burst_timestamps)
                allowed_span = args.burst_span_seconds + 1e-6
                if observed_span > allowed_span:
                    burst_span_violation_count += 1
            burst_video_ids.clear()
            burst_timestamps.clear()

    uniform_bin_probability = 1.0 / args.bins
    normalized_histogram = [count / args.samples for count in normalized_hist]
    global_abs_deviation = sum(abs(value - uniform_bin_probability) for value in normalized_histogram) / args.bins
    global_max_deviation = max(abs(value - uniform_bin_probability) for value in normalized_histogram)

    averaged_per_video_hist = [0.0 for _ in range(args.bins)]
    included_videos = 0
    for video_id, histogram in per_video_hist.items():
        total = per_video_timestamp_counts[video_id]
        if total < args.min_video_samples:
            continue
        included_videos += 1
        for idx, count in enumerate(histogram):
            averaged_per_video_hist[idx] += count / total
    if included_videos > 0:
        averaged_per_video_hist = [value / included_videos for value in averaged_per_video_hist]
    per_video_abs_deviation = (
        sum(abs(value - uniform_bin_probability) for value in averaged_per_video_hist) / args.bins
        if included_videos > 0
        else None
    )
    per_video_max_deviation = (
        max(abs(value - uniform_bin_probability) for value in averaged_per_video_hist)
        if included_videos > 0
        else None
    )

    expected_probabilities = {
        record.video_id: weight / dataset._total_weight
        for record, weight in zip(dataset.records, dataset._weights, strict=True)
    }
    video_deviation_rows = []
    for video_id, probability in expected_probabilities.items():
        observed_count = sampled_video_counts.get(video_id, 0)
        observed_probability = observed_count / args.samples
        expected_count = probability * args.samples
        relative_error = 0.0 if expected_count == 0 else (observed_count - expected_count) / expected_count
        video_deviation_rows.append(
            {
                "video_id": video_id,
                "expected_probability": probability,
                "observed_probability": observed_probability,
                "expected_count": expected_count,
                "observed_count": observed_count,
                "relative_error": relative_error,
            }
        )
    video_deviation_rows.sort(key=lambda row: abs(row["relative_error"]), reverse=True)

    result = {
        "config": {
            "manifest": args.manifest,
            "split": args.split,
            "samples": args.samples,
            "seed": args.seed,
            "sampling_weight": args.sampling_weight,
            "video_burst_size": args.video_burst_size,
            "burst_span_seconds": args.burst_span_seconds,
            "max_sequential_gap_frames": args.max_sequential_gap_frames,
            "bins": args.bins,
        },
        "manifest_summary": summarize_records(records),
        "split_audit": {
            "requested_split": args.split,
            "sampled_split_counts": dict(sampled_split_counts),
            "split_leaks": args.samples - sampled_split_counts.get(args.split, 0),
        },
        "video_weight_audit": {
            "videos_in_split": len(records),
            "sampled_videos": len(sampled_video_counts),
            "top_relative_errors": video_deviation_rows[: args.top_videos],
        },
        "timestamp_coverage": {
            "global_normalized_histogram": normalized_histogram,
            "global_abs_deviation_from_uniform": global_abs_deviation,
            "global_max_deviation_from_uniform": global_max_deviation,
            "average_per_video_histogram": averaged_per_video_hist if included_videos > 0 else None,
            "average_per_video_abs_deviation_from_uniform": per_video_abs_deviation,
            "average_per_video_max_deviation_from_uniform": per_video_max_deviation,
            "videos_used_for_per_video_histogram": included_videos,
        },
        "burst_audit": {
            "burst_size": args.video_burst_size,
            "bursts_checked": args.samples // max(args.video_burst_size, 1),
            "video_mismatch_bursts": burst_mismatch_count,
            "span_violation_bursts": burst_span_violation_count,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
