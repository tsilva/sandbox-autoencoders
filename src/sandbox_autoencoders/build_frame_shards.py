from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sandbox_autoencoders.data import ImageSpec
from sandbox_autoencoders.frame_shards import (
    DEFAULT_TARGET_SHARD_SIZE_BYTES,
    build_frame_shards,
    summarize_frame_shards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite a video manifest into WebDataset-style frame shards.")
    parser.add_argument("--manifest", required=True, help="Path to the JSONL video manifest.")
    parser.add_argument("--output-dir", default="outputs/frame-shards/frames-256")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--image-format", choices=("webp", "png"), default="webp")
    parser.add_argument("--target-shard-size-mb", type=float, default=DEFAULT_TARGET_SHARD_SIZE_BYTES / (1024 * 1024))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory already exists and is not empty: {output_dir} (pass --overwrite to rebuild)")

    result = build_frame_shards(
        manifest_path=args.manifest,
        output_dir=output_dir,
        image_spec=ImageSpec(width=args.width, height=args.height),
        sample_fps=args.sample_fps,
        image_format=args.image_format,
        target_shard_size_bytes=max(1, round(args.target_shard_size_mb * 1024 * 1024)),
        splits=tuple(args.splits),
        fail_on_error=args.fail_on_error,
    )
    summary = summarize_frame_shards(result.shard_records)
    print(f"wrote shard manifest: {output_dir / 'shards.jsonl'}")
    print(
        f"videos_processed={result.videos_processed} "
        f"videos_failed={result.videos_failed} "
        f"samples_written={result.samples_written} "
        f"shards_written={len(result.shard_records)}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
