from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from sandbox_autoencoders.local_video_data import (
    VIDEO_EXTENSIONS,
    VideoRecord,
    assign_split,
    discover_videos,
    probe_video,
    summarize_records,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan a local video directory and build a JSONL manifest.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", default="outputs/video-manifests/local-videos.jsonl")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extensions", nargs="+", default=list(VIDEO_EXTENSIONS))
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.video_dir).expanduser().resolve()
    print(f"scanning video_dir={root}")
    records: list[VideoRecord] = []
    failures = 0
    paths = discover_videos(root, extensions=args.extensions)
    if not paths:
        extension_list = ", ".join(args.extensions)
        raise RuntimeError(f"no video files found under {root} matching extensions: {extension_list}")
    with tqdm(total=len(paths), desc="probe", unit="video") as progress:
        progress.set_postfix(written=0, failed=0)
        for path in paths:
            try:
                probed = probe_video(path)
            except Exception as exc:
                failures += 1
                if args.fail_on_error:
                    raise
                tqdm.write(f"skip {path}: {exc}")
            else:
                relative_path = path.expanduser().resolve().relative_to(Path(root).expanduser().resolve()).as_posix()
                records.append(
                    VideoRecord(
                        video_id=relative_path,
                        path=probed.path,
                        relative_path=relative_path,
                        split=assign_split(
                            relative_path,
                            train_fraction=args.train_fraction,
                            val_fraction=args.val_fraction,
                            seed=args.seed,
                        ),
                        duration_seconds=probed.duration_seconds,
                        fps=probed.fps,
                        width=probed.width,
                        height=probed.height,
                        frame_count=probed.frame_count,
                        size_bytes=probed.size_bytes,
                    )
                )
            finally:
                progress.update(1)
                progress.set_postfix(written=len(records), failed=failures)

    output_path = write_manifest(records, args.output)
    summary = summarize_records(records)
    print(f"wrote manifest: {output_path}")
    print(f"videos_discovered={len(paths)} videos_written={len(records)} failures={failures}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
