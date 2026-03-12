from __future__ import annotations

import functools
import hashlib
import http.server
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from sandbox_autoencoders.data import ImageSpec, collate_frames
from sandbox_autoencoders.frame_shards import (
    FrameShardDataset,
    build_frame_shards,
    load_frame_shard_manifest,
    resolve_frame_shard_urls,
)
from sandbox_autoencoders.local_video_data import VideoRecord, write_manifest


class FrameShardsTest(unittest.TestCase):
    def test_build_frame_shards_is_deterministic(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self._write_manifest(root)

            output_a = root / "out-a"
            output_b = root / "out-b"
            result_a = build_frame_shards(
                manifest_path=manifest_path,
                output_dir=output_a,
                image_spec=ImageSpec(width=32, height=24),
                sample_fps=2.0,
                image_format="png",
            )
            result_b = build_frame_shards(
                manifest_path=manifest_path,
                output_dir=output_b,
                image_spec=ImageSpec(width=32, height=24),
                sample_fps=2.0,
                image_format="png",
            )

            self.assertEqual(result_a.videos_failed, 0)
            self.assertEqual(result_b.videos_failed, 0)
            self.assertEqual((output_a / "shards.jsonl").read_text(), (output_b / "shards.jsonl").read_text())

            shard_hashes_a = self._hash_tree(output_a)
            shard_hashes_b = self._hash_tree(output_b)
            self.assertEqual(shard_hashes_a, shard_hashes_b)

    def test_frame_shard_dataset_decodes_expected_samples(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self._write_manifest(root)
            output_dir = root / "out"
            build_frame_shards(
                manifest_path=manifest_path,
                output_dir=output_dir,
                image_spec=ImageSpec(width=32, height=24),
                sample_fps=2.0,
                image_format="png",
            )

            shard_urls = resolve_frame_shard_urls(
                shard_manifest=output_dir / "shards.jsonl",
                split="train",
            )
            dataset = FrameShardDataset(
                shard_urls=shard_urls,
                image_spec=ImageSpec(width=32, height=24),
                samples_per_epoch=2,
                seed=7,
                shuffle_shards=False,
                shuffle_buffer_size=0,
            )
            samples = list(dataset)

            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["frame_id"], "train.avi:0:0")
            self.assertEqual(samples[1]["frame_id"], "train.avi:2:500")
            self.assertEqual(tuple(samples[0]["image"].shape), (3, 24, 32))
            first_center = (samples[0]["image"][:, 12, 16] * 255).round().to(dtype=torch.int32).tolist()
            second_center = (samples[1]["image"][:, 12, 16] * 255).round().to(dtype=torch.int32).tolist()
            self.assertGreater(first_center[0], first_center[1])
            self.assertGreater(first_center[0], first_center[2])
            self.assertGreater(second_center[2], second_center[0])
            self.assertGreater(second_center[2], second_center[1])

    def test_remote_shards_can_be_reused_from_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self._write_manifest(root)
            output_dir = root / "out"
            build_frame_shards(
                manifest_path=manifest_path,
                output_dir=output_dir,
                image_spec=ImageSpec(width=32, height=24),
                sample_fps=2.0,
                image_format="png",
            )
            shard_record = load_frame_shard_manifest(output_dir / "shards.jsonl", split="train")[0]
            cache_dir = root / "cache"

            server, thread = self._start_http_server(output_dir)
            try:
                shard_url = f"http://127.0.0.1:{server.server_port}/{shard_record.relative_path}"
                warm_dataset = FrameShardDataset(
                    shard_urls=[shard_url],
                    image_spec=ImageSpec(width=32, height=24),
                    samples_per_epoch=2,
                    seed=7,
                    cache_dir=str(cache_dir),
                    shuffle_shards=False,
                    shuffle_buffer_size=0,
                )
                warm_samples = list(warm_dataset)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertTrue(any(cache_dir.iterdir()))
            cached_dataset = FrameShardDataset(
                shard_urls=[shard_url],
                image_spec=ImageSpec(width=32, height=24),
                samples_per_epoch=2,
                seed=7,
                cache_dir=str(cache_dir),
                shuffle_shards=False,
                shuffle_buffer_size=0,
            )
            cached_samples = list(cached_dataset)
            self.assertEqual([sample["frame_id"] for sample in warm_samples], [sample["frame_id"] for sample in cached_samples])

    def test_samples_per_epoch_with_more_workers_than_shards(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self._write_manifest(root)
            output_dir = root / "out"
            build_frame_shards(
                manifest_path=manifest_path,
                output_dir=output_dir,
                image_spec=ImageSpec(width=32, height=24),
                sample_fps=2.0,
                image_format="png",
            )
            shard_urls = resolve_frame_shard_urls(
                shard_manifest=output_dir / "shards.jsonl",
                split="train",
            )
            dataset = FrameShardDataset(
                shard_urls=shard_urls,
                image_spec=ImageSpec(width=32, height=24),
                samples_per_epoch=4,
                seed=7,
                shuffle_shards=False,
                shuffle_buffer_size=0,
            )
            loader = DataLoader(
                dataset,
                batch_size=2,
                num_workers=2,
                persistent_workers=True,
                collate_fn=collate_frames,
            )
            total_samples = 0
            for batch in loader:
                total_samples += int(batch["image"].shape[0])
            self.assertEqual(total_samples, 4)

    def _write_manifest(self, root: Path) -> Path:
        train_path = root / "train.avi"
        val_path = root / "val.avi"
        self._write_video(
            train_path,
            colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
        )
        self._write_video(
            val_path,
            colors=[(255, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0)],
        )
        records = [
            self._video_record(train_path, split="train"),
            self._video_record(val_path, split="val"),
        ]
        return write_manifest(records, root / "videos.jsonl")

    def _video_record(self, path: Path, *, split: str) -> VideoRecord:
        return VideoRecord(
            video_id=path.name,
            path=str(path.resolve()),
            relative_path=path.name,
            split=split,
            duration_seconds=1.0,
            fps=4.0,
            width=32,
            height=24,
            frame_count=4,
            size_bytes=path.stat().st_size,
        )

    def _write_video(self, path: Path, *, colors: list[tuple[int, int, int]]) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (32, 24))
        self.assertTrue(writer.isOpened(), msg=f"failed to open video writer for {path}")
        try:
            for color in colors:
                frame = np.full((24, 32, 3), color[::-1], dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def _hash_tree(self, root: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def _start_http_server(self, directory: Path):
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A003
                return None

        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            functools.partial(QuietHandler, directory=str(directory)),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread


if __name__ == "__main__":
    unittest.main()
