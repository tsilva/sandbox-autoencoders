from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
import subprocess
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
from PIL import Image
from torch.utils.data import Dataset

from sandbox_autoencoders.data import ImageSpec, resize_with_padding


VIDEO_EXTENSIONS = (".mkv", ".mp4", ".mov", ".webm", ".avi")


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    path: str
    relative_path: str
    split: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_count: int
    size_bytes: int


@dataclass
class CaptureState:
    capture: cv2.VideoCapture
    next_frame_index: int | None = None


def discover_videos(video_dir: str | Path, extensions: Iterable[str] = VIDEO_EXTENSIONS) -> list[Path]:
    raw_root = Path(video_dir).expanduser()
    if not raw_root.exists():
        raise FileNotFoundError(f"video directory does not exist: {raw_root}")
    if not raw_root.is_dir():
        raise NotADirectoryError(f"video directory is not a directory: {raw_root}")

    root = raw_root.resolve()
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    matches: list[Path] = []
    for current_root, _, filenames in os.walk(root, followlinks=True):
        current_path = Path(current_root)
        for filename in filenames:
            path = current_path / filename
            if path.suffix.lower() in allowed:
                matches.append(path)
    return sorted(matches)


def probe_video(path: str | Path) -> VideoRecord:
    video_path = Path(path).expanduser().resolve()
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError(f"no video stream found in {video_path}")

    duration_seconds = _parse_float(payload.get("format", {}).get("duration"))
    fps = _parse_frame_rate(video_stream.get("avg_frame_rate")) or _parse_frame_rate(video_stream.get("r_frame_rate"))
    frame_count = _parse_int(video_stream.get("nb_frames"))
    if frame_count <= 0 and duration_seconds > 0 and fps > 0:
        frame_count = max(1, round(duration_seconds * fps))

    return VideoRecord(
        video_id=video_path.stem,
        path=str(video_path),
        relative_path=video_path.name,
        split="",
        duration_seconds=duration_seconds,
        fps=fps,
        width=_parse_int(video_stream.get("width")),
        height=_parse_int(video_stream.get("height")),
        frame_count=frame_count,
        size_bytes=_parse_int(payload.get("format", {}).get("size")),
    )


def build_manifest(
    video_dir: str | Path,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
    extensions: Iterable[str] = VIDEO_EXTENSIONS,
) -> list[VideoRecord]:
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction must be > 0 and train_fraction + val_fraction must be < 1")

    root = Path(video_dir).expanduser().resolve()
    records: list[VideoRecord] = []
    for path in discover_videos(root, extensions=extensions):
        record = probe_video(path)
        relative_path = path.relative_to(root).as_posix()
        video_id = relative_path
        split = assign_split(video_id, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)
        records.append(
            VideoRecord(
                video_id=video_id,
                path=record.path,
                relative_path=relative_path,
                split=split,
                duration_seconds=record.duration_seconds,
                fps=record.fps,
                width=record.width,
                height=record.height,
                frame_count=record.frame_count,
                size_bytes=record.size_bytes,
            )
        )
    return records


def write_manifest(records: Iterable[VideoRecord], output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return destination


def load_manifest(manifest_path: str | Path, split: str | None = None) -> list[VideoRecord]:
    manifest = Path(manifest_path).expanduser().resolve()
    records: list[VideoRecord] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            record = VideoRecord(**payload)
            if split is None or record.split == split:
                records.append(record)
    return records


def summarize_records(records: Iterable[VideoRecord]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for record in records:
        bucket = summary.setdefault(record.split, {"videos": 0, "duration_seconds": 0.0, "size_bytes": 0.0})
        bucket["videos"] += 1
        bucket["duration_seconds"] += record.duration_seconds
        bucket["size_bytes"] += record.size_bytes
    return summary


class SampledVideoFrameDataset(Dataset):
    def __init__(
        self,
        records: list[VideoRecord],
        image_spec: ImageSpec,
        samples_per_epoch: int,
        sampling_weight: str = "duration",
        seed: int = 42,
        video_burst_size: int = 1,
        burst_span_seconds: float = 0.0,
        max_open_captures: int = 8,
        max_decode_attempts: int = 3,
        max_sequential_gap_frames: int = 120,
    ) -> None:
        if not records:
            raise ValueError("at least one video record is required")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if video_burst_size <= 0:
            raise ValueError("video_burst_size must be positive")
        if burst_span_seconds < 0:
            raise ValueError("burst_span_seconds must be non-negative")
        if max_open_captures <= 0:
            raise ValueError("max_open_captures must be positive")
        if max_decode_attempts <= 0:
            raise ValueError("max_decode_attempts must be positive")
        if max_sequential_gap_frames < 0:
            raise ValueError("max_sequential_gap_frames must be non-negative")

        cv2.setNumThreads(1)

        self.records = records
        self.image_spec = image_spec
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.video_burst_size = video_burst_size
        self.burst_span_seconds = burst_span_seconds
        self.max_open_captures = max_open_captures
        self.max_decode_attempts = max_decode_attempts
        self.max_sequential_gap_frames = max_sequential_gap_frames
        self.sampling_weight = sampling_weight

        self._capture_cache: OrderedDict[str, CaptureState] = OrderedDict()
        self._weights = self._build_weights(records, sampling_weight)
        total_weight = sum(self._weights)
        if total_weight <= 0:
            raise ValueError("sampling weights must sum to a positive number")
        cumulative = []
        running_total = 0.0
        for weight in self._weights:
            running_total += weight
            cumulative.append(running_total)
        self._cumulative_weights = cumulative
        self._total_weight = total_weight

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        last_error: Exception | None = None
        for attempt in range(self.max_decode_attempts):
            record, timestamp = self._sample_candidate(index=index, attempt=attempt)
            try:
                image = self._decode_frame(record, timestamp)
            except Exception as exc:  # pragma: no cover - exercised by real videos
                last_error = exc
                continue
            return {
                "image": image,
                "frame_id": f"{record.video_id}:{timestamp:.3f}",
                "timestamp": timestamp,
                "video_title": record.video_id,
                "index": index,
            }
        raise RuntimeError(f"failed to decode frame after {self.max_decode_attempts} attempts") from last_error

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_capture_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        for state in self._capture_cache.values():
            state.capture.release()
        self._capture_cache.clear()

    def _sample_candidate(self, index: int, attempt: int) -> tuple[VideoRecord, float]:
        burst_index = index // self.video_burst_size
        offset_in_burst = index % self.video_burst_size
        rng = random.Random(self.seed + burst_index + attempt * 1_000_003)
        record = self._sample_record(rng)
        timestamp = self._sample_timestamp(
            record=record,
            rng=rng,
            offset_in_burst=offset_in_burst,
        )
        return record, timestamp

    def _sample_record(self, rng: random.Random) -> VideoRecord:
        needle = rng.random() * self._total_weight
        index = bisect.bisect_left(self._cumulative_weights, needle)
        return self.records[min(index, len(self.records) - 1)]

    def _sample_timestamp(
        self,
        record: VideoRecord,
        rng: random.Random,
        offset_in_burst: int,
    ) -> float:
        if record.duration_seconds <= 0:
            return 0.0
        max_timestamp = max(0.0, record.duration_seconds - 1e-3)
        if self.video_burst_size <= 1 or self.burst_span_seconds <= 0:
            return rng.uniform(0.0, max_timestamp)

        span = min(self.burst_span_seconds, max_timestamp)
        if span <= 0:
            return rng.uniform(0.0, max_timestamp)

        start = rng.uniform(0.0, max(0.0, max_timestamp - span))
        if self.video_burst_size == 1:
            return start
        step = span / max(1, self.video_burst_size - 1)
        timestamp = start + offset_in_burst * step
        return min(max(timestamp, 0.0), max_timestamp)

    def _decode_frame(self, record: VideoRecord, timestamp: float):
        state = self._get_capture_state(record.path)
        frame_index = self._resolve_frame_index(record, timestamp)
        ok, frame = self._read_frame(state=state, frame_index=frame_index, timestamp=timestamp)
        if not ok or frame is None:
            state.capture.release()
            self._capture_cache.pop(record.path, None)
            state = self._get_capture_state(record.path)
            ok, frame = self._read_frame(state=state, frame_index=frame_index, timestamp=timestamp)
        if not ok or frame is None:
            raise RuntimeError(f"failed to decode frame from {record.path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        return resize_with_padding(image, self.image_spec)

    def _resolve_frame_index(self, record: VideoRecord, timestamp: float) -> int:
        if record.frame_count > 0 and record.duration_seconds > 0:
            return min(record.frame_count - 1, max(0, round(timestamp / record.duration_seconds * (record.frame_count - 1))))
        if record.fps > 0:
            return max(0, round(timestamp * record.fps))
        return 0

    def _read_frame(
        self,
        state: CaptureState,
        frame_index: int,
        timestamp: float,
    ) -> tuple[bool, object]:
        if state.next_frame_index is not None:
            gap = frame_index - state.next_frame_index
            if 0 <= gap <= self.max_sequential_gap_frames:
                for _ in range(gap):
                    ok, _ = state.capture.read()
                    if not ok:
                        break
                else:
                    ok, frame = state.capture.read()
                    if ok and frame is not None:
                        state.next_frame_index = frame_index + 1
                        return ok, frame

        if frame_index > 0:
            state.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            state.capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)

        ok, frame = state.capture.read()
        if ok and frame is not None:
            state.next_frame_index = frame_index + 1
        else:
            state.next_frame_index = None
        return ok, frame

    def _get_capture_state(self, path: str) -> CaptureState:
        state = self._capture_cache.pop(path, None)
        if state is None:
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise RuntimeError(f"failed to open video {path}")
            state = CaptureState(capture=capture)
        self._capture_cache[path] = state
        while len(self._capture_cache) > self.max_open_captures:
            _, stale_state = self._capture_cache.popitem(last=False)
            stale_state.capture.release()
        return state

    @staticmethod
    def _build_weights(records: list[VideoRecord], sampling_weight: str) -> list[float]:
        if sampling_weight == "uniform":
            return [1.0 for _ in records]
        if sampling_weight == "duration":
            return [max(record.duration_seconds, 1.0) for record in records]
        if sampling_weight == "sqrt_duration":
            return [max(math.sqrt(max(record.duration_seconds, 1.0)), 1.0) for record in records]
        raise ValueError(f"unsupported sampling_weight: {sampling_weight}")


def assign_split(video_id: str, train_fraction: float, val_fraction: float, seed: int = 42) -> str:
    bucket = _stable_bucket(video_id, seed=seed)
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + val_fraction:
        return "val"
    return "test"


def _stable_bucket(value: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).digest()
    numerator = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return numerator / float(1 << 64)


def _parse_frame_rate(value: object) -> float:
    if value in (None, "", "0/0"):
        return 0.0
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value
    return _parse_float(text)


def _parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
