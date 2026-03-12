from __future__ import annotations

import contextlib
import fcntl
import glob
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import fsspec
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import functional as TF

from sandbox_autoencoders.data import ImageSpec, resize_image_with_padding, resize_with_padding
from sandbox_autoencoders.local_video_data import VideoRecord, load_manifest, resolve_frame_index
from sandbox_autoencoders.utils import ensure_dir


IMAGE_EXTENSIONS = (".png", ".webp")
DEFAULT_TARGET_SHARD_SIZE_BYTES = 512 * 1024 * 1024
DEFAULT_CACHE_SIZE_BYTES = 30 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class FrameShardRecord:
    split: str
    relative_path: str
    sample_count: int
    size_bytes: int
    first_frame_id: str
    last_frame_id: str


@dataclass(frozen=True)
class BuildFrameShardsResult:
    shard_records: list[FrameShardRecord]
    videos_processed: int
    videos_failed: int
    samples_written: int


def load_frame_shard_manifest(manifest_path: str | Path, split: str | None = None) -> list[FrameShardRecord]:
    manifest = Path(manifest_path).expanduser().resolve()
    records: list[FrameShardRecord] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            record = FrameShardRecord(**payload)
            if split is None or record.split == split:
                records.append(record)
    return records


def write_frame_shard_manifest(records: list[FrameShardRecord], manifest_path: str | Path) -> Path:
    destination = Path(manifest_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return destination


def summarize_frame_shards(records: list[FrameShardRecord]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for record in records:
        bucket = summary.setdefault(record.split, {"shards": 0, "samples": 0, "size_bytes": 0.0})
        bucket["shards"] += 1
        bucket["samples"] += record.sample_count
        bucket["size_bytes"] += record.size_bytes
    return summary


def resolve_frame_shard_urls(
    *,
    shard_manifest: str | Path | None = None,
    shard_root: str | None = None,
    split: str | None = None,
    shards: list[str] | None = None,
) -> list[str]:
    if shard_manifest and shards:
        raise ValueError("pass either shard_manifest or shards, not both")
    if shard_manifest:
        records = load_frame_shard_manifest(shard_manifest, split=split)
        if not records:
            raise RuntimeError(f"no shard records found for split={split!r} in manifest={shard_manifest}")
        root = shard_root or str(Path(shard_manifest).expanduser().resolve().parent)
        return [_join_root(root, record.relative_path) for record in records]
    if shards:
        urls = _expand_local_patterns(shards)
        if not urls:
            raise RuntimeError("no shard files matched the provided shard patterns")
        return urls
    raise ValueError("either shard_manifest or shards is required")


def build_frame_shards(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    image_spec: ImageSpec,
    sample_fps: float,
    image_format: str = "webp",
    target_shard_size_bytes: int = DEFAULT_TARGET_SHARD_SIZE_BYTES,
    splits: tuple[str, ...] = ("train", "val", "test"),
    fail_on_error: bool = False,
) -> BuildFrameShardsResult:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if target_shard_size_bytes <= 0:
        raise ValueError("target_shard_size_bytes must be positive")

    output_root = Path(output_dir).expanduser().resolve()
    records = [record for record in load_manifest(manifest_path) if record.split in set(splits)]
    writers = {
        split: _ShardWriter(
            output_root=output_root,
            split=split,
            image_format=image_format,
            target_shard_size_bytes=target_shard_size_bytes,
        )
        for split in splits
    }

    videos_processed = 0
    videos_failed = 0
    samples_written = 0
    for record in sorted(records, key=lambda row: (row.split, row.relative_path)):
        writer = writers[record.split]
        try:
            samples_written += _write_video_record(
                writer=writer,
                record=record,
                image_spec=image_spec,
                sample_fps=sample_fps,
            )
        except Exception:
            videos_failed += 1
            if fail_on_error:
                for shard_writer in writers.values():
                    shard_writer.close()
                raise
        else:
            videos_processed += 1

    shard_records: list[FrameShardRecord] = []
    for writer in writers.values():
        shard_records.extend(writer.close())
    write_frame_shard_manifest(shard_records, output_root / "shards.jsonl")
    return BuildFrameShardsResult(
        shard_records=shard_records,
        videos_processed=videos_processed,
        videos_failed=videos_failed,
        samples_written=samples_written,
    )


class FrameShardDataset(IterableDataset):
    def __init__(
        self,
        *,
        shard_urls: list[str],
        image_spec: ImageSpec,
        samples_per_epoch: int | None = None,
        seed: int = 42,
        cache_dir: str | None = None,
        cache_size_bytes: int = DEFAULT_CACHE_SIZE_BYTES,
        shuffle_shards: bool = True,
        shuffle_buffer_size: int = 2048,
    ) -> None:
        if not shard_urls:
            raise ValueError("at least one shard URL is required")
        if samples_per_epoch is not None and samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if cache_size_bytes <= 0:
            raise ValueError("cache_size_bytes must be positive")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be non-negative")

        super().__init__()
        self.shard_urls = list(shard_urls)
        self.image_spec = image_spec
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer_size = shuffle_buffer_size
        self._epoch = mp.get_context("spawn").Value("q", 0)
        self._cache = _ShardCache(cache_dir=cache_dir, cache_size_bytes=cache_size_bytes)

    def __len__(self) -> int:
        if self.samples_per_epoch is None:
            raise TypeError("FrameShardDataset length is only defined when samples_per_epoch is set")
        return self.samples_per_epoch

    def close(self) -> None:
        return None

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch.value = epoch

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        epoch = int(self._epoch.value)
        rng = random.Random(self.seed + epoch * 10_000_019 + worker_id)
        shard_urls = list(self.shard_urls)
        if self.shuffle_shards:
            rng.shuffle(shard_urls)
        active_workers = max(1, min(num_workers, len(shard_urls)))
        if worker_id >= active_workers:
            return
        target_samples = _split_worker_limit(self.samples_per_epoch, worker_id, active_workers)
        if target_samples == 0:
            return
        assigned_urls = shard_urls[worker_id::active_workers]
        if not assigned_urls:
            return

        yielded = 0
        sample_index = 0
        repeat = target_samples is not None
        while target_samples is None or yielded < target_samples:
            yielded_this_pass = 0
            sample_iter = self._iter_assigned_samples(assigned_urls, rng)
            for sample in sample_iter:
                sample["index"] = sample_index
                yield sample
                yielded += 1
                yielded_this_pass += 1
                sample_index += 1
                if target_samples is not None and yielded >= target_samples:
                    return
            if yielded_this_pass == 0 or not repeat:
                return

    def _iter_assigned_samples(self, assigned_urls: list[str], rng: random.Random):
        urls = list(assigned_urls)
        if self.shuffle_shards:
            rng.shuffle(urls)
        yield from _shuffle_buffer(self._iter_samples(urls), rng=rng, buffer_size=self.shuffle_buffer_size)

    def _iter_samples(self, shard_urls: list[str]):
        for shard_url in shard_urls:
            with self._cache.open(shard_url) as handle:
                yield from _iter_tar_samples(handle, image_spec=self.image_spec)


class _ShardWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        split: str,
        image_format: str,
        target_shard_size_bytes: int,
    ) -> None:
        self.output_root = output_root
        self.split = split
        self.image_format = image_format.lower()
        self.target_shard_size_bytes = target_shard_size_bytes
        self.records: list[FrameShardRecord] = []
        self.shard_index = 0
        self._archive: tarfile.TarFile | None = None
        self._archive_path: Path | None = None
        self._archive_relative_path: str | None = None
        self._sample_count = 0
        self._logical_size_bytes = 0
        self._first_frame_id = ""
        self._last_frame_id = ""

    def add_sample(self, *, sample_key: str, image_bytes: bytes, metadata: dict[str, object]) -> None:
        image_ext = f".{self.image_format}"
        image_name = f"{sample_key}{image_ext}"
        metadata_name = f"{sample_key}.json"
        metadata_bytes = _stable_json_bytes(metadata)
        projected_size = self._logical_size_bytes
        projected_size += _tar_member_size(len(image_bytes)) + _tar_member_size(len(metadata_bytes))
        if self._archive is None or (self._sample_count > 0 and projected_size > self.target_shard_size_bytes):
            self._rotate()
        assert self._archive is not None
        self._write_member(image_name, image_bytes)
        self._write_member(metadata_name, metadata_bytes)
        self._logical_size_bytes += _tar_member_size(len(image_bytes)) + _tar_member_size(len(metadata_bytes))
        frame_id = str(metadata["frame_id"])
        if not self._first_frame_id:
            self._first_frame_id = frame_id
        self._last_frame_id = frame_id
        self._sample_count += 1

    def close(self) -> list[FrameShardRecord]:
        self._finalize_archive()
        return list(self.records)

    def _rotate(self) -> None:
        self._finalize_archive()
        split_dir = ensure_dir(self.output_root / self.split)
        filename = f"{self.split}-{self.shard_index:06d}.tar"
        self._archive_path = split_dir / filename
        self._archive_relative_path = f"{self.split}/{filename}"
        self._archive = tarfile.open(self._archive_path, mode="w")
        self._sample_count = 0
        self._logical_size_bytes = 0
        self._first_frame_id = ""
        self._last_frame_id = ""
        self.shard_index += 1

    def _finalize_archive(self) -> None:
        if self._archive is None or self._archive_path is None or self._archive_relative_path is None:
            return
        self._archive.close()
        self.records.append(
            FrameShardRecord(
                split=self.split,
                relative_path=self._archive_relative_path,
                sample_count=self._sample_count,
                size_bytes=self._archive_path.stat().st_size,
                first_frame_id=self._first_frame_id,
                last_frame_id=self._last_frame_id,
            )
        )
        self._archive = None
        self._archive_path = None
        self._archive_relative_path = None

    def _write_member(self, name: str, payload: bytes) -> None:
        assert self._archive is not None
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        info.mode = 0o644
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        self._archive.addfile(info, io.BytesIO(payload))


class _ShardCache:
    def __init__(self, *, cache_dir: str | None, cache_size_bytes: int) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.cache_size_bytes = cache_size_bytes

    @contextlib.contextmanager
    def open(self, shard_url: str):
        if _is_local_path_like(shard_url):
            local_path = _local_path_from_url(shard_url)
            with local_path.open("rb") as handle:
                yield handle
            return
        if self.cache_dir is None:
            with fsspec.open(shard_url, "rb").open() as handle:
                yield handle
            return
        local_path = self._materialize(shard_url)
        with local_path.open("rb") as handle:
            yield handle

    def _materialize(self, shard_url: str) -> Path:
        assert self.cache_dir is not None
        ensure_dir(self.cache_dir)
        destination = self.cache_dir / _cache_filename(shard_url)
        lock_path = self.cache_dir / f"{destination.name}.lock"
        with _exclusive_lock(lock_path):
            if destination.exists() and destination.stat().st_size > 0:
                os.utime(destination, None)
                return destination
            temp_path = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
            with fsspec.open(shard_url, "rb").open() as source:
                with temp_path.open("wb") as handle:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            temp_path.replace(destination)
            os.utime(destination, None)
        self._evict(exclude={destination.name, lock_path.name})
        return destination

    def _evict(self, *, exclude: set[str]) -> None:
        assert self.cache_dir is not None
        with _exclusive_lock(self.cache_dir / ".evict.lock"):
            all_files = [path for path in self.cache_dir.iterdir() if path.is_file() and not path.name.endswith(".lock")]
            files = [path for path in all_files if path.name not in exclude]
            total_size = sum(path.stat().st_size for path in all_files)
            if total_size <= self.cache_size_bytes:
                return
            for path in sorted(files, key=lambda item: item.stat().st_mtime):
                size = path.stat().st_size
                path.unlink(missing_ok=True)
                total_size -= size
                if total_size <= self.cache_size_bytes:
                    break


def _write_video_record(
    *,
    writer: _ShardWriter,
    record: VideoRecord,
    image_spec: ImageSpec,
    sample_fps: float,
) -> int:
    capture = cv2.VideoCapture(record.path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video {record.path}")
    samples_written = 0
    try:
        for frame_index, timestamp_seconds in _iter_sample_points(record, sample_fps=sample_fps):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if (not ok or frame is None) and timestamp_seconds >= 0:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
                ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to decode frame {frame_index} from {record.path}")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prepared = resize_image_with_padding(image, image_spec)
            metadata = {
                "frame_id": _frame_id(record.video_id, frame_index, timestamp_seconds),
                "frame_idx": frame_index,
                "source_path": record.relative_path,
                "split": record.split,
                "timestamp_ms": round(timestamp_seconds * 1000.0),
                "video_id": record.video_id,
            }
            writer.add_sample(
                sample_key=_sample_key(metadata["frame_id"]),
                image_bytes=_encode_image(prepared, writer.image_format),
                metadata=metadata,
            )
            samples_written += 1
    finally:
        capture.release()
    return samples_written


def _iter_tar_samples(handle, *, image_spec: ImageSpec):
    pending: dict[str, dict[str, object]] = {}
    with tarfile.open(fileobj=handle, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS and suffix != ".json":
                continue
            sample_key = member.name.rsplit(".", maxsplit=1)[0]
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            sample = pending.setdefault(sample_key, {})
            if suffix == ".json":
                sample["metadata"] = json.loads(payload.decode("utf-8"))
            else:
                sample["image_bytes"] = payload
            if "metadata" in sample and "image_bytes" in sample:
                yield _decode_sample(sample, image_spec=image_spec)
                del pending[sample_key]


def _decode_sample(sample: dict[str, object], *, image_spec: ImageSpec) -> dict[str, object]:
    metadata = dict(sample["metadata"])
    image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
    if image.size == (image_spec.width, image_spec.height):
        tensor = TF.to_tensor(image)
    else:
        tensor = resize_with_padding(image, image_spec)
    timestamp_ms = float(metadata["timestamp_ms"])
    return {
        "image": tensor,
        "frame_id": str(metadata["frame_id"]),
        "timestamp": timestamp_ms / 1000.0,
        "video_title": str(metadata["video_id"]),
    }


def _iter_sample_points(record: VideoRecord, *, sample_fps: float):
    duration_seconds = record.duration_seconds
    if duration_seconds <= 0 and record.frame_count > 0 and record.fps > 0:
        duration_seconds = record.frame_count / record.fps
    max_timestamp = max(duration_seconds - 1e-6, 0.0)
    sample_count = max(1, int(math.floor(max_timestamp * sample_fps)) + 1)
    last_frame_index: int | None = None
    for sample_idx in range(sample_count):
        timestamp_seconds = min(sample_idx / sample_fps, max_timestamp)
        frame_index = resolve_frame_index(record, timestamp_seconds)
        if frame_index == last_frame_index:
            continue
        yield frame_index, timestamp_seconds
        last_frame_index = frame_index


def _frame_id(video_id: str, frame_index: int, timestamp_seconds: float) -> str:
    return f"{video_id}:{frame_index}:{round(timestamp_seconds * 1000.0)}"


def _sample_key(frame_id: str) -> str:
    return hashlib.sha1(frame_id.encode("utf-8")).hexdigest()


def _encode_image(image: Image.Image, image_format: str) -> bytes:
    buffer = io.BytesIO()
    if image_format == "webp":
        image.save(buffer, format="WEBP", lossless=True, method=6)
    elif image_format == "png":
        image.save(buffer, format="PNG", optimize=False)
    else:
        raise ValueError(f"unsupported image_format: {image_format}")
    return buffer.getvalue()


def _stable_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tar_member_size(payload_size: int) -> int:
    return 512 + ((payload_size + 511) // 512) * 512


def _shuffle_buffer(samples, *, rng: random.Random, buffer_size: int):
    if buffer_size <= 0:
        yield from samples
        return
    buffer: list[dict[str, object]] = []
    for sample in samples:
        buffer.append(sample)
        if len(buffer) >= buffer_size:
            index = rng.randrange(len(buffer))
            yield buffer.pop(index)
    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)


def _split_worker_limit(total: int | None, worker_id: int, num_workers: int) -> int | None:
    if total is None:
        return None
    base = total // num_workers
    remainder = total % num_workers
    return base + (1 if worker_id < remainder else 0)


def _join_root(root: str, relative_path: str) -> str:
    if _is_local_path_like(root):
        return str((Path(root).expanduser() / relative_path).resolve())
    return f"{root.rstrip('/')}/{relative_path.lstrip('/')}"


def _expand_local_patterns(patterns: list[str]) -> list[str]:
    urls: list[str] = []
    for pattern in patterns:
        if not _is_local_path_like(pattern):
            if any(token in pattern for token in "*?[]"):
                raise ValueError("remote glob patterns are not supported; use a shard manifest instead")
            urls.append(pattern)
            continue
        if any(token in pattern for token in "*?[]"):
            matches = sorted(glob.glob(str(Path(pattern).expanduser())))
            urls.extend(str(Path(match).resolve()) for match in matches)
        else:
            urls.append(str(_local_path_from_url(pattern).resolve()))
    return urls


def _cache_filename(shard_url: str) -> str:
    digest = hashlib.sha1(shard_url.encode("utf-8")).hexdigest()
    basename = Path(urlsplit(shard_url).path).name or "shard.tar"
    return f"{digest}-{basename}"


def _is_local_path_like(path_or_url: str) -> bool:
    return urlsplit(path_or_url).scheme in ("", "file")


def _local_path_from_url(path_or_url: str) -> Path:
    parsed = urlsplit(path_or_url)
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve()
    return Path(path_or_url).expanduser().resolve()


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
