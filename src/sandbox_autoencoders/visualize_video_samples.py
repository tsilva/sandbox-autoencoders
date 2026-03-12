from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import functional as TF

from sandbox_autoencoders.data import ImageSpec, resize_with_padding
from sandbox_autoencoders.local_video_data import VideoRecord, load_manifest


PREVIEW_BACKGROUND = (12, 12, 12)
LABEL_BACKGROUND = (24, 24, 24)
LABEL_COLOR = (235, 235, 235)
PANEL_PADDING = 12
LABEL_HEIGHT = 44
GAP = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw and post-processed video frames from a manifest split.")
    parser.add_argument("--manifest", required=True, help="Path to the JSONL manifest created by build_video_manifest.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to visualize.")
    parser.add_argument("--output", required=True, help="Output PNG path for the contact sheet.")
    parser.add_argument("--count", type=int, default=16, help="Number of distinct videos to sample.")
    parser.add_argument("--columns", type=int, default=2, help="Number of sample rows per image column group.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=256, help="Model input width.")
    parser.add_argument("--height", type=int, default=192, help="Model input height.")
    return parser.parse_args()


def _resolve_frame_index(record: VideoRecord, timestamp: float) -> int:
    if record.frame_count > 0 and record.duration_seconds > 0:
        return min(record.frame_count - 1, max(0, round(timestamp / record.duration_seconds * (record.frame_count - 1))))
    if record.fps > 0:
        return max(0, round(timestamp * record.fps))
    return 0


def _decode_raw_frame(record: VideoRecord, timestamp: float) -> Image.Image:
    capture = cv2.VideoCapture(record.path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video {record.path}")

    frame_index = _resolve_frame_index(record, timestamp)
    if frame_index > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    else:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)

    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to decode frame from {record.path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def _fit_preview(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.pad(
        image.convert("RGB"),
        size,
        method=Image.Resampling.BICUBIC,
        color=PREVIEW_BACKGROUND,
        centering=(0.5, 0.5),
    )


def _choose_records(records: list[VideoRecord], count: int, seed: int) -> list[VideoRecord]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    if count >= len(records):
        chosen = list(records)
        rng.shuffle(chosen)
        return chosen
    return rng.sample(records, count)


def _make_panel(record: VideoRecord, raw_image: Image.Image, processed_image: Image.Image, timestamp: float, spec: ImageSpec) -> Image.Image:
    original_preview = _fit_preview(raw_image, (spec.width, spec.height))
    processed_preview = processed_image.convert("RGB")
    panel_width = spec.width * 2 + PANEL_PADDING * 2 + GAP
    panel_height = spec.height + LABEL_HEIGHT + PANEL_PADDING * 2
    panel = Image.new("RGB", (panel_width, panel_height), PREVIEW_BACKGROUND)

    panel.paste(original_preview, (PANEL_PADDING, PANEL_PADDING))
    panel.paste(processed_preview, (PANEL_PADDING + spec.width + GAP, PANEL_PADDING))

    draw = ImageDraw.Draw(panel)
    label_top = PANEL_PADDING + spec.height
    draw.rectangle((0, label_top, panel_width, panel_height), fill=LABEL_BACKGROUND)
    label = f"{Path(record.relative_path).stem[:52]} | t={timestamp:.2f}s"
    draw.text((PANEL_PADDING, label_top + 8), label, fill=LABEL_COLOR)
    draw.text((PANEL_PADDING, label_top + 24), "left: raw frame    right: training preprocess", fill=LABEL_COLOR)
    return panel


def _compose_grid(panels: list[Image.Image], columns: int) -> Image.Image:
    if not panels:
        raise ValueError("at least one panel is required")
    columns = max(1, columns)
    rows = (len(panels) + columns - 1) // columns
    panel_width, panel_height = panels[0].size
    canvas = Image.new(
        "RGB",
        (
            columns * panel_width + (columns + 1) * GAP,
            rows * panel_height + (rows + 1) * GAP,
        ),
        PREVIEW_BACKGROUND,
    )
    for index, panel in enumerate(panels):
        row = index // columns
        column = index % columns
        x = GAP + column * (panel_width + GAP)
        y = GAP + row * (panel_height + GAP)
        canvas.paste(panel, (x, y))
    return canvas


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest, split=args.split)
    if not records:
        raise ValueError(f"no records found for split={args.split}")

    spec = ImageSpec(width=args.width, height=args.height)
    selected_records = _choose_records(records, args.count, args.seed)
    rng = random.Random(args.seed)

    panels: list[Image.Image] = []
    for record in selected_records:
        if record.duration_seconds > 0:
            timestamp = rng.uniform(0.0, max(0.0, record.duration_seconds - 1e-3))
        else:
            timestamp = 0.0
        raw_image = _decode_raw_frame(record, timestamp)
        processed_tensor = resize_with_padding(raw_image, spec)
        processed_image = TF.to_pil_image(processed_tensor)
        panels.append(_make_panel(record, raw_image, processed_image, timestamp, spec))

    grid = _compose_grid(panels, args.columns)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
