# ZX Spectrum VAE

This repo trains a convolutional variational autoencoder on the Hugging Face dataset [`tsilva/zx-spectrum-worldoflongplays-te8t6fzk-2i`](https://huggingface.co/datasets/tsilva/zx-spectrum-worldoflongplays-te8t6fzk-2i).

Dataset observations from a quick inspection on March 11, 2026:

- `2190` RGB frames in the `train` split
- frames are roughly `256x183`
- the current dataset appears to be a single ZX Spectrum longplay video: `Paperboy 2`

The training pipeline preserves aspect ratio, pads images to `256x192`, and learns a latent space that can be interpolated between dataset frames.

The preferred cloud-native path is now:

1. keep raw videos in object storage or a local `raw/` directory
2. build a per-video JSONL manifest
3. rewrite the manifest into `frames-256/` tar shards
4. train from shard manifests with a bounded local cache

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Train From Frame Shards

Rewrite the raw manifest into training-optimized frame shards:

```bash
python -m sandbox_autoencoders.build_frame_shards \
  --manifest outputs/video-manifests/local-videos.jsonl \
  --output-dir outputs/frame-shards/frames-256 \
  --sample-fps 1.0 \
  --overwrite
```

This writes:

- `outputs/frame-shards/frames-256/shards.jsonl`
- `outputs/frame-shards/frames-256/train/*.tar`
- `outputs/frame-shards/frames-256/val/*.tar`
- `outputs/frame-shards/frames-256/test/*.tar`

Train against the shard manifest:

```bash
python -m sandbox_autoencoders.train \
  --shard-manifest outputs/frame-shards/frames-256/shards.jsonl \
  --cache-dir .cache/frame-shards \
  --epochs 40 \
  --batch-size 32 \
  --latent-dim 128 \
  --beta 0.01 \
  --output-dir outputs/run-01
```

Artifacts:

- checkpoints in `outputs/run-01/checkpoints/`
- sample reconstructions after each epoch in `outputs/run-01/reconstructions/`
- training history in `outputs/run-01/history.json`

If your shards live in object storage, point the manifest at a different root:

```bash
python -m sandbox_autoencoders.train \
  --shard-manifest outputs/frame-shards/frames-256/shards.jsonl \
  --shard-root s3://my-bucket/frames-256 \
  --cache-dir /mnt/nvme/frame-shards \
  --epochs 40 \
  --batch-size 32 \
  --output-dir outputs/run-cloud
```

`--shard-root` also works with `hf://` roots when the shards have been synced to a Hugging Face storage bucket.

## Interpolate

After training, interpolate between two dataset indices:

```bash
python -m sandbox_autoencoders.interpolate \
  --checkpoint outputs/run-01/checkpoints/best.pt \
  --index-a 10 \
  --index-b 1800 \
  --steps 9 \
  --output outputs/run-01/interpolation_10_1800.png
```

The output strip contains the first source frame, latent interpolation frames, and the second source frame.

## Explorer UI

Launch a local interpolation explorer against the current best checkpoint:

```bash
python -m sandbox_autoencoders.explorer \
  --checkpoint outputs/full-run-es/checkpoints/best.pt
```

The UI lets you:

- switch checkpoints
- scrub frame indices
- compare original frames against endpoint reconstructions
- play back the decoded latent path as a looping video

## Local Video Manifest

Build a JSONL manifest from a local directory of videos:

```bash
python -m sandbox_autoencoders.build_video_manifest \
  --video-dir ~/Desktop/videos \
  --output outputs/video-manifests/local-videos.jsonl
```

The manifest stores one row per video with duration, fps, dimensions, byte size, and a deterministic `train`/`val`/`test` split.

`ffprobe` must be available on your `PATH`.

The raw-video manifest remains useful for inspection, preprocessing, and debugging. The trainer can still read it directly with `--manifest`, but that path is slower because it samples from compressed videos on demand.

## Benchmark Loader Throughput

Measure dataloader throughput against the shard manifest:

```bash
python -m sandbox_autoencoders.benchmark_video_loader \
  --shard-manifest outputs/frame-shards/frames-256/shards.jsonl \
  --split train \
  --samples 4096 \
  --batch-size 32 \
  --num-workers 4 \
  --cache-dir .cache/frame-shards \
  --sample-shuffle-buffer 2048 \
  --pin-memory \
  --persistent-workers
```

Benchmark the legacy raw-video path by swapping `--shard-manifest` for `--manifest`. That keeps the old OpenCV random-seek loader available when you need to compare preprocessing quality or debug source videos.

The shard benchmark reports effective samples/sec so you can measure cold-cache and warm-cache behavior on a cloud node.
