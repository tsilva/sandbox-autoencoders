# ZX Spectrum VAE

This repo trains a convolutional variational autoencoder on the Hugging Face dataset [`tsilva/zx-spectrum-worldoflongplays-te8t6fzk-2i`](https://huggingface.co/datasets/tsilva/zx-spectrum-worldoflongplays-te8t6fzk-2i).

Dataset observations from a quick inspection on March 11, 2026:

- `2190` RGB frames in the `train` split
- frames are roughly `256x183`
- the current dataset appears to be a single ZX Spectrum longplay video: `Paperboy 2`

The training pipeline preserves aspect ratio, pads images to `256x192`, and learns a latent space that can be interpolated between dataset frames.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Train

```bash
python -m sandbox_autoencoders.train \
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

## Benchmark Local Video Loading

Measure dataloader throughput against the manifest:

```bash
python -m sandbox_autoencoders.benchmark_video_loader \
  --manifest outputs/video-manifests/local-videos.jsonl \
  --split train \
  --samples 4096 \
  --batch-size 32 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers
```

This benchmark samples random frames from held-out videos on demand and reports effective samples/sec so you can decide whether offline frame extraction is necessary.
