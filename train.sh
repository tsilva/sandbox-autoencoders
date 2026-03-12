#!/usr/bin/env bash
set -euo pipefail

python3 -m sandbox_autoencoders.train \
  --manifest outputs/video-manifests/local-videos.jsonl \
  --output-dir outputs/zx-local-run \
  --epochs 40 \
  --batch-size 32 \
  --num-workers 4 \
  --persistent-workers \
  --learning-rate 3e-4 \
  --latent-dim 128 \
  --beta 0.001 \
  --beta-warmup-epochs 10 \
  --train-samples-per-epoch 100000 \
  --val-samples 4096 \
  --sampling-weight sqrt_duration \
  --video-burst-size 16 \
  --burst-span-seconds 1.5 \
  --val-video-burst-size 1 \
  --val-burst-span-seconds 0.0 \
  --max-open-captures 8 \
  --max-decode-attempts 4 \
  --max-sequential-gap-frames 120 \
  --hf-repo-id tsilva/zx-spectrum-vae
