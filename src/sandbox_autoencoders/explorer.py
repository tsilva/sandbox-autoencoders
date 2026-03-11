from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from sandbox_autoencoders.data import DEFAULT_DATASET, collate_frames, load_full_dataset
from sandbox_autoencoders.interpolate import _load_model, interpolate_tensors
from sandbox_autoencoders.utils import choose_device


@dataclass
class LoadedArtifacts:
    checkpoint_path: Path
    dataset_name: str
    cache_dir: str | None
    device: torch.device
    model: torch.nn.Module
    dataset: object


class ExplorerState:
    def __init__(self) -> None:
        self.loaded: LoadedArtifacts | None = None

    def load(
        self,
        checkpoint_path: str,
        dataset_name: str,
        cache_dir: str | None,
        device_name: str | None,
    ) -> LoadedArtifacts:
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        normalized_cache_dir = cache_dir or None
        device = choose_device(device_name)

        if (
            self.loaded
            and self.loaded.checkpoint_path == resolved_checkpoint
            and self.loaded.dataset_name == dataset_name
            and self.loaded.cache_dir == normalized_cache_dir
            and self.loaded.device == device
        ):
            return self.loaded

        model, image_spec = _load_model(str(resolved_checkpoint), device)
        dataset = load_full_dataset(
            dataset_name=dataset_name,
            image_spec=image_spec,
            cache_dir=normalized_cache_dir,
        )
        self.loaded = LoadedArtifacts(
            checkpoint_path=resolved_checkpoint,
            dataset_name=dataset_name,
            cache_dir=normalized_cache_dir,
            device=device,
            model=model,
            dataset=dataset,
        )
        return self.loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an interpolation explorer for a trained VAE.")
    parser.add_argument("--checkpoint", default="outputs/full-run-es/checkpoints/best.pt")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _discover_checkpoints() -> list[str]:
    checkpoint_paths = sorted(Path("outputs").glob("**/checkpoints/best.pt"))
    if not checkpoint_paths:
        checkpoint_paths = sorted(Path("outputs").glob("**/checkpoints/epoch-*.pt"))
    return [str(path) for path in checkpoint_paths]


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(tensor.clamp(0.0, 1.0))


@torch.inference_mode()
def _render(
    state: ExplorerState,
    checkpoint_path: str,
    dataset_name: str,
    cache_dir: str | None,
    device_name: str | None,
    index_a: int,
    index_b: int,
    steps: int,
) -> tuple[str, Image.Image, Image.Image, Image.Image, Image.Image, list[Image.Image]]:
    loaded = state.load(
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        device_name=device_name,
    )
    dataset = loaded.dataset
    dataset_size = len(dataset)

    if dataset_size < 2:
        raise ValueError("dataset must contain at least two frames")

    index_a = max(0, min(int(index_a), dataset_size - 1))
    index_b = max(0, min(int(index_b), dataset_size - 1))

    frame_a = collate_frames([dataset[index_a]])
    frame_b = collate_frames([dataset[index_b]])
    image_a = frame_a["image"]
    image_b = frame_b["image"]

    reconstruction_a = loaded.model(image_a.to(loaded.device)).reconstruction.cpu()
    reconstruction_b = loaded.model(image_b.to(loaded.device)).reconstruction.cpu()
    strip = interpolate_tensors(
        model=loaded.model,
        image_a=image_a,
        image_b=image_b,
        steps=steps,
        device=loaded.device,
    )

    gallery_images = [_tensor_to_image(frame) for frame in strip]
    metadata = torch.load(str(loaded.checkpoint_path), map_location="cpu", weights_only=False)
    epoch = metadata.get("epoch", "?")
    metrics = metadata.get("metrics") or {}
    val_loss = metrics.get("val_loss")
    recon_mse = metrics.get("val_recon_mse")
    info = "\n".join(
        [
            f"checkpoint: `{loaded.checkpoint_path}`",
            f"dataset: `{loaded.dataset_name}`",
            f"frames: `{dataset_size}`",
            f"indices: `{index_a}` -> `{index_b}`",
            f"epoch: `{epoch}`",
            f"val_loss: `{val_loss:.4f}`" if isinstance(val_loss, (int, float)) else "val_loss: `n/a`",
            f"val_recon_mse: `{recon_mse:.6f}`" if isinstance(recon_mse, (int, float)) else "val_recon_mse: `n/a`",
        ]
    )

    return (
        info,
        _tensor_to_image(image_a[0]),
        _tensor_to_image(reconstruction_a[0]),
        _tensor_to_image(image_b[0]),
        _tensor_to_image(reconstruction_b[0]),
        gallery_images,
    )


def _load_checkpoint_summary(
    state: ExplorerState,
    checkpoint_path: str,
    dataset_name: str,
    cache_dir: str | None,
    device_name: str | None,
) -> tuple[gr.Slider, gr.Slider, str]:
    loaded = state.load(
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        device_name=device_name,
    )
    dataset_size = len(loaded.dataset)
    maximum_index = max(0, dataset_size - 1)
    summary = f"Loaded `{Path(checkpoint_path)}` with `{dataset_size}` frames."
    return (
        gr.Slider(minimum=0, maximum=maximum_index, value=min(0, maximum_index), step=1),
        gr.Slider(minimum=0, maximum=maximum_index, value=min(100, maximum_index), step=1),
        summary,
    )


def _randomize_indices(
    state: ExplorerState,
    checkpoint_path: str,
    dataset_name: str,
    cache_dir: str | None,
    device_name: str | None,
) -> tuple[int, int]:
    loaded = state.load(
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        device_name=device_name,
    )
    dataset_size = len(loaded.dataset)
    if dataset_size < 2:
        return 0, 0
    index_a, index_b = random.sample(range(dataset_size), 2)
    return index_a, index_b


def build_app(
    initial_checkpoint: str,
    dataset_name: str,
    cache_dir: str | None,
    device_name: str | None,
) -> tuple[gr.Blocks, gr.themes.ThemeClass, str]:
    state = ExplorerState()
    checkpoint_choices = _discover_checkpoints()
    if initial_checkpoint not in checkpoint_choices:
        checkpoint_choices.insert(0, initial_checkpoint)

    theme = gr.themes.Base(
        primary_hue="amber",
        secondary_hue="orange",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "sans-serif"],
    )

    css = """
    body, .gradio-container {
      background:
        radial-gradient(circle at top, rgba(255, 186, 73, 0.12), transparent 35%),
        linear-gradient(180deg, #15120d 0%, #090806 100%);
    }
    .app-shell {
      border: 1px solid rgba(255, 196, 92, 0.18);
      border-radius: 20px;
      background: rgba(19, 16, 11, 0.88);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.4);
    }
    .eyebrow {
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #f4bb59;
      font-size: 0.8rem;
    }
    """

    with gr.Blocks(title="ZX Spectrum VAE Explorer") as app:
        gr.Markdown(
            """
            <div class="app-shell">
            <p class="eyebrow">ZX Spectrum VAE</p>
            <h1>Interpolation Explorer</h1>
            <p>Inspect endpoint reconstructions and the latent path between any two frames.</p>
            </div>
            """
        )

        with gr.Row():
            checkpoint = gr.Dropdown(
                label="Checkpoint",
                choices=checkpoint_choices,
                value=initial_checkpoint,
                allow_custom_value=True,
            )
            dataset = gr.Textbox(label="Dataset", value=dataset_name)
            cache = gr.Textbox(label="Cache Dir", value=cache_dir or "")
            device = gr.Textbox(label="Device", value=device_name or "")

        load_status = gr.Markdown()

        with gr.Row():
            index_a = gr.Slider(label="Frame A", minimum=0, maximum=2189, value=0, step=1)
            index_b = gr.Slider(label="Frame B", minimum=0, maximum=2189, value=100, step=1)
            steps = gr.Slider(label="Interpolation Steps", minimum=3, maximum=15, value=9, step=1)

        with gr.Row():
            load_button = gr.Button("Load Checkpoint", variant="secondary")
            random_button = gr.Button("Random Pair", variant="secondary")
            swap_button = gr.Button("Swap Frames", variant="secondary")
            render_button = gr.Button("Render Interpolation", variant="primary")

        info = gr.Markdown()

        with gr.Row():
            original_a = gr.Image(label="Original A", type="pil")
            recon_a = gr.Image(label="Reconstruction A", type="pil")
            original_b = gr.Image(label="Original B", type="pil")
            recon_b = gr.Image(label="Reconstruction B", type="pil")

        gallery = gr.Gallery(label="Latent Path", columns=6, object_fit="contain", height="auto")

        load_button.click(
            fn=lambda cp, ds, cd, dev: _load_checkpoint_summary(state, cp, ds, cd or None, dev or None),
            inputs=[checkpoint, dataset, cache, device],
            outputs=[index_a, index_b, load_status],
        )
        random_button.click(
            fn=lambda cp, ds, cd, dev: _randomize_indices(state, cp, ds, cd or None, dev or None),
            inputs=[checkpoint, dataset, cache, device],
            outputs=[index_a, index_b],
        )
        swap_button.click(fn=lambda a, b: (b, a), inputs=[index_a, index_b], outputs=[index_a, index_b])
        render_button.click(
            fn=lambda cp, ds, cd, dev, a, b, s: _render(state, cp, ds, cd or None, dev or None, a, b, s),
            inputs=[checkpoint, dataset, cache, device, index_a, index_b, steps],
            outputs=[info, original_a, recon_a, original_b, recon_b, gallery],
        )

        app.load(
            fn=lambda cp, ds, cd, dev, a, b, s: _render(state, cp, ds, cd or None, dev or None, a, b, s),
            inputs=[checkpoint, dataset, cache, device, index_a, index_b, steps],
            outputs=[info, original_a, recon_a, original_b, recon_b, gallery],
        )

    return app, theme, css


def main() -> None:
    args = parse_args()
    app, theme, css = build_app(
        initial_checkpoint=args.checkpoint,
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        device_name=args.device,
    )
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    main()
