from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from sandbox_autoencoders.data import ImageSpec


@dataclass(frozen=True)
class VAEOutput:
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=8, num_channels=channels)


class ConvVAE(nn.Module):
    def __init__(self, image_spec: ImageSpec, latent_dim: int = 128) -> None:
        super().__init__()
        if image_spec.width % 32 != 0 or image_spec.height % 32 != 0:
            raise ValueError("image dimensions must be divisible by 32")
        self.image_spec = image_spec
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            _group_norm(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            _group_norm(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            _group_norm(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            _group_norm(256),
            nn.SiLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            _group_norm(512),
            nn.SiLU(),
        )

        self.feature_shape = (512, image_spec.height // 32, image_spec.width // 32)
        flattened_features = self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2]
        self.fc_mu = nn.Linear(flattened_features, latent_dim)
        self.fc_logvar = nn.Linear(flattened_features, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, flattened_features)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            _group_norm(256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            _group_norm(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            _group_norm(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            _group_norm(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            _group_norm(16),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        mu = self.fc_mu(features)
        logvar = torch.clamp(self.fc_logvar(features), min=-8.0, max=8.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_input(z)
        features = features.view(z.size(0), *self.feature_shape)
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return VAEOutput(reconstruction=reconstruction, mu=mu, logvar=logvar)


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = target.size(0)
    recon_mse = F.mse_loss(reconstruction, target, reduction="mean")
    recon_loss = recon_mse * target[0].numel()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = recon_loss + beta * kl_loss
    return total_loss, {
        "loss": total_loss.item(),
        "recon_loss": recon_loss.item(),
        "recon_mse": recon_mse.item(),
        "kl_loss": kl_loss.item(),
    }
