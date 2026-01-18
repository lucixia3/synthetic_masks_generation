from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * -(torch.log(torch.tensor(10000.0)) / (half - 1))
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = F.relu(h)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.conv2(h)
        return F.relu(h + self.residual(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> List[torch.Tensor]:
        h = self.res(x, t)
        pooled = self.pool(h)
        return [h, pooled]


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t)


class DiffusionUNet(nn.Module):
    """Lightweight U-Net for unconditional diffusion on segmentation masks."""

    def __init__(self, channels: int, base_dim: int = 64, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.inc = ResidualBlock(channels, base_dim, time_dim)
        self.down1 = DownBlock(base_dim, base_dim * 2, time_dim)
        self.down2 = DownBlock(base_dim * 2, base_dim * 4, time_dim)

        self.mid = ResidualBlock(base_dim * 4, base_dim * 4, time_dim)

        self.up1 = UpBlock(base_dim * 4 + base_dim * 4, base_dim * 2, time_dim)
        self.up2 = UpBlock(base_dim * 2 + base_dim * 2, base_dim, time_dim)
        self.outc = nn.Conv2d(base_dim + base_dim, channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x1 = self.inc(x, t_emb)
        x2_skip, x2 = self.down1(x1, t_emb)
        x3_skip, x3 = self.down2(x2, t_emb)

        mid = self.mid(x3, t_emb)

        x = self.up1(mid, x3_skip, t_emb)
        x = self.up2(x, x2_skip, t_emb)
        x = torch.cat([x, x1], dim=1)
        return self.outc(x)
