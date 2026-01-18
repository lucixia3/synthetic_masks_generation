from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class LinearNoiseScheduler:
    """Simple linear beta scheduler for DDPM-style training."""

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self) -> None:
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register(betas, alphas, alphas_cumprod)

    def register(self, betas: torch.Tensor, alphas: torch.Tensor, alphas_cumprod: torch.Tensor) -> None:
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x0.device)[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(x0.device)[t][:, None, None, None]
        noisy = sqrt_alpha * x0 + sqrt_one_minus * noise
        return noisy, noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
