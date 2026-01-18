

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import MaskDataset


# ============================================================
# PALETA CORREGIDA (CLAVE)
# ============================================================
# canal → significado → color
PALETTE = [
    
    (0, 0, 0), #negro
    (0, 0, 255),     #azul 
    (0, 255, 0),   #verde
    (255, 0, 255),   #magenta
    (255, 255, 0),     #amarillo
    (255, 0, 0),  #rojo
    (128, 128, 128), #gris
]
NUM_CLASSES = 7


# ============================================================
# MASK UTILS
# ============================================================
def to_one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    if mask.ndim == 3 and mask.size(0) == 1:
        mask = mask[0]
    oh = F.one_hot(mask.long(), num_classes=num_classes)
    return oh.permute(2, 0, 1).float()


class RemapMaskValues:
    def __init__(self, num_classes: int, mask_values: Optional[Tuple[int, ...]] = None):
        self.num_classes = num_classes
        self.mask_values = mask_values

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = x.squeeze(0).long()
        if self.mask_values is not None:
            out = torch.empty_like(m)
            for i, v in enumerate(self.mask_values):
                out[m == v] = i
            return out
        vals = torch.sort(torch.unique(m)).values
        out = torch.empty_like(m)
        for i, v in enumerate(vals):
            out[m == v] = i
        return out


class ToOneHot:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return to_one_hot(x, self.n_classes)


def get_mask_transforms(image_size: int, num_classes: int, mask_values=None):
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.PILToTensor(),
        RemapMaskValues(num_classes, mask_values),
        ToOneHot(num_classes),
    ])


def mask_logits_to_rgb(mask_logits: torch.Tensor) -> Image.Image:
    probs = torch.softmax(mask_logits, dim=0)
    ids = torch.argmax(probs, dim=0).cpu().numpy()
    h, w = ids.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, col in enumerate(PALETTE):
        rgb[ids == i] = col

    return Image.fromarray(rgb, mode="RGB")


# ============================================================
# VAE
# ============================================================
class MaskVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 16 * 16, latent_dim * 2)

        self.dec_fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, NUM_CLASSES, 4, 2, 1),
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        mu, logvar = self.fc(h).chunk(2, dim=1)
        return mu, logvar

    def reparam(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        h = self.dec_fc(z).view(z.size(0), 256, 16, 16)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(mask_oh, logits, mu, logvar, beta):
    target = torch.argmax(mask_oh, dim=1)
    recon = F.cross_entropy(logits, target)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld


# ============================================================
# LATENT DIFFUSION
# ============================================================
class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, z, t):
        t = t.float().unsqueeze(1) / 1000.0
        return self.net(torch.cat([z, t], dim=1))

class LatentScheduler:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.device = device

        self.betas = torch.linspace(1e-4, 0.02, T, device=device)
        self.alphas = 1 - self.betas
        self.ab = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, z0, t):
        eps = torch.randn_like(z0)
        a = self.ab[t].unsqueeze(1)
        return torch.sqrt(a) * z0 + torch.sqrt(1 - a) * eps, eps

    def step(self, zt, eps_hat, t):
        beta = self.betas[t]
        alpha = 1 - beta
        ab = self.ab[t]
        return (zt - beta / torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(alpha)



# ============================================================
# TRAIN VAE
# ============================================================
def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_t = get_mask_transforms(args.image_size, NUM_CLASSES, args.mask_values)
    ds = MaskDataset(args.mask_dir, transforms=mask_t)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    vae = MaskVAE(args.latent).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        for batch in dl:
            batch = batch.to(device)
            logits, mu, logvar = vae(batch)
            loss = vae_loss(batch, logits, mu, logvar, args.beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"[VAE] epoch {ep+1}/{args.epochs} loss={loss.item():.4f}")
        torch.save({"vae": vae.state_dict()}, out / "vae_last.pt")

@torch.no_grad()
def sample_from_ldm(
    vae: MaskVAE,
    denoiser: LatentDenoiser,
    scheduler: LatentScheduler,
    latent_dim: int,
    timesteps: int,
    out_path: Path,
    device: torch.device,
):
    """
    Genera UNA máscara desde ruido y la guarda como PNG
    """
    z = torch.randn(1, latent_dim, device=device)

    for t in reversed(range(timesteps)):
        tt = torch.tensor([t], device=device)
        eps_hat = denoiser(z, tt)
        z = scheduler.step(z, eps_hat, t)

    logits = vae.decode(z)[0].cpu()
    img = mask_logits_to_rgb(logits)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

# ============================================================
# TRAIN LATENT DIFFUSION
# ============================================================
def train_ldm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_t = get_mask_transforms(args.image_size, NUM_CLASSES, args.mask_values)
    ds = MaskDataset(args.mask_dir, transforms=mask_t)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # ----- cargar VAE (freeze) -----
    vae = MaskVAE(args.latent).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ----- diffusion -----
    denoiser = LatentDenoiser(args.latent).to(device)
    scheduler = LatentScheduler(args.timesteps, device=device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=args.lr)

    out = Path(args.outdir)
    sample_dir = out / "samples"
    out.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for ep in range(args.epochs):
        for batch in dl:
            batch = batch.to(device)

            with torch.no_grad():
                mu, logvar = vae.encode(batch)
                z0 = vae.reparam(mu, logvar)

            t = torch.randint(0, args.timesteps, (z0.size(0),), device=device)
            zt, eps = scheduler.add_noise(z0, t)
            eps_hat = denoiser(zt, t)

            loss = F.mse_loss(eps_hat, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # ---------- LOG ----------
            if step % 100 == 0:
                print(f"[LDM] step={step} loss={loss.item():.4f}")

            # ---------- SAMPLE ----------
            if step % args.sample_every == 0 and step > 0:
                sample_path = sample_dir / f"sample_step_{step}.png"
                sample_from_ldm(
                    vae=vae,
                    denoiser=denoiser,
                    scheduler=scheduler,
                    latent_dim=args.latent,
                    timesteps=args.timesteps,
                    out_path=sample_path,
                    device=device,
                )
                print(f"🖼️  sample guardado en {sample_path}")

            step += 1

        torch.save({"denoiser": denoiser.state_dict()}, out / "ldm_last.pt")

    print("✅ train_ldm terminado")

# ============================================================
# SAMPLE
# ============================================================
@torch.no_grad()
def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = MaskVAE(args.latent).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["vae"])
    vae.eval()

    denoiser = LatentDenoiser(args.latent).to(device)
    denoiser.load_state_dict(torch.load(args.ldm_ckpt, map_location=device)["denoiser"])
    denoiser.eval()

    sched = LatentScheduler(args.timesteps)
    z = torch.randn(args.n, args.latent, device=device)

    for t in reversed(range(args.timesteps)):
        tt = torch.full((args.n,), t, device=device, dtype=torch.long)
        z = sched.step(z, denoiser(z, tt), t)

    logits = vae.decode(z)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        img = mask_logits_to_rgb(logits[i].cpu())
        img.save(out.with_stem(out.stem + f"_{i}"))

    print("✅ máscaras generadas")


# ============================================================
# CLI
# ============================================================
def build_parser():
    p = argparse.ArgumentParser("Latent diffusion masks (palette fixed)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train_vae")
    p1.add_argument("--mask-dir", required=True)
    p1.add_argument("--outdir", default="ckpts_vae")
    p1.add_argument("--epochs", type=int, default=50)
    p1.add_argument("--batch", type=int, default=8)
    p1.add_argument("--lr", type=float, default=1e-4)
    p1.add_argument("--beta", type=float, default=0.01)
    p1.add_argument("--latent", type=int, default=256)
    p1.add_argument("--image-size", type=int, default=256)
    p1.add_argument("--mask-values", type=str, default=None)

    p2 = sub.add_parser("train_ldm")
    p2.add_argument("--mask-dir", required=True)
    p2.add_argument("--vae-ckpt", required=True)
    p2.add_argument("--outdir", default="ckpts_ldm")
    p2.add_argument("--epochs", type=int, default=50)
    p2.add_argument("--batch", type=int, default=64)
    p2.add_argument("--lr", type=float, default=1e-4)
    p2.add_argument("--latent", type=int, default=256)
    p2.add_argument("--timesteps", type=int, default=1000)
    p2.add_argument("--image-size", type=int, default=256)
    p2.add_argument("--mask-values", type=str, default=None)
    p2.add_argument("--sample-every", type=int, default=2000,
                help="Guardar samples cada N steps durante train_ldm")


    p3 = sub.add_parser("sample")
    p3.add_argument("--vae-ckpt", required=True)
    p3.add_argument("--ldm-ckpt", required=True)
    p3.add_argument("--out", default="generated/mask.png")
    p3.add_argument("--n", type=int, default=1)
    p3.add_argument("--latent", type=int, default=256)
    p3.add_argument("--timesteps", type=int, default=1000)

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "train_vae":
        train_vae(args)
    elif args.cmd == "train_ldm":
        train_ldm(args)
    elif args.cmd == "sample":
        sample(args)


if __name__ == "__main__":
    main()
