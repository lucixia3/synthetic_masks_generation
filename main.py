import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import MaskDataset

# DATASET CONFIG (ESCALA DE GRISES)
NUM_CLASSES = 7
LESION_ID = 6   # INFARTO

PALETTE = [
    [0, 0, 0],          # Background
    [140, 128, 115],    # Soft tissue
    [255, 255, 255],    # Bone
    [64, 115, 140],     # CSF
    [217, 217, 217],    # White matter
    [153, 153, 153],    # Gray matter
    [179, 38, 38],      # Infarct
]


# PROMPTS
PROMPTS: Dict[str, int] = {
    "no_lesion": 0,
    "lesion_visible": 1,
}
N_PROMPTS = len(PROMPTS)
PROMPT_DIM_DEFAULT = 16
MAX_RESAMPLES_LESION = 15

def to_one_hot(mask_hw: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = F.one_hot(mask_hw.long(), num_classes=num_classes)
    return oh.permute(2, 0, 1).float()




class SqueezeToLong:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1,H,W]
        return x.squeeze(0).long()

class ToOneHot:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return to_one_hot(x, self.n_classes)

def get_mask_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.PILToTensor(),   # [1,H,W]
        SqueezeToLong(),            # [H,W]
        ToOneHot(NUM_CLASSES),      # [C,H,W]
    ])



def mask_logits_to_rgb(mask_logits: torch.Tensor) -> Image.Image:
    ids = torch.argmax(mask_logits, dim=0).cpu().numpy()
    h, w = ids.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, col in enumerate(PALETTE):
        rgb[ids == i] = col
    return Image.fromarray(rgb, mode="RGB")

# VAE
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

# PROMPT + diffu

class PromptEncoder(nn.Module):
    def __init__(self, n_prompts: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_prompts, dim)

    def forward(self, ids):
        return self.emb(ids)


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim=256, prompt_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1 + prompt_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, z, t, p):
        # t: [B] -> [B,1] normalizado
        t = t.float().unsqueeze(1) / 1000.0
        return self.net(torch.cat([z, t, p], dim=1))


class LatentScheduler:
    def __init__(self, T=1000, device="cpu"):
        self.betas = torch.linspace(1e-4, 0.02, T, device=device)
        self.alphas = 1 - self.betas
        self.ab = torch.cumprod(self.alphas, 0)

    def add_noise(self, z0, t):
        eps = torch.randn_like(z0)
        a = self.ab[t].unsqueeze(1)
        return torch.sqrt(a) * z0 + torch.sqrt(1 - a) * eps, eps

    def step(self, zt, eps_hat, t_int: int):
        beta = self.betas[t_int]
        alpha = 1 - beta
        ab = self.ab[t_int]
        return (zt - beta / torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(alpha)

# ============================================================
# TRAIN VAE
# ============================================================
def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = MaskDataset(args.mask_dir, transforms=get_mask_transforms(args.image_size))

    # --------------------------------------------------
    # build sampler to rebalance lesion / no-lesion
    # --------------------------------------------------
    weights = []
    for i in range(len(ds)):
        mask_oh = ds[i]                    # [C,H,W]
        ids = torch.argmax(mask_oh, dim=0) # [H,W]
        has_lesion = (ids == LESION_ID).any().item()
        weights.append(5.0 if has_lesion else 1.0)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=sampler,         
        shuffle=False,            
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # sanity check
    x0 = ds[0]
    print("Sanity check clases:", torch.unique(torch.argmax(x0, dim=0)).tolist())

    vae = MaskVAE(args.latent).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        vae.train()
        last_loss = None

        for i, batch in enumerate(dl):
            batch = batch.to(device, non_blocking=True)
            logits, mu, logvar = vae(batch)
            loss = vae_loss(batch, logits, mu, logvar, args.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = loss.item()

            if i % args.log_every == 0:
                print(f"[VAE] epoch {ep+1}/{args.epochs} | step {i} | loss={loss.item():.4f}")

        torch.save({"vae": vae.state_dict()}, out / "vae_last.pt")
        print(f"[VAE] ✅ guardado vae_last.pt | epoch {ep+1}/{args.epochs} | last_loss={last_loss:.4f}")

#sampling during training

@torch.no_grad()
def sample_one(
    vae: MaskVAE,
    denoiser: LatentDenoiser,
    prompt_encoder: PromptEncoder,
    sched: LatentScheduler,
    latent_dim: int,
    timesteps: int,
    prompt_id: int,
    device: torch.device,
):
    z = torch.randn(1, latent_dim, device=device)
    p = prompt_encoder(torch.tensor([prompt_id], device=device))  # [1, prompt_dim]

    for t in reversed(range(timesteps)):
        tt = torch.tensor([t], device=device, dtype=torch.long)
        eps_hat = denoiser(z, tt, p)
        z = sched.step(z, eps_hat, t)

    logits = vae.decode(z)[0].cpu()  # [C,H,W]
    return mask_logits_to_rgb(logits)

# TRAIN LDM
def train_ldm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = MaskDataset(args.mask_dir, transforms=get_mask_transforms(args.image_size))
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- load VAE (freeze) ----
    vae = MaskVAE(args.latent).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ---- diffusion ----
    prompt_encoder = PromptEncoder(N_PROMPTS, args.prompt_dim).to(device)
    denoiser = LatentDenoiser(args.latent, args.prompt_dim).to(device)
    scheduler = LatentScheduler(args.timesteps, device=device)

    opt = torch.optim.AdamW(
        list(prompt_encoder.parameters()) + list(denoiser.parameters()),
        lr=args.lr
    )

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sample_dir = out / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for ep in range(args.epochs):
        denoiser.train()
        prompt_encoder.train()
        last_loss = None

        for i, batch in enumerate(dl):
            batch = batch.to(device, non_blocking=True)

            with torch.no_grad():
                ids = torch.argmax(batch, dim=1)  # [B,H,W]
                has_lesion = (ids == LESION_ID).any(dim=(1, 2)).long()  # [B]
                mu, logvar = vae.encode(batch)
                z0 = vae.reparam(mu, logvar)

            t = torch.randint(0, args.timesteps, (z0.size(0),), device=device)
            zt, eps = scheduler.add_noise(z0, t)

            p = prompt_encoder(has_lesion)              # [B, prompt_dim]
            eps_hat = denoiser(zt, t, p)                # [B, latent_dim]
            loss = F.mse_loss(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = loss.item()

            if global_step % args.log_every == 0:
                # stats rapida para debug
                frac_lesion = has_lesion.float().mean().item()
                print(f"[LDM] ep {ep+1}/{args.epochs} | step {global_step} | loss={last_loss:.4f} | frac_lesion_batch={frac_lesion:.2f}")

            
            if args.sample_every > 0 and global_step > 0 and (global_step % args.sample_every == 0):
                denoiser.eval()
                prompt_encoder.eval()
                img0 = sample_one(
                    vae, denoiser, prompt_encoder, scheduler,
                    latent_dim=args.latent,
                    timesteps=args.timesteps,
                    prompt_id=PROMPTS["no_lesion"],
                    device=device
                )
                img1 = sample_one(
                    vae, denoiser, prompt_encoder, scheduler,
                    latent_dim=args.latent,
                    timesteps=args.timesteps,
                    prompt_id=PROMPTS["lesion_visible"],
                    device=device
                )
                p0 = sample_dir / f"step_{global_step:07d}_no_lesion.png"
                p1 = sample_dir / f"step_{global_step:07d}_lesion_visible.png"
                img0.save(p0)
                img1.save(p1)
                print(f"🖼️ samples guardados: {p0.name}, {p1.name}")

                denoiser.train()
                prompt_encoder.train()

            global_step += 1

        torch.save(
            {"denoiser": denoiser.state_dict(), "prompt_encoder": prompt_encoder.state_dict()},
            out / "ldm_last.pt"
        )
        print(f"[LDM]  guardado ldm_last.pt | epoch {ep+1}/{args.epochs} | last_loss={last_loss:.4f}")

    print("train_ldm terminado")

# SAMPLE
@torch.no_grad()
def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    vae = MaskVAE(args.latent).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["vae"])
    vae.eval()

    ckpt = torch.load(args.ldm_ckpt, map_location=device)
    prompt_encoder = PromptEncoder(N_PROMPTS, args.prompt_dim).to(device)
    prompt_encoder.load_state_dict(ckpt["prompt_encoder"])
    prompt_encoder.eval()

    denoiser = LatentDenoiser(args.latent, args.prompt_dim).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()

    sched = LatentScheduler(args.timesteps, device=device)

    def run_diffusion(prompt_ids: torch.Tensor) -> torch.Tensor:
        z = torch.randn(prompt_ids.size(0), args.latent, device=device)
        p = prompt_encoder(prompt_ids)
        for t in reversed(range(args.timesteps)):
            tt = torch.full((prompt_ids.size(0),), t, device=device, dtype=torch.long)
            eps_hat = denoiser(z, tt, p)
            z = sched.step(z, eps_hat, t)
        return vae.decode(z)

    prompt_ids = torch.full((args.n,), PROMPTS[args.prompt], device=device, dtype=torch.long)
    logits = run_diffusion(prompt_ids)

    if args.prompt == "lesion_visible":
        for _ in range(MAX_RESAMPLES_LESION):
            ids = torch.argmax(logits, dim=1)
            has_lesion = (ids == LESION_ID).any(dim=(1, 2))
            if has_lesion.all():
                break
            missing = (~has_lesion).nonzero(as_tuple=False).flatten().tolist()
            if not missing:
                break
            for idx in missing:
                logits[idx] = run_diffusion(prompt_ids[idx:idx + 1])[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        img = mask_logits_to_rgb(logits[i].cpu())
        img.save(out.with_stem(out.stem + f"_{args.prompt}_{i}"))

    print("✅ máscaras generadas")

# CLI
def build_parser():
    p = argparse.ArgumentParser("Latent diffusion masks (GRAY labels)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train_vae")
    p1.add_argument("--mask-dir", required=True)
    p1.add_argument("--epochs", type=int, default=50)
    p1.add_argument("--batch", type=int, default=8)
    p1.add_argument("--lr", type=float, default=1e-4)
    p1.add_argument("--beta", type=float, default=0.01)
    p1.add_argument("--latent", type=int, default=256)
    p1.add_argument("--image-size", type=int, default=256)
    p1.add_argument("--outdir", default="ckpts_vae")
    p1.add_argument("--log-every", type=int, default=20)
    p1.add_argument("--num-workers", type=int, default=2)

    p2 = sub.add_parser("train_ldm")
    p2.add_argument("--mask-dir", required=True)
    p2.add_argument("--vae-ckpt", required=True)
    p2.add_argument("--epochs", type=int, default=50)
    p2.add_argument("--batch", type=int, default=16)
    p2.add_argument("--lr", type=float, default=1e-4)
    p2.add_argument("--latent", type=int, default=256)
    p2.add_argument("--timesteps", type=int, default=100)  # recomendado para masks
    p2.add_argument("--image-size", type=int, default=256)
    p2.add_argument("--prompt-dim", type=int, default=PROMPT_DIM_DEFAULT)
    p2.add_argument("--outdir", default="ckpts_ldm")
    p2.add_argument("--log-every", type=int, default=20)
    p2.add_argument("--sample-every", type=int, default=500, help="guardar samples cada N steps (0 desactiva)")
    p2.add_argument("--num-workers", type=int, default=2)

    p3 = sub.add_parser("sample")
    p3.add_argument("--vae-ckpt", required=True)
    p3.add_argument("--ldm-ckpt", required=True)
    p3.add_argument("--out", default="generated/mask.png")
    p3.add_argument("--n", type=int, default=4)
    p3.add_argument("--latent", type=int, default=256)
    p3.add_argument("--timesteps", type=int, default=100)
    p3.add_argument("--prompt-dim", type=int, default=PROMPT_DIM_DEFAULT)
    p3.add_argument("--prompt", choices=list(PROMPTS.keys()), default="lesion_visible")

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
