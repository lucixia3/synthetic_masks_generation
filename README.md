# 🧠 Anatomy-Preserving Latent Diffusion for Generation of Brain Segmentation Masks with Ischemic Infarct

> **📄 Submitted to CBMS 2026** · [arXiv:2602.10167](https://arxiv.org/abs/2602.10167)

A generative framework for synthesizing anatomically coherent **multi-class brain segmentation masks** — including ischemic infarcts — using a VAE + Latent Diffusion Model pipeline trained exclusively on masks, with no CT images required.

---

## 🔬 Abstract

Manual annotation of segmentation masks in non-contrast CT (NCCT) neuroimaging is costly, time-consuming, and highly variable across annotators. To address this data scarcity bottleneck, we propose an **anatomy-preserving generative framework** for the unconditional synthesis of multi-class brain segmentation masks, including ischemic infarcts.

The approach combines:
- A **Variational Autoencoder (VAE)** trained exclusively on segmentation masks to learn a compact anatomical latent representation
- A **Latent Diffusion Model (LDM)** that generates new samples from pure Gaussian noise within that latent space

At inference, synthetic masks are decoded from denoised latent vectors through the frozen VAE decoder, with optional coarse control over **lesion presence via a binary prompt**. Generated masks preserve global brain anatomy, discrete tissue semantics, and realistic variability — while avoiding the structural artifacts typically seen in pixel-space generative models.

---

## ✨ Key Contributions

- **Latent-space diffusion over masks** — by decoupling anatomical structure learning (VAE) from stochastic generation (LDM), the framework avoids pixel-space artifacts and reduces computational cost
- **Anatomy as a prior** — the VAE implicitly defines the space of anatomically valid configurations through categorical reconstruction and latent regularization
- **Lesion-conditioned sampling** — a binary prompt controls whether generated masks include an ischemic infarct, with automatic retry logic to ensure lesion label presence
- **No paired data needed** — inference requires only the trained checkpoints, no CT images or conditioning inputs

---

## 🏷️ Segmentation Classes

| ID | Class |
|----|-------|
| 0 | Background |
| 1 | Soft tissue |
| 2 | Bone |
| 3 | CSF |
| 4 | White matter |
| 5 | Gray matter |
| 6 | Infarct (lesion) |

---

## 📦 Resources

| Resource | Link |
|----------|------|
| 📄 Paper | [arXiv:2602.10167](https://arxiv.org/abs/2602.10167) |
| 🤗 Generated dataset | [IISM\_brain on Hugging Face](https://huggingface.co/datasets/lborrego/IISM_brain) |
| 🔬 Pretrained checkpoints (VAE + LDM) | [Google Drive](https://drive.google.com/drive/folders/1m-xTYYYOmgO_v1_betD7p-hkR6Z6Rkfq?usp=drive_link) |

---

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: Scripts\activate
pip install -r requirements.txt
```

Create checkpoint directories before downloading weights:

```bash
mkdir ckpts_vae ckpts_ldm
```

> **Windows users:** append `--num-workers 0` to all commands to avoid dataloader pickling issues.

---

## 🚀 Pipeline

The full pipeline runs in three stages via `main.py`.

### 1 — Train the mask VAE

Learns a compact latent manifold encoding global anatomical structure and inter-tissue relationships.

```bash
python main.py train_vae \
  --mask-dir data/masks \
  --epochs 50 \
  --batch 8 \
  --image-size 256 \
  --outdir ckpts_vae \
  --num-workers 0
```

**Output:** `ckpts_vae/vae_last.pt`

---

### 2 — Train the Latent Diffusion Model

Trains a denoising diffusion model in the VAE's latent space, conditioned on lesion presence.

```bash
python main.py train_ldm \
  --mask-dir data/masks \
  --vae-ckpt ckpts_vae/vae_last.pt \
  --epochs 50 \
  --batch 16 \
  --timesteps 100 \
  --outdir ckpts_ldm \
  --sample-every 500 \
  --num-workers 0
```

**Output:** `ckpts_ldm/ldm_last.pt` + periodic samples in `ckpts_ldm/samples/` (if `--sample-every > 0`)

---

### 3 — Generate Synthetic Masks

Sample new masks from pure Gaussian noise, optionally conditioned on lesion presence.

```bash
python main.py sample \
  --vae-ckpt ckpts_vae/vae_last.pt \
  --ldm-ckpt ckpts_ldm/ldm_last.pt \
  --prompt lesion_visible \
  --n 4 \
  --out generated/mask.png \
  --timesteps 100
```

| Prompt | Description |
|--------|-------------|
| `lesion_visible` | Includes ischemic infarct label (auto-retries up to 15× per sample) |
| `no_lesion` | Generates healthy brain masks without lesion |

**Output:** `generated/mask_{prompt}_{i}.png` — colored PNG files, one per sample.

---

## 🗂️ Project Structure

```
synthetic_masks_generation/
├── main.py                        # Entry point: train_vae / train_ldm / sample
├── train.py                       # Training loops
├── infer_diffusion.py             # Sampling & denoising logic
├── distribution_synth_masks.py    # Class distribution analysis
├── requirements.txt
├── ckpts_vae/                     # VAE checkpoints
├── ckpts_ldm/                     # LDM checkpoints
└── generated/                     # Synthetic mask outputs
```

---

## 🔧 Configuration Notes

- `NUM_CLASSES=7` and `LESION_ID=6` are fixed in `main.py` — adjust if your label schema differs
- Tune `--timesteps`, `--latent`, and `--prompt-dim` to scale model capacity up or down
- See the full argument parser in `main.py` for all available hyperparameters

---

## 📄 Citation

If you use this code or the generated dataset, please cite:

```bibtex
@article{synthetic_masks_cbms2026,
  title   = {Anatomy-Preserving Latent Diffusion for Generation of Brain Segmentation Masks with Ischemic Infarct},
  journal = {CBMS 2026},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.10167}
}
```

---

## 📜 License

See `LICENSE` for details.
