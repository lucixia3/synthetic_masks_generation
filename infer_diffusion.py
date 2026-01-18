from train import infer_diffusion_mask_rgb

infer_diffusion_mask_rgb(
    ckpt_path="checkpoints/diffusion/diffusion_step_10000.pt",
    out_path="generated_masks/mask_001.png",
    image_size=256,
)
