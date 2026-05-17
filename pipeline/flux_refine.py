"""Stage 2 of the pipeline: low-denoise Flux img2img refinement with subject LoRA."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image


DEFAULT_PROMPT = (
    "preserve exact identity, preserve facial structure, preserve expression, "
    "preserve composition, natural skin texture, realistic fine detail, "
    "no facial reshaping, no plastic skin, no beauty filter"
)

_FLUX_VAE_FACTOR = 16  # Flux VAE downscale is 8, plus 2x patch → 16-pixel grid.


def _crop_to_grid(img: Image.Image, grid: int = _FLUX_VAE_FACTOR) -> Image.Image:
    """Center-crop image dimensions down to a multiple of grid (16 for Flux)."""
    w, h = img.size
    new_w = (w // grid) * grid
    new_h = (h // grid) * grid
    if (new_w, new_h) == (w, h):
        return img
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


class FluxRefiner:
    """Wraps FluxImg2ImgPipeline + subject LoRA. Loaded once, called many times."""

    def __init__(
        self,
        lora_path: str | Path,
        lora_scale: float,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        enable_vae_tiling: bool = True,
    ):
        # Imported lazily so `--help` and the smoke test don't drag in diffusers.
        from diffusers import FluxImg2ImgPipeline

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token is None:
            raise RuntimeError(
                "HF_TOKEN env var is required to load Flux.1-dev (gated repo)."
            )

        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token,
        ).to(device)

        if enable_vae_tiling:
            # Cheap insurance for very large output sizes — no quality cost.
            self.pipe.vae.enable_tiling()

        self.adapter_name = "subject"
        self.pipe.load_lora_weights(
            str(lora_path), adapter_name=self.adapter_name
        )
        self.pipe.set_adapters([self.adapter_name], adapter_weights=[lora_scale])
        self.lora_scale = lora_scale

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        denoise: float,
        steps: int,
        guidance_scale: float,
        seed: int | None = None,
    ) -> Image.Image:
        img = _crop_to_grid(image.convert("RGB"))
        w, h = img.size
        generator = (
            torch.Generator(device=self.pipe.device).manual_seed(seed)
            if seed is not None else None
        )
        out = self.pipe(
            prompt=prompt,
            image=img,
            height=h,
            width=w,
            strength=denoise,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return out.images[0]
