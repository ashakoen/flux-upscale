"""Stage 2 of the pipeline: low-denoise Flux img2img refinement with subject LoRA.

Refinement is *tiled* by default: the upscaled image is split into ~1 MP tiles,
each refined at Flux's native resolution, then feather-blended back together.
This keeps every tile inside Flux.1-dev's training resolution (~1 MP) — refining
the whole multi-megapixel image in a single pass pushes Flux far out of
distribution, which produces global softness and identity drift even at low
denoise. Pass tile=0 to force the legacy whole-image pass (for comparison).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
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


def _tile_starts(length: int, tile: int, overlap: int, grid: int = _FLUX_VAE_FACTOR):
    """Plan tile placements along one axis.

    Returns a list of (start, size) covering [0, length]. Every size is a
    multiple of `grid`. For axes longer than `tile`, tiles are full `tile`-sized
    and evenly placed (the last one is shifted back to stay in-bounds, so the
    actual overlap may exceed the requested minimum). Shorter axes get a single
    grid-cropped tile.
    """
    tile = (tile // grid) * grid
    if length <= tile:
        return [(0, (length // grid) * grid)]
    stride = max(grid, tile - overlap)
    n = math.ceil((length - tile) / stride) + 1
    starts = []
    seen = set()
    for i in range(n):
        s = min(i * stride, length - tile)
        if s not in seen:
            seen.add(s)
            starts.append((s, tile))
    return starts


def _feather_mask(h: int, w: int, overlap: int) -> np.ndarray:
    """Separable ramp window: 1.0 in the interior, ramping toward (but never
    reaching) 0 over `overlap` px at each edge. Used as the blend weight so
    overlapping tiles cross-fade with no visible seams; the minimum weight stays
    > 0 so the normalising sum is never zero."""
    def ramp(n: int) -> np.ndarray:
        m = np.ones(n, dtype=np.float32)
        f = min(overlap, n // 2)
        if f > 0:
            r = (np.arange(f, dtype=np.float32) + 1.0) / (f + 1.0)  # in (0, 1]
            m[:f] = r
            m[-f:] = r[::-1]
        return m
    return np.outer(ramp(h), ramp(w))


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

    def _refine_one(self, img, prompt, denoise, steps, guidance_scale, generator):
        """Run a single Flux img2img pass over a grid-aligned image."""
        w, h = img.size
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

    def _gen(self, seed, offset):
        if seed is None:
            return None
        return torch.Generator(device=self.pipe.device).manual_seed(seed + offset)

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        denoise: float,
        steps: int,
        guidance_scale: float,
        seed: int | None = None,
        tile: int = 1024,
        tile_overlap: int = 128,
    ) -> Image.Image:
        img = image.convert("RGB")

        # Whole-image (legacy) path: tiling disabled, or image already fits a tile.
        if tile <= 0 or (img.width <= tile and img.height <= tile):
            return self._refine_one(
                _crop_to_grid(img), prompt, denoise, steps, guidance_scale,
                self._gen(seed, 0),
            )

        # Tiled path: refine each ~1 MP tile at native resolution, feather-blend.
        grid = _FLUX_VAE_FACTOR
        xs = _tile_starts(img.width, tile, tile_overlap, grid)
        ys = _tile_starts(img.height, tile, tile_overlap, grid)
        out_w = max(s + sz for s, sz in xs)
        out_h = max(s + sz for s, sz in ys)

        acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
        wsum = np.zeros((out_h, out_w, 1), dtype=np.float32)

        total = len(xs) * len(ys)
        idx = 0
        for ty, th in ys:
            for tx, tw in xs:
                crop = img.crop((tx, ty, tx + tw, ty + th))
                refined = self._refine_one(
                    crop, prompt, denoise, steps, guidance_scale,
                    # vary noise per tile so identical patterns don't repeat,
                    # while staying deterministic for a given seed.
                    self._gen(seed, idx),
                )
                arr = np.asarray(refined, dtype=np.float32)
                mask = _feather_mask(th, tw, tile_overlap)[..., None]
                acc[ty:ty + th, tx:tx + tw] += arr * mask
                wsum[ty:ty + th, tx:tx + tw] += mask
                idx += 1
                print(f"[upscale]   tile {idx}/{total} "
                      f"({tw}x{th} @ {tx},{ty})", flush=True)

        blended = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)
