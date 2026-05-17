"""Stage 1 of the pipeline: non-generative base upscale.

Uses spandrel to load either Real-ESRGAN x4plus or SwinIR Real x4 GAN weights,
runs the model at its native x4 scale, then resamples down to the requested
target scale (2 or 3).
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

# Default cache root: /workspace/.cache/flux-upscale on RunPod, ~/.cache otherwise.
_DEFAULT_CACHE = (
    Path(os.environ.get("FLUX_UPSCALE_CACHE", ""))
    or (Path("/workspace") / ".cache" / "flux-upscale"
        if Path("/workspace").exists()
        else Path.home() / ".cache" / "flux-upscale")
)

# Model weight URLs — pinned to releases that are known to load with spandrel.
_MODELS = {
    "realesrgan": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
    },
    "swinir": {
        # Real-world SR, large model, x4, GAN-trained. Best quality SwinIR variant.
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "filename": "SwinIR_realSR_x4_GAN_large.pth",
    },
}


def _ensure_weight(kind: str, cache_dir: Path) -> Path:
    if kind not in _MODELS:
        raise ValueError(f"Unknown upscaler {kind!r}. Known: {sorted(_MODELS)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / _MODELS[kind]["filename"]
    if not dest.exists():
        url = _MODELS[kind]["url"]
        print(f"[base_upscale] downloading {kind} weights from {url}")
        urlretrieve(url, dest)
    return dest


class BaseUpscaler:
    """Loads an x4 upscaler and runs it, downsampling to the requested scale."""

    def __init__(self, kind: str, device: str = "cuda", cache_dir: Path | None = None):
        self.kind = kind
        self.device = device
        cache = cache_dir or _DEFAULT_CACHE
        weight_path = _ensure_weight(kind, cache)
        loaded = ModelLoader().load_from_file(str(weight_path))
        if not isinstance(loaded, ImageModelDescriptor):
            raise RuntimeError(
                f"spandrel returned non-image model for {kind}: {type(loaded)}"
            )
        self.model: ImageModelDescriptor = loaded.eval().to(device)
        self.native_scale = self.model.scale  # 4 for both bundled models

    @torch.no_grad()
    def __call__(self, image: Image.Image, target_scale: int) -> Image.Image:
        if target_scale not in (2, 3, 4):
            raise ValueError(f"target_scale must be 2, 3, or 4; got {target_scale}")

        # PIL RGB → tensor [1, 3, H, W] in 0..1, on device.
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        out = self.model(tensor)  # → [1, 3, H*4, W*4]
        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))

        # If target_scale != native (4), resample down to target size.
        if target_scale != self.native_scale:
            target_w = rgb.width * target_scale
            target_h = rgb.height * target_scale
            out_img = out_img.resize((target_w, target_h), Image.LANCZOS)
        return out_img
