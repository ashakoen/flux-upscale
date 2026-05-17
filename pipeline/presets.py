"""Named parameter presets for the upscale pipeline.

Values are starting points — meant to be tuned per-LoRA via the A/B/C grid
described in BUILD_PLAN.md, not validated targets.
"""
from dataclasses import dataclass, asdict
from typing import Literal

BaseUpscaler = Literal["realesrgan", "swinir"]


@dataclass
class Preset:
    name: str
    base_upscaler: BaseUpscaler
    denoise: float
    lora_scale: float
    guidance_scale: float
    steps: int


PRESETS: dict[str, Preset] = {
    "identity": Preset(
        name="identity",
        base_upscaler="swinir",
        denoise=0.10,
        lora_scale=0.30,
        guidance_scale=2.5,
        steps=28,
    ),
    "balanced": Preset(
        name="balanced",
        base_upscaler="realesrgan",
        denoise=0.15,
        lora_scale=0.45,
        guidance_scale=3.5,
        steps=28,
    ),
    "detail": Preset(
        name="detail",
        base_upscaler="realesrgan",
        denoise=0.22,
        lora_scale=0.60,
        guidance_scale=3.5,
        steps=32,
    ),
}


def apply_preset(preset_name: str, overrides: dict) -> dict:
    """Resolve a preset by name, then apply any non-None CLI overrides on top."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset {preset_name!r}. Known: {sorted(PRESETS)}"
        )
    params = asdict(PRESETS[preset_name])
    for k, v in overrides.items():
        if v is not None:
            params[k] = v
    return params
