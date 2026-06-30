"""Named parameter presets for the upscale pipeline.

`portrait` and `fullbody` are the two content modes (pick with --preset). Each
bundles a content-appropriate Flux prompt plus the tuned knobs. The older
identity/balanced/detail presets are kept for manual experimentation.

Values for fullbody are a starting point — tune per-LoRA. The portrait values
were validated on a real subject + realism LoRA (denoise 0.40 + realism LoRA +
light unsharp held identity/teeth while adding skin detail).
"""
from dataclasses import dataclass, asdict
from typing import Literal

BaseUpscaler = Literal["realesrgan", "swinir"]


# Content-appropriate prompts. Note: a wrong prompt actively harms output (e.g.
# "skin pores" injected into a car), so each mode names only what it should add.
PORTRAIT_PROMPT = (
    "extreme close-up portrait photograph, visible skin pores, fine skin texture, "
    "natural skin detail, subsurface scattering, fabric texture, sharp focus, "
    "no plastic skin, no beauty filter, preserve exact identity and facial structure"
)
FULLBODY_PROMPT = (
    "full body photograph, natural skin texture, detailed clothing and fabric "
    "texture, realistic materials, fine environmental detail, sharp focus, "
    "photorealistic, no plastic skin, no beauty filter, "
    "preserve exact identity, body proportions and composition"
)
_GENERIC_PROMPT = (
    "high quality, highly detailed, sharp focus, photorealistic, natural texture, "
    "no plastic skin, no beauty filter, preserve identity and composition"
)


@dataclass
class Preset:
    name: str
    base_upscaler: BaseUpscaler
    denoise: float
    lora_scale: float
    guidance_scale: float
    steps: int
    prompt: str
    realism_scale: float = 0.0   # weight for an optional stacked realism LoRA
    sharpen: float = 0.0         # optical unsharp-mask amount applied at the end
    tile_overlap: int = 128


PRESETS: dict[str, Preset] = {
    # --- Content modes (the two intended entry points) ---------------------
    "portrait": Preset(
        name="portrait",
        base_upscaler="realesrgan",
        denoise=0.40,            # teeth/fine-structure safe; detail comes from realism LoRA + sharpen
        lora_scale=0.45,
        guidance_scale=2.8,
        steps=34,
        prompt=PORTRAIT_PROMPT,
        realism_scale=0.25,      # only active if --realism-lora is supplied
        sharpen=0.30,
        tile_overlap=64,
    ),
    "fullbody": Preset(
        name="fullbody",
        base_upscaler="realesrgan",
        denoise=0.38,            # starting point — tune
        lora_scale=0.45,
        guidance_scale=3.0,
        steps=34,
        prompt=FULLBODY_PROMPT,
        realism_scale=0.20,
        sharpen=0.30,
        tile_overlap=64,
    ),
    # --- Legacy manual-experiment presets ----------------------------------
    "identity": Preset(
        name="identity", base_upscaler="swinir", denoise=0.10,
        lora_scale=0.30, guidance_scale=2.5, steps=28, prompt=_GENERIC_PROMPT,
    ),
    "balanced": Preset(
        name="balanced", base_upscaler="realesrgan", denoise=0.15,
        lora_scale=0.45, guidance_scale=3.5, steps=28, prompt=_GENERIC_PROMPT,
    ),
    "detail": Preset(
        name="detail", base_upscaler="realesrgan", denoise=0.22,
        lora_scale=0.60, guidance_scale=3.5, steps=32, prompt=_GENERIC_PROMPT,
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
