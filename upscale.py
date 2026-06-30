#!/usr/bin/env python
"""flux-upscale CLI — quality, identity-preserving 2x/3x upscale of Flux outputs.

Stage 1: conservative non-generative upscale (Real-ESRGAN or SwinIR).
Stage 2: low-denoise Flux img2img refinement with the subject LoRA at reduced
         strength to keep identity stable.

Designed for H100/H200/B200 RunPod pods — no memory-saver tricks.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Only the lightweight preset module is imported at module top so that
# `python upscale.py --help` works before `pip install -r requirements.txt`.
from pipeline.presets import PRESETS, apply_preset


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Identity-preserving 2x/3x upscale for Flux+LoRA images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="Source image path.")
    p.add_argument("--output", required=True, type=Path, help="Destination image path.")
    p.add_argument(
        "--lora", type=Path, default=None,
        help="Subject LoRA .safetensors (ai-toolkit / diffusers-keyed). "
             "Required unless --no-refine is set.",
    )
    p.add_argument(
        "--preset", choices=sorted(PRESETS), default="balanced",
        help="Named preset — sets all the knobs; individual flags override.",
    )
    p.add_argument("--scale", type=int, choices=[2, 3], default=2,
                   help="Final upscale factor relative to the input size.")
    # Per-flag overrides — left at None so the preset value wins by default.
    p.add_argument("--denoise", type=float, default=None,
                   help="Override preset denoise (strength) for Stage 2.")
    p.add_argument("--lora-scale", type=float, default=None,
                   help="Override preset LoRA scale for Stage 2.")
    p.add_argument("--base-upscaler", choices=["realesrgan", "swinir"], default=None,
                   help="Override preset base upscaler (Stage 1).")
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="Override preset guidance scale for Stage 2.")
    p.add_argument("--steps", type=int, default=None,
                   help="Override preset diffusion steps.")
    p.add_argument(
        "--prompt", type=str, default=None,
        help=(
            "Refinement prompt. If omitted, uses the default restraint prompt. "
            "Pass your LoRA trigger token here if your LoRA requires one."
        ),
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    p.add_argument("--tile", type=int, default=1024,
                   help="Stage 2 refines in tiles of this size (px) at Flux's "
                        "native resolution, then feather-blends. 0 = legacy "
                        "single whole-image pass (blurs/drifts at high res).")
    p.add_argument("--tile-overlap", type=int, default=128,
                   help="Overlap (px) between refine tiles; blended to hide seams.")
    p.add_argument("--no-refine", action="store_true",
                   help="Run only Stage 1 (base upscale). Skips Flux refinement.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    if not args.no_refine:
        if args.lora is None:
            print("--lora is required unless --no-refine is set.", file=sys.stderr)
            return 2
        if not args.lora.exists():
            print(f"LoRA file not found: {args.lora}", file=sys.stderr)
            return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Deferred imports — these pull in numpy/torch/diffusers/spandrel/etc.
    from PIL import Image
    from pipeline.base_upscale import BaseUpscaler
    from pipeline.flux_refine import DEFAULT_PROMPT, FluxRefiner

    params = apply_preset(args.preset, {
        "base_upscaler": args.base_upscaler,
        "denoise": args.denoise,
        "lora_scale": args.lora_scale,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
    })
    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT

    print(f"[upscale] preset={args.preset} resolved params:")
    for k, v in params.items():
        print(f"           {k} = {v}")
    print(f"[upscale] scale={args.scale}x  prompt={'<custom>' if args.prompt else '<default>'}")

    src = Image.open(args.input)
    print(f"[upscale] input: {args.input} ({src.width}x{src.height})")

    # --- Stage 1: base upscale -------------------------------------------------
    t0 = time.time()
    base = BaseUpscaler(kind=params["base_upscaler"])
    upscaled = base(src, target_scale=args.scale)
    t1 = time.time()
    print(f"[upscale] stage 1 ({params['base_upscaler']}) done in {t1 - t0:.1f}s → "
          f"{upscaled.width}x{upscaled.height}")

    # Free the upscaler before loading Flux — they don't need to coexist.
    del base
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.no_refine:
        upscaled.save(args.output)
        print(f"[upscale] saved (Stage 1 only): {args.output}")
        return 0

    # --- Stage 2: Flux refinement ---------------------------------------------
    refiner = FluxRefiner(lora_path=args.lora, lora_scale=params["lora_scale"])
    t2 = time.time()
    final = refiner(
        image=upscaled,
        prompt=prompt,
        denoise=params["denoise"],
        steps=params["steps"],
        guidance_scale=params["guidance_scale"],
        seed=args.seed,
        tile=args.tile,
        tile_overlap=args.tile_overlap,
    )
    t3 = time.time()
    print(f"[upscale] stage 2 (Flux refine) done in {t3 - t2:.1f}s → "
          f"{final.width}x{final.height}")

    final.save(args.output)
    print(f"[upscale] saved: {args.output}  (total {t3 - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
