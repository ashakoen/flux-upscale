# flux-upscale

Identity-preserving 2x / 3x upscale for Flux-generated portraits that used a subject LoRA.

Two-stage CLI pipeline:

1. **Stage 1 — Base upscale.** Real-ESRGAN or SwinIR (non-generative, no hallucination).
2. **Stage 2 — Flux img2img refinement.** Low denoise + the same subject LoRA at reduced strength to keep identity stable.

Designed for an H100 / H200 / B200 RunPod pod. No memory-saver tricks; full bf16 Flux.1-dev.

See [`BUILD_PLAN.md`](BUILD_PLAN.md) for the design rationale.

---

## Quick start on RunPod

Spin up a pod with a PyTorch 2.x + CUDA 12.x image (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) and an attached network volume mounted at `/workspace`. Then:

```bash
# 1. Clone into /workspace so the cache survives pod restarts.
cd /workspace
git clone <your-repo-url> flux-upscale
cd flux-upscale

# 2. Install Python deps. Torch is preinstalled in the RunPod image — don't
#    reinstall it from requirements.txt.
pip install -r requirements.txt

# 3. Export your HF token (Flux.1-dev is gated).
export HF_TOKEN=hf_xxx

# 4. (Optional) pre-warm the model cache. Otherwise first run downloads on demand.
python scripts/download_models.py --flux

# 5. Run.
python upscale.py \
  --input  /workspace/in/photo.png \
  --output /workspace/out/photo_2x.png \
  --lora   /workspace/loras/subject.safetensors \
  --preset balanced \
  --scale  2
```

The model cache lives under `/workspace/.cache/flux-upscale/` (and `~/.cache/huggingface/` for Flux), so it persists across pod boots if the volume is reused.

---

## CLI

```
python upscale.py \
  --input  INPUT.png \
  --output OUTPUT.png \
  --lora   subject.safetensors \
  [--preset {identity,balanced,detail}]    # default: balanced
  [--scale {2,3}]                          # default: 2
  [--denoise FLOAT]                        # override preset
  [--lora-scale FLOAT]                     # override preset
  [--base-upscaler {realesrgan,swinir}]    # override preset
  [--guidance-scale FLOAT]                 # override preset
  [--steps INT]                            # override preset
  [--prompt "trigger_word, restraint phrases"]
  [--seed INT]
  [--no-refine]                            # Stage 1 only
```

`--preset` sets all the knobs at once; individual flags override preset values.

`--prompt` is optional. If omitted, the default restraint prompt is used (see `pipeline/flux_refine.py`). If your LoRA needs a trigger token, pass it via `--prompt`.

---

## Presets

All three presets share the same two-stage pipeline; only the parameters differ. Numbers are **starting points** to be tuned per-LoRA.

| Preset      | Base upscaler | Denoise | LoRA scale | Guidance | Steps | When to use |
|-------------|---------------|---------|------------|----------|-------|-------------|
| `identity`  | SwinIR        | 0.10    | 0.30       | 2.5      | 28    | Likeness is the only thing that matters. |
| `balanced`  | Real-ESRGAN   | 0.15    | 0.45       | 3.5      | 28    | Default. Good detail without identity risk. |
| `detail`    | Real-ESRGAN   | 0.22    | 0.60       | 3.5      | 32    | Softer sources that need more help. Verify identity. |

---

## Layout

```
flux-upscale/
├── README.md                # this file
├── BUILD_PLAN.md            # design doc / assumptions
├── requirements.txt
├── upscale.py               # CLI entrypoint
├── pipeline/
│   ├── __init__.py
│   ├── presets.py
│   ├── base_upscale.py      # Real-ESRGAN / SwinIR via spandrel
│   └── flux_refine.py       # Flux img2img + subject LoRA
└── scripts/
    └── download_models.py   # pre-fetch upscaler / Flux weights
```

---

## What's not in v1

- Face-only refinement pass (insightface detect → crop → refine → composite).
- ArcFace identity-drift metrics for A/B testing.
- Batch processing.
- API server.

Each of these is straightforward to add once v1 parameters are validated. See `BUILD_PLAN.md` § "Known Unknowns".
