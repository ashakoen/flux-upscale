#!/usr/bin/env python
"""Pre-fetch upscaler weights and (optionally) the Flux.1-dev model.

Run on a fresh pod to warm the cache before the first interactive upscale:

    python scripts/download_models.py            # upscalers only (no HF token needed)
    python scripts/download_models.py --flux     # also download Flux.1-dev (~24 GB)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running this script from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.base_upscale import _MODELS, _ensure_weight, _DEFAULT_CACHE


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flux", action="store_true",
                   help="Also download black-forest-labs/FLUX.1-dev (gated, ~24GB).")
    p.add_argument(
        "--only", choices=sorted(_MODELS), default=None,
        help="Only download this base upscaler (default: all).",
    )
    args = p.parse_args()

    cache = _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    print(f"[download] cache directory: {cache}")

    kinds = [args.only] if args.only else list(_MODELS)
    for kind in kinds:
        path = _ensure_weight(kind, cache)
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"[download] {kind}: {path} ({size_mb:.1f} MB)")

    if args.flux:
        from huggingface_hub import snapshot_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token is None:
            print("HF_TOKEN required for --flux (Flux.1-dev is gated).",
                  file=sys.stderr)
            return 2

        print("[download] fetching black-forest-labs/FLUX.1-dev (~24 GB)...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            token=token,
        )
        print("[download] Flux.1-dev ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
