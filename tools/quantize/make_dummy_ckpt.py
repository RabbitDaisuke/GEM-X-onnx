#!/usr/bin/env python3
"""make_dummy_ckpt.py — test fixture generator (NOT for real models).

Creates a tiny dummy checkpoint in /tmp/gemx-quant-test/ with:
  * float32 weight tensors (should become float16)
  * an int64 tensor (must stay int64)
  * a bool tensor (must stay bool)
  * fake Adam optimizer state (exp_avg / exp_avg_sq — must be skipped)

Uses safetensors+numpy only, so it runs on machines without torch
(like this Linux VM).  If torch is available it also writes a .pt
version with the same contents plus a top-level {"state_dict": ...}
wrapper and a top-level "optimizer" dict, to exercise the torch path.
"""

from __future__ import annotations

import os
import sys

OUT_DIR = "/tmp/gemx-quant-test"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except ImportError as exc:
        print(f"ERROR: numpy + safetensors required: {exc}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(0)
    tensors = {
        "backbone.blocks.0.weight": rng.standard_normal((64, 64)).astype(np.float32),
        "backbone.blocks.0.bias": rng.standard_normal((64,)).astype(np.float32),
        "head.weight": rng.standard_normal((32, 64)).astype(np.float32),
        "head.scale": np.array([1.5, 2.5], dtype=np.float64),
        "meta.num_joints": np.array([77], dtype=np.int64),
        "meta.valid": np.array([True], dtype=np.bool_),
        # Fake Adam state — the quantizer must skip these.
        "backbone.blocks.0.weight.exp_avg": rng.standard_normal((64, 64)).astype(np.float32),
        "backbone.blocks.0.weight.exp_avg_sq": rng.standard_normal((64, 64)).astype(np.float32),
    }
    out = os.path.join(OUT_DIR, "dummy_fp32.safetensors")
    save_file(tensors, out, metadata={"format": "pt"})
    print(f"Wrote {out} ({os.path.getsize(out)} bytes, {len(tensors)} tensors)")

    try:
        import torch
    except ImportError:
        print("torch not installed — skipping .pt dummy (safetensors dummy is enough here).")
        return 0

    torch_tensors = {k: torch.from_numpy(v) for k, v in tensors.items()}
    ckpt = {
        "state_dict": torch_tensors,
        "optimizer": {"state": {0: {"exp_avg": torch.zeros(4),
                                     "exp_avg_sq": torch.zeros(4)}}},
        "epoch": 240000,
    }
    out_pt = os.path.join(OUT_DIR, "dummy_fp32.pt")
    torch.save(ckpt, out_pt)
    print(f"Wrote {out_pt} ({os.path.getsize(out_pt)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
