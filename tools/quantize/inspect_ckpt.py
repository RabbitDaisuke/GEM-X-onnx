#!/usr/bin/env python3
"""inspect_ckpt.py — list what's inside a GEM-X checkpoint, CPU-only.

Prints every tensor's name, dtype, shape, element count and byte size,
classifies each entry as weight / optimizer-state / other, and estimates
the FP32 and FP16 totals.  Works without writing anything.

Supported inputs:
  * ``.safetensors`` — header is parsed with the standard library only
    (no torch needed); tensor data is never loaded.
  * ``.pt`` / ``.ckpt`` / ``.pth`` — needs ``torch`` (``torch.load``,
    ``weights_only=True`` first, with a warned fallback).

This is the first thing to run on ``egohand_s060000.ckpt`` on the
Windows PC: it answers, in seconds, whether the 10.66 GB file is
weights-only or also contains Adam optimizer state.

Usage:
    python tools\\quantize\\inspect_ckpt.py weights\\egohand_s060000.ckpt
    python tools\\quantize\\inspect_ckpt.py weights\\body_s240000.ckpt --top 30
    python tools\\quantize\\inspect_ckpt.py weights\\body_fp16.safetensors --json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

OPTIMIZER_HINTS = (
    "exp_avg", "exp_avg_sq", "optimizer", "param_groups", "scheduler",
    "momentum_buffer", "adam",
)

DTYPE_SIZES = {
    "float32": 4, "float64": 8, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
    "bool": 1, "F32": 4, "F64": 8, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def _classify(name: str) -> str:
    low = name.lower()
    if any(h in low for h in OPTIMIZER_HINTS) or low.startswith("state."):
        return "optimizer"
    if low in ("epoch", "global_step", "step", "iteration", "lr"):
        return "other"
    return "weight"


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024.0 or unit == "GiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.2f} GiB"


def inspect_safetensors(path: str):
    """Parse only the safetensors header (stdlib, no tensor data loaded)."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    entries = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta.get("dtype", "?")
        shape = tuple(int(d) for d in meta.get("shape", []))
        numel = 1
        for d in shape:
            numel *= d
        itemsize = DTYPE_SIZES.get(dtype, 4)
        entries.append({
            "name": name, "dtype": dtype, "shape": list(shape),
            "numel": numel, "bytes": numel * itemsize,
            "class": _classify(name),
        })
    return entries, header.get("__metadata__", {})


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}{k}." if prefix else f"{k}")
    else:
        yield (prefix.rstrip("."), obj)


def inspect_torch_checkpoint(path: str):
    """Load a .pt/.ckpt with torch and list its tensors."""
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required to inspect .pt/.ckpt files.\n"
              "  pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu", file=sys.stderr)
        raise SystemExit(2)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"weights_only load failed ({exc}). Retrying with "
              "weights_only=False — only for files you downloaded yourself.",
              file=sys.stderr)
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(obj, dict):
        print(f"Top level is {type(obj).__name__}, not a dict.", file=sys.stderr)
        return [], {}
    # Unwrap state_dict/model wrappers for reporting, but note it.
    meta = {}
    for wrapper in ("state_dict", "model", "net"):
        if wrapper in obj and isinstance(obj[wrapper], dict):
            meta["unwrapped"] = wrapper
            obj = obj[wrapper]
            break
    entries = []
    for name, value in _flatten(obj):
        if not hasattr(value, "dtype"):
            entries.append({"name": name, "dtype": type(value).__name__,
                            "shape": [], "numel": 0, "bytes": 0,
                            "class": "other"})
            continue
        dtype_name = str(value.dtype).replace("torch.", "")
        shape = tuple(int(d) for d in value.shape)
        numel = int(value.numel())
        itemsize = int(value.element_size())
        entries.append({"name": name, "dtype": dtype_name, "shape": list(shape),
                        "numel": numel, "bytes": numel * itemsize,
                        "class": _classify(name)})
    return entries, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Inspect a GEM-X checkpoint.")
    ap.add_argument("input", help="Path to .ckpt / .pt / .safetensors")
    ap.add_argument("--top", type=int, default=0,
                    help="Show only the N largest tensors (0 = all)")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = ap.parse_args(argv)

    path = args.input
    if not os.path.isfile(path):
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 2
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        entries, meta = inspect_safetensors(path)
    elif ext in (".pt", ".ckpt", ".pth", ".bin"):
        entries, meta = inspect_torch_checkpoint(path)
    else:
        print(f"ERROR: unsupported extension '{ext}'.", file=sys.stderr)
        return 2

    total = sum(e["bytes"] for e in entries)
    w_total = sum(e["bytes"] for e in entries if e["class"] == "weight")
    o_total = sum(e["bytes"] for e in entries if e["class"] == "optimizer")
    # FP16 estimate: float32/64 weights halve; everything else unchanged.
    # (safetensors reports dtypes as F32/F64, torch as float32/float64.)
    fp16_total = 0
    for e in entries:
        if e["class"] != "weight":
            continue
        fp16_total += e["bytes"] // 2 if e["dtype"] in (
            "float32", "float64", "F32", "F64") else e["bytes"]

    summary = {
        "file": path,
        "file_size_bytes": os.path.getsize(path),
        "entries": len(entries),
        "weight_tensors": sum(1 for e in entries if e["class"] == "weight"),
        "optimizer_entries": sum(1 for e in entries if e["class"] == "optimizer"),
        "other_entries": sum(1 for e in entries if e["class"] == "other"),
        "total_tensor_bytes": total,
        "weight_bytes": w_total,
        "optimizer_bytes": o_total,
        "estimated_fp16_weight_bytes": fp16_total,
        "metadata": meta,
    }
    if args.json:
        print(json.dumps({**summary, "tensors": entries}, indent=2))
        return 0

    print(f"File: {path}  ({_fmt_bytes(os.path.getsize(path))})")
    if meta:
        print(f"Metadata: {meta}")
    print(f"Entries: {summary['entries']}  "
          f"(weights {summary['weight_tensors']}, "
          f"optimizer {summary['optimizer_entries']}, "
          f"other {summary['other_entries']})")
    print(f"Tensor bytes total: {_fmt_bytes(total)}  "
          f"(weights {_fmt_bytes(w_total)}, optimizer {_fmt_bytes(o_total)})")
    print(f"Estimated FP16 weight size: {_fmt_bytes(fp16_total)}")
    if o_total > 0:
        print("NOTE: this file contains optimizer state. Run "
              "quantize_ckpt_to_fp16.py (default skips optimizer state) "
              "to get weights-only FP16.")
    print()
    print(f"{'name':<64} {'dtype':<9} {'shape':<28} {'bytes':>12}  class")
    print("-" * 130)
    shown = sorted(entries, key=lambda e: -e["bytes"])
    if args.top:
        shown = shown[: args.top]
    for e in shown:
        print(f"{e['name']:<64} {e['dtype']:<9} {str(e['shape']):<28} "
              f"{_fmt_bytes(e['bytes']):>12}  {e['class']}")
    if args.top and len(entries) > args.top:
        print(f"... and {len(entries) - args.top} more entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
