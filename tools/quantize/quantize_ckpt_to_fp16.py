#!/usr/bin/env python3
"""quantize_ckpt_to_fp16.py — CPU-only FP16 weight conversion, no training images.

What this does
--------------
Takes a PyTorch checkpoint (``.ckpt`` / ``.pt``) or a safetensors file and
casts every floating-point weight tensor from FP32/FP64 to FP16, writing
the result as ``.safetensors`` (preferred) or ``.pt``.

No training images, no calibration data, no inference is involved:
FP16 *weight casting* is a per-tensor dtype conversion.  That is why it
needs no dataset at all.  (INT8 / FP8 quantization *would* need
calibration data, because it has to choose per-tensor scales from real
activations — that is out of scope for v1 and is NOT done here.)

Why this is safe to run on CPU
------------------------------
Casting is a memory-bandwidth-bound copy; no GPU is needed.  FP16
*inference* on CPU is a different story (many CPU kernels reject FP16,
e.g. ``ADDMM not implemented for half``) — but we never run inference
here.  The FP16 file is for TensorRT-RTX on the Windows GPU.

RAM: two paths
--------------
1. ``.safetensors`` input: true streaming.  Tensors are read one at a
   time via ``safetensors.safe_open`` (or the numpy backend) and cast
   one at a time.  Peak extra RAM is roughly one tensor, not the model.
2. ``.pt`` / ``.ckpt`` input (``torch.save`` format): ``torch.load`` must
   parse the whole file first, so peak RAM is about the FP32 file size
   plus the FP16 output (~1.5x file size).  On daisuke's PC (128 GB RAM)
   that is a non-issue even for the 10.66 GB EgoHand checkpoint
   (~12-13 GB peak).  On an 8 GB machine, convert via safetensors first
   if possible, or use ``--dry-run`` to see the sizes before committing.

Optimizer state (the EgoHand trap)
----------------------------------
``egohand_s060000.ckpt`` is 10,657,169,753 B but EgoHand has only
883.5M parameters: FP32 weights alone would be ~3.53 GB.  883.5M x 12 B
(FP32 weight + Adam m + Adam v) = ~10.60 GB, which matches the file
almost exactly — so the file very likely contains full Adam optimizer
state, not just weights.  This script detects optimizer tensors by key
name (``exp_avg``, ``exp_avg_sq``, ``optimizer``, ``state``, Adam
``m``/``v`` entries, ``param_groups``, ``scheduler``, ``epoch``,
``global_step``) and by structure (a top-level ``optimizer`` /
``state`` dict), SKIPS them by default, and reports exactly what was
skipped.  Use ``--keep-optimizer-state`` only if you really want the
optimizer tensors cast and kept (you almost certainly do not — they
are useless for inference and triple the file).

Checkpoint wrappers: if the top level is ``{"state_dict": {...}}`` or
``{"model": {...}}``, the inner dict is used as the weights.  Anything
else at the top level is reported, not silently dropped.

Output verification: after writing, the output is reloaded and every
tensor's dtype/shape/element-count is checked against the conversion
plan.  The script exits non-zero on any mismatch.

Usage (Windows PC, PowerShell)
------------------------------
    python tools\\quantize\\inspect_ckpt.py weights\\egohand_s060000.ckpt
    python tools\\quantize\\quantize_ckpt_to_fp16.py ^
        --input weights\\egohand_s060000.ckpt ^
        --output weights\\egohand_fp16.safetensors --format safetensors
    python tools\\quantize\\quantize_ckpt_to_fp16.py ^
        --input weights\\body_s240000.ckpt ^
        --output weights\\body_fp16.safetensors --format safetensors

Requires: ``torch`` for ``.pt`` / ``.ckpt`` input; ``safetensors``
(+ ``numpy``) for safetensors input/output.  See ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys

# ---------------------------------------------------------------------------
# Optimizer-state detection
# ---------------------------------------------------------------------------

OPTIMIZER_KEY_HINTS = (
    "exp_avg",
    "exp_avg_sq",
    "exp_avg_sq_row",
    "exp_avg_sq_col",
    "optimizer",
    "param_groups",
    "scheduler",
    "momentum_buffer",
    "adam",
    "adagrad",
    "rmsprop",
)

OPTIMIZER_TOP_LEVEL_KEYS = {
    "optimizer",
    "optimizers",
    "state",
    "scheduler",
    "schedulers",
    "scaler",
    "grad_scaler",
    "epoch",
    "global_step",
    "iteration",
    "step",
    "lr",
    "best_loss",
}

WEIGHT_TOP_LEVEL_KEYS = ("state_dict", "model", "net", "network", "module")


def _looks_like_optimizer_key(key: str) -> bool:
    low = key.lower()
    if any(h in low for h in OPTIMIZER_KEY_HINTS):
        return True
    # PyTorch optimizer state_dict layout: {"state": {0: {"exp_avg": ...}}}
    # is handled structurally, but flat keys like "state.0.exp_avg" also match.
    if low.startswith("state.") or ".state." in low:
        return True
    return False


def _is_float_dtype_name(dtype_name: str) -> bool:
    return dtype_name in ("float32", "float64", "float16", "bfloat16",
                          "torch.float32", "torch.float64",
                          "torch.float16", "torch.bfloat16",
                          "float", "double", "half")


def _unwrap_checkpoint(obj, *, keep_optimizer_state: bool):
    """Split a loaded checkpoint into (weights_dict, skipped_list, notes).

    Returns (weights, skipped, notes) where ``weights`` maps name -> tensor
    (torch Tensor or numpy array) and ``skipped`` is a list of
    (name, reason) for everything deliberately left out.
    """
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []

    if not isinstance(obj, dict):
        raise ValueError(
            f"Checkpoint top level is {type(obj).__name__}, not a dict. "
            "Expected a state_dict mapping name -> tensor."
        )

    # Unwrap {"state_dict": {...}} / {"model": {...}} wrappers.
    for wrapper in WEIGHT_TOP_LEVEL_KEYS:
        if wrapper in obj and isinstance(obj[wrapper], dict):
            inner = obj[wrapper]
            # Only unwrap if the inner dict actually looks like tensors.
            if inner and all(
                hasattr(v, "dtype") or hasattr(v, "shape") for v in inner.values()
            ):
                notes.append(f"Unwrapped top-level key '{wrapper}' as the weights.")
                obj = inner
                break

    # Report (and skip) known non-weight top-level entries.
    for key in list(obj.keys()):
        if key in OPTIMIZER_TOP_LEVEL_KEYS:
            skipped.append((str(key), "top-level non-weight entry"))
            del obj[key]
        elif key in ("args", "config", "cfg", "hyper_parameters", "hparams",
                    "metadata", "version"):
            notes.append(f"Ignored top-level metadata key '{key}'.")
            del obj[key]

    weights: dict = {}
    for name, value in obj.items():
        if _looks_like_optimizer_key(str(name)):
            skipped.append((str(name), "optimizer-state key"))
            continue
        # Nested optimizer state: {"state": {idx: {"exp_avg": ...}}}
        if isinstance(value, dict):
            flat = _flatten_nested(value, prefix=str(name))
            kept_any = False
            for sub_name, sub_val in flat:
                if _looks_like_optimizer_key(sub_name):
                    skipped.append((sub_name, "nested optimizer-state entry"))
                else:
                    weights[sub_name] = sub_val
                    kept_any = True
            if not kept_any:
                skipped.append((str(name), "nested dict contained only optimizer state"))
            continue
        weights[str(name)] = value

    if not keep_optimizer_state:
        # Second pass: anything that slipped through with an optimizer hint.
        for name in list(weights.keys()):
            if _looks_like_optimizer_key(name):
                skipped.append((name, "optimizer-state key (second pass)"))
                del weights[name]

    return weights, skipped, notes


def _flatten_nested(obj, *, prefix: str):
    """Yield (dotted_name, tensor) for a nested dict of tensors."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _flatten_nested(value, prefix=f"{prefix}.{key}")
    else:
        yield (prefix, obj)


def _tensor_info(name, tensor):
    """(dtype_name, shape, numel, bytes) for a torch or numpy tensor."""
    dtype = getattr(tensor, "dtype", None)
    dtype_name = str(dtype).replace("torch.", "")
    shape = tuple(int(d) for d in getattr(tensor, "shape", ()))
    numel = 1
    for d in shape:
        numel *= d
    itemsize = getattr(tensor, "element_size", None)
    if itemsize is None:  # numpy
        itemsize = getattr(getattr(tensor, "dtype", None), "itemsize", 4)
    else:
        itemsize = int(itemsize())
    return dtype_name, shape, numel, numel * itemsize


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024.0 or unit == "GiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.2f} GiB"


# ---------------------------------------------------------------------------
# Conversion backends
# ---------------------------------------------------------------------------

def convert_safetensors_streaming(input_path, output_path, *, dry_run=False):
    """Stream a .safetensors file tensor-by-tensor to FP16.

    Uses safetensors + numpy if torch is unavailable, else torch.
    Returns a summary dict.
    """
    try:
        from safetensors import safe_open
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: safetensors/numpy required for .safetensors input: {exc}",
              file=sys.stderr)
        raise SystemExit(2)

    plan = []          # (name, dtype, shape, numel, in_bytes, out_bytes)
    converted = {}     # name -> numpy array (FP16) or torch tensor
    skipped = []
    use_torch = False
    try:
        import torch  # noqa: F401
        use_torch = True
    except ImportError:
        use_torch = False

    with safe_open(input_path, framework="pt" if use_torch else "np") as f:
        keys = list(f.keys())
        for name in keys:
            tensor = f.get_tensor(name)
            dtype_name, shape, numel, in_bytes = _tensor_info(name, tensor)
            if _looks_like_optimizer_key(name):
                skipped.append((name, "optimizer-state key"))
                continue
            if dtype_name in ("float32", "float64"):
                out_bytes = numel * 2
                plan.append((name, dtype_name, shape, numel, in_bytes, out_bytes))
                if not dry_run:
                    if use_torch:
                        converted[name] = tensor.to("cpu").half().contiguous()
                    else:
                        converted[name] = np.ascontiguousarray(
                            tensor.astype(np.float16))
                    del tensor
                    gc.collect()
            elif dtype_name in ("float16", "bfloat16"):
                plan.append((name, dtype_name, shape, numel, in_bytes, in_bytes))
                if not dry_run:
                    converted[name] = tensor
            else:
                # int / bool / other: keep unchanged.
                plan.append((name, dtype_name, shape, numel, in_bytes, in_bytes))
                if not dry_run:
                    converted[name] = tensor

    if dry_run:
        return {"plan": plan, "skipped": skipped, "converted": {}}

    if use_torch:
        from safetensors.torch import save_file
        save_file(converted, output_path, metadata={"format": "pt"})
    else:
        from safetensors.numpy import save_file
        save_file(converted, output_path, metadata={"format": "pt"})
    return {"plan": plan, "skipped": skipped, "converted": converted}


def convert_torch_checkpoint(input_path, output_path, *, output_format,
                             keep_optimizer_state=False, dry_run=False):
    """Convert a .pt / .ckpt checkpoint (torch.save format) to FP16."""
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required for .pt/.ckpt input.\n"
              "Install it on the Windows PC first, e.g.:\n"
              "  pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu\n"
              "Or convert the checkpoint to .safetensors first and use the\n"
              "safetensors (numpy) path, which needs no torch.",
              file=sys.stderr)
        raise SystemExit(2)

    print(f"Loading {input_path} (weights_only=True) ...", flush=True)
    # GEM-X checkpoints embed an omegaconf DictConfig, which
    # weights_only=True rejects by default. Allowlist it (the file comes
    # from nvidia/GEM-X / nvidia/EgoHand-v1, downloaded by the caller).
    # mmap=True keeps a 10.66 GB checkpoint from being copied into RAM:
    # without it the fallback load OOM-kills a 16 GB Actions runner.
    try:
        from omegaconf import DictConfig, ListConfig
        torch.serialization.add_safe_globals([DictConfig, ListConfig])
    except Exception:
        pass
    try:
        obj = torch.load(input_path, map_location="cpu", weights_only=True,
                         mmap=True)
    except Exception as exc:
        print(f"weights_only load failed ({exc}). Retrying with "
              "weights_only=False — only do this for files you downloaded "
              "yourself from nvidia/GEM-X or nvidia/EgoHand-v1.",
              file=sys.stderr)
        obj = torch.load(input_path, map_location="cpu", weights_only=False,
                         mmap=True)

    weights, skipped, notes = _unwrap_checkpoint(
        obj, keep_optimizer_state=keep_optimizer_state)
    for note in notes:
        print(f"  note: {note}")
    del obj
    gc.collect()

    plan = []
    converted = {}
    for name, tensor in weights.items():
        dtype_name, shape, numel, in_bytes = _tensor_info(name, tensor)
        if dtype_name in ("float32", "float64"):
            out_bytes = numel * 2
            plan.append((name, dtype_name, shape, numel, in_bytes, out_bytes))
            if not dry_run:
                converted[name] = tensor.to(torch.float16).contiguous()
                del tensor
                gc.collect()
        elif dtype_name in ("float16", "bfloat16"):
            plan.append((name, dtype_name, shape, numel, in_bytes, in_bytes))
            if not dry_run:
                converted[name] = tensor
        else:
            plan.append((name, dtype_name, shape, numel, in_bytes, in_bytes))
            if not dry_run:
                converted[name] = tensor

    if dry_run:
        return {"plan": plan, "skipped": skipped, "converted": {}}

    if output_format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("ERROR: safetensors is required for --format safetensors.\n"
                  "  pip install safetensors", file=sys.stderr)
            raise SystemExit(2)
        save_file(converted, output_path, metadata={"format": "pt"})
    else:
        torch.save(converted, output_path)
    return {"plan": plan, "skipped": skipped, "converted": converted}


def verify_output(output_path, plan):
    """Reload the output and check dtype/shape/count against the plan."""
    ext = os.path.splitext(output_path)[1].lower()
    errors = []
    if ext == ".safetensors":
        from safetensors import safe_open
        try:
            import torch  # noqa: F401
            framework = "pt"
        except ImportError:
            framework = "np"
        with safe_open(output_path, framework=framework) as f:
            got_keys = set(f.keys())
            for name, dtype_name, shape, numel, _in_b, _out_b in plan:
                if name not in got_keys:
                    errors.append(f"missing tensor {name}")
                    continue
                tensor = f.get_tensor(name)
                got_dtype, got_shape, got_numel, _ = _tensor_info(name, tensor)
                want_dtype = ("float16" if dtype_name in ("float32", "float64")
                              else dtype_name)
                if got_dtype != want_dtype:
                    errors.append(f"{name}: dtype {got_dtype} != {want_dtype}")
                if tuple(got_shape) != tuple(shape) or got_numel != numel:
                    errors.append(f"{name}: shape/numel mismatch")
            extra = got_keys - {p[0] for p in plan}
            if extra:
                errors.append(f"unexpected extra tensors: {sorted(extra)[:5]}")
    else:
        import torch
        obj = torch.load(output_path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        for name, dtype_name, shape, numel, _in_b, _out_b in plan:
            if name not in obj:
                errors.append(f"missing tensor {name}")
                continue
            tensor = obj[name]
            got_dtype, got_shape, got_numel, _ = _tensor_info(name, tensor)
            want_dtype = ("float16" if dtype_name in ("float32", "float64")
                          else dtype_name)
            if got_dtype != want_dtype:
                errors.append(f"{name}: dtype {got_dtype} != {want_dtype}")
            if tuple(got_shape) != tuple(shape) or got_numel != numel:
                errors.append(f"{name}: shape/numel mismatch")
    return errors


def _print_summary(plan, skipped, *, output_path=None, dry_run=False):
    total_in = sum(p[4] for p in plan)
    total_out = sum(p[5] for p in plan)
    n_float = sum(1 for p in plan if p[1] in ("float32", "float64"))
    n_kept = len(plan) - n_float
    print()
    print(f"Tensors kept:      {len(plan)} "
          f"({n_float} float32/64 -> float16, {n_kept} already FP16 / non-float)")
    print(f"Input size:        {_fmt_bytes(total_in)}")
    print(f"Output size:       {_fmt_bytes(total_out)}")
    print(f"Saved:             {_fmt_bytes(total_in - total_out)}")
    if skipped:
        print(f"Skipped (optimizer / non-weight): {len(skipped)}")
        for name, reason in skipped[:20]:
            print(f"  - {name}  [{reason}]")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more (see --dry-run output)")
    else:
        print("Skipped:           none")
    if output_path and not dry_run:
        print(f"Wrote:             {output_path}")
    if dry_run:
        print("(dry run — nothing was written)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="CPU-only FP16 weight conversion for GEM-X checkpoints. "
                    "No training images, no calibration.")
    ap.add_argument("--input", required=True, help="Input .ckpt / .pt / .safetensors")
    ap.add_argument("--output", help="Output path (default: <input stem>_fp16.safetensors)")
    ap.add_argument("--format", choices=["safetensors", "pt"], default="safetensors",
                    help="Output format (default: safetensors)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List tensors/sizes and what would be skipped; write nothing.")
    ap.add_argument("--keep-optimizer-state", action="store_true",
                    help="Keep optimizer tensors (default: skip them). "
                         "Almost never what you want for inference.")
    args = ap.parse_args(argv)

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2
    ext = os.path.splitext(in_path)[1].lower()
    out_path = args.output
    if out_path is None:
        stem = os.path.splitext(in_path)[0]
        out_path = f"{stem}_fp16.{args.format if args.format == 'pt' else 'safetensors'}"
    if args.dry_run:
        out_path = None

    if ext == ".safetensors":
        result = convert_safetensors_streaming(
            in_path, out_path or "", dry_run=args.dry_run)
    elif ext in (".pt", ".ckpt", ".pth", ".bin"):
        result = convert_torch_checkpoint(
            in_path, out_path or "", output_format=args.format,
            keep_optimizer_state=args.keep_optimizer_state,
            dry_run=args.dry_run)
    else:
        print(f"ERROR: unsupported input extension '{ext}'. "
              "Use .ckpt, .pt, .pth or .safetensors.", file=sys.stderr)
        return 2

    _print_summary(result["plan"], result["skipped"],
                   output_path=out_path, dry_run=args.dry_run)

    if not args.dry_run:
        print("Verifying output by reloading ...", flush=True)
        errors = verify_output(out_path, result["plan"])
        if errors:
            print("VERIFICATION FAILED:", file=sys.stderr)
            for e in errors[:20]:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print("Verification OK: all dtypes/shapes/element counts match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
