#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrated FP8 Q/DQ export for GEM-X Body/Hand FP16 ONNX.

This is post-training quantization, NOT a weight-only cast. The input
calibration NPZ must contain REAL normalized model inputs (see
prepare_fp8_calibration.py); random calibration is intentionally forbidden.

Example:
  python tools/quantize/fp8_quantize_onnx.py \
      --task body --onnx out/body_fp16.onnx \
      --calibration calibration/body.npz --output out/body_fp8.onnx
"""
import argparse
import json
import os
from pathlib import Path
import sys

INPUTS = {
    "body": {"img": (3, 256, 256), "bbx_xys": (3,), "K_fullimg": (3, 3)},
    "hand": {"img": (3, 256, 256), "bbx_xys": (3,), "K_fullimg": (3, 3),
             "ray_feats": (256, 32)},
}


def read_calibration(path, task):
    import numpy as np

    expected = INPUTS[task]
    with np.load(path, allow_pickle=False) as data:
        supplied = set(data.files)
        if supplied != set(expected):
            raise ValueError(
                f"{path}: input names {sorted(supplied)}; expected "
                f"{sorted(expected)}")
        calib = {}
        n = None
        for name, sample_shape in expected.items():
            arr = data[name]
            if arr.ndim != len(sample_shape) + 1 or tuple(arr.shape[1:]) != sample_shape:
                raise ValueError(
                    f"{name}: expected (N,{','.join(map(str, sample_shape))}), "
                    f"got {arr.shape}")
            if n is None:
                n = arr.shape[0]
            if arr.shape[0] != n or n == 0:
                raise ValueError(f"{name}: inconsistent or empty sample count")
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError(f"{name}: expected floating input, got {arr.dtype}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name}: non-finite calibration inputs")
            # ModelOpt/ORT must see the same input dtypes as FP16 GEM-X.
            calib[name] = np.ascontiguousarray(arr, dtype=np.float16)
    if n < 32:
        print(f"WARNING: only {n} calibration samples; 128-500 varied "
              "samples are preferable for reliable FP8 calibration.",
              file=sys.stderr)
    return calib, n


def inspect_fp8(path):
    """Validate that the output has actual FLOAT8 Q/DQ, not only FP16."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    q = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)
    dq = sum(node.op_type == "DequantizeLinear" for node in model.graph.node)
    f8_types = {onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E5M2}
    f8 = sum(t.data_type in f8_types for t in model.graph.initializer)
    if q == 0 or dq == 0 or f8 == 0:
        raise RuntimeError(
            f"FP8 export is not verified: Q={q}, DQ={dq}, FP8 initializers={f8}. "
            "Do NOT deploy as FP8.")
    return {"quantize_nodes": q, "dequantize_nodes": dq,
            "fp8_initializers": f8, "opset": max(x.version for x in model.opset_import
                                                 if x.domain in ("", "ai.onnx"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(INPUTS))
    ap.add_argument("--onnx", required=True, type=Path,
                    help="existing unmodified FP16 ONNX")
    ap.add_argument("--calibration", required=True, type=Path,
                    help="NPZ with actual GEM-X model feeds")
    ap.add_argument("--output", required=True, type=Path,
                    help="separate FP8 Q/DQ ONNX destination")
    ap.add_argument("--method", choices=("max", "entropy"), default="max")
    ap.add_argument("--calibration-ep", choices=("cpu", "cuda:0"), default="cpu")
    ap.add_argument("--calibrate-per-node", action="store_true",
                    help="reduce peak memory at the cost of extra work")
    args = ap.parse_args(argv)
    if not args.onnx.is_file() or not args.calibration.is_file():
        ap.error("existing --onnx and --calibration files are required")
    if args.onnx.resolve() == args.output.resolve():
        ap.error("refusing to overwrite the FP16 source")
    if args.output.exists() or args.output.with_suffix(
            args.output.suffix + ".data").exists():
        ap.error("output already exists; refusing to overwrite")
    calib, count = read_calibration(args.calibration, args.task)
    print(f"FP8 calibration: {args.task}, {count} real samples, "
          f"EP={args.calibration_ep}, method={args.method}", flush=True)

    try:
        from modelopt.onnx.quantization import quantize
    except ImportError as exc:
        raise SystemExit(
            "Missing ModelOpt: install nvidia-modelopt[onnx] in this "
            f"environment. {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize(onnx_path=str(args.onnx), quantize_mode="fp8",
             calibration_data=calib, calibration_method=args.method,
             calibration_eps=[args.calibration_ep],
             high_precision_dtype="fp16", use_external_data_format=True,
             calibrate_per_node=args.calibrate_per_node,
             output_path=str(args.output))
    if not args.output.is_file():
        raise RuntimeError("ModelOpt did not produce the requested ONNX")
    info = inspect_fp8(args.output)
    info.update(task=args.task, calibration_samples=count,
                calibration_method=args.method, calibration_ep=args.calibration_ep,
                source_onnx=str(args.onnx), calibration_file=str(args.calibration),
                output_onnx=str(args.output))
    with open(str(args.output) + ".quantization.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
