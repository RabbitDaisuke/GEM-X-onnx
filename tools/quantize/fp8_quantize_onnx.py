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
    """Verify Q/DQ, FP8 tensor types and every Q/DQ scale's finiteness.

    Inspect small scale tensors through offsets in ONNX external data files.
    Do not load multi-GB GEM-X model weights into RAM just to check scales.
    """
    import onnx
    import numpy as np
    from onnx import numpy_helper

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    nodes = list(model.graph.node)
    q = sum(node.op_type == "QuantizeLinear" for node in nodes)
    dq = sum(node.op_type == "DequantizeLinear" for node in nodes)
    f8_types = {onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E5M2}
    f8 = sum(t.data_type in f8_types for t in model.graph.initializer)
    initializers = {t.name: t for t in model.graph.initializer}
    scale_names = {node.input[1] for node in nodes
                   if node.op_type in ("QuantizeLinear", "DequantizeLinear")
                   and len(node.input) > 1}
    nonfinite = []
    nonpositive = []
    checked = 0
    for name in sorted(scale_names):
        tensor = initializers.get(name)
        if tensor is None:
            raise RuntimeError(f"Q/DQ scale is not a constant initializer: {name}")
        if tensor.external_data:
            external = {entry.key: entry.value for entry in tensor.external_data}
            location = external.get("location")
            if not location or Path(location).is_absolute() or ".." in Path(location).parts:
                raise RuntimeError(f"unsafe or missing ONNX external scale location: {name}")
            with (path.parent / location).open("rb") as handle:
                handle.seek(int(external.get("offset", "0")))
                raw = handle.read(int(external["length"]))
            dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
            values = np.frombuffer(raw, dtype=dtype)
        else:
            values = numpy_helper.to_array(tensor)
        if values.size != int(np.prod(tensor.dims)):
            raise RuntimeError(f"Q/DQ scale size is inconsistent: {name}")
        checked += values.size
        if not np.isfinite(values).all():
            nonfinite.append(name)
        elif not np.all(values > 0):
            nonpositive.append(name)
    info = {"quantize_nodes": q, "dequantize_nodes": dq,
            "fp8_initializers": f8, "scale_count": len(scale_names),
            "scale_values_checked": checked, "nonfinite_scale_count": len(nonfinite),
            "nonfinite_scale_examples": nonfinite[:12],
            "nonpositive_scale_count": len(nonpositive),
            "nonpositive_scale_examples": nonpositive[:12],
            "opset": max(x.version for x in model.opset_import
                         if x.domain in ("", "ai.onnx"))}
    if not q or not dq or not f8 or nonfinite or nonpositive:
        raise RuntimeError(
            f"FP8 export is invalid or unsafe: {json.dumps(info)}. "
            "Do NOT deploy as FP8.")
    return info


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
