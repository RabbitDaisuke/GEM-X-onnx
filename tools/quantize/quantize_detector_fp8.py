#!/usr/bin/env python3
"""FP8 Q/DQ post-training quantization of an ONNX detector.

Requires an NPZ containing real feed arrays for *every* model input; the
input shape and dtype are checked against ONNX, so no person/hand detector
preprocessing is guessed or silently altered.

Example:
    python tools/quantize/quantize_detector_fp8.py \
        --onnx weights/hand_detector.onnx \
        --calibration out/calibration/hand_detector.npz \
        --output out/hand_detector_fp8.onnx

Keep the source ONNX and original detector for FP16/FP32 compatibility.
This export is only usable after numerical and TensorRT-RTX validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def validate_feeds(path, model):
    import onnx
    from onnx import helper

    required = [entry for entry in model.graph.input
                if entry.name not in {i.name for i in model.graph.initializer}]
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {x.name for x in required}:
            raise ValueError(
                f"calibration keys {sorted(archive.files)} do not match "
                f"model inputs {sorted(x.name for x in required)}")
        feeds = {}
        samples = None
        for input_info in required:
            raw = archive[input_info.name]
            tensor = input_info.type.tensor_type
            expected_type = helper.tensor_dtype_to_np_dtype(tensor.elem_type)
            if raw.dtype != expected_type:
                raise ValueError(
                    f"{input_info.name}: expected {expected_type}, got {raw.dtype}")
            if raw.ndim != len(tensor.shape.dim):
                raise ValueError(f"{input_info.name}: wrong rank {raw.shape}")
            for index, dim in enumerate(tensor.shape.dim):
                # axis 0 holds multiple calibration samples, not the ONNX
                # model's per-inference batch (normally 1); ModelOpt
                # splits these into individual inference batches.
                if index == 0:
                    continue
                if dim.HasField("dim_value") and raw.shape[index] != dim.dim_value:
                    raise ValueError(f"{input_info.name}: axis {index} must be "
                                     f"{dim.dim_value}, got {raw.shape[index]}")
            if raw.ndim == 0 or raw.shape[0] == 0:
                raise ValueError(f"{input_info.name}: no batch samples")
            if samples is None:
                samples = raw.shape[0]
            if raw.shape[0] != samples:
                raise ValueError(f"{input_info.name}: mismatched batch size")
            if np.issubdtype(raw.dtype, np.floating) and not np.isfinite(raw).all():
                raise ValueError(f"{input_info.name}: non-finite values")
            feeds[input_info.name] = np.ascontiguousarray(raw)
    return feeds, samples


def validate_fp8(path):
    import onnx
    from onnx import numpy_helper

    # Unlike the multi-GB GEM-X pose models, the hand detector is ~80 MB,
    # so loading external weights into RAM here is practical.
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=True)
    initializers = {t.name: t for t in model.graph.initializer}
    q = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    dq = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    fp8_types = {onnx.TensorProto.FLOAT8E4M3FN,
                 onnx.TensorProto.FLOAT8E5M2}
    f8 = sum(t.data_type in fp8_types for t in model.graph.initializer)
    scales = {node.input[1] for node in q + dq if len(node.input) > 1}
    bad = []
    for name in scales:
        if name in initializers:
            a = numpy_helper.to_array(initializers[name])
            if not np.isfinite(a).all():
                bad.append(name)
    info = {"q_nodes": len(q), "dq_nodes": len(dq),
            "fp8_initializers": f8, "scale_count": len(scales),
            "nonfinite_scales": len(bad), "bad_scale_examples": sorted(bad)[:12]}
    if not q or not dq or not f8 or bad:
        raise ValueError("FP8 ONNX failed structural/scale validation: "
                         + json.dumps(info))
    return info


def main(argv=None):
    import onnx
    from modelopt.onnx.quantization import quantize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-ep", choices=("cpu", "cuda:0"),
                        default="cpu")
    args = parser.parse_args(argv)
    if not args.onnx.is_file() or not args.calibration.is_file():
        parser.error("existing ONNX and calibration NPZ are required")
    if args.output.resolve() == args.onnx.resolve():
        parser.error("cannot replace the original detector")
    if (args.output.exists() or
            args.output.with_name(args.output.name + "_data").exists()):
        parser.error("FP8 output already exists; no overwrite")
    model = onnx.load(str(args.onnx), load_external_data=False)
    data, n = validate_feeds(args.calibration, model)
    print(f"Calibrating {args.onnx.name} with {n} actual samples "
          f"on {args.calibration_ep}; inputs={list(data)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize(onnx_path=str(args.onnx), quantize_mode="fp8",
             calibration_data=data, calibration_method="max",
             calibration_eps=[args.calibration_ep],
             high_precision_dtype="fp16", use_external_data_format=True,
             output_path=str(args.output))
    info = validate_fp8(args.output)
    info.update(source=str(args.onnx), output=str(args.output),
                calibration=str(args.calibration), sample_count=n,
                verification="ONNX structure and finite Q/DQ scales; "
                             "NOT GPU inference accuracy")
    metadata = Path(str(args.output) + ".quantization.json")
    metadata.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
