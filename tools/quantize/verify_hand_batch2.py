#!/usr/bin/env python3
"""Verify a real GEM-X Hand batch-2 FP16 ONNX against two batch-1 runs.

A marker is written ONLY after numerical equivalence is confirmed. The
Windows bench_two_optimizations.py requires this marker before running
its Hand batch-2 benchmark.

Example:
  python tools/quantize/verify_hand_batch2.py \
    --single out/hand_fp16.onnx \
    --batch out/hand_batch2_fp16.onnx \
    --calibration out/calibration-refined/hand_calibration.npz

The existing batch-1 ONNX is read only. Never edit ONNX tensor shapes
as a substitute for exporting the real batch-2 PyTorch model.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

EXPECTED_INPUTS = ("img", "bbx_xys", "K_fullimg", "ray_feats")
ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def new_session(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def prepare_feeds(session, samples, start, count):
    feed = {}
    for info in session.get_inputs():
        if info.name not in samples:
            raise ValueError(f"Calibration NPZ missing {info.name}")
        dtype = ORT_DTYPES.get(info.type)
        if dtype is None:
            raise ValueError(f"Unsupported model input type: {info.name} {info.type}")
        arr = np.ascontiguousarray(
            samples[info.name][start:start + count], dtype=dtype
        )
        if arr.shape[0] != count:
            raise ValueError(f"Not enough calibration samples for {info.name}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite calibration input: {info.name}")
        feed[info.name] = arr
    return feed


def check_model_shape(session, batch_size):
    inputs = session.get_inputs()
    if tuple(x.name for x in inputs) != EXPECTED_INPUTS:
        raise ValueError(
            f"Input names differ from GEM-X Hand: {[x.name for x in inputs]}"
        )
    for inp in inputs:
        actual = inp.shape[0]
        if actual != batch_size:
            raise ValueError(
                f"{inp.name}: expected static batch={batch_size}, found {actual}"
            )


def verify(single, batch, calibration, atol, rtol):
    if single.resolve() == batch.resolve():
        raise ValueError("Single and batch ONNX must be different files")
    with np.load(calibration, allow_pickle=False) as data:
        missing = [k for k in EXPECTED_INPUTS if k not in data.files]
        if missing:
            raise ValueError(f"Calibration inputs missing: {missing}")
        samples = {
            name: np.ascontiguousarray(data[name][:2])
            for name in EXPECTED_INPUTS
        }
    if any(value.shape[0] < 2 for value in samples.values()):
        raise ValueError("At least two real Hand calibration samples are required")

    # Only hold one multi-GB ORT session at a time on the 16GiB Codespace.
    one = new_session(single)
    check_model_shape(one, 1)
    output_names = [x.name for x in one.get_outputs()]
    if not output_names:
        raise ValueError("Batch-1 ONNX has no outputs")
    reference = [
        one.run(None, prepare_feeds(one, samples, i, 1)) for i in range(2)
    ]
    del one
    gc.collect()

    two = new_session(batch)
    check_model_shape(two, 2)
    if [x.name for x in two.get_outputs()] != output_names:
        raise ValueError("Batch-2 ONNX output names differ from batch-1")
    actual = two.run(None, prepare_feeds(two, samples, 0, 2))
    del two
    gc.collect()

    stats = {}
    for index, name in enumerate(output_names):
        expected = np.concatenate(
            [reference[0][index], reference[1][index]], axis=0
        )
        got = actual[index]
        if got.shape != expected.shape:
            raise ValueError(
                f"{name}: batch2 output shape {got.shape}, expected {expected.shape}"
            )
        if not (np.isfinite(expected).all() and np.isfinite(got).all()):
            raise ValueError(f"{name}: model output has inf/NaN")
        delta = np.abs(got.astype(np.float64) - expected.astype(np.float64))
        max_abs = float(delta.max(initial=0.0))
        reference_abs = np.abs(expected.astype(np.float64))
        ok = bool(np.allclose(got, expected, rtol=rtol, atol=atol))
        stats[name] = {
            "shape": list(got.shape),
            "max_abs_error": max_abs,
            "max_reference_abs": float(reference_abs.max(initial=0.0)),
            "numerically_equal": ok,
        }
        print(f"{name}: max_abs_error={max_abs:.7g}, equal={ok}", flush=True)
        if not ok:
            raise ValueError(
                f"{name}: batch-2 output differs from two batch-1 outputs. "
                "Not validated for the Windows benchmark."
            )
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single", type=Path, required=True)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--atol", type=float, default=0.01)
    ap.add_argument("--rtol", type=float, default=0.01)
    args = ap.parse_args()
    for path in (args.single, args.batch, args.calibration):
        if not path.is_file() or path.stat().st_size == 0:
            ap.error(f"Required input absent or empty: {path}")
    if args.atol < 0 or args.rtol < 0:
        ap.error("Numerical tolerances must be nonnegative")
    marker = args.batch.with_suffix(".batch2_verified.json")
    if marker.exists():
        ap.error(f"Refusing to replace existing verification marker: {marker}")
    stats = verify(args.single, args.batch, args.calibration,
                   args.atol, args.rtol)
    info = {
        "source_batch1_fp16": str(args.single),
        "exported_batch2_fp16": str(args.batch),
        "calibration": str(args.calibration),
        "atol": args.atol,
        "rtol": args.rtol,
        "outputs": stats,
        "note": "Numerical equivalence on two inputs, not general accuracy validation",
    }
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"BATCH2_VERIFIED {marker}", flush=True)


if __name__ == "__main__":
    main()
