#!/usr/bin/env python3
"""export_onnx_fp16.py — TEMPLATE / STUB (untested) for the ONNX export step.

Recommended path (from the research report):
    FP32 ONNX export (PyTorch, CPU OK, no training data)
      -> NVIDIA ModelOpt AutoCast to FP16 ONNX (CPU, graph rewrite only)
      -> TensorRT-RTX engine build on the Windows GPU (SCG 48K)
      -> numeric compare vs PyTorch FP32 (MPJPE / max abs error)

This file is deliberately a *template*, not a finished exporter, because
the GEM-X PoseModel source is not in this repo and the export has never
been run anywhere in this project.  It does the honest thing instead:

  1. Checks for torch / onnx / modelopt and prints the exact install
     commands when something is missing.
  2. If everything is present, it tries the generic export flow with a
     user-supplied model-builder module (--model-module /
     --build-function).  The GEM-X-specific builder (constructing
     PoseModel from body_config.json + checkpoint, feeding a 512x512
     crop + K_fullimg / bbx_xys) still has to be written against the
     NVlabs/GEM-X source on the Windows PC — that part is marked TODO
     below with the exact places to fill in.

DO NOT claim this script "works" until step 2 has been run on the
Windows PC against the real GEM-X code and the ONNX has been parsed
by ``tensorrt_rtx`` / ``trtexec`` with no unsupported ops.

Why no training images are needed here either: ONNX export traces the
model with dummy inputs (a random 512x512 crop tensor, a K matrix,
a bounding box).  The *weights* come from the checkpoint; no dataset
is read.  FP16 AutoCast likewise needs no data — it rewrites the
graph's dtypes.  (INT8 would need calibration data; not v1.)

Usage on the Windows PC (after the template is completed):
    python tools\\quantize\\export_onnx_fp16.py ^
        --checkpoint weights\\body_s240000.ckpt ^
        --config weights\\body_config.json ^
        --model-module gemx_export --build-function build_body_pose ^
        --output body_fp32.onnx --fp16-output body_fp16.onnx
    python -m modelopt.onnx.autocast --onnx_path body_fp32.onnx ^
        --output_path body_fp16.onnx
    tensorrt_rtx.exe --onnx=body_fp16.onnx --saveEngine=body_fp16.trt ^
        --memPoolSize=tacticSharedMem:48K
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys


def _check(cmd, *, what):
    try:
        found = subprocess.run(
            [sys.executable, "-c", cmd], capture_output=True, text=True)
    except Exception as exc:
        print(f"MISSING: {what} ({exc})")
        return False
    if found.returncode != 0:
        print(f"MISSING: {what}")
        return False
    print(f"OK: {what}")
    return True


def check_dependencies():
    print("Checking dependencies (CPU-only is fine for this step) ...")
    ok = True
    ok &= _check("import torch; print(torch.__version__)",
                 what="torch")
    ok &= _check("import onnx; print(onnx.__version__)", what="onnx")
    ok &= _check("import onnxruntime; print(onnxruntime.__version__)",
                 what="onnxruntime (for numeric comparison)")
    has_modelopt = _check(
        "import modelopt; print(modelopt.__version__)",
        what="nvidia-modelopt (AutoCast)")
    if not has_modelopt:
        print("  Install with: pip install nvidia-modelopt")
        print("  (AutoCast can be run separately afterwards; see README.md.)")
    if not ok:
        print("\nInstall the missing packages first, e.g.:\n"
              "  pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu\n"
              "  pip install onnx onnxruntime nvidia-modelopt")
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="TEMPLATE: export a GEM-X PoseModel to FP32 ONNX "
                    "then AutoCast to FP16. UNTESTED — see module docstring.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True,
                    help="body_config.json (or hand config)")
    ap.add_argument("--model-module", required=True,
                    help="Python module that builds the PoseModel, e.g. gemx_export")
    ap.add_argument("--build-function", default="build_pose_model",
                    help="Function in --model-module returning an eval-mode nn.Module")
    ap.add_argument("--output", default="model_fp32.onnx")
    ap.add_argument("--fp16-output", default="model_fp16.onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--skip-autocast", action="store_true",
                    help="Only export FP32 ONNX; run AutoCast separately")
    args = ap.parse_args(argv)

    print("=" * 72)
    print("WARNING: this exporter is a TEMPLATE and has never been run")
    print("against the real GEM-X code.  It will fail until the TODOs in")
    print("the module docstring / --model-module builder are completed")
    print("on the Windows PC.  Do not report it as working.")
    print("=" * 72)

    if not check_dependencies():
        return 2
    for path in (args.checkpoint, args.config):
        if not os.path.isfile(path):
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 2

    # TODO (Windows PC, against NVlabs/GEM-X source):
    #   - Implement --model-module with build_pose_model(checkpoint, config)
    #     that returns an eval-mode torch.nn.Module whose forward() is the
    #     PoseModel inference path ONLY (no SOMA FK, no translation solve,
    #     no projection — those stay in the C++ DLL).
    #   - forward() inputs must be STATIC shapes: crop image (e.g.
    #     1x3x512x512), K_fullimg, bbx_xys.  No dynamic batch.
    #   - If RoPE uses complex64, replace it with real-valued cos/sin
    #     buffers before export (TensorRT cannot consume complex ops).
    try:
        module = importlib.import_module(args.model_module)
        build = getattr(module, args.build_function)
    except Exception as exc:
        print(f"ERROR: cannot load model builder "
              f"{args.model_module}.{args.build_function}: {exc}", file=sys.stderr)
        print("Fill in the TODO in this file's docstring first.", file=sys.stderr)
        return 2

    import torch
    model = build(args.checkpoint, args.config)
    model.eval()
    # TODO: dummy inputs must match the real forward() signature.
    # Placeholder shapes for a 512x512 Body crop — ADJUST to the builder.
    dummy_crop = torch.randn(1, 3, 512, 512)
    dummy_k = torch.eye(3).unsqueeze(0)
    dummy_bbx = torch.tensor([[0.0, 0.0, 512.0, 512.0, 1.0]])
    print(f"Exporting FP32 ONNX -> {args.output} ...")
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy_crop, dummy_k, dummy_bbx), args.output,
            input_names=["crop", "K_fullimg", "bbx_xys"],
            output_names=["pose"],
            opset_version=args.opset, dynamo=False)
    print(f"Wrote {args.output}")

    if not args.skip_autocast:
        print(f"Running ModelOpt AutoCast -> {args.fp16_output} ...")
        res = subprocess.run(
            [sys.executable, "-m", "modelopt.onnx.autocast",
             "--onnx_path", args.output,
             "--output_path", args.fp16_output])
        if res.returncode != 0:
            print("AutoCast failed. Run it manually:\n"
                  f"  python -m modelopt.onnx.autocast --onnx_path "
                  f"{args.output} --output_path {args.fp16_output}",
                  file=sys.stderr)
            return res.returncode
        print(f"Wrote {args.fp16_output}")
    print("Next: parse the ONNX with tensorrt_rtx / trtexec and check "
          "for unsupported ops before building the engine:")
    print(f"  tensorrt_rtx.exe --onnx={args.fp16_output} "
          f"--saveEngine=model_fp16.trt --memPoolSize=tacticSharedMem:48K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
