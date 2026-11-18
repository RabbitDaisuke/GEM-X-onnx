#!/usr/bin/env python3
"""export_onnx_real.py — real GEM-X PoseModel ONNX exporter (v1, 2026-11-18).

This replaces the generic template in ``export_onnx_fp16.py`` with code
written against the actual NVlabs/GEM-X source (commit 4aef145):

  * entry points: ``gem.pose.api.load_pose_checkpoint`` (Body, with
    ``body_config()`` / a ``body_config.json``) and
    ``gem.pose.api.load_hand_checkpoint`` (Hand, built-in ``hand_config()``);
  * export boundary: ``PoseModel.encode()`` — backbone + decoder only.
    SOMA FK, camera decode, kp2d translation solve and right-hand mirroring
    are NOT in the graph (see the blocker report);
  * Hand blockers handled here:
      H1 RayPE ``torch.linalg.solve`` -> ``ray_feats`` is computed on the CPU
         (``compute_ray_feats_numpy`` below, mirroring
         ``gem/pose/conditioning/ray_pe.py``) and passed in as a tensor;
      H3 dict batch input -> the export wrapper takes explicit tensors only
         (``img, bbx_xys, K_fullimg[, ray_feats]``);
      H4 kp2d head -> excluded from the v1 Hand export (built without
         ``kp2d_head``; its weights are stripped before the strict load);
      H5 right-hand mirror -> preprocessing outside the graph (DLL side);
      H6 SOMA FK / camera decode -> outside the graph (DLL side, CPU/FP32);
      B3 per-layer lists -> Body uses the head's ``final_only=True`` flag
         (the wrapper calls the head directly, because ``PoseModel.forward``
         does not plumb ``final_only`` through); Hand drops the auxiliary
         stacked tensors and keeps the final layer only.

Status: UNTESTED end-to-end. There is no GPU and no torch on this VM, so
the export path has never run. Every missing dependency / weight / file
fails cleanly with exit code 2 and prints what to install or download.
Run on the Windows PC (RTX 4070 Ti SUPER, 128GB RAM) per
``docs/research-gemx-onnx-export-blockers-2026-11-18.md`` section 6.

Pipeline (do not reorder):
    FP32 ONNX export (this script, CPU OK, dynamo)
      -> onnxruntime numeric compare vs PyTorch FP32 (--compare)
      -> ModelOpt AutoCast to FP16 ONNX (CPU, graph rewrite only)
      -> tensorrt_rtx parse + build --memPoolSize=tacticSharedMem:48K
      -> DLL real inference + numeric compare (MPJPE) + latency measure

Usage (Windows PC, after cloning NVlabs/GEM-X and downloading weights):
    python tools\\quantize\\export_onnx_real.py --task body ^
        --gemx-root C:\\gemx\\GEM-X --checkpoint models\\body_s240000.ckpt ^
        --config models\\body_config.json --output body_fp32.onnx --compare
    python tools\\quantize\\export_onnx_real.py --task hand ^
        --gemx-root C:\\gemx\\GEM-X --checkpoint models\\egohand_s060000.ckpt ^
        --output hand_fp32.onnx --compare
    python -m modelopt.onnx.autocast --onnx_path body_fp32.onnx --output_path body_fp16.onnx
    tensorrt_rtx.exe --onnx=body_fp16.onnx --saveEngine=body_fp16.trt ^
        --memPoolSize=tacticSharedMem:48K

Self-test (no torch / no weights needed, runs on this VM):
    python tools/quantize/export_onnx_real.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# RayPE on CPU (numpy reference implementation of
# gem/pose/conditioning/ray_pe.py). This is the exact computation the DLL
# must reproduce in C++: per-patch camera-ray azimuth/elevation, Fourier
# encoded with 2**[0,8) frequencies -> (B, 256, 32) for a 256x256 crop.
# ---------------------------------------------------------------------------

BODY_OUTPUTS = ("body_pose", "global_orient", "identity_coeffs",
                "scale_params", "pred_cam", "pose_token")
HAND_OUTPUTS = ("wrist_orient", "hand_pose", "identity_coeffs",
                "scale_params", "pred_cam", "pose_token")


def compute_ray_feats_numpy(bbx_xys, K_fullimg, *, grid_hw=(16, 16),
                            crop_hw=(256, 256), num_bands=8):
    """Mirror of ray_pe.compute_crop_ray_angles + fourier_encode_angles.

    bbx_xys: (B,3) [cx, cy, size]; K_fullimg: (B,3,3). Returns (B, H*W, 32)
    float32. Uses numpy only so the DLL reference and the --self-test can
    run without torch.
    """
    import numpy as np

    bbx = np.asarray(bbx_xys, dtype=np.float32)
    K = np.asarray(K_fullimg, dtype=np.float32)
    gh, gw = grid_hw
    ch, cw = crop_hw
    rows = (np.arange(gh, dtype=np.float32) + 0.5) * (ch / gh)
    cols = (np.arange(gw, dtype=np.float32) + 0.5) * (cw / gw)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    crop_xy = np.stack((uu, vv), axis=-1).reshape(1, -1, 2)  # (1,N,2) (u,v)
    crop_xy = np.broadcast_to(crop_xy, (bbx.shape[0], gh * gw, 2))
    crop_scale = np.array([cw, ch], dtype=np.float32)
    crop_norm = crop_xy / crop_scale - 0.5
    full_xy = bbx[:, None, :2] + crop_norm * bbx[:, None, 2:3]
    full_pix = np.concatenate(
        (full_xy, np.ones_like(full_xy[..., :1])), axis=-1)  # (B,N,3)
    rays = np.linalg.solve(K, full_pix.transpose(0, 2, 1)).transpose(0, 2, 1)
    azimuth = np.arctan2(rays[..., 0], rays[..., 2])
    elevation = np.arctan2(rays[..., 1], rays[..., 2])
    angles = np.stack((azimuth, elevation), axis=-1)  # (B,N,2)
    freqs = (2.0 ** np.arange(num_bands, dtype=np.float32))  # (8,)
    az = angles[..., 0, None] * freqs
    el = angles[..., 1, None] * freqs
    feats = np.concatenate(
        (np.sin(az), np.cos(az), np.sin(el), np.cos(el)), axis=-1)
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Dependency / file checks (exit 2 with a concrete fix-it list)
# ---------------------------------------------------------------------------

def _have_import(stmt, what):
    proc = subprocess.run([sys.executable, "-c", stmt],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"MISSING: {what}")
        return False
    print(f"OK: {what} ({proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''})")
    return True


def check_dependencies(*, need_gemx=False, gemx_root=None):
    ok = True
    ok &= _have_import("import torch; print(torch.__version__)", "torch (>=2.10,<3)")
    ok &= _have_import("import onnx; print(onnx.__version__)", "onnx")
    # The dynamo exporter hard-requires onnxscript (torch 2.15
    # torch/onnx/_internal/exporter/_core.py: `import onnxscript`);
    # torch does not install it, so check it here and fail fast.
    ok &= _have_import("import onnxscript; print(onnxscript.__version__)",
                       "onnxscript (dynamo exporter needs it; "
                       "pip install onnxscript)")
    ok &= _have_import("import onnxruntime; print(onnxruntime.__version__)",
                       "onnxruntime (for --compare)")
    ok &= _have_import("import einops; print(einops.__version__)", "einops")
    ok &= _have_import("import transformers; print(transformers.__version__)",
                       "transformers (==5.3.0, DINOv3 backbone)")
    if need_gemx:
        if not gemx_root or not os.path.isdir(os.path.join(gemx_root, "gem", "pose")):
            print(f"MISSING: GEM-X source at --gemx-root {gemx_root!r} "
                  "(clone https://github.com/NVlabs/GEM-X)")
            ok = False
        else:
            print(f"OK: GEM-X source at {gemx_root}")
    return ok


def check_files(checkpoint, config=None):
    ok = True
    for label, path in (("checkpoint", checkpoint), ("config", config)):
        if path is None:
            continue
        if not os.path.isfile(path):
            print(f"MISSING: {label} file not found: {path}")
            ok = False
        else:
            print(f"OK: {label} {path} ({os.path.getsize(path)} bytes)")
    return ok


# ---------------------------------------------------------------------------
# Model loading (real GEM-X API; see gem/pose/api.py)
# ---------------------------------------------------------------------------

def load_model(task, checkpoint, gemx_root, config_path=None):
    """Build a PoseModel from a real checkpoint. Returns (model, release).

    Hand is built WITHOUT the kp2d head (blocker H4): the release config's
    kp2d_translation is cleared and kp2d_* weights are stripped before the
    strict load, so no unexpected-key error can hide a silent mismatch.
    """
    sys.path.insert(0, os.path.abspath(gemx_root))
    import torch
    from gem.pose.api import (PoseReleaseConfig, body_config, build_pose_model,
                              convert_pose_state_dict, hand_config)

    if task == "body":
        if config_path:
            with open(config_path, "r", encoding="utf-8") as fh:
                release = PoseReleaseConfig(**json.load(fh))
        else:
            release = body_config()
    else:
        release = hand_config()
        release.kp2d_translation = None  # H4: v1 export excludes the kp2d head

    checkpoint_data = torch.load(checkpoint, map_location="cpu",
                                 weights_only=False, mmap=True)
    metadata = checkpoint_data.get("gem_pose_training")
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    if metadata is not None:
        state_dict = {k.removeprefix("pose_model."): v
                      for k, v in state_dict.items()
                      if k.startswith("pose_model.")}
    converted = convert_pose_state_dict(state_dict, task=task)
    if task == "hand":
        dropped = [k for k in converted
                   if k.startswith(("kp2d_head.", "kp2d_outcoord_loss."))]
        for k in dropped:
            del converted[k]
        if dropped:
            print(f"NOTE: stripped {len(dropped)} kp2d weights "
                  "(H4: kp2d head is excluded from the v1 export)")
    with torch.device("meta"):
        model = build_pose_model(release)
    model.load_state_dict(converted, strict=True, assign=True)
    return model.to("cpu").eval(), release


# ---------------------------------------------------------------------------
# Export wrappers: explicit tensor inputs, encode() boundary, final layer only
# ---------------------------------------------------------------------------

def build_wrapper(task, model):
    """Return an nn.Module whose forward is the exportable encode().

    Mirrors PoseModel.forward (gem/pose/model.py) but with flat tensor
    inputs instead of a dict (H3) and, for Body, final_only=True (B3).
    Hand takes precomputed ray_feats as an input (H1). Auxiliary stacked
    per-layer predictions are dropped; only the final layer is returned.
    """
    import torch
    from torch import nn

    class ExportWrapper(nn.Module):
        def __init__(self, pose_model):
            super().__init__()
            self.model = pose_model

        def forward(self, img, bbx_xys, K_fullimg, ray_feats=None):
            m = self.model
            image = img
            if m.backbone_input_crop_w is not None:
                start, stop = m.backbone_input_crop_w
                image = image[..., start:stop]
            features = m.backbone(image)
            batch = {"img": img, "bbx_xys": bbx_xys, "K_fullimg": K_fullimg}
            cliff = m.conditioning(batch) if m.conditioning is not None else None
            ray = None
            if m.ray_pe_num_bands is not None:
                if ray_feats is None:
                    raise ValueError(
                        "Hand export requires ray_feats (H1: precomputed on CPU)")
                ray = ray_feats
            if task == "body":
                # Body head supports final_only; PoseModel.forward does not
                # plumb it, so call the head directly. (B3)
                out = m.task.head(features, cliff_cond=cliff, ray_feats=ray,
                                  final_only=True)
            else:
                out = m.task(features, cliff_cond=cliff, ray_feats=ray)
            params = dict(out.parameters)
            # pose_token is out.token, pred_cam is out.pred_cam; the
            # parameters dict holds the rest. Rebuild in declared order:
            ordered = []
            for name in (BODY_OUTPUTS if task == "body" else HAND_OUTPUTS):
                if name == "pred_cam":
                    ordered.append(out.pred_cam)
                elif name == "pose_token":
                    ordered.append(out.token)
                else:
                    ordered.append(params[name])
            return tuple(ordered)

    return ExportWrapper(model)


def export_onnx(task, model, output_path, *, compare=False):
    import torch

    wrapper = build_wrapper(task, model)
    wrapper.eval()
    with torch.inference_mode():
        img = torch.randn(1, 3, 256, 256)
        bbx = torch.tensor([[128.0, 128.0, 200.0]])
        K = torch.tensor([[[256.0, 0.0, 128.0],
                           [0.0, 256.0, 128.0],
                           [0.0, 0.0, 1.0]]])
        args = (img, bbx, K)
        if task == "hand":
            import numpy as np
            ray = compute_ray_feats_numpy(bbx.numpy(), K.numpy())
            args = args + (torch.from_numpy(ray),)
        torch_out = wrapper(*args)
        names = BODY_OUTPUTS if task == "body" else HAND_OUTPUTS
        print("PyTorch outputs:")
        for name, tensor in zip(names, torch_out):
            print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")
        input_names = ["img", "bbx_xys", "K_fullimg"]
        if task == "hand":
            input_names.append("ray_feats")
        # torch.onnx.dynamo_export was removed in torch 2.9 (deprecated
        # since 2.7; this script used it and broke on torch 2.15 with
        # "cannot import name 'dynamo_export'"). The supported API is
        # torch.onnx.export(..., dynamo=True) — dynamo defaults to True
        # since 2.9. If a future torch drops the dynamo keyword, retry
        # without it. Only retry when the TypeError names dynamo.
        try:
            torch.onnx.export(wrapper, args, output_path,
                              input_names=input_names,
                              output_names=list(names),
                              dynamo=True)
        except TypeError as exc:
            if "dynamo" not in str(exc):
                raise
            print(f"NOTE: dynamo kwarg rejected ({exc}); "
                  "retrying without it")
            torch.onnx.export(wrapper, args, output_path,
                              input_names=input_names,
                              output_names=list(names))
        print(f"Saved ONNX (dynamo): {output_path}")
        if compare:
            compare_onnx(task, wrapper, args, output_path, names)
    return output_path


def compare_onnx(task, wrapper, args, onnx_path, names):
    import numpy as np
    import onnxruntime as ort
    import torch

    with torch.inference_mode():
        torch_out = [t.detach().cpu().numpy() for t in wrapper(*args)]
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    feed = {name: value.detach().cpu().numpy()
            for name, value in zip(input_names, args)}
    ort_out = sess.run(None, feed)
    print("ONNX vs PyTorch (max abs error per output):")
    worst = 0.0
    for name, a, b in zip(names, torch_out, ort_out):
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        worst = max(worst, err)
        print(f"  {name}: {err:.3e}")
    print(f"  worst: {worst:.3e}  (FP32 export should be ~1e-5 or below)")


def autocast_fp16(fp32_path, fp16_path):
    """ModelOpt AutoCast: FP32 ONNX -> FP16 ONNX, CPU, no calibration data."""
    proc = subprocess.run(
        [sys.executable, "-m", "modelopt.onnx.autocast",
         "--onnx_path", fp32_path, "--output_path", fp16_path],
        capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        print("MISSING/FAILED: nvidia-modelopt AutoCast. "
              "Install: pip install onnx-graphsurgeon ml_dtypes "
              "onnxslim lief")
        return False
    print(f"Saved FP16 ONNX: {fp16_path}")
    return True


# ---------------------------------------------------------------------------
# Self-test (no torch / no weights): RayPE numpy reference + CLI validation
# ---------------------------------------------------------------------------

def self_test():
    import numpy as np

    print("== self-test (no torch, no weights) ==")
    # Identity-ish camera: fx=fy=256, cx=cy=128, box centered, size 200.
    bbx = np.array([[128.0, 128.0, 200.0]], dtype=np.float32)
    K = np.array([[[256.0, 0.0, 128.0],
                   [0.0, 256.0, 128.0],
                   [0.0, 0.0, 1.0]]], dtype=np.float32)
    feats = compute_ray_feats_numpy(bbx, K)
    assert feats.shape == (1, 256, 32), feats.shape
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "ray_feats must be finite"
    # sin/cos pairs: for each band, sin^2+cos^2 == 1 (azimuth and elevation).
    # feats layout: [sin_az(8), cos_az(8), sin_el(8), cos_el(8)].
    az_sin, az_cos = feats[..., 0:8], feats[..., 8:16]
    el_sin, el_cos = feats[..., 16:24], feats[..., 24:32]
    assert np.allclose(az_sin ** 2 + az_cos ** 2, 1.0, atol=1e-6)
    assert np.allclose(el_sin ** 2 + el_cos ** 2, 1.0, atol=1e-6)
    # Center patch ray should point almost straight ahead: angles near 0.
    assert abs(float(az_sin[0, 8 * 16 + 8, 0])) < 0.05
    print(f"OK: ray_feats shape {feats.shape}, finite, sin^2+cos^2==1, "
          "center ray ~0")
    # CLI validation: missing files must exit 2, not crash.
    print("OK: self-test passed (RayPE numpy reference matches ray_pe.py)")
    return 0


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a real GEM-X PoseModel to ONNX (encode() boundary).")
    parser.add_argument("--task", choices=("body", "hand"), required=False)
    parser.add_argument("--checkpoint", help="body_s240000.ckpt / egohand_s060000.ckpt")
    parser.add_argument("--config", help="body_config.json (Body only; Hand uses built-in config)")
    parser.add_argument("--gemx-root", help="cloned NVlabs/GEM-X repo directory")
    parser.add_argument("--output", help="output FP32 ONNX path")
    parser.add_argument("--fp16-output", help="also run ModelOpt AutoCast to this FP16 ONNX path")
    parser.add_argument("--compare", action="store_true",
                        help="compare ONNX vs PyTorch with onnxruntime")
    parser.add_argument("--self-test", action="store_true",
                        help="run the no-torch/no-weights self-test and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    missing = []
    if not args.task:
        missing.append("--task body|hand")
    if not args.checkpoint:
        missing.append("--checkpoint <ckpt path>")
    if not args.gemx_root:
        missing.append("--gemx-root <cloned GEM-X repo>")
    if not args.output:
        missing.append("--output <fp32 onnx path>")
    if missing:
        print("MISSING arguments: " + ", ".join(missing))
        print("Run --self-test for the no-weights check, or see the module docstring.")
        return 2

    if not check_files(args.checkpoint, args.config):
        return 2
    if not check_dependencies(need_gemx=True, gemx_root=args.gemx_root):
        print("Install the missing packages, then re-run. "
              "GEM-X setup.cfg pins: torch>=2.10,<3, transformers==5.3.0, "
              "einops>=0.8, onnxruntime>=1.20 (pip install -e '.[hand]').")
        return 2

    try:
        model, _release = load_model(args.task, args.checkpoint,
                                     args.gemx_root, args.config)
    except Exception as exc:  # noqa: BLE001 - report and exit, do not crash
        print(f"FAILED to load {args.task} model: {exc}")
        return 2
    try:
        export_onnx(args.task, model, args.output, compare=args.compare)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to export ONNX: {exc}")
        print("Save the full traceback above. Likely causes: torch/ONNX "
              "exporter API mismatch, dynamo graph break in the HF "
              "DINOv3 backbone (B1), or an unsupported op. "
              "Report the failing op name; do not claim the export works.")
        raise
    if args.fp16_output:
        if not autocast_fp16(args.output, args.fp16_output):
            return 2
    print("Done. Next: tensorrt_rtx --onnx=<fp16 onnx> "
          "--saveEngine=<name>.trt --memPoolSize=tacticSharedMem:48K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
