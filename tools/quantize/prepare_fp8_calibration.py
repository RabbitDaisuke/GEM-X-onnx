#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare genuine GEM-X Body/Hand FP8 calibration feeds from inference logs.

Run on the Windows GPU PC where the original video exists, then transfer
the resulting small NPZ files into the Codespace for ModelOpt FP8 PTQ.

    python tools/quantize/prepare_fp8_calibration.py \
      --video IMG_0174.MOV --inference-json out/gpu-profile.json \
      --output-dir out/calibration --max-samples 256

Uses the actual image crop/normalization and hand mirror conventions of
gemx-python/gpu_pre.py, *not* synthetic random calibration inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from export_onnx_real import compute_ray_feats_numpy

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
INV_STD = np.float32(1.0) / np.array([0.229, 0.224, 0.225],
                                      dtype=np.float32)


def crop_matrix(bbox, side=256, mirror=False):
    cx, cy, box_side = [float(x) for x in bbox]
    scale = box_side / side
    row0 = [-scale, 0., cx + box_side / 2] if mirror else [
        scale, 0., cx - box_side / 2]
    return np.array([row0, [0., scale, cy - box_side / 2]], np.float32)


def mirror_params(bbox, K, frame_width):
    limit = float(frame_width - 1)
    bbox = np.array(bbox, dtype=np.float32).copy()
    bbox[0] = limit - bbox[0]
    flip = np.eye(3, dtype=np.float32)
    flip[0, 0] = -1.
    flip[0, 2] = limit
    K = (flip @ np.asarray(K, dtype=np.float32)) * np.array(
        [-1., 1., 1.], dtype=np.float32)
    return bbox, K


def crop_normalized(img_bgr, bbox, mirror=False):
    """Reference to GPU affine warp + bilinear + BGR->RGB + ImageNet."""
    h, w = img_bgr.shape[:2]
    M = crop_matrix(bbox, 256, mirror)
    x = np.arange(256, dtype=np.float32)[None, :]
    y = np.arange(256, dtype=np.float32)[:, None]
    sx = (M[0, 0] * x + M[0, 1] * y + M[0, 2]).astype(np.float32)
    sy = (M[1, 0] * x + M[1, 1] * y + M[1, 2]).astype(np.float32)
    x0, y0 = np.floor(sx), np.floor(sy)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    rgb = img_bgr[:, :, ::-1].astype(np.float32)

    def tap(xx, yy):
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        px = rgb[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]
        return np.where(valid[..., None], px, np.float32(0))

    one = np.float32(1.)
    a, b = tap(x0, y0), tap(x0 + 1, y0)
    d, e = tap(x0, y0 + 1), tap(x0 + 1, y0 + 1)
    pixels = (a * (one - fx) + b * fx) * (one - fy) + (
        d * (one - fx) + e * fx) * fy
    norm = (pixels.astype(np.float32) * np.float32(1. / 255.)
            - MEAN) * INV_STD
    return np.ascontiguousarray(norm.transpose(2, 0, 1), dtype=np.float16)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--inference-json", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args(argv)
    if args.max_samples <= 0 or args.stride <= 0:
        ap.error("--max-samples and --stride must be positive")
    if not args.video.is_file() or not args.inference_json.is_file():
        ap.error("video and inference JSON must exist")
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python-headless is required") from exc

    with open(args.inference_json, encoding="utf-8") as fh:
        log = json.load(fh)
    frames = log.get("frames", [])
    if not frames:
        raise SystemExit("inference JSON has no frames")
    selected = {}
    for index, item in enumerate(frames):
        frame_number = item.get("frame_index")
        if (not isinstance(frame_number, int) or frame_number < 0
                or index % args.stride):
            continue
        selected[frame_number] = item
    if not selected:
        raise SystemExit("no usable frames")

    store = {"body": {k: [] for k in ("img", "bbx_xys", "K_fullimg")},
             "hand": {k: [] for k in ("img", "bbx_xys", "K_fullimg",
                                      "ray_feats")}}
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"failed to decode {args.video}")
    found = 0
    last = max(selected)
    try:
        index = 0
        while index <= last:
            ok, frame = cap.read()
            if not ok:
                break
            record = selected.get(index)
            index += 1
            if record is None:
                continue
            K = np.array(record["K"], dtype=np.float32)
            if K.shape != (3, 3):
                raise ValueError(f"frame {index - 1}: invalid K shape")
            height, width = frame.shape[:2]
            image_size = record.get("image_size")
            if image_size and list(image_size) != [width, height]:
                raise ValueError(f"frame {index - 1}: video dimensions do not match")
            for task, detections in (("body", record.get("persons", [])),
                                     ("hand", record.get("hands", []))):
                if len(store[task]["img"]) >= args.max_samples:
                    continue
                for entry in detections:
                    bbox = entry.get("crop_xys")
                    if not bbox or len(bbox) != 3:
                        continue
                    if entry.get("pose") is None and task == "body":
                        # The JSON contains the crop even for failed frames.
                        # Calibrate only on successfully processed persons.
                        continue
                    mirror = task == "hand" and entry.get("side") == "right"
                    img = crop_normalized(frame, bbox, mirror)
                    bbx = np.asarray(bbox, np.float32)
                    cam = K
                    if mirror:
                        bbx, cam = mirror_params(bbx, K, width)
                    store[task]["img"].append(img)
                    store[task]["bbx_xys"].append(bbx.astype(np.float16))
                    store[task]["K_fullimg"].append(
                        np.asarray(cam, np.float16))
                    if task == "hand":
                        rays = compute_ray_feats_numpy(
                            bbx[None], cam[None])[0]
                        store[task]["ray_feats"].append(
                            rays.astype(np.float16))
                    if len(store[task]["img"]) >= args.max_samples:
                        break
            found += 1
            if all(len(store[t]["img"]) >= args.max_samples
                   for t in store):
                break
    finally:
        cap.release()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task, arrays in store.items():
        n = len(arrays["img"])
        if not n:
            raise RuntimeError(f"no {task} samples found; check inference JSON")
        path = args.output_dir / f"{task}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        np.savez_compressed(path, **{
            name: np.stack(items, axis=0).astype(np.float16)
            for name, items in arrays.items()})
        print(f"{task}: {n} genuine model input samples -> {path}", flush=True)
    print(f"decoded and matched {found} logged frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
