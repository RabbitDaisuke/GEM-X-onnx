#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provisional GEM-X FP8 calibration from upstream's real demonstration videos.

Creates inputs from REAL camera frames in the NVlabs/GEM-X repository.
The source clips include visualized pose overlays and manually annotated
coarse crops; this is a smoke-test calibration set, NOT production-grade
calibration for the user's own unannotated capture domain.

Run only after the upstream source has been cloned:
    .venv/bin/python tools/quantize/prepare_demo_calibration.py \
        --gemx-root GEM-X --output-dir out/calibration --samples 32
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from export_onnx_real import compute_ray_feats_numpy
from prepare_fp8_calibration import crop_normalized, mirror_params

# The public NVlabs demo videos show original camera views alongside
# graphics. These are manually measured approximate squares around the
# photographed subject/hand in the original 1280x720 canvas, not
# object-detector results. Keep their limitations explicit in the manifest.
BODY_VIDEOS = (
    ("body_pose_motion.mp4", (260.0, 340.0, 550.0)),
    ("body_motion_night.mp4", (250.0, 342.0, 525.0)),
)
HAND_VIDEO = "hand_pose.mp4"
HAND_BOXES = (
    ("left", (290.0, 385.0, 355.0)),
    ("right", (560.0, 380.0, 365.0)),
)


def frames_from_video(path, count):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to read upstream demo clip: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < count:
            raise RuntimeError(f"{path}: only {total} frames for {count} requested")
        for pos in np.linspace(0, total - 1, count, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{path}: cannot decode frame {pos}")
            yield frame
    finally:
        cap.release()


def intrinsics(frame):
    height, width = frame.shape[:2]
    return np.array([[width, 0., width / 2.],
                     [0., width, height / 2.],
                     [0., 0., 1.]], dtype=np.float32)


def accumulate(stack, frame, bbox, *, side=None):
    K = intrinsics(frame)
    mirror = side == "right"
    bbx = np.asarray(bbox, dtype=np.float32)
    img = crop_normalized(frame, bbx, mirror=mirror)
    if mirror:
        bbx, K = mirror_params(bbx, K, frame.shape[1])
    stack["img"].append(img)
    stack["bbx_xys"].append(np.asarray(bbx, dtype=np.float16))
    stack["K_fullimg"].append(np.asarray(K, dtype=np.float16))
    if side is not None:
        stack["ray_feats"].append(np.asarray(
            compute_ray_feats_numpy(bbx[None], K[None])[0], dtype=np.float16))


def write_npz(destination, data):
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing {destination}")
    if not data["img"]:
        raise RuntimeError(f"no real frames for {destination}")
    np.savez_compressed(destination, **{
        name: np.stack(values, axis=0) for name, values in data.items()})
    print(f"{destination}: {len(data['img'])} real frames "
          f"(shape={data['img'][0].shape})", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemx-root", type=Path, default=Path("GEM-X"))
    ap.add_argument("--output-dir", type=Path, default=Path("out/calibration"))
    ap.add_argument("--samples", type=int, default=32,
                    help="minimum 32; total per task (split between Body clips)")
    args = ap.parse_args(argv)
    if args.samples < 32 or args.samples % 2:
        ap.error("--samples must be even and at least 32")
    assets = args.gemx_root / "assets"
    body = {"img": [], "bbx_xys": [], "K_fullimg": []}
    for name, bbox in BODY_VIDEOS:
        for frame in frames_from_video(assets / name, args.samples // 2):
            accumulate(body, frame, bbox)
    hand = {"img": [], "bbx_xys": [], "K_fullimg": [], "ray_feats": []}
    for frame in frames_from_video(assets / HAND_VIDEO, args.samples // 2):
        for side, bbox in HAND_BOXES:
            accumulate(hand, frame, bbox, side=side)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_npz(args.output_dir / "body_calibration.npz", body)
    write_npz(args.output_dir / "hand_calibration.npz", hand)
    manifest = {
        "source": "NVlabs/GEM-X official demonstration videos",
        "source_commit": "4aef1452e5df171deff4f293c08e643eb450981e",
        "samples_per_model": args.samples,
        "manual_crops": {"body": BODY_VIDEOS, "hand": HAND_BOXES},
        "disclosure": "Visualized pose overlays; coarse fixed crops. "
                      "Preliminary FP8 PTQ only. Replace with true GPU "
                      "inference inputs for production accuracy checks.",
    }
    with open(args.output_dir / "demo_manifest.json", "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print("Calibration is genuine upstream video, NOT synthetic data; "
          "its overlay-domain limitations are recorded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
