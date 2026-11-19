#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo calibration feeds for the ACTUAL Windows RT-DETR R18 person detector.

Source: onnx-community/rtdetr_r18vd_coco_o365
Runtime: gemx-python/detectors.py + gpu_pre.py.
The runtime uses BGR->RGB, centered letterbox 114, /255 and NCHW float32.
The 1280x720 upstream demo frames lead to an integer 140-pixel vertical pad.
Other frame sizes use rounded placement; final production calibration should
be captured from the running GPU pipeline on the actual user's video.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def frames(path, n):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < n:
            raise RuntimeError(f"{path} has only {count} frames, need {n}")
        for position in np.linspace(0, count - 1, n, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {position} of {path}")
            yield frame
    finally:
        cap.release()


def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(640.0 / h, 640.0 / w)
    resized_w = round(w * scale)
    resized_h = round(h * scale)
    scaled = cv2.resize(frame, (resized_w, resized_h),
                        interpolation=cv2.INTER_LINEAR)
    left = (640 - resized_w) // 2
    top = (640 - resized_h) // 2
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[top:top + resized_h, left:left + resized_w] = scaled
    # RT-DETR R18 detector consumes RGB float32 in [0,1], no ImageNet norm.
    rgb = padded[:, :, ::-1].astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; not overwriting")
    if args.samples < len(args.videos):
        parser.error("samples must be at least the video count")
    for file in args.videos:
        if not file.is_file():
            parser.error(f"video does not exist: {file}")
    quotas = [args.samples // len(args.videos)] * len(args.videos)
    for i in range(args.samples % len(args.videos)):
        quotas[i] += 1
    batch = [preprocess(frame) for path, n in zip(args.videos, quotas)
             for frame in frames(path, n)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
                        pixel_values=np.stack(batch).astype(np.float32))
    print(f"RT-DETR R18 calibration: {len(batch)} frames, "
          f"{(len(batch), 3, 640, 640)} float32 RGB, -> {args.output}")
    print("NOTICE: overlayed demo footage / CPU bilinear approximation; "
          "for accuracy use input captured from Windows GPU inference.")


if __name__ == "__main__":
    main()
