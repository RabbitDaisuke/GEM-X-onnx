#!/usr/bin/env python3
"""Prepare real YOLOX-X HumanArt calibration from demonstration video frames.

Exactly matches NVlabs/GEM-X gem/common/detectors/yolox.py::_preprocess:
BGR frame; scale to fit in 640x640, top-left placement, uint8 114
padding, transpose to CHW, float32 WITHOUT division by 255.
This is 32-sample smoke-test calibration, not accuracy certification.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def iter_video_samples(file_path, count):
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {file_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"no video frames: {file_path}")
        for i in np.linspace(0, total - 1, count, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {i} in {file_path}")
            yield frame
    finally:
        cap.release()


def preprocess(frame):
    h, w = frame.shape[:2]
    ratio = min(640 / h, 640 / w)
    resized = cv2.resize(frame, (int(w * ratio), int(h * ratio)),
                         interpolation=cv2.INTER_LINEAR)
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return padded.transpose(2, 0, 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; refusing to overwrite")
    if args.samples < len(args.videos):
        parser.error("--samples must be >= number of videos")
    if not all(v.is_file() for v in args.videos):
        parser.error("one or more video sources are missing")
    counts = [args.samples // len(args.videos)] * len(args.videos)
    for i in range(args.samples % len(args.videos)):
        counts[i] += 1
    samples = []
    for path, count in zip(args.videos, counts):
        samples.extend(preprocess(frame) for frame in iter_video_samples(path, count))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, input=np.stack(samples, axis=0))
    print(f"YOLOX input: {len(samples)} samples, "
          f"{(len(samples), 3, 640, 640)} float32 BGR 114-pad -> {args.output}")
    print("NOTE: source demo frames contain overlays and limited poses; "
          "not production calibration or FP8 accuracy validation")


if __name__ == "__main__":
    main()
