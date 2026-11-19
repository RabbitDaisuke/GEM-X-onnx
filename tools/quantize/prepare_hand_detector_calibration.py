#!/usr/bin/env python3
"""Create real RT-DETR Hand Detector calibration inputs from video frames.

The RT-DETR preprocessing is copied from GEM-X/gem/common/detectors/hand.py:
RGB, PIL bilinear resize 640x640, [0,1] float32, NCHW; original image
dimensions [width,height] as int64. Upstream demo video frames contain
visual overlays and are for preliminary FP8 conversion only.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sample_video(path, count):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open calibration source {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"Cannot determine frame count of {path}")
        for frame_number in np.linspace(max(0, total // 10),
                                        max(0, total - 1 - total // 10),
                                        count, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{path} frame {frame_number} not decodable")
            yield frame
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.samples < len(args.videos) or args.samples < 1:
        parser.error("samples must be >= number of videos")
    if args.output.exists():
        parser.error("refusing to overwrite calibration data")
    if any(not v.is_file() for v in args.videos):
        parser.error("all video files must exist")
    images = []
    orig = []
    counts = [(args.samples // len(args.videos)) for _ in args.videos]
    for i in range(args.samples % len(args.videos)):
        counts[i] += 1
    for path, count in zip(args.videos, counts):
        for frame in sample_video(path, count):
            h, w = frame.shape[:2]
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            resized = image.resize((640, 640), Image.Resampling.BILINEAR)
            pixels = np.asarray(resized, dtype=np.float32) / 255.0
            images.append(pixels.transpose(2, 0, 1))
            orig.append([w, h])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, images=np.ascontiguousarray(np.stack(images), np.float32),
        orig_target_sizes=np.asarray(orig, dtype=np.int64))
    print(f"{args.output}: {len(images)} samples, images="
          f"{(len(images), 3, 640, 640)} float32; orig_target_sizes int64")
    print("NOTE: official demo footage contains pose overlays. This is "
          "conversion calibration, not production accuracy validation.")


if __name__ == "__main__":
    main()
