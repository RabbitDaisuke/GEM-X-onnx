#!/usr/bin/env python3
"""Create real RT-DETR Hand Detector calibration inputs from video frames.

Calibrate against the ACTUAL Windows gemx-python/detectors.py GPU path:
BGR->RGB, 640x640 aspect-preserving centered gray-114 letterbox,
/255 float32 NCHW. The original target sizes input is [640,640] int64
because the detector sees the already-letterboxed image, and Windows
unletterboxes boxes after inference. The previous stretched image +
original [width,height] calibration was for the upstream NVlabs path
and was mismatched to this Windows inference route. These sample frames
contain overlays and remain preliminary calibration only.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
 

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
            scale = min(640.0 / h, 640.0 / w)
            resized_w, resized_h = round(w * scale), round(h * scale)
            resized = cv2.resize(
                frame, (resized_w, resized_h),
                interpolation=cv2.INTER_LINEAR)
            left = (640 - resized_w) // 2
            top = (640 - resized_h) // 2
            padded = np.full((640, 640, 3), 114, dtype=np.uint8)
            padded[top:top+resized_h, left:left+resized_w] = resized
            rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
            images.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            orig.append([640, 640])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, images=np.ascontiguousarray(np.stack(images), np.float32),
        orig_target_sizes=np.asarray(orig, dtype=np.int64))
    print(f"{args.output}: {len(images)} samples, images="
          f"{(len(images), 3, 640, 640)} float32; orig_target_sizes int64")
    print("NOTE: inputs match Windows centered letterbox + orig sizes "
          "[640,640]. Demo footage has pose overlays and does NOT "
          "establish production accuracy.")


if __name__ == "__main__":
    main()
