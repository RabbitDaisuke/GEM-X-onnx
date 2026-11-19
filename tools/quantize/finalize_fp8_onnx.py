#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize verified GEM-X FP8 ONNX as canonical filenames.

The outputs made with the fixed ModelOpt scale calculation were named
body_fp8_repaired.onnx and hand_fp8_repaired.onnx to avoid overwriting
the original faulty conversions. This publishes canonical filenames
body_fp8.onnx / hand_fp8.onnx with matching external weight filenames
for the Windows TensorRT-RTX runtime.

Only ONNX GRAPH METADATA (about 1 MiB) is written. The ~1.7 GiB external
weights are hard-linked on the same filesystem, NOT copied or converted.
The original repaired graph, data and FP16 ONNX are all retained.

Usage:
  python tools/quantize/finalize_fp8_onnx.py --task body --out-dir out
  python tools/quantize/finalize_fp8_onnx.py --task hand --out-dir out
"""
import argparse
import json
import os
from pathlib import Path

import onnx

from fp8_quantize_onnx import inspect_fp8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("body", "hand"), required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    args = parser.parse_args(argv)

    src = args.out_dir / f"{args.task}_fp8_repaired.onnx"
    dst = args.out_dir / f"{args.task}_fp8.onnx"
    src_data = Path(str(src) + "_data")
    dst_data = Path(str(dst) + "_data")
    dst_meta = Path(str(dst) + ".quantization.json")
    src_meta = Path(str(src) + ".quantization.json")
    staging = Path(str(dst) + ".metadata-part")
    for required in (src, src_data, src_meta):
        if not required.is_file():
            parser.error(f"missing required repaired FP8 file: {required}")
    for target in (dst, dst_data, dst_meta, staging):
        if target.exists():
            parser.error(f"will not replace existing destination: {target}")

    old_info = json.loads(src_meta.read_text(encoding="utf-8"))
    if old_info.get("nonfinite_scale_count") != 0:
        raise ValueError("the repaired model has not passed all-scale validation")

    graph = onnx.load(str(src), load_external_data=False)
    replaced = 0
    external_count = 0
    for initializer in graph.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        external_count += 1
        entries = {item.key: item for item in initializer.external_data}
        location = entries.get("location")
        if location is None or location.value != src_data.name:
            raise ValueError(
                f"external weight {initializer.name!r} does not reference "
                f"expected sidecar {src_data.name!r}")
        location.value = dst_data.name
        replaced += 1
    if external_count == 0 or replaced != external_count:
        raise ValueError(f"no external data pointers to finalize: {src}")
    # A safe, small graph rewrite. No external weight tensors are loaded.
    onnx.save_model(graph, str(staging))
    try:
        # os.link is zero-copy and makes the canonical pair portable when
        # both files are copied together to Windows. If unsupported by the
        # filesystem, fail instead of silently copying gigabytes.
        os.link(src_data, dst_data)
        os.replace(staging, dst)
        info = inspect_fp8(dst)
        info.update(task=args.task, source_onnx=old_info.get("source_onnx"),
                    output_onnx=str(dst), calibration_samples=old_info.get(
                        "calibration_samples"), calibration_ep=old_info.get(
                        "calibration_ep"), calibration_method=old_info.get(
                        "calibration_method"))
        dst_meta.write_text(json.dumps(info, ensure_ascii=False, indent=2)
                            + "\n", encoding="utf-8")
    except Exception:
        # Preserve source and any diagnostic files. Never remove the source
        # graph or data. A partly created target is reported explicitly.
        raise

    print(f"FP8 {args.task}: {dst} + {dst_data}")
    print(f"Repointed {replaced} external tensors without duplicating weights")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
