#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an Apache-2.0 OpenCV Palm Detection NCHW FP16 ONNX release.

Run with an HF_TOKEN-bearing login shell for --push. The model is only
downloaded in the build environment; client L1 downloads the release ONNX.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen

import numpy as np
import onnx
import onnxruntime as ort

UPSTREAM = "opencv/palm_detection_mediapipe"
SOURCE = "palm_detection_mediapipe_2023feb.onnx"
SOURCE_SHA = "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c"
TARGET = "palm_detection_fp16.onnx"
REPOSITORY = "RabbitDaisuke/gemx-palm-fp16"


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as inp:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve(url, path):
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".partial")
    with urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(tmp, path)


def inspect(path, fp16=False):
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise RuntimeError("Palm ONNX must have one input")
    inp = model.graph.input[0]
    shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
    required_shape = [1, 3, 192, 192] if fp16 else [1, 192, 192, 3]
    required_type = onnx.TensorProto.FLOAT16 if fp16 else onnx.TensorProto.FLOAT
    if shape != required_shape or inp.type.tensor_type.elem_type != required_type:
        raise RuntimeError(f"Incorrect Palm ONNX input: {inp.name} {shape}")
    outputs = [([d.dim_value for d in x.type.tensor_type.shape.dim],
                x.type.tensor_type.elem_type) for x in model.graph.output]
    required_outputs = [([1, 2016, 18], required_type),
                        ([1, 2016, 1], required_type)]
    if outputs != required_outputs:
        raise RuntimeError(f"Incorrect Palm ONNX output: {outputs}")
    return model


def convert(source, target, directory):
    original = inspect(source)
    intermediate = directory / "palm_nhwc_fp16.onnx"
    if not intermediate.exists():
        subprocess.run([sys.executable, "-m", "modelopt.onnx.autocast",
                        "--onnx_path", str(source), "--output_path",
                        str(intermediate), "--data_max", "inf"], check=True)
    model = onnx.load(str(intermediate))
    old = model.graph.input[0]
    if old.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
        raise RuntimeError("ModelOpt did not change input to FP16")
    old_name = old.name
    del model.graph.input[:]
    model.graph.input.append(onnx.helper.make_tensor_value_info(
        "images", onnx.TensorProto.FLOAT16, [1, 3, 192, 192]))
    model.graph.node.insert(0, onnx.helper.make_node(
        "Transpose", ["images"], [old_name],
        name="gemx_nchw_adapter", perm=[0, 2, 3, 1]))
    partial = directory / (TARGET + ".partial")
    onnx.save(model, str(partial))
    inspect(partial, fp16=True)
    os.replace(partial, target)


def verify(source, target):
    base = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    half = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    assert [o.name for o in base.get_outputs()] == [
        o.name for o in half.get_outputs()]
    comparisons = []
    for seed in (101, 202, 303):
        frame = np.random.default_rng(seed).uniform(
            0, 1, (1, 192, 192, 3)).astype(np.float32)
        a = base.run(None, {base.get_inputs()[0].name: frame})
        b = half.run(None, {"images":
            np.ascontiguousarray(frame.transpose(0, 3, 1, 2)).astype(np.float16)})
        measurements = []
        for expected, actual in zip(a, b):
            expected, actual = expected.astype(np.float32), actual.astype(np.float32)
            if expected.shape != actual.shape or not np.isfinite(actual).all():
                raise RuntimeError("Invalid output from FP16 model")
            delta = np.abs(expected - actual)
            relative = delta / np.maximum(1.0, np.abs(expected))
            metrics = {"max_abs": float(delta.max()),
                       "p99_abs": float(np.percentile(delta, 99)),
                       "max_relative_scaled": float(relative.max())}
            if metrics["max_relative_scaled"] > 0.05:
                raise RuntimeError(f"FP16 numerical mismatch: {metrics}")
            measurements.append(metrics)
        comparisons.append({"seed": seed, "output_metrics": measurements})
    return comparisons


def release(out, original, converted, measurements):
    folder = out / "release"
    folder.mkdir(exist_ok=True)
    shutil.copyfile(converted, folder / TARGET)
    shutil.copyfile(out / "LICENSE", folder / "LICENSE")
    metadata = {
        "original_model": UPSTREAM, "original_file": SOURCE,
        "original_sha256": SOURCE_SHA,
        "upstream_url": f"https://huggingface.co/{UPSTREAM}",
        "license": "Apache-2.0",
        "conversion": "NVIDIA ModelOpt ONNX AutoCast --data_max inf",
        "input_adapter": "ONNX Transpose NCHW to NHWC (for SCG GPU warp)",
        "result_sha256": hash_file(converted),
        "result_bytes": converted.stat().st_size,
        "cpu_onnxruntime_comparison": measurements,
        "validation_note": "Synthetic numeric samples, NOT accuracy or SCG verification",
    }
    (folder / "provenance.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card = f"""---
license: apache-2.0
base_model: {UPSTREAM}
tags:
- onnx
- fp16
- hand-detection
---
# GEM-X Palm Detection FP16 ONNX
Converted from OpenCV MediaPipe Palm Detection ({UPSTREAM}).
Original source file: {SOURCE}. Original SHA-256: {SOURCE_SHA}.
The unchanged upstream Apache-2.0 LICENSE is included. This is a derivative and not an NVIDIA RT-DETR Hand Detector.
## Conversion
NVIDIA ModelOpt ONNX AutoCast to FP16; ONNX Transpose node added to accept NCHW input.
Input: images, float16 [1,3,192,192], RGB in 0..1 with aspect-preserving centred black letterbox.
Outputs: Identity [1,2016,18] raw SSD box/keypoint offsets; Identity_1 [1,2016,1] raw logits.
The model does not output left/right labels. The consumer must decode anchors and apply NMS.
## Verification
See provenance.json for exact hashes and FP32/FP16 numerical comparison.
CPU numerical agreement does not prove TensorRT for RTX/SCG support or real-world hand detection accuracy.
"""
    (folder / "README.md").write_text(card, encoding="utf-8")
    return folder


def publish(folder, repository):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN absent from login shell")
    api = HfApi(token=token)
    identity = api.whoami()
    names = {identity.get("name")}
    names.update(entry.get("name") for entry in identity.get("orgs", []))
    if repository.split("/", 1)[0] not in names:
        raise RuntimeError("HF_TOKEN account cannot publish to requested owner")
    try:
        api.repo_info(repo_id=repository, repo_type="model")
        print("HF repo exists: creation skipped", flush=True)
    except RepositoryNotFoundError:
        print("Creating new HF model repo", flush=True)
        api.create_repo(repo_id=repository, repo_type="model",
                        private=False, exist_ok=False)
    api.upload_folder(folder_path=str(folder), repo_id=repository,
                      repo_type="model",
                      commit_message="Add Apache-2.0 OpenCV Palm FP16 ONNX and provenance")
    names = set(api.list_repo_files(repo_id=repository, repo_type="model"))
    if not {TARGET, "README.md", "LICENSE", "provenance.json"} <= names:
        raise RuntimeError("HF upload incomplete")
    print("HF publish verified: https://huggingface.co/" + repository, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/palm-fp16"))
    ap.add_argument("--repo", default=REPOSITORY)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    work = args.out.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = work / SOURCE
    retrieve(f"https://huggingface.co/{UPSTREAM}/resolve/main/{SOURCE}", source)
    if hash_file(source) != SOURCE_SHA:
        raise RuntimeError("Source SHA-256 mismatch")
    retrieve(f"https://huggingface.co/{UPSTREAM}/resolve/main/LICENSE",
             work / "LICENSE")
    converted = work / TARGET
    if not converted.exists():
        convert(source, converted, work)
    inspect(converted, fp16=True)
    comparisons = verify(source, converted)
    print("FP16 verified:", json.dumps(comparisons), flush=True)
    folder = release(work, source, converted, comparisons)
    print("FP16 digest:", hash_file(converted), flush=True)
    if args.push:
        publish(folder, args.repo)


if __name__ == "__main__":
    main()
