#!/usr/bin/env python3
"""GEM-X FP16 quantization + ONNX export — ONE script, run on a Codespace.

Usage (in a GitHub Codespace created from this repo, 4-core / 16 GB machine):

    python3 run_gemx_onnx.py            # Body + Hand, end to end
    python3 run_gemx_onnx.py --task body
    python3 run_gemx_onnx.py --task hand --keep-weights --keep-fp32-onnx
    python3 run_gemx_onnx.py --self-test   # preflight only, no downloads

What it does, in order, one model at a time (Body, then Hand):
  1. Preflight: Python version, RAM, free disk. Warns, does not guess.
  2. Clones NVlabs/GEM-X at commit 4aef145 (if not already there).
  3. Creates .venv if needed, installs CPU torch + GEM-X deps, then
     re-runs itself inside the venv. You run this script exactly once.
  4. Downloads that model's weights from Hugging Face (no login first;
     EgoHand falls back to the `hf` CLI with HF_TOKEN if it 401/403s).
  5. Inspects the checkpoint, exports ONNX (FP32 + FP16, with an
     onnxruntime-vs-PyTorch compare), and FP16-casts the checkpoint.
  6. Deletes the downloaded checkpoint and the FP32 ONNX afterwards to
     stay inside Codespace disk (32 GB), unless --keep-weights /
     --keep-fp32-onnx is given. The FP16 ONNX + FP16 checkpoint stay in
     out/ either way.

Why one model at a time: a 16 GB machine cannot hold both checkpoints
(3.5 GB + 10.7 GB) plus both FP32 ONNX files at once. Each tool runs as
its own process, so RAM is freed between steps. All torch.load calls in
the tools use mmap=True (the 10.66 GB Hand checkpoint is never copied
wholesale into RAM) and omegaconf DictConfig is allowlisted for
weights_only loads — the two bugs that OOM-killed the old Actions run.

Disk budget (Codespace disk is 32 GB — that is the hard limit):
  .venv + deps ~1.5 GB, GEM-X clone ~0.1 GB,
  Body peak ~10 GB, Hand peak ~21 GB (with Body's outputs kept),
  final out/ ~7.3 GB. Fits, with headroom, as long as each model's
  checkpoint and FP32 ONNX are deleted right after use (the default).
  --keep-weights + --keep-fp32-onnx together need ~30 GB and do NOT
  fit on a 32 GB disk — keep at most one of them.

Cores: the 4-core machine is used fully — OMP/MKL/OpenBLAS threads are
set to os.cpu_count() for every step (torch, onnxruntime, numpy).

What this does NOT do: TensorRT-RTX engine build (needs a GPU machine)
and real latency measurement. The next step after this script is the
tensorrt_rtx build on the GPU machine, using out/*_fp16.onnx.

Codespace note: create the Codespace with the 4-core / 16 GB machine
type. The 2-core / 8 GB type does not have enough RAM for this.
HF_TOKEN: only needed if the no-login EgoHand download fails. Set it as
a Codespace secret or `export HF_TOKEN=...` before running.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

GEMX_REPO = "https://github.com/NVlabs/GEM-X.git"
GEMX_REF = "4aef1452e5df171deff4f293c08e643eb450981e"

TASKS = {
    "body": {
        "repo": "nvidia/GEM-X",
        "checkpoint": "body_s240000.ckpt",
        "extra_download": ["body_config.json"],
        "config_arg": ["--config", "weights/body_config.json"],
    },
    "hand": {
        "repo": "nvidia/EgoHand-v1",
        "checkpoint": "egohand_s060000.ckpt",
        "extra_download": [],
        "config_arg": [],
    },
}

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
DEPS = [
    "transformers==5.3.0", "py-soma-x==0.3.3",
    "numpy>=1.26,<3", "scipy>=1.11,<2", "einops>=0.8,<1",
    "pillow>=10,<13", "opencv-python-headless>=4.10,<5",
    "omegaconf>=2.3,<2.4", "onnx", "onnxruntime>=1.20,<2",
    "nvidia-modelopt", "huggingface_hub", "safetensors",
]


def log(msg):
    print(f"[run_gemx_onnx] {msg}", flush=True)


def log_disk():
    usage = shutil.disk_usage(".")
    log(f"disk free: {gb(usage.free):.1f} GB")


def run(cmd, *, check=True, env=None):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=check, env=env)


def gb(n):
    return n / (1024 ** 3)


def preflight():
    """Report Python / RAM / disk. Returns False only on a hard blocker."""
    log(f"Python {sys.version.split()[0]} (need >= 3.11)")
    if sys.version_info < (3, 11):
        print("ERROR: Python 3.11+ is required.", file=sys.stderr)
        return False
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            mem = {l.split(":")[0]: l.split(":")[1].strip()
                   for l in fh if ":" in l}
        total_gb = int(mem["MemTotal"].split()[0]) / (1024 ** 2)
        log(f"RAM total: {total_gb:.1f} GB (16 GB machine recommended)")
        if total_gb < 12:
            print("WARNING: less than 12 GB RAM — the Hand export may OOM.",
                  file=sys.stderr)
    except Exception:
        log("RAM: unknown (not /proc/meminfo)")
    usage = shutil.disk_usage(".")
    log(f"Disk free: {gb(usage.free):.1f} GB (Codespace disk is 32 GB; "
        "this pipeline peaks at ~21 GB, ~30 GB with --keep-weights "
        "--keep-fp32-onnx)")
    if gb(usage.free) < 18:
        print("ERROR: less than 18 GB free disk — the Hand export will "
              "not fit on this disk.", file=sys.stderr)
        return False
    if gb(usage.free) < 22:
        print("WARNING: less than 22 GB free disk — tight for the Hand "
              "phase.", file=sys.stderr)
    if shutil.which("git") is None:
        print("ERROR: git is required.", file=sys.stderr)
        return False
    return True


def ensure_gemx(root):
    """Clone NVlabs/GEM-X at GEMX_REF, or verify an existing checkout."""
    if os.path.isdir(os.path.join(root, "gem")):
        got = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        if got.stdout.strip() == GEMX_REF:
            log(f"GEM-X already present at {GEMX_REF}")
            return
        print(f"ERROR: {root} exists but is not at {GEMX_REF} "
              f"(got {got.stdout.strip()}). Remove it and re-run.",
              file=sys.stderr)
        raise SystemExit(2)
    run(["git", "clone", "--depth", "1", GEMX_REPO, root])
    run(["git", "-C", root, "rev-parse", "HEAD"])
    got = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    if got.stdout.strip() != GEMX_REF:
        print(f"ERROR: cloned GEM-X is {got.stdout.strip()}, "
              f"expected {GEMX_REF}.", file=sys.stderr)
        raise SystemExit(2)


def ensure_venv(root):
    """Create .venv with CPU torch + deps if torch is not importable.

    Returns the python executable to use for the tool steps. If the
    current interpreter already has torch, it is used as-is.
    """
    try:
        import torch  # noqa: F401
        log("torch already importable in this interpreter")
        return sys.executable
    except ImportError:
        pass
    venv = os.path.join(root, ".venv")
    venv_py = os.path.join(venv, "bin", "python")
    if not os.path.exists(venv_py):
        log("Creating .venv ...")
        run([sys.executable, "-m", "venv", venv])
        run([venv_py, "-m", "pip", "install", "--upgrade", "pip",
             "--no-cache-dir"])
        run([venv_py, "-m", "pip", "install", "--no-cache-dir",
             "torch>=2.10,<3", "--index-url", CPU_TORCH_INDEX])
        run([venv_py, "-m", "pip", "install", "--no-cache-dir", *DEPS])
        run([venv_py, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
             "--no-build-isolation", "-e", os.path.join(root, "GEM-X")])
    else:
        log(".venv already exists, reusing it")
    # Re-exec this script inside the venv so every later step uses it.
    if os.environ.get("GEMX_ONNX_IN_VENV") != "1":
        log("Re-running inside .venv ...")
        env = dict(os.environ, GEMX_ONNX_IN_VENV="1")
        raise SystemExit(subprocess.run(
            [venv_py, os.path.abspath(__file__), *sys.argv[1:]],
            env=env).returncode)
    return venv_py


def download(task, spec, root, tools_dir):
    """Download one task's weights (no login first, HF_TOKEN fallback)."""
    ckpt = spec["checkpoint"]
    dl = os.path.join(tools_dir, "hf_download.py")
    files = [ckpt, *spec["extra_download"]]
    if task == "hand":
        # Hand detector is small and needed later for gem-infer.
        run([sys.executable, dl, "--repo", "nvidia/RT-DETR-Hand-Detector-v1",
             "--file", "hand_detector.onnx", "--out-dir", "weights"],
            check=True)
    try:
        run([sys.executable, dl, "--repo", spec["repo"],
             "--file", ",".join(files), "--out-dir", "weights"], check=True)
    except subprocess.CalledProcessError:
        token = os.environ.get("HF_TOKEN")
        if task == "hand" and token:
            log("No-login download failed, retrying with HF_TOKEN ...")
            run(["hf", "download", spec["repo"], ckpt,
                 "--local-dir", "weights"], check=True)
        else:
            print("ERROR: download failed. For EgoHand, set HF_TOKEN "
                  "(Codespace secret or export HF_TOKEN=...) and re-run.",
                  file=sys.stderr)
            raise
    for name in files:
        path = os.path.join("weights", name)
        log(f"downloaded {path}: {os.path.getsize(path) / (1024 ** 2):.0f} MB")
    log_disk()


def run_task(task, spec, root, tools_dir, args):
    """Download -> inspect -> export ONNX -> FP16-cast one model."""
    download(task, spec, root, tools_dir)
    ckpt_path = os.path.join("weights", spec["checkpoint"])
    out_dir = os.path.join("out")
    os.makedirs(out_dir, exist_ok=True)

    log(f"== {task}: inspect {ckpt_path} ==")
    with open(os.path.join(out_dir, f"inspect_{task}.txt"), "w",
              encoding="utf-8") as fh:
        subprocess.run([sys.executable,
                        os.path.join(tools_dir, "inspect_ckpt.py"),
                        ckpt_path], stdout=fh, stderr=subprocess.STDOUT,
                       check=True)

    log(f"== {task}: export ONNX (FP32 + FP16, compare) ==")
    export_cmd = [sys.executable,
                  os.path.join(tools_dir, "export_onnx_real.py"),
                  "--task", task, "--gemx-root", os.path.join(root, "GEM-X"),
                  "--checkpoint", ckpt_path, *spec["config_arg"],
                  "--output", os.path.join(out_dir, f"{task}_fp32.onnx"),
                  "--fp16-output", os.path.join(out_dir, f"{task}_fp16.onnx"),
                  "--compare"]
    with open(os.path.join(out_dir, f"export_{task}.log"), "w",
              encoding="utf-8") as fh:
        subprocess.run(export_cmd, stdout=fh, stderr=subprocess.STDOUT,
                       check=True)
    log_path = os.path.join(out_dir, f"export_{task}.log")
    if os.path.exists(log_path):
        lines = [l for l in open(log_path, encoding="utf-8")
                 .read().splitlines() if l.strip()]
        log("export last line: " + (lines[-1] if lines else "(empty log)"))
    log_disk()

    # Delete the FP32 ONNX right away (before quantize) to keep the
    # 32 GB disk headroom: the FP16 ONNX is what the TRT build uses.
    if not args.keep_fp32_onnx:
        fp32 = os.path.join(out_dir, f"{task}_fp32.onnx")
        if os.path.exists(fp32):
            log(f"removing {fp32} (use --keep-fp32-onnx to keep it)")
            os.remove(fp32)
            log_disk()

    if not args.skip_quantize:
        log(f"== {task}: FP16-cast checkpoint ==")
        with open(os.path.join(out_dir, f"quantize_{task}.log"), "w",
                  encoding="utf-8") as fh:
            subprocess.run([sys.executable,
                            os.path.join(tools_dir,
                                         "quantize_ckpt_to_fp16.py"),
                            "--input", ckpt_path,
                            "--output", os.path.join(
                                out_dir, f"{task}_fp16.safetensors")],
                           stdout=fh, stderr=subprocess.STDOUT, check=True)

    # Free disk for the next model: the checkpoint goes now (it is
    # re-downloadable). The FP32 ONNX was already removed above.
    if not args.keep_weights:
        log(f"removing downloaded checkpoint {ckpt_path} "
            "(re-downloadable; use --keep-weights to keep it)")
        os.remove(ckpt_path)
    log_disk()


def summary(root):
    log("== done. out/ ==")
    total = 0
    for name in sorted(os.listdir("out")):
        path = os.path.join("out", name)
        size = os.path.getsize(path)
        total += size
        log(f"  {name}: {size / (1024 ** 2):.0f} MB")
    log(f"  total: {total / (1024 ** 2):.0f} MB")
    print("Next step (GPU machine, TensorRT-RTX 1.7.1.116, CUDA 13.4):")
    print("  tensorrt_rtx --onnx=out/body_fp16.onnx "
          "--saveEngine=body_fp16.trt --memPoolSize=tacticSharedMem:48K")
    print("  tensorrt_rtx --onnx=out/hand_fp16.onnx "
          "--saveEngine=hand_fp16.trt --memPoolSize=tacticSharedMem:48K")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="GEM-X FP16 + ONNX export on a 16 GB Codespace.")
    ap.add_argument("--task", choices=("all", "body", "hand"),
                    default="all")
    ap.add_argument("--keep-weights", action="store_true",
                    help="keep the downloaded .ckpt files in weights/")
    ap.add_argument("--keep-fp32-onnx", action="store_true",
                    help="keep the FP32 ONNX files in out/")
    ap.add_argument("--skip-quantize", action="store_true",
                    help="skip the FP16 checkpoint cast")
    ap.add_argument("--self-test", action="store_true",
                    help="preflight only: environment + exporter self-test")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    # Use all CPU cores (4 on the 16 GB Codespace machine) for every step:
    # torch, onnxruntime and numpy all obey these thread env vars, and
    # every tool runs as a subprocess inheriting them.
    nproc = os.cpu_count() or 4
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(nproc))
    log(f"CPU cores: {nproc} (threads set to {nproc} for all steps)")

    tools_dir = os.path.join(root, "tools", "quantize")
    for name in ("export_onnx_real.py", "hf_download.py",
                 "inspect_ckpt.py", "quantize_ckpt_to_fp16.py"):
        if not os.path.exists(os.path.join(tools_dir, name)):
            print(f"ERROR: missing tools/quantize/{name}. "
                  "Run this script from the repo root.", file=sys.stderr)
            return 2

    if not preflight():
        return 2
    if args.self_test:
        run([sys.executable,
             os.path.join(tools_dir, "export_onnx_real.py"), "--self-test"],
            check=True)
        log("self-test OK")
        return 0

    ensure_gemx(os.path.join(root, "GEM-X"))
    ensure_venv(root)
    tasks = ("body", "hand") if args.task == "all" else (args.task,)
    for task in tasks:
        run_task(task, TASKS[task], root, tools_dir, args)
    summary(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
