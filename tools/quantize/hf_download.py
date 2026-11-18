#!/usr/bin/env python3
"""hf_download.py — download GEM-X files from Hugging Face with NO login.

Uses only the Python standard library (urllib).  No token, no
``huggingface_hub`` required (that package is an optional fallback, see
below).  Two endpoints, both verified 2026-11-17 with no token:

  * List files:  https://huggingface.co/api/models/<repo>/tree/main?recursive=true
  * Download:    https://huggingface.co/<repo>/resolve/main/<path>
    (follows the 307 redirect to the CDN automatically — urllib does
    this by default; plain ``curl`` needs ``-L`` for the same reason.)

Repos used by this project (all ``gated: false`` as of 2026-11-17):
  * nvidia/GEM-X                  — body_s240000.ckpt (3.54 GB) etc.
  * nvidia/EgoHand-v1             — egohand_s060000.ckpt (10.66 GB)
  * nvidia/RT-DETR-Hand-Detector-v1 — hand_detector.onnx (80 MB)
  * nvidia/SOMA-X  (revision v0.3.3) — SOMA assets

One discrepancy: the GEM-X docs say EgoHand needs "a Hugging Face
account that has been granted access", but the API reports
``gated: false``.  Try no-login first.  If the server answers 401/403,
this script prints the fallback: ``hf auth login`` then
``hf download nvidia/EgoHand-v1 egohand_s060000.ckpt``.

YOLOX-X HumanArt is NOT on Hugging Face; it comes from OpenMMLab
(see README.md in this folder).

Usage:
    python tools\\quantize\\hf_download.py --list nvidia/GEM-X
    python tools\\quantize\\hf_download.py --repo nvidia/GEM-X ^
        --file body_s240000.ckpt --out-dir weights
    python tools\\quantize\\hf_download.py --repo nvidia/EgoHand-v1 ^
        --file egohand_s060000.ckpt --out-dir weights
    python tools\\quantize\\hf_download.py --repo nvidia/SOMA-X ^
        --revision v0.3.3 --file SOMA_neutral.npz --out-dir assets\\soma
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_BASE = "https://huggingface.co/api/models"
RESOLVE_BASE = "https://huggingface.co"


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "gemx-quantize/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            print("ERROR: Hugging Face returned "
                  f"{exc.code} — this repo may require login after all.\n"
                  "Fallback:\n"
                  "  hf auth login\n"
                  "  hf download <repo> <file> --local-dir <dir>",
                  file=sys.stderr)
        else:
            print(f"ERROR: HTTP {exc.code} for {url}", file=sys.stderr)
        raise SystemExit(1)


def repo_info(repo: str):
    return _get_json(f"{API_BASE}/{repo}")


def list_files(repo: str, revision: str = "main"):
    data = _get_json(
        f"{API_BASE}/{repo}/tree/{revision}?recursive=true")
    files = []
    for entry in data:
        if entry.get("type") != "file":
            continue
        files.append({
            "path": entry["path"],
            "size": entry.get("size", 0),
            "lfs": bool(entry.get("lfs")),
        })
    return files


def download_file(repo: str, remote_path: str, out_path: str,
                  revision: str = "main") -> int:
    """Download one file, resuming a partial download if present.

    Returns the final byte count.  Verifies the size against the
    Content-Length header when the server sends one.
    """
    url = f"{RESOLVE_BASE}/{repo}/resolve/{revision}/{remote_path}"
    existing = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    headers = {"User-Agent": "gemx-quantize/1.0"}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"Resuming {out_path} from byte {existing} ...")
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            print("ERROR: Hugging Face returned "
                  f"{exc.code} — this file may require login.\n"
                  "Fallback:\n"
                  "  hf auth login\n"
                  f"  hf download {repo} {remote_path}",
                  file=sys.stderr)
        elif exc.code == 416:
            print("Server says the existing file is already complete.")
            return existing
        else:
            print(f"ERROR: HTTP {exc.code} for {url}", file=sys.stderr)
        raise SystemExit(1)

    # A server that ignores Range returns 200 with the full file; in that
    # case restart from scratch rather than appending a duplicate.
    if resp.status == 200 and existing > 0:
        print("Server ignored Range; restarting download from scratch.")
        mode, existing = "wb", 0
    total = resp.headers.get("Content-Length")
    total = int(total) if total else None
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    downloaded = existing
    chunk = 1024 * 1024
    with open(out_path, mode) as fh:
        while True:
            data = resp.read(chunk)
            if not data:
                break
            fh.write(data)
            downloaded += len(data)
            if total:
                pct = 100.0 * downloaded / (total + (existing if mode == "ab" else 0))
                print(f"\r  {downloaded / 1e9:.2f} GB ({pct:.1f}%)",
                      end="", flush=True)
    print()
    if total is not None:
        expected = total + (existing if mode == "ab" and resp.status == 206 else 0)
        if downloaded != expected:
            print(f"WARNING: size mismatch: got {downloaded}, "
                  f"expected {expected}", file=sys.stderr)
    return downloaded


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024.0 or unit == "GiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.2f} GiB"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Download GEM-X files from Hugging Face without login.")
    ap.add_argument("--repo", default="nvidia/GEM-X",
                    help="HF repo, e.g. nvidia/GEM-X (default) or nvidia/EgoHand-v1")
    ap.add_argument("repo_arg", nargs="?",
                    help="Optional positional repo (same as --repo)")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--list", action="store_true", help="List files and sizes; download nothing")
    ap.add_argument("--file", help="Remote file path to download (repeatable via comma list)")
    ap.add_argument("--out-dir", default="weights", help="Local directory (default: weights)")
    args = ap.parse_args(argv)
    if args.repo_arg:
        args.repo = args.repo_arg

    if args.list:
        info = repo_info(args.repo)
        print(f"{args.repo}: gated={info.get('gated')}, "
              f"private={info.get('private')}, downloads={info.get('downloads')}")
        files = list_files(args.repo, args.revision)
        total = sum(f["size"] for f in files)
        print(f"{len(files)} files, {_fmt_bytes(total)} total "
              f"(do NOT download all of it — pick named files):")
        for f in sorted(files, key=lambda x: -x["size"]):
            print(f"  {_fmt_bytes(f['size']):>12}  {f['path']}"
                  f"{'  [LFS]' if f['lfs'] else ''}")
        return 0

    if not args.file:
        ap.error("--file is required unless --list is given")
    wanted = [f.strip() for f in args.file.split(",") if f.strip()]
    # Verify names against the repo listing first (cheap, catches typos
    # before a multi-GB download starts).
    available = {f["path"]: f["size"] for f in list_files(args.repo, args.revision)}
    for name in wanted:
        if name not in available:
            print(f"ERROR: '{name}' is not in {args.repo}@{args.revision}. "
                  f"Run --list to see the names.", file=sys.stderr)
            return 2
        print(f"{name}: {_fmt_bytes(available[name])}")
    for name in wanted:
        out_path = os.path.join(args.out_dir, os.path.basename(name))
        print(f"Downloading {args.repo}:{name} -> {out_path}")
        got = download_file(args.repo, name, out_path, args.revision)
        print(f"Done: {out_path} ({_fmt_bytes(got)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
