# GEM-X FP8 PTQ — 量子化の実現性と精度の検証
FP8 conversion is an optional experiment, not a new hardware requirement or a mandatory runtime mode. Keep the existing FP16/FP32 inference paths as the default while evaluating accuracy and practical costs. This repository originally produces FP16 ONNX. The FP8 branch converts **existing real Body/Hand FP16 ONNX** into FP8 Q/DQ ONNX using NVIDIA Model Optimizer. No overwriting FP16 files and no pretend FP8 fallback. ModelOpt can generate random calibration inputs when data are omitted, but this is only a conversion smoke test and does not establish accuracy; the GEM-X converter deliberately requires a supplied NPZ.
## Collect calibration on the Windows GPU PC
Use the existing GEM-X inference code with input capture enabled. It records the exact GPU-preprocessed image tensors plus bbox, K, and Hand RayPE tensors:
~~~bat
.\.venv\Scripts\python.exe l1_video.py "C:\path\to\IMG_0174.MOV" --out out\calib-run --pose-precision fp16 --collect-fp8-calibration out\calibration
~~~
The output is `out\calibration\body_calibration.npz` / `hand_calibration.npz`. The input-collection version of `gemx-python` is required; otherwise use the standalone `tools/quantize/prepare_fp8_calibration.py` (CPU reconstruction from video and exported inference JSON):
~~~bat
python tools\quantize\prepare_fp8_calibration.py --video "C:\path\to\IMG_0174.MOV" --inference-json out\gpu-profile.json --output-dir out\calibration --max-samples 256
~~~
The standalone generator writes `body.npz` / `hand.npz` (not `*_calibration.npz`). Prefer actual GPU input capture where available.
## Quantize on a Codespace with the existing FP16 ONNX
Transfer the calibration NPZ files into `out/calibration/` in the Codespace. This command intentionally does **not** run `run_gemx_onnx.py` again and does not regenerate the FP16 models.
Before allocating a 64GB swap file, check that `/tmp` has at least 64GB available:
~~~bash
df -h --total
free -h
sudo fallocate -l 64G /tmp/swapfile
sudo chmod 600 /tmp/swapfile
sudo mkswap /tmp/swapfile
sudo swapon /tmp/swapfile
free -h
~~~
Run the calibrated FP8 path (CPU-only Codespace; very slow calibration may occur):
~~~bash
.venv/bin/python -c 'from modelopt.onnx.quantization import quantize; print("ModelOpt ready")'
.venv/bin/python tools/quantize/fp8_quantize_onnx.py --task body --onnx out/body_fp16.onnx --calibration out/calibration/body_calibration.npz --output out/body_fp8.onnx --calibration-ep cpu
.venv/bin/python tools/quantize/fp8_quantize_onnx.py --task hand --onnx out/hand_fp16.onnx --calibration out/calibration/hand_calibration.npz --output out/hand_fp8.onnx --calibration-ep cpu
~~~
The exporter venv already installs ModelOpt/ONNX dependencies; do not install GPU-only extras in a CPU Codespace. First run ordinary full-graph calibration. If RAM plus swap is still insufficient, retry with `--calibrate-per-node` (substantially slower). For a preliminary smoke test only, `prepare_demo_calibration.py` can obtain 32 samples each from NVlabs demo videos (including visual overlays), with limitations written to `demo_manifest.json`. For accuracy evaluation, replace these with actual GPU inference feeds.

If using standalone generation instead, replace the calibration paths with `body.npz` and `hand.npz`. ModelOpt may require a GPU Execution Provider for FP16 nodes unsupported by ORT CPU; that is an actual calibration failure, not permission to fabricate calibration values. On an NVIDIA GPU host use `--calibration-ep cuda:0`.
Output `body_fp8.onnx` and `hand_fp8.onnx` have Q/DQ and must pass the FP8 type check. Original FP16 ONNX files are kept unchanged.
## Optional runtime compatibility — not a quantization requirement
**Important model distinction:** the Windows `gemx-python` package uses **RT-DETR R18 COCO** for person detection, not YOLOX-X HumanArt (which is used by NVlabs's upstream path). The four ONNX conversion results are separate from any decision to use FP8 in production:
| Model | Experimental FP8 ONNX + external weights | Existing FP16 baseline |
| --- | --- | --- |
| GEM-X Body | `body_fp8.onnx` + `body_fp8.onnx_data` | `body_fp16.onnx` |
| GEM-X Hand | `hand_fp8.onnx` + `hand_fp8.onnx_data` | `hand_fp16.onnx` |
| RT-DETR R18 person | `rtdetr_r18vd_coco_fp8.onnx` + `rtdetr_r18vd_coco_fp8.onnx_data` | `rtdetr_r18vd_coco_fp16.onnx` |
| RT-DETR hand | `hand_detector_fp8.onnx` + `hand_detector_fp8.onnx_data` | `hand_detector_fp16.onnx` |
These files demonstrate conversion feasibility, **not a requirement to modify the Windows inference runtime**. Do not force FP8 by GPU generation or require the user to obtain new hardware as part of the quantization task. The existing FP16 route remains usable regardless of FP8 evaluation. The separate `yolox_humanart_fp8.onnx` is not a replacement for the RT-DETR R18 person detector.
The corrected GEM-X Body/Hand results were first written as `*_fp8_repaired.onnx`. Use `tools/quantize/finalize_fp8_onnx.py` to publish canonical `*_fp8.onnx` with updated external-data references. **Do not hard-link external ONNX weight files:** `onnx.checker` rejects multiple hard links for security. The finalizer uses an OS copy (reflink where supported) while retaining original models.
## Status
Codespace has produced and checked all four canonical Windows-runtime FP8 ONNX files and their matching external weight sidecars. The full ONNX model checker and every scale's finiteness/positive-value checks were performed; the GEM-X Body and Hand originally each had 38 infinite scales before the float32-intermediate fix. CPU ONNXRuntime inference comparisons were performed for Body/Hand and the two RT-DETR detector models. Body and Hand on one demo sample showed rotation-angle differences near 1 degree on average. Detector FP8 results had some changes in scores, detections, and bounding boxes, notably in the hand detector. **Do not mark FP8 as production-ready or automatically enable it.** The comparison demonstrates that FP8 is technically convertible; its practical benefit and acceptable accuracy are not established. Reporting that FP8 is impractical for this workflow is an acceptable outcome. No new GPU requirements are imposed. Demo footage (32 samples per original conversion, 96 for the corrected Windows hand detector) includes overlays and does not establish production accuracy.


## FP8 scale overflow correction and rejection of nonfinite scales

The installed NVIDIA ModelOpt version initially generated **38 infinite FP8 Q/DQ scales in Body and 38 in Hand**, even though the quantization step returned successfully. The root cause was the float16 intermediate multiplication in `modelopt/onnx/quantization/fp8.py`:
```python
np_fp8_scale = (np_scale * 448.0) / 127.0
```
The repository patch changes the intermediate arithmetic to float32 while keeping the source ONNX scale dtype:
```bash
patch --dry-run -d .venv/lib/python3.14/site-packages -p1 -i tools/quantize/modelopt-fp8-intermediate-overflow.patch
patch -d .venv/lib/python3.14/site-packages -p1 -i tools/quantize/modelopt-fp8-intermediate-overflow.patch
```
With this patch, `fp8_quantize_onnx.py` validates **every Q/DQ scale** including values in external tensor-data files and refuses success if any value is `inf` or `nan`. It does not clamp invalid values. The tested source was ModelOpt in Python 3.14 Codespace; confirm patch applicability after dependency upgrades. **Passing this check does not establish inference accuracy.**

## FP8 detector quantization (person and hand)

Detector FP8 PTQ is a separate operation from GEM-X Body / Hand *pose* FP8 PTQ. The detection ONNX and inference preprocessing are kept intact.
* Hand detection: `nvidia/RT-DETR-Hand-Detector-v1` / `hand_detector.onnx`, ONNX inputs `images` (RGB float32 /255, 640x640) and `orig_target_sizes` (int64).
* Person detection: YOLOX-X HumanArt from the official OpenMMLab ONNX SDK, input `input` (BGR float32, 640x640, top-left fit-and-pad at 114, **no** /255).

A 32-sample preliminary calibration feed can be constructed using official demonstration videos under `GEM-X/assets`:
```bash
.venv/bin/python tools/quantize/prepare_hand_detector_calibration.py --output out/calibration/hand_detector.npz --samples 32 --videos GEM-X/assets/hand_pose.mp4 GEM-X/assets/body_pose_motion.mp4 GEM-X/assets/gemx_overview.mp4
.venv/bin/python tools/quantize/prepare_yolox_detector_calibration.py --output out/calibration/yolox_humanart.npz --samples 32 --videos GEM-X/assets/body_pose_motion.mp4 GEM-X/assets/body_motion_night.mp4 GEM-X/assets/gemx_overview.mp4
.venv/bin/python tools/quantize/quantize_detector_fp8.py --onnx weights/hand_detector.onnx --calibration out/calibration/hand_detector.npz --output out/hand_detector_fp8.onnx
.venv/bin/python tools/quantize/quantize_detector_fp8.py --onnx weights/yolox_humanart.onnx --calibration out/calibration/yolox_humanart.npz --output out/yolox_humanart_fp8.onnx
```
Here `weights/hand_detector.onnx` and `weights/yolox_humanart.onnx` represent the actual user-provided or officially downloaded originals; they are not included in this repository. **YOLOX HumanArt commercial licensing must be independently verified before deployment or redistribution.** The exporter performs ONNX structural and finite-scale checks. TensorRT-RTX GPU compatibility, inference accuracy and speed still need an RTX machine.

**RT-DETR R18 person detection:** the Windows runtime's actual person detector is `rtdetr_r18vd_coco_fp16.onnx` from `onnx-community/rtdetr_r18vd_coco_o365` (revision `37122d8b9b89eb8384fba4d688d05051eaf7bb78`, SHA-256 `540c30402fb56fdbc1b2c6256b981d53b47d103b256474cf617bf6e9b3e321f8`). The matching FP8 is generated from the existing FP16 source using the `pixel_values` input (RGB float32 /255, 640x640 centered gray-114 letterbox). Preliminary real-video samples can be prepared using `tools/quantize/prepare_rtdetr_person_calibration.py`. Do not replace that model with YOLOX to claim the Windows detection path is quantized.
