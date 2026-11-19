# GEM-X FP8 PTQ (RTX 40/50) / FP16 compatibility (RTX 30)
This repository originally produces FP16 ONNX. The FP8 branch converts **existing real Body/Hand FP16 ONNX** into FP8 Q/DQ ONNX using NVIDIA Model Optimizer. No synthetic calibration, no overwriting FP16 files, and no pretend FP8 fallback.
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
.venv/bin/python -m pip install 'nvidia-modelopt[onnx]'
.venv/bin/python tools/quantize/fp8_quantize_onnx.py --task body --onnx out/body_fp16.onnx --calibration out/calibration/body_calibration.npz --output out/body_fp8.onnx --calibration-ep cpu --calibrate-per-node
.venv/bin/python tools/quantize/fp8_quantize_onnx.py --task hand --onnx out/hand_fp16.onnx --calibration out/calibration/hand_calibration.npz --output out/hand_fp8.onnx --calibration-ep cpu --calibrate-per-node
~~~
If using standalone generation instead, replace the calibration paths with `body.npz` and `hand.npz`. ModelOpt may require a GPU Execution Provider for FP16 nodes unsupported by ORT CPU; that is an actual calibration failure, not permission to fabricate calibration values. On an NVIDIA GPU host use `--calibration-ep cuda:0`.
Output `body_fp8.onnx` and `hand_fp8.onnx` have Q/DQ and must pass the FP8 type check. Original FP16 ONNX files are kept unchanged.
## Deploy on RTX GPUs
RTX 40/50 (Ada/Blackwell) are FP8-capable. RTX 30 (Ampere) uses FP16. Person/hand detectors remain FP16 in both configurations. The runtime must detect the GPU's compute capability and select Body+Hand FP8 only for SM >= 8.9.
~~~powershell
tensorrt_rtx.exe --onnx=body_fp8.onnx --saveEngine=body_fp8.trt --memPoolSize=tacticSharedMem:48K
tensorrt_rtx.exe --onnx=hand_fp8.onnx --saveEngine=hand_fp8.trt --memPoolSize=tacticSharedMem:48K
~~~
The build must be run on the target RTX GPU. The 48KiB tactic shared-memory constraint is mandatory for RTX 40 in SCG mode. TensorRT for RTX is strongly typed; an FP16 ONNX does **not** become FP8 simply by requesting faster kernels.
## Status
GitHub-side FP8 converter and video/JSON calibration generator are implemented. Real GEM-X FP8 output, numeric accuracy, engine build, and real SCG latency require target-machine execution. A successful Q/DQ export alone is not proof of FP8 Tensor Core utilization or numeric equivalence.
