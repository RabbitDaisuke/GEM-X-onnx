# GEM-X Hand：静的バッチ2 ONNXの生成・検証
元のHandチェックポイントからバッチ2のONNXを新規エクスポートする。既存の`hand_fp16.onnx`は変更しない。ONNXの入力次元だけを2に書き換える方式は使わない。
## Codespaceでの生成
```bash
cd /workspaces/GEM-X-onnx
.venv/bin/python tools/quantize/export_onnx_real.py --task hand --batch-size 2 --checkpoint /tmp/gemx-onnx-work/weights/egohand_s060000.ckpt --gemx-root GEM-X --output /tmp/gemx-onnx-work/out/hand_batch2_fp32.onnx --fp16-output /tmp/gemx-onnx-work/out/hand_batch2_fp16.onnx
```
中間FP32と最終FP16のONNXは`/tmp/gemx-onnx-work/out/`に置く。モデルの巨大な重みをGitHubにcommitしない。FP16変換時の警告・失敗を無視しない。
## 数値検証
```bash
.venv/bin/python tools/quantize/verify_hand_batch2.py --single /tmp/gemx-onnx-work/out/hand_fp16.onnx --batch /tmp/gemx-onnx-work/out/hand_batch2_fp16.onnx --calibration /tmp/gemx-onnx-work/out/calibration-refined/hand_calibration.npz
```
バッチ1推論を2回行った結果と、バッチ2推論を1回行った結果を、出力名・形状・有限値・数値誤差で比較する。失敗時は成功マーカーを生成しない。成功時のみ`hand_batch2_fp16.batch2_verified.json`を作成する。検証はモデル変換の同値性であり、実際のWindows TensorRT for RTX / SCG実行速度・精度の保証ではない。
## Windows側でのベンチマーク
検証済み`hand_batch2_fp16.onnx`と`hand_batch2_fp16.batch2_verified.json`をWindows側の`models/`へ配置する。TensorRTエンジンを生成する前に数値検証マーカーを確認する。従来のHandバッチ1に対して、Batch2・CPUデコード先読み・通常実行を同一FP16精度、同じ動画とSCG条件で比較する。
