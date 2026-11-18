# ATTRIBUTIONS — GEM-X-onnx

## 親（上流）

- 上流リポジトリ：[NVlabs/GEM-X](https://github.com/NVlabs/GEM-X)
- 上流コミット：`4aef1452e5df171deff4f293c08e643eb450981e`
- このリポジトリはフォークではない。GEM-X のソースコードや重みを同梱していない。
  run_gemx_onnx.py が実行時に上流を clone する。

## このリポジトリ自身のライセンス

- `run_gemx_onnx.py`、`tools/quantize/`、ドキュメント類：Apache License 2.0（`LICENSE` 参照）。

## 実行時に取得するものと、そのライセンス

| もの | 取得元 | ライセンス | 商用利用 |
| --- | --- | --- | --- |
| GEM-X Body 重み `body_s240000.ckpt`、`body_config.json` | `nvidia/GEM-X`（Hugging Face） | NVIDIA Open Model License | モデルカード曰く可 |
| EgoHand 重み `egohand_s060000.ckpt` | `nvidia/EgoHand-v1`（Hugging Face） | OpenMDW 1.1（`LICENSES/LicenseRef-OpenMDW-1.1.txt`） | モデルカード曰く可。配布時は本ライセンス文と著作権表示の保持が必要 |
| RT-DETR Hand Detector `hand_detector.onnx` | `nvidia/RT-DETR-Hand-Detector-v1`（Hugging Face） | OpenMDW 1.1 | モデルカード曰く可 |
| GEM-X ソースコード（実行時に clone） | `NVlabs/GEM-X` | Apache 2.0。ただし `ATTRIBUTIONS.md` にある一部モジュールは OpenMDW 1.1 / MIT / BSD 3-Clause など、元の表示を保持する | 上流の表示通り |
| SOMA-X 資産 | このスクリプトでは使わない | Apache 2.0 | — |

## 含めていないもの（意図的な除外）

- **YOLOX-X HumanArt 検出器**：ライセンスの商用可否が未確認のため、含めないし、スクリプトでも落とさない。ONNX エクスポートには不要。
- **重み・ONNX・TensorRT エンジン・SOMA 資産全般**：このリポジトリには同梱しない。実行時に公式ソースから取得する。`.gitignore` で誤コミットを防ぐ。
- **GEM-X のデモ用 `assets/`（動画・GIF、69MB）**：実行に不要のため含めない。
