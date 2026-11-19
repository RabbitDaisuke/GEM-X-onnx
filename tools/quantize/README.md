# tools/quantize — GEM-X の FP16・FP8 ツール

FP16 の重みキャストには学習画像・キャリブレーションデータは不要。
FP8 ONNX は別経路であり、ModelOptはデータ省略時にランダム入力を生成できるが、それは変換可否の確認用で精度は保証されない。実用性評価には実入力のキャリブレーションが望ましい。
Body・HandをFP8 Q/DQ化する実験用ツールも追加した。
既存FP16経路は保持し、FP8採用や新しいハードウェア要件は強制しない。FP8手順と検証結果は [docs/fp8.md](../../docs/fp8.md) を参照。

## daisuke の PC (2026-11-17 の Task Manager より)

- RAM **128 GB** — FP16 変換のピーク 8〜10 GB はまったく問題にならない。
  ストリーミング方式 (1テンソルずつ) は残してあるが、必須ではない。
  Body 3.54 GB + EgoHand 10.66 GB を両方 RAM に載せても 15 GB 弱で収まる。
- GPU RTX 4070 Ti SUPER、ドライバ **32.0.16.1692 (2026/09/04)**、
  DirectX 12 (FL 12.2)。CiG / SCG に必要なドライバ 555 以上は満たす。
  (Windows の `nvidia-smi` で出る 5xx/6xx 番号でも確認すること)
- C: / D: は NVMe SSD。重み 15 GB の読み書きは数分で終わる見込み。
- アイドル時 GPU 1%、VRAM 1.3/16.0 GB 使用、57°C。推論用に
  約 14.7 GB が空いている計算になる。

## ファイル

| ファイル | 役割 |
|---|---|
| `quantize_ckpt_to_fp16.py` | `.ckpt` / `.pt` / `.safetensors` を FP16 化。オプティマイザ状態は既定で除外 |
| `inspect_ckpt.py` | チェックポイントの中身一覧。EgoHand の 10.66 GB の正体確認用 |
| `hf_download.py` | ログインなしで Hugging Face から落とす (stdlib のみ) |
| `export_onnx_fp16.py` | **テンプレート (未検証)**。FP32 ONNX → ModelOpt AutoCast で FP16 ONNX |
| `requirements.txt` | torch (CPU) / safetensors / huggingface_hub / numpy / onnx / onnxruntime |
| `fp8_quantize_onnx.py` | 実入力NPZを使ったModelOpt FP8 Q/DQ ONNX生成（元FP16は保持） |
| `prepare_fp8_calibration.py` | 動画と既存推論JSONからBody/Hand校正入力を再生成 |

## セットアップ (Windows PC、PowerShell)

```powershell
python -m pip install -r tools\quantize\requirements.txt
python -m pip install nvidia-modelopt   # AutoCast 用。別途
```

## 手順 (明日やること)

### 1. 落とす (ログインなしで試す)

```powershell
python tools\quantize\hf_download.py --list nvidia/GEM-X
python tools\quantize\hf_download.py --repo nvidia/GEM-X --file body_s240000.ckpt --out-dir weights
python tools\quantize\hf_download.py --repo nvidia/EgoHand-v1 --file egohand_s060000.ckpt --out-dir weights
```

401/403 が返ってきたら `hf auth login` してから `hf download` に
切り替えること。YOLOX-X HumanArt は Hugging Face ではなく OpenMMLab
から (ZIP 351 MiB)。詳しくは親ディレクトリの調査レポート参照。

### 2. 中身を確認する (EgoHand の 10.66 GB の正体)

```powershell
python tools\quantize\inspect_ckpt.py weights\egohand_s060000.ckpt --top 30
python tools\quantize\inspect_ckpt.py weights\body_s240000.ckpt --top 30
```

EgoHand は 883.5M パラメータなのに 10.66 GB ある。FP32 の重みだけ
なら約 3.53 GB のはずで、残りは Adam の `exp_avg` / `exp_avg_sq`
(重み + m + v で 12 B/パラメータ = 約 10.60 GB) の可能性が高い。
`inspect_ckpt.py` が optimizer として分類したバイト数がそれを
示していれば、重みだけ抜き出して FP16 化する (次の手順が自動でやる)。

### 3. FP16 化する (CPU だけ、学習画像なし)

```powershell
python tools\quantize\quantize_ckpt_to_fp16.py --input weights\body_s240000.ckpt --output weights\body_fp16.safetensors
python tools\quantize\quantize_ckpt_to_fp16.py --input weights\egohand_s060000.ckpt --output weights\egohand_fp16.safetensors
```

- `--dry-run` で、書かずにテンソル一覧とサイズ、除外される
  オプティマイザ状態だけ確認できる。まずこれをやること。
- 出力は既定で `.safetensors`。`--format pt` も選べる。
- 変換後に自動で再読み込みして、全テンソルの dtype / shape /
  要素数を照合する。不一致があれば終了コード 1 で失敗する。
- 期待サイズ: Body 約 1.68 GB、Hand (重みのみ) 約 1.77 GB。

### 4. ONNX に出す (テンプレート — まだ未検証)

`export_onnx_fp16.py` は GEM-X の PoseModel を組み立てる部分が
TODO のテンプレート。NVlabs/GEM-X のソースに対して
`--model-module` / `--build-function` を書いてから使うこと。
完成形の流れ:

```powershell
python tools\quantize\export_onnx_fp16.py --checkpoint weights\body_s240000.ckpt --config weights\body_config.json --model-module gemx_export --build-function build_body_pose --output body_fp32.onnx --fp16-output body_fp16.onnx
python -m modelopt.onnx.autocast --onnx_path body_fp32.onnx --output_path body_fp16.onnx
tensorrt_rtx.exe --onnx=body_fp16.onnx --saveEngine=body_fp16.trt --memPoolSize=tacticSharedMem:48K
```

ONNX Runtime で PyTorch FP32 と数値比較 (MPJPE / 最大絶対誤差)
してからエンジンをビルドすること。RoPE / Attention /
`grid_sample` / `einsum` が通るかは、ONNX をパースさせて初めて
わかる。

## SCG (Simultaneous Compute and Graphics) について

- SCG は **推論とグラフィックス描画を同時に走らせる**機能。
  モデルの切り替え機能ではない。混同しないこと。
- 4070 Ti SUPER は Ada (CC 8.x) なので共有メモリ上限は **48 KiB**。
  エンジンビルドは `--memPoolSize=tacticSharedMem:48K`。
  Blackwell の 84 KiB は使えない。
- CiG モードは CUDA 12.6 以上 / ドライバ 555 以上が必要。
  この PC のドライバ (2026/09/04) は満たす。TensorRT-RTX は
  DirectX のみ対応で、Vulkan は将来予定。
- SCG では単独推論より遅くなる (共有メモリ制限で選べるカーネルが
  減る)。Body 25〜40ms という見積もりは単独時の参考値として扱う。

## モデル切り替えについて (GPT-6 free の話の検証結果)

「アンロードせずにモデルを切り替える公式機能」という名前の機能は、
NVIDIA の TensorRT-RTX ドキュメントには**見つからなかった**。
本当にあるのは、次の別々の仕組みの組み合わせ:

1. **複数エンジン常駐 + 複数 IExecutionContext** — 標準パターン。
   Body と Hand を別々のエンジンとして VRAM に両方載せておき、
   切り替えは別のコンテキストで `enqueueV3` を呼ぶだけ。
   VRAM に入る限りアンロード/リロードは発生しない。
   エンジンはデシリアライズ後はイミュータブルでスレッド間で共有でき、
   コンテキストはスレッドごとに1つ作る、というのが公式の契約。
2. **Deferred weights loading (1.6+)** — 複数エンジンを、重みを
   VRAM に載せずにデシリアライズして常駐させておける。
   「多くのエンジンを常駐させるマルチモデルアプリ」向けと明記されている。
3. **IWeightsManager の offload / restore** — *同じ*エンジンの重みを
   VRAM から外して、後で再ビルド/再 refit なしに復元する機能。
   アーキテクチャが違う Body と Hand の切り替えには使えない。
4. **Refit / weightless engine** — *同じ*アーキテクチャで重みだけ
   差し替える機能 (LoRA 切り替えなど)。Body と Hand は
   アーキテクチャが違うので refit の対象外。
5. **Runtime cache** — JIT コンパイル済みカーネルをディスクに保存して
   次回起動を速くするもの。切り替え機能ではない。

つまり GPT-6 free の言っていたことは、名前のある単一機能ではなく、
「(1) の常駐パターン」か「(2)+(3) あたり」の話をまとめたものと
考えられる。このプロジェクトの推奨設計は (1): Body と Hand の
エンジンを両方 VRAM に常駐させ、単一 CUDA ストリームで
Body → Hand を直列に enqueue する。切り替えコストは実質ゼロ
(次の enqueue を呼ぶだけ)。VRAM に入らなくなったときだけ、
アンロードが避けられなくなる。詳しくは
`docs/research-trt-switching-and-quantize-2026-11-17.md` を参照。

## テスト結果 (この Linux VM、GPU なし)

- ダミーの `.pt` チェックポイント (float32 テンソル + int64 テンソル +
  偽の Adam 状態 `exp_avg` / `exp_avg_sq` を含む) を作り、
  `inspect_ckpt.py` と `quantize_ckpt_to_fp16.py` を実行。
- 結果: float テンソルは float16 になり、int64 は変わらず、
  Adam 状態は除外され、出力の再読み込み検証も通った。
  詳細なコマンドと出力は下の「VM でのテスト記録」を参照。
- 本物の重み (数 GB) はこの VM ではダウンロードしていない
  (GPU なし、/tmp 512 MB、RAM 空きわずか)。Windows PC 側でやること。
- `export_onnx_fp16.py` は未検証のテンプレート。Windows PC で
  GEM-X ソースと組み合わせて初めて動かせる。

## VM でのテスト記録 (2026-11-17、この Linux VM、GPU なし)

環境: Python 3.12.3、RAM 7.7 GB、/tmp 512 MB。torch は無いため
venv (`/tmp/gemx-venv`) に `safetensors 0.8.0` + `numpy 2.5.5` だけ
入れてテストした。`.pt` 経路は torch が無いので未実行
(Windows PC で torch を入れてから試すこと)。

```bash
python3 -m venv /tmp/gemx-venv
/tmp/gemx-venv/bin/pip install safetensors numpy
/tmp/gemx-venv/bin/python -m py_compile tools/quantize/*.py   # OK
/tmp/gemx-venv/bin/python tools/quantize/make_dummy_ckpt.py
# Wrote /tmp/gemx-quant-test/dummy_fp32.safetensors (58305 bytes, 8 tensors)
/tmp/gemx-venv/bin/python tools/quantize/inspect_ckpt.py /tmp/gemx-quant-test/dummy_fp32.safetensors
# Entries: 8 (weights 6, optimizer 2, other 0)
# Tensor bytes total: 56.27 KiB (weights 24.27 KiB, optimizer 32.00 KiB)
# Estimated FP16 weight size: 12.14 KiB
/tmp/gemx-venv/bin/python tools/quantize/quantize_ckpt_to_fp16.py --input /tmp/gemx-quant-test/dummy_fp32.safetensors --dry-run
# Tensors kept: 6 (4 float32/64 -> float16, 2 already FP16 / non-float)
# Skipped (optimizer / non-weight): 2 — exp_avg, exp_avg_sq
/tmp/gemx-venv/bin/python tools/quantize/quantize_ckpt_to_fp16.py --input /tmp/gemx-quant-test/dummy_fp32.safetensors --output /tmp/gemx-quant-test/dummy_fp16.safetensors
# Wrote: /tmp/gemx-quant-test/dummy_fp16.safetensors
# Verifying output by reloading ... Verification OK
/tmp/gemx-venv/bin/python tools/quantize/inspect_ckpt.py /tmp/gemx-quant-test/dummy_fp16.safetensors
# Entries: 6 (weights 6, optimizer 0, other 0) — F32/F64 は F16 に、I64/BOOL は変わらず
/tmp/gemx-venv/bin/python tools/quantize/hf_download.py --list nvidia/GEM-X
# nvidia/GEM-X: gated=False, private=False, downloads=163 — 19 files, 9.81 GiB total
/tmp/gemx-venv/bin/python tools/quantize/hf_download.py --repo nvidia/EgoHand-v1 --list
# nvidia/EgoHand-v1: gated=False, private=False — 7 files, 9.93 GiB total (egohand_s060000.ckpt のみ)
/tmp/gemx-venv/bin/python tools/quantize/export_onnx_fp16.py --checkpoint /tmp/nonexistent.ckpt --config /tmp/nonexistent.json --model-module gemx_export
# MISSING: torch / onnx / onnxruntime / nvidia-modelopt、終了コード 2 (想定通り。テンプレート)
```

テスト中に見つけて直した不具合 2件: `inspect_ckpt.py` の FP16
見積もりが safetensors の dtype 表記 (`F32`) を拾えていなかった件と、
`export_onnx_fp16.py` が `python` コマンド (この VM には無い) を
呼んでトレースバックを吐いていた件。どちらも修正済みで、上の
記録は修正後のもの。

## 2026-11-18 追加：本物の ONNX エクスポート（`export_onnx_real.py`）

`export_onnx_fp16.py` は汎用テンプレートのまま。実際の GEM-X ソース
（NVlabs/GEM-X、コミット `4aef145`）に対するエクスポートは
`export_onnx_real.py` が担当する。詳しくは
`docs/research-gemx-onnx-export-blockers-2026-11-18.md` と
`docs/dll-spec-2026-11-18.md`（DLL 仕様 v2、こちらが優先）。

- エクスポート境界は `PoseModel.encode()`。SOMA FK・カメラ復号・kp2d ソルブ・右手ミラーは ONNX の外。
- Hand は `ray_feats` を CPU 計算で入力にし、kp2d ヘッドの `kp2d_mu` と `kp2d_presence` を出力する。カメラ並進のソルブは ONNX の外で実装する。以前配布された ONNX にはこれらの出力がないため再生成が必要。
- 切り出しは 256x256（過去資料の 512 は誤り）。
- この VM では torch が無いためエクスポート自体は未実行。
  `--self-test`（RayPE の numpy 参照実装の検証、重み・torch 不要）は通る。
