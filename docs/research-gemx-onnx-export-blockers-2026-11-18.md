# GEM-X ONNX エクスポートのブロッカー調査（2026-11-18）

> 調査対象：NVlabs/GEM-X（GitHub、コミット `4aef145`、2026-11-18 時点の main）。
> 取得方法：`git clone --depth 1`（sparse checkout の不具合で中断したが、GitHub API のツリー一覧と `raw.githubusercontent.com` から必要な .py を全て取得して読んだ。重みは一切ダウンロードしていない）。
> 検証状況：この VM には GPU が無く、torch も入っていないため、**エクスポート自体は一度も実行していない**。以下の「ブロッカー」はソースコード読解によるもので、「実行して確認した」ものではない。各項目に検証状況を明記する。
> 前提の訂正：切り出しサイズは **256×256**（`PoseInferenceConfig.image_size = 256`、`gem/pose/data/crop.py`）。過去の資料の「512」は誤り。

## 0. 結論（3行）

1. **GEM-X リポジトリに ONNX エクスポートのコードは存在しない。** `torch.onnx.export` / dynamo の呼び出しは 1 か所も無い（`onnx` という文字列が出るのは RT-DETR 手検出器の onnxruntime 実行だけ）。2025-06 のニュース「ONNX/TensorRT acceleration」は旧 GEM デモの話で、GEM-X の Pose ではない。
2. **エクスポート境界は `PoseModel.encode()`**（`gem/pose/model.py`）。SOMA FK・カメラ復号・kp2d ソルブ・右手ミラーはすべて ONNX の外に出す。これができないと Hand は変換できない。
3. **Hand の「そのまま変換できない」は 1 つの理由ではなく 6 つのブロッカーの束**（H1〜H6）。どれも回避策はあるが、回避策なしで `torch.onnx.export(model, …)` を呼んでも通らない。

## 1. エクスポートの入口（Body / Hand 共通）

### 1.1 モデルの作り方（公式 API）

```python
from gem.pose.api import load_pose_checkpoint, load_hand_checkpoint, body_config, hand_config
body = load_pose_checkpoint("models/body_s240000.ckpt", body_config(), device="cpu")
hand = load_hand_checkpoint("models/egohand_s060000.ckpt", device="cpu")
```

- `load_pose_checkpoint`（`gem/pose/api.py`）：チェックポイントを読み、`convert_pose_state_dict` で名前空間を統一し、meta デバイスでモデルを構築して `strict=True` で重みを入れる。`body_config.json`（HF）はこの `body_config()` と同一内容（`task: body`、head 76 関節・identity 128・scale 57 packed、`dinov3_vith16plus`、CLIFF `sam3db_perspective`）。
- Hand は `hand_config()` がプリセットとしてコード内にある（HF に hand 用の config JSON は無い）。`ray_pe_num_bands=8`、`kp2d_translation`（separate、25 関節）が有効。
- 依存バージョン（`setup.cfg`）：`torch>=2.10,<3`、`transformers==5.3.0`、`py-soma-x==0.3.3`、`einops>=0.8,<1`、`onnxruntime>=1.20,<2`（`[video]` / `[hand]` extra）。Python >= 3.10。

### 1.2 forward の実体

`PoseModel.forward(batch: dict[str, Tensor]) -> PoseHeadOutput`（`gem/pose/model.py:44`）。

```
img (B,3,256,256) → DinoV3Backbone → 特徴 (B,1280,16,16)
                → CliffConditioning(batch) → cliff (B,3)
                → [Handのみ] RayPE(bbx_xys, K_fullimg) → ray_feats (B,256,32)
                → Task head (TransformerDecoder 6層) → PoseHeadOutput
                → [Handのみ] kp2d_head(特徴) → auxiliary["kp2d"]
```

`encode()`（同ファイル）は `@torch.inference_mode()` で `forward` を呼び、`ImageCondition(pose_tokens, pred_cam, parameters, task, release_id)` を返す。**これがエクスポートすべき単位**。`PosePredictor`（`gem/pose/predictor.py`）はこの上に FK・カメラ復号・ソルブを載せる別レイヤーで、ONNX に含めない。

### 1.3 入出力テンソル（エクスポート後の姿）

**Body**（`body_config()`、head = `SOMABodyHeadV2`）

| 入力 | 形状 | dtype | 出典 |
|---|---|---|---|
| `img` | (1, 3, 256, 256) | FP32→FP16 | 正規化済み切り出し（ImageNet 平均/分散） |
| `bbx_xys` | (1, 3) | FP32 | `[cx, cy, size]`（CLIFF 条件付け用） |
| `K_fullimg` | (1, 3, 3) | FP32 | フル画像 K（CLIFF 用。`sam3db_perspective`） |
| 出力 | 形状 | dtype | 出典 |
| `body_pose` | (1, 76, 3, 3) | FP32 | `soma_body_head.py: _params_from_pose_shape` |
| `global_orient` | (1, 1, 3, 3) | FP32 | 同上 |
| `identity_coeffs` | (1, 128) | FP32 | 同上 |
| `scale_params` | (1, 57) | FP32 | packed `[GS=1.0, local56]`（`_scale_params_from_shape`、direct＋fixed_global_scale=1.0） |
| `pred_cam` | (1, 3) | FP32 | legacy `[s, dx, dy]` |
| `pose_token` | (1, 1024) | FP32 | 最終クエリ |

**Hand**（`hand_config()`、head = `SOMAHandHeadV2`、kp2d 有効）

| 入力 | 形状 | dtype | 出典 |
|---|---|---|---|
| `img` | (1, 3, 256, 256) | FP32→FP16 | 正準左手の正規化済み切り出し |
| `bbx_xys` | (1, 3) | FP32 | ミラー済みボックス |
| `K_fullimg` | (1, 3, 3) | FP32 | ミラー済み K |
| `ray_feats` | (1, 256, 32) | FP32 | **CPU で事前計算**（H1 の回避策） |
| 出力 | 形状 | dtype | 出典 |
| `wrist_orient` | (1, 1, 3, 3) | FP32 | `soma_hand_head.py: _params_from_pose_shape` |
| `hand_pose` | (1, 24, 3, 3) | FP32 | 同上 |
| `identity_coeffs` | (1, 5) | FP32 | 同上 |
| `scale_params` | (1, 27) | FP32 | `[global1, local26]` |
| `pred_cam` | (1, 3) | FP32 | log_lambda `[log_lambda, dx, dy]` |
| `pose_token` | (1, 1024) | FP32 | 最終クエリ |
| `kp2d_mu` / `kp2d_presence` | (1, 25, 2) / (1, 25) | FP32 | 任意。v1 では出さない（H4） |

- 注意：Body の `scale_params` はリリース版では packed 57 だが、学習用の native 版は raw local56。`body_config.json` の `scale_parameterization: packed` を確認すること。間違えると FK の scale が 1 個ずれる。
- 注意：Hand の `pred_cam` は log_lambda 空間。legacy `[s, dx, dy]` への変換は DLL 側（`depth_parameter_to_legacy_camera` と同一式）。

## 2. Hand のブロッカー（H1〜H6）

### H1 — RayPE の `torch.linalg.solve`（最重要・Hand 固有）

- 場所：`gem/pose/conditioning/ray_pe.py: crop_points_to_ray_angles`（`torch.linalg.solve(K_fullimg, …)`）。
- 内容：Hand は `ray_pe_num_bands=8` のため、`PoseModel.forward` が `bbx_xys` と `K_fullimg` からパッチごとのカメラレイ角度を計算し、フーリエ符号化して 32 次元の `ray_feats (B,256,32)` を作る。その計算の核心が 3×3 連立一次方程式の求解（`K⁻¹·pixel`）。
- なぜ通らないか：標準 ONNX opset（18／21）には `Solve` op が無い。`torch.onnx.export`（dynamo）は `linalg.solve` を逆行列＋行列積に分解する場合があるが、保証は無い。TensorRT-RTX の ONNX パーサが通るかは未確認。
- 回避策（推奨）：**RayPE を ONNX の外に出す。** DLL が CPU で `ray_feats` を計算し、ONNX の入力として渡す。計算式は `ray_pe.py` そのまま（`compute_crop_ray_angles` → `fourier_encode_angles`、8 バンド）。3×3 逆行列は解析式（余因子行列）で書けるので CPU 実装は簡単。`ray_feats` は 256×32×4B = 32KB／フレームで、入力コストは無視できる。
- 検証状況：未実行。Windows で `ray_feats` の CPU 計算結果と PyTorch の `ray_feats` の最大絶対誤差を比較する必要がある。

### H2 — `torch.autocast` コンテキストが forward 経路に埋まっている

- 場所：`gem/pose/conditioning/ray_pe.py`（`with torch.autocast(..., enabled=False)`）、`gem/pose/heads/soma_body_head.py`（`torch.amp.autocast(..., enabled=False)`、sigma 計算部）、`gem/pose/predictor.py`（`predict_batch`、`_decode_prediction`）。
- 内容：`autocast` コンテキストマネージャは dynamo のグラフ分割点になりやすく、エクスポート時にグラフブレークや「autocast 内の演算が FP32 固定になる」挙動を招く。
- 回避策：エクスポート用ラッパーでは `model.eval()`＋`torch.inference_mode()` で呼び、`autocast` を一切使わない。`enabled=False` の箇所は計算内容が FP32 そのものなので、ラッパー側で同等の演算を素直に書けばよい。Body head の sigma 計算（`decsigma`）は v1 の出力には不要（`parameters` に含まれない）ため、ラッパーで切り捨てる。
- 検証状況：未実行。dynamo export 時のグラフブレーク報告を確認すること。

### H3 — 入力が `dict[str, Tensor]` でキーが任意

- 場所：`gem/pose/model.py: forward(batch)`、`gem/pose/conditioning/cliff.py: CliffConditioning.forward`（`"bbx_xys" in batch` で分岐し、無ければ `box_center`／`box_size` から合成）、`gem/pose/predictor.py`（`batch.get("right_hand", …)`、`batch.get("original_bbx_xys", …)`）。
- 内容：ONNX の入出力は固定のテンソルリストでなければならない。dict 入力・キー存在による分岐・`batch.get` のフォールバックはそのままではエクスポートできない。
- 回避策：**エクスポート用ラッパー `nn.Module` を書く。** 引数を明示的なテンソル（`img, bbx_xys, K_fullimg`、Hand は `ray_feats` も）だけにし、dict を作らない。`box_center`／`box_size` 経路は使わない（DLL は常に `bbx_xys` を渡す）。
- 検証状況：未実行（ラッパーの設計は §5 のスクリプトに反映済み）。

### H4 — kp2d ヘッド（RLE 2D 関節）が第二のデコーダ＋dict 出力

- 場所：`gem/pose/heads/rletr_occ_head.py`（`RLETransformerOccHead`：`nn.TransformerDecoder` 3 層＋`RLEOccReadout`）、`gem/pose/api.py: build_pose_model`（Hand のみ `kp2d_head` を構築）。
- 内容：Hand には ViT 特徴から 25 関節の 2D 座標・不確かさ・存在確率を出す第二のデコーダがある。`nn.TransformerDecoder`（PyTorch ネイティブ）は dynamo エクスポートで扱いにくい部類で、`forward` が `dict[str, Tensor]`（`mu, sigma, evidence, mass, pres_logit, presence`）を返す。さらに `RLEVisualMemory` は特徴マップがちょうど (16,16) でないと例外を投げる（固定サイズ前提）。
- 回避策（推奨）：**v1 では kp2d ヘッドをエクスポートしない。** `kp2d_head=None` の Hand モデルを別途構築し、`auxiliary["kp2d"]` を出さない。kp2d の用途は並進の補正（`_keypoint_translation`、2×2 最小二乗ソルブ）だけで、無くても `decode_camera` の並進で動作する（精度は落ちる可能性があるが、動作する）。kp2d が必要になったら v1.1 で別 ONNX グラフとして切り出す。
- 注意：`convert_pose_state_dict` は `kp2d_head.` の重みを許容するが、`strict=True` で読む場合、kp2d 無しのモデルに kp2d 重みを渡すと余る。スクリプト側で `kp2d_head.`／`kp2d_outcoord_loss.` のキーを除いてから `load_state_dict` する（`strict=False` にしない。余りキーの見落としを防ぐため、除外リストを明示する）。
- 検証状況：未実行。kp2d 無しでの Hand 精度劣化は未測定。

### H5 — 右手ミラーが前処理（画像反転＋K のミラー）

- 場所：`gem/pose/tasks/hand.py: mirror_hand_camera`、`gem/pose/data/crop.py: prepare_pose_batch`（`full_image` 規約：画像を `W−1−x` で反転してから切り出し、K は `K' = F K M`）。
- 内容：Hand モデルは**正準左手**しか知らない。右手は画像とカメラをミラーしてから入れる。ミラーは画像の反転（`np.ascontiguousarray(image[:, ::-1])`）を含むため、ONNX グラフの中には入れない。
- 回避策：DLL の前処理でやる（DLL 仕様 §6.2）。ONNX の入力は常に正準左手。FK 後の物理的手への復元も DLL 側（`HandGeometry._right_parameters` と同一の考え方。単なる X 反転では T-pose の向きとキラリティ補正が失われるため、2 つのリグを通す）。
- 検証状況：未実行。ミラー前後の Hand 出力が左右対称になることの確認が必要。

### H6 — SOMA FK・カメラ復号・ソルブはそもそも ONNX の対象外

- 場所：`gem/pose/geometry/hand.py`（`HandGeometry`：SOMA-X 0.3.3 の `SOMAHandLayer` を遅延ロード、左右リグ、`torch.linalg.solve` を `rotate_wrist_orientation` でも使う）、`gem/pose/geometry/body.py`（`BodyGeometry`：`SomaFixedGs1BodyAdapter`、correctives＋procedural transforms `legacy_pose021`、FP32 固定）、`gem/pose/predictor.py: _decode_prediction`／`_keypoint_translation`（`solve_lateral_translation_full_perspective`：2×2 最小二乗、`torch.linalg.eigvalsh`＋`torch.linalg.solve`、FP64 で解く）、`gem/pose/camera.py: decode_camera`。
- 内容：FK は外部の SOMA ランタイム（`py-soma-x==0.3.3`）と資産ファイル（`MHR/mhr_model_lod1.pt` 等）に依存し、行ごとの真偽マスク分岐（左右）や遅延初期化を含む。ONNX にできないし、するべきでもない。
- 回避策：DLL 側で C++（または CPU）実装する（DLL 仕様 §6.5、決定 D-2）。FK に必要なのは関節の位置・回転のみで、メッシュ頂点は v1 では不要。
- 検証状況：未実行。SOMA 資産の中身（リグの関節階層・バインドポーズ）の確認が D-2 の前提。

## 3. Body の状況（Hand との対比）

| 項目 | Body | Hand |
|---|---|---|
| RayPE | 無し（`ray_pe_num_bands=None`） | 有り（8 バンド、H1） |
| kp2d ヘッド | 無し | 有り（H4） |
| 右手ミラー | 不要 | 必須（H5） |
| カメラ復号 | legacy `[s,dx,dy]`（そのまま） | log_lambda → legacy 変換が必要 |
| SOMA FK | Body 用（77 関節、correctives あり） | Hand 用（25 関節×2、左右リグ） |
| 入力 | img／bbx_xys／K_fullimg | img／bbx_xys／K_fullimg／ray_feats（CPU 計算） |
| 判定 | **ラッパーさえ書けばエクスポート可能見込み** | **H1〜H6 の回避策を全部入れたラッパーが必要** |

### Body 側の残りのブロッカー（共通・軽め）

- **B1 — HF Transformers の DINOv3ViTModel**（`gem/common/backbones/dinov3.py`）：`DINOv3ViTModel`（32 層、hidden 1280、20 heads、gated MLP＝SwiGLU、RoPE、レジスタトークン 4 個）をそのまま `forward` している。`return_dict=True` で `ModelOutput`（dataclass）を返し、`last_hidden_state[:, 1+4:]` を切り出している。レガシー `torch.onnx.export`（TorchScript）では HF モデリングコードはほぼ通らない。**dynamo export（`torch.onnx.dynamo_export`、torch ≥ 2.10）を使う**。それでも `transformers==5.3.0` のモデリングコード内のデータ依存分岐がグラフブレークを起こす可能性がある。最初の確認は `dynamo_export` の成否と、出力グラフの op 一覧。
- **B2 — `einops.rearrange`**（`soma_body_head.py`、`soma_hand_head.py`、`transformer_decoder.py` の Attention／CrossAttention）：`rearrange` 自体は reshape／permute に分解されるため dynamo では通る見込みが高い。ただし `transformer_decoder.py` の `Attention`／`CrossAttention` は `attention_implementation="eager"`（head 側で指定）で、`torch.matmul`＋Softmax の素直な実装のため ONNX 化しやすい。バックボーン側は SDPA（`_attn_implementation="sdpa"`）。SDPA は ONNX の `Attention` op に落ちるか、分解されるかはパーサ次第。**TensorRT-RTX でのパース結果で確認する。**
- **B3 — per-layer 予測のリスト出力**：両 head とも `per_layer_predictions=True` で 6 層分の中間予測を `torch.cat` で積む。Body head には `final_only=True` フラグがある（`soma_body_head.py: forward(..., final_only=False)`）。**Body は `final_only=True` でエクスポートする。** Hand head には `final_only` が無いため、ラッパー側で最終層の出力だけを取り、auxiliary（積み上げた中間予測）は捨てる。
- **B4 — `ste_clamp` の detach トリック**：`clamped.detach() + (raw − raw.detach())`（`soma_hand_head.py`、`soma_body_head.py` の `apply_camera_scale_constraint`）。推論時（`inference_mode`）は detach が消えるため、順方向の値は clamp そのもの。ONNX 化の問題にはならない見込み。ただし Hand は `camera_depth_mode=log_lambda` のため `apply_camera_scale_constraint` は素通し（`camera_depth_mode != "s"` で早期 return）する。
- **B5 — `rot6d_to_rotmat`**（`gem/common/geometry/rotations.py`）：Gram-Schmidt（normalize＋einsum＋cross）。すべて標準 op で、ONNX 化しやすい。問題になるのは `einsum` の実装がパーサに通るかだけで、通らなければ手書きの積和に置き換える。
- **B6 — 動的形状**：切り出しが 256 固定のため特徴グリッドは 16×16 固定。`ray_feats` の形状チェック（`(B, H*W, 32)`）も固定。バッチは B=1 固定。動的軸は使わない。`torch.arange` で作る位置（RayPE のグリッド座標）は定数として畳み込める。

## 4. パイプライン全体（推奨順序）

```
1. ckpt inspect（tools/quantize/inspect_ckpt.py）
     body_s240000.ckpt（FP32 ~3.54GB）、egohand_s060000.ckpt（FP32 ~10.66GB。
     Hand は Adam 状態を含む疑い。重みのみ FP16 で ~1.77GB）
        ↓
2. FP16 化（tools/quantize/quantize_ckpt_to_fp16.py、CPU、学習画像なし）
     重みだけを FP16 に。オプティマイザ状態は捨てる
        ↓
3. FP32 ONNX エクスポート（tools/quantize/export_onnx_real.py、CPU、dynamo）
     ラッパー：明示的テンソル入力、encode() 相当、最終層出力のみ。
     Hand は ray_feats を CPU 計算で入力に、kp2d_head は除外
        ↓
4. onnxruntime で PyTorch FP32 と数値比較（最大絶対誤差）
     通らなければ 3 に戻る（op の置き換え）
        ↓
5. ModelOpt AutoCast で FP16 ONNX（CPU、グラフ書き換えのみ、学習画像なし）
        ↓
6. tensorrt_rtx でパース＋ビルド
     tensorrt_rtx --onnx=body_fp16.onnx --saveEngine=body_fp16.trt --memPoolSize=tacticSharedMem:48K
     サポート外 op が出たら 3 に戻る
        ↓
7. DLL で実推論＋SOMA FK、PyTorch との数値比較（MPJPE）とレイテンシ計測
```

- FP16 化（2）と ONNX エクスポート（3）の順序に注意：**エクスポートは FP32 の重みで一度行い、FP16 化は ModelOpt AutoCast（5）で行う。** 先に重みを FP16 にしてからエクスポートすると、エクスポート時の数値誤差と AutoCast の誤差が区別できなくなる。重みの FP16 化スクリプトはチェックポイントの検査・保管用であり、エクスポートの入力は FP32 のままがよい。
- INT8 は v1 の対象外（キャリブレーション画像が必要になるため）。

## 5. 検証状況の正直な表

| 項目 | 状況 | 根拠 |
|---|---|---|
| GEM-X ソース読解（PoseModel／head／RayPE／kp2d／FK／crop） | 検証済み（読解） | 本レポートのファイル：行引用 |
| リポジトリに ONNX エクスポートコードが無い | 検証済み（grep） | `onnx` の出現は RT-DETR 検出器のみ |
| 切り出し 256×256 | 検証済み（読解） | `PoseInferenceConfig.image_size = 256` |
| Body／Hand の入出力テンソル形状 | 検証済み（読解） | §1.3 の表 |
| H1〜H6 のブロッカー | 未実行（ソース読解による推定） | 実行環境（torch／GPU）がこの VM に無い |
| dynamo export が通るか | 未実行 | Windows PC での実行待ち |
| tensorrt_rtx のパース（RoPE／SDPA／einsum） | 未実行 | Windows PC での実行待ち |
| FP16 化の数値影響 | 未実行 | 重みをダウンロードしていない |
| レイテンシ（Body／Hand の実測） | 未実行（推定値のみ） | 25–40ms は RTX 5070 Ti 級の推定で、4070 Ti SUPER の実測ではない |
| SOMA FK の C++ 実装可否（D-2） | 未実行 | 資産の中身未確認 |

## 6. Windows PC でのチェックリスト（daisuke 向け）

環境：RTX 4070 Ti SUPER、128GB RAM、ドライバ 32.0.16.1692、Windows。Python 3.10+、torch ≥ 2.10、transformers == 5.3.0、einops、onnxruntime、onnx、nvidia-modelopt、TensorRT-RTX 1.7.1.116、CUDA 13.4。

```
[ ] 1. NVlabs/GEM-X を clone し、pip install -e .（[hand] extra も。onnxruntime が入る）
[ ] 2. HF からログインなしで DL：body_s240000.ckpt、body_config.json、
        egohand_s060000.ckpt（tools/quantize/hf_download.py --list で先に確認）
        30GB 空ける。401/403 なら hf auth login
[ ] 3. inspect_ckpt.py で両 ckpt の中身を確認（EgoHand の Adam 状態の有無と重みサイズ）
[ ] 4. Python で公式 API が動くことを確認（CPU でよい）：
        load_pose_checkpoint / load_hand_checkpoint でモデル構築＋
        ダミー 256x256 切り出しで encode() が通ること
[ ] 5. export_onnx_real.py --task body --export-only を実行（dynamo export）
        → body_fp32.onnx ができること。失敗したらエラーメッセージ全文を保存
[ ] 6. onnxruntime で body_fp32.onnx と PyTorch の出力を比較（最大絶対誤差を記録）
[ ] 7. ModelOpt AutoCast → body_fp16.onnx
[ ] 8. tensorrt_rtx --onnx=body_fp16.onnx --saveEngine=body_fp16.trt
        --memPoolSize=tacticSharedMem:48K（パースエラーの全文を保存）
[ ] 9. Hand も同じ手順（5〜8）。ray_feats は CPU 計算、kp2d_head は除外
[ ] 10. 測定して報告：各ステップの成否・最大絶対誤差・MPJPE・
        body_ms / hand_ms / 合計レイテンシ・VRAM 使用量
```

測定してほしいもの（数値の捏造を避けるため、必ず実測値を報告する）：
- ONNX vs PyTorch の最大絶対誤差（`body_pose`、`hand_pose`、`pred_cam` それぞれ）
- FK 後の MPJPE（Body 77 関節、Hand 25 関節×2）
- レイテンシ：Body 推論 ms、Hand 推論 ms（左右合計）、前処理 ms、FK ms、合計 ms、fps
- VRAM：エンジン常駐時の使用量（Body＋Hand＋検出器）

## 7. 出典（URL・ファイル）

- NVlabs/GEM-X（GitHub）：https://github.com/NVlabs/GEM-X（コミット `4aef145`）
- `gem/pose/model.py`（PoseModel／encode）、`gem/pose/api.py`（body_config／hand_config／load_pose_checkpoint）、`gem/pose/tasks/hand.py`（mirror_hand_camera）、`gem/pose/tasks/body.py`
- `gem/pose/heads/soma_body_head.py`、`gem/pose/heads/soma_hand_head.py`、`gem/pose/heads/transformer_decoder.py`、`gem/pose/heads/rletr_occ_head.py`
- `gem/common/backbones/dinov3.py`（DINOv3ViTModel、RoPE、gated MLP）、`gem/pose/conditioning/ray_pe.py`（`torch.linalg.solve`）、`gem/pose/conditioning/cliff.py`、`gem/pose/conditioning/camera_depth.py`
- `gem/pose/predictor.py`（FK・カメラ復号・kp2d ソルブ）、`gem/pose/geometry/hand.py`／`body.py`（SOMA FK）、`gem/pose/data/crop.py`（256 切り出し、ImageNet 正規化）、`gem/pose/camera.py`（decode_camera）、`gem/pose/conditioning/fixed_depth_translation.py`（2×2 最小二乗）
- `gem/common/detectors/hand.py`（RT-DETR 手検出器、onnxruntime）、`gem/inference/hand.py`
- HF：`nvidia/GEM-X`（`body_config.json`、`body_s240000.ckpt`）、`nvidia/EgoHand-v1`（`egohand_s060000.ckpt`）、`nvidia/SOMA-X`（SOMA 資産）
- TensorRT-RTX SCG：https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/compute-graphics.html（Ada 48 KiB、CiG は CUDA 12.6+／ドライバ 555+、DirectX のみ）
- TensorRT-RTX 複数エンジン／実行コンテキスト：https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/architecture/how-trt-rtx-works.html
