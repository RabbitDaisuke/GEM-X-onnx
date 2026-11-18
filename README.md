# GEM-X-onnx

GEM-X の Body / Hand モデルを、GitHub Actions（CPU ランナー）上で FP16 量子化して ONNX にエクスポートするためのリポジトリ。**自分の PC では何も実行しない。** ダウンロードした重みやコードを、パスワードやログイン済みセッションのある PC で動かすサプライチェーンリスクを避けるための構成。

## 親（上流）の指定

- 上流：[NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) コミット `4aef1452e5df171deff4f293c08e643eb450981e`
- このリポジトリはフォークではない。GEM-X のソースコードを同梱していない。Actions が実行時に上流を clone する。
- GEM-X のライセンス：コードは Apache 2.0、モデルは NVIDIA Open Model License（モデルカード曰く商用・非商用どちらも可）。

## 何をするものか

1. GitHub Actions が NVlabs/GEM-X を clone する。
2. Python 3.11 + CPU 版 torch + GEM-X の依存関係を入れる。
3. Hugging Face から重みをダウンロードする（ログインなしが先。EgoHand が 401/403 なら `HF_TOKEN` シークレットでリトライ）。
4. 両チェックポイントを inspect して、Body → Hand の順で ONNX エクスポートする（`--compare` と `--fp16-output` 付き）。
5. チェックポイント自体も FP16 化する（学習画像なし、キャリブレーションなしの重みキャスト）。
6. ONNX とログを成果物としてアップロードする（7日間保持）。

TensorRT-RTX のエンジンビルド（SCG 48K）とレイテンシ測定は、GitHub のランナーに GPU がないのでここではできない。そこだけは GPU のあるマシン（RTX 4070 Ti SUPER + CUDA 13.4 + TensorRT-RTX 1.7.1.116）でやる。ONNX までここで作って、`.trt` だけ別途ビルドする形。

## 含まれているもの・含まれていないもの

| もの | 扱い | ライセンス / 備考 |
| --- | --- | --- |
| このリポジトリのスクリプト・ワークフロー（`tools/quantize/`、`.github/workflows/`） | 同梱 | Apache 2.0（`LICENSE` 参照） |
| GEM-X Body 重み `body_s240000.ckpt` | 実行時に HF からダウンロード。同梱しない | NVIDIA Open Model License。モデルカード曰く商用可 |
| EgoHand 重み `egohand_s060000.ckpt` | 実行時に HF からダウンロード。同梱しない | OpenMDW 1.1。モデルカード曰く商用可。配布時はライセンス文と著作権表示の保持が必要（`LICENSES/LicenseRef-OpenMDW-1.1.txt` 参照） |
| RT-DETR Hand Detector `hand_detector.onnx` | 実行時に HF からダウンロード。同梱しない | OpenMDW 1.1 |
| SOMA-X 資産（`SOMA_neutral.npz` など） | このワークフローでは使わないので落とさない。同梱しない | Apache 2.0。エクスポートには不要（FK は対象外） |
| YOLOX-X HumanArt 検出器 | **含まない。落とさない。** ライセンスの商用可否が未確認のため | エクスポートには不要。推論時に使う場合は各自でライセンスを確認すること |
| 重み・ONNX・エンジン・SOMA 資産全般 | **同梱しない。** `.gitignore` で誤ってコミットされないようにしてある | 実行時に公式ソースから取得する |

## 使い方

### 初回のみ：リポジトリ作成と初回 push

1. GitHub でリポジトリ `RabbitDaisuke/GEM-X-onnx` を作る（初回のみ）。
2. この zip の中身を push する。
3. 任意：EgoHand のログインなしダウンロードが 401/403 で失敗する場合だけ、リポジトリの Settings > Secrets に `HF_TOKEN`（Hugging Face のトークン）を追加する。
4. Actions タブで「GEM-X ONNX export (CPU)」を選んで Run workflow を押す。
5. 終わったら成果物 `gemx-onnx` から ONNX とログをダウンロードする。

### それ以降：更新

リポジトリを作り直さない。通常の push で更新する。ワークフローは `tools/quantize/**` と `.github/workflows/gemx-onnx.yml` の変更でも自動で走る。

## 失敗したとき

そのステップのログ全文が結果。`body_fp32.onnx` ができるまでは「エクスポート成功」とは言わない。ログを貼って相談すること。
"# GEM-X-onnx" 
