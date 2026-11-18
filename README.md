# GEM-X-onnx

GEM-X の Body / Hand モデルを、GitHub Codespace（16GB マシン）上で FP16 量子化して ONNX にエクスポートするためのリポジトリ。**自分の PC では何も実行しない。** ダウンロードした重みやコードを、パスワードやログイン済みセッションのある PC で動かすサプライチェーンリスクを避けるための構成。GitHub Actions はやめた（16GB ランナーが OOM で死ぬため）。

## 親（上流）の指定

- 上流：[NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) コミット `4aef1452e5df171deff4f293c08e643eb450981e`
- このリポジトリはフォークではない。GEM-X のソースコードを同梱していない。スクリプトが実行時に上流を clone する。
- GEM-X のライセンス：コードは Apache 2.0、モデルは NVIDIA Open Model License（モデルカード曰く商用・非商用どちらも可）。

## 何をするものか

`run_gemx_onnx.py` を1回走らせるだけ。Body → Hand の順に、1モデルずつ処理する。

1. NVlabs/GEM-X をコミット `4aef145` で clone する。
2. `.venv` を作って CPU 版 torch + GEM-X の依存関係を入れ、自分自身を venv 内で再実行する。
3. Hugging Face からそのモデルの重みをダウンロードする（ログインなしが先。EgoHand が 401/403 なら `HF_TOKEN` でリトライ）。
4. チェックポイントを inspect して、ONNX エクスポートする（`--compare` と `--fp16-output` 付き）。
5. チェックポイント自体も FP16 化する（学習画像なし、キャリブレーションなしの重みキャスト）。
6. 次のモデルのために、落としたチェックポイントと FP32 ONNX を消す（`--keep-weights` / `--keep-fp32-onnx` を付ければ残す）。FP16 ONNX と FP16 チェックポイントは `out/` に残る。

TensorRT-RTX のエンジンビルド（SCG 48K）とレイテンシ測定は、Codespace に GPU がないのでここではできない。そこだけは GPU のあるマシン（RTX 4070 Ti SUPER + CUDA 13.4 + TensorRT-RTX 1.7.1.116）でやる。ONNX までここで作って、`.trt` だけ別途ビルドする形。

## 含まれているもの・含まれていないもの

| もの | 扱い | ライセンス / 備考 |
| --- | --- | --- |
| このリポジトリのスクリプト（`run_gemx_onnx.py`、`tools/quantize/`） | 同梱 | Apache 2.0（`LICENSE` 参照） |
| GEM-X Body 重み `body_s240000.ckpt` | 実行時に HF からダウンロード。同梱しない | NVIDIA Open Model License。モデルカード曰く商用可 |
| EgoHand 重み `egohand_s060000.ckpt` | 実行時に HF からダウンロード。同梱しない | OpenMDW 1.1。モデルカード曰く商用可。配布時はライセンス文と著作権表示の保持が必要（`LICENSES/LicenseRef-OpenMDW-1.1.txt` 参照） |
| RT-DETR Hand Detector `hand_detector.onnx` | 実行時に HF からダウンロード。同梱しない | OpenMDW 1.1 |
| SOMA-X 資産（`SOMA_neutral.npz` など） | このスクリプトでは使わないので落とさない。同梱しない | Apache 2.0。エクスポートには不要（FK は対象外） |
| YOLOX-X HumanArt 検出器 | **含まない。落とさない。** ライセンスの商用可否が未確認のため | エクスポートには不要。推論時に使う場合は各自でライセンスを確認すること |
| 重み・ONNX・エンジン・SOMA 資産全般 | **同梱しない。** `.gitignore` で誤ってコミットされないようにしてある | 実行時に公式ソースから取得する |

## 使い方

### 初回のみ：リポジトリ作成と初回 push

1. GitHub でリポジトリ `RabbitDaisuke/GEM-X-onnx` を作る（初回のみ）。
2. この zip の中身を push する。

### それ以降：更新

リポジトリを作り直さない。通常の push で更新する。

### Codespace で実行

1. リポジトリのページで Code > Codespaces > Create codespace を選ぶ。**マシンタイプは 4-core / 16GB を選ぶこと。** 2-core / 8GB ではメモリが足りない。
2. ターミナルで1コマンドだけ：
   ```
   python3 run_gemx_onnx.py
   ```
   途中で止まったら、ログの最後の行を見て、同じコマンドをもう一度走らせる（ダウンロード済みのものは再利用される）。
3. 終わったら `out/` に `body_fp16.onnx`、`hand_fp16.onnx`、`body_fp16.safetensors`、`hand_fp16.safetensors` と各ログが残る。VS Code からダウンロードする。

ディスクについて：`/workspaces` は 32GB しかなくて空きは ~19GB だけど、`/tmp` は別ディスク 118GB で空き ~108GB ある。スクリプトは重み（Hand で 10.7GB）を自動で `/tmp` に置くから、容量は足りる。成果物の `out/` だけリポジトリ内に残る。`/tmp` が狭い環境では重みはリポジトリ内に置かれて、代わりに不要なツール類の削除で空きを作る。

EgoHand のログインなしダウンロードが 401/403 で失敗する場合だけ、Codespace シークレットか `export HF_TOKEN=...` でトークンを渡す。トークンはコードやコミットには絶対に入れないこと。

### オプション

- `python3 run_gemx_onnx.py --task body`：Body だけ。
- `python3 run_gemx_onnx.py --keep-weights --keep-fp32-onnx`：全部残す。両方付けると ~30GB 要る。Codespace のディスクは 32GB なので、片方だけ残すのが無難。
- `python3 run_gemx_onnx.py --self-test`：環境確認だけ。ダウンロードなし。

### Hugging Face に push する

終わったあと `out/` を Hugging Face のモデルリポジトリに上げる。トークンは引数で渡す（`--hf-token`、なければ `HF_TOKEN` 環境変数）。ログにトークンは出ない（`***` でマスクされる）。

```
python3 run_gemx_onnx.py --hf-push RabbitDaisuke/gemx-onnx-fp16 --hf-token $HF_TOKEN
```

リポジトリがなければ作られる。`out/README.md`（モデルカード）も一緒に上がる。上がるのは FP16 ONNX・FP16 チェックポイント・ログだけ。FP32 ONNX は含まれない。

## 失敗したとき

そのステップのログ全文が結果。`out/body_fp16.onnx` ができるまでは「エクスポート成功」とは言わない。ログを貼って相談すること。
