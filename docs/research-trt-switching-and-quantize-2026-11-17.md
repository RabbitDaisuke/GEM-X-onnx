# TensorRT-RTX のモデル切り替えと FP16 量子化の検証

- 調査日: 2026-11-17
- 対象: daisuke の Windows PC (RTX 4070 Ti SUPER, Ada, 16GB VRAM,
  **RAM 128 GB**、ドライバ **32.0.16.1692 (2026/09/04)**、
  DirectX 12 FL 12.2、C:/D: NVMe SSD、アイドル時 GPU 1%・VRAM 1.3/16.0 GB・57°C)
- TensorRT-RTX 1.7.1.116 / CUDA 13.4
- きっかけ: GPT-6 chat free が「アンロードせずにモデルを切り替える
  公式機能がある」と言った、という話の検証 + Python 量子化ツールの作成

## 結論の要約

1. **「アンロードせずに切り替える」という名前の公式機能は、NVIDIA の
   TensorRT-RTX ドキュメントには見つからなかった。** GPT-6 free の話は、
   別々の実在する仕組みをひとつにまとめたものと考えられる。
   本当にあるのは次の組み合わせ:
   - **複数エンジン常駐 + 複数 IExecutionContext (標準パターン)。**
     Body と Hand を別々のエンジンとして VRAM に両方載せておき、
     切り替えは別のコンテキストで `enqueueV3` を呼ぶだけ。
     VRAM に入る限りアンロード/リロードは発生しない。
   - **Deferred weights loading (1.6+)。** 複数エンジンを重みを載せずに
     デシリアライズして常駐させておける。「多くのエンジンを常駐させる
     マルチモデルアプリ」向けと明記されている。
   - **IWeightsManager の offload / restore。** *同じ*エンジンの重みを
     VRAM から外し、後で再ビルド/再 refit なしに復元する。
     アーキテクチャの違う Body と Hand の切り替えには使えない。
   - **Refit / weightless engine。** *同じ*アーキテクチャで重みだけ
     差し替える機能 (LoRA 切り替えなど)。Body と Hand はアーキテクチャが
     違うので対象外。
   - **Runtime cache。** JIT コンパイル済みカーネルをディスクに保存して
     次回起動を速くするもの。切り替え機能ではない。
2. **SCG は切り替え機能ではない。** SCG (Simultaneous Compute and
   Graphics) は推論とグラフィックス描画を同時に走らせる機能。
   Ada では共有メモリ上限 48 KiB、CiG は CUDA 12.6+/ドライバ 555+、
   DirectX のみ。エンジンビルドは `--memPoolSize=tacticSharedMem:48K`。
3. **推奨設計は変わらない: 単一ストリームで Body → Hand を直列に
   enqueue する。** 切り替えコストは実質ゼロ (次の enqueue を呼ぶだけ)。
   並列に走らせる必要はないし、1.4 の既知の問題 (下記) を踏まえると
   並列化は v1 では避けるのが正しい。
4. **FP16 化は CPU でできる。学習画像はいらない。** ツールを
   `tools/quantize/` に作った。ダミーでのテスト済み。
5. **128 GB RAM のおかげで、FP16 変換の RAM 問題は消えた。**
   素直な `.half()` でもピーク 8〜10 GB。ストリーミング方式は
   残してあるが必須ではない。ドライバも CiG の 555+ を満たす。

---

## 1. 何を検索したか

検索クエリ (2026-11-17、browser.search + browser.open で原文を確認):

- `TensorRT-RTX multiple execution contexts multiple engines simultaneously`
- `TensorRT concurrent inference multiple engines switch without unloading`
- `TensorRT-RTX runtime cache multiple engines`
- `TensorRT weight streaming refit switch model weights without rebuilding engine`
- `"TensorRT-RTX" "multiple engines" "parallel CUDA graph capture" unique stream per context`
- `site:docs.nvidia.com TensorRT-RTX "deferred weights loading" multi-model engines resident`
- `TensorRT-RTX 1.4 release notes segmentation fault concurrent inference multiple engines crash`

読んだ原文ページ:

- How TensorRT-RTX Works (Threading / Memory / Weight Streaming / Runtime Cache)
- Simultaneous Compute and Graphics (SCG)
- CPU-Only AOT and TensorRT-RTX Engines (Weightless / Refit / Offloading and Restoring Engine Weights / Deferred Weights Loading)
- Working with Runtime Cache
- TensorRT-RTX 1.4 Release Notes (Fixed Issues / Known Issues)
- TensorRT-RTX 1.6 Release Notes (Stability Fixes)
- IExecutionContext (TensorRT-RTX Python API 1.7.1)
- Refitting an Engine (TensorRT 10.x docs)

---

## 2. TensorRT-RTX Runtime は複数エンジンを保持できるのか — 検証結果

**できる。公式ドキュメントに明記されている。** ただしそれは
「切り替え専用の新機能」ではなく、ランタイムの基本設計である。

### 2.1 IRuntime / ICudaEngine / IExecutionContext の契約

How TensorRT-RTX Works の Threading 節より:

> "The TensorRT-RTX runtime supports concurrent inference across
> multiple threads and GPUs" / "Engines are immutable after
> deserialization and can be safely shared across multiple threads.
> A single engine can back many execution contexts running
> concurrently on the same or different GPUs." /
> "An execution context is **not** safe to use concurrently from
> multiple threads. Create one execution context per thread." /
> "The recommended pattern for multi-threaded inference is **one
> execution context per thread, sharing a single deserialized engine**"

Python API の IExecutionContext も同じことを言っている:

> "Multiple `IExecutionContext` s may exist for one `ICudaEngine`
> instance, allowing the same `ICudaEngine` to be used for the
> execution of multiple batches simultaneously."

つまり API の形はこうなる:

```cpp
// 1. ランタイムを1つ作る
IRuntime* runtime = createInferRuntime(logger);
// 2. エンジンを2つデシリアライズして、両方 VRAM に常駐させる
ICudaEngine* bodyEngine = runtime->deserializeCudaEngine(bodyBlob, bodySize);
ICudaEngine* handEngine = runtime->deserializeCudaEngine(handBlob, handSize);
// 3. エンジンごとに実行コンテキストを1つ作る (スレッドごとに1つが契約)
IExecutionContext* bodyCtx = bodyEngine->createExecutionContext();
IExecutionContext* handCtx = handEngine->createExecutionContext();
// 4. 「切り替え」は、他のコンテキストで enqueueV3 を呼ぶだけ
bodyCtx->enqueueV3(bodyStream);   // Body 推論
handCtx->enqueueV3(handStream);   // Hand 推論 — アンロードもリロードも無い
```

エンジンはデシリアライズ時に重み分のデバイスメモリを確保する
("An engine allocates device memory to store model weights upon
deserialization. The serialized engine size approximates the weight
memory required.")。だから **VRAM に両方入る限り、切り替えで
アンロード/リロードは一切発生しない**。これが GPT-6 free の話の
実態にいちばん近い。名前のある専用機能ではなく、ランタイムの
標準的な使い方である。

### 2.2 Deferred weights loading (1.6+) — 常駐を軽くする仕組み

CPU-Only AOT のページより:

> "Use deferred weights loading to deserialize a TensorRT-RTX engine
> without allocating the engine weights in host or GPU memory. ...
> It is useful when an application deserializes several engines before
> running any of them, such as a multi-model application that keeps
> many engines resident."

つまり起動時に Body + Hand + (将来 Motion) のエンジンを先に
デシリアライズだけしておき、重みは推論直前に `loadWeights()` で
載せる、という使い方が公式に想定されている。JIT コンパイルも
コンテキスト作成時に走るので、ランタイムキャッシュと組み合わせれば
複数モデルのキャッシュを並行して作れる。ただし weight streaming とは
排他 (両方有効にすると API 使用エラー)。

このプロジェクトでは必須ではない (Body + Hand の FP16 重みは
合計 ~3.45 GB で、16 GB VRAM に十分入る) が、将来 Motion
(ViTPose 1.69 GB 追加) まで載せるときの選択肢になる。

### 2.3 IWeightsManager の offload / restore — 同じエンジンの重みの一時退避

同じページより:

> "After you refit a weightless engine, use `IWeightsManager` to save
> its loaded weights, release their physical GPU memory, and restore
> them later without rebuilding or refitting the engine. An inactive
> engine can therefore give up its weight memory so that another engine
> can use it."

これは「使っていないエンジンの重みを VRAM から外して、後で
再ビルドなしに復元する」機能で、**同じエンジン**の話である。
Body のエンジンと Hand のエンジンを行き来する切り替えには使えない。
VRAM が足りなくなったときの最終手段として知っておく価値はあるが、
v1 の設計には不要。

### 2.4 Refit / weightless engine — 同じアーキテクチャの重み差し替え

> "TensorRT can *refit* an engine with new weights without having to
> rebuild it." / weight-stripped engine は `kSTRIP_PLAN` + `kREFIT`
> で作り、実行時に重みを戻す。

これも **同じアーキテクチャで重みだけ違う**場合の機能
(LoRA 切り替えが典型例)。Body (DINOv3 + SOMA body head) と
Hand (DINOv3 + SOMA hand head) はヘッドも入出力も違う別モデルなので、
1つのエンジンを refit して Body と Hand を切り替えることはできない。
別々のエンジンを作るのが正しい。

### 2.5 Runtime cache — 起動高速化であって切り替えではない

ランタイムキャッシュは JIT コンパイル済みカーネルをディスクに保存し、
次回起動時の再コンパイルを省くもの。CiG 状態・GPU・TensorRT-RTX
バージョン・ドライバが一致しないと無視されて JIT し直す (自動で
フォールバックするのでエラーにはならない)。Body と Hand で
キャッシュを共有する話ではない。

### 2.6 Parallel CUDA graph capture — 複数コンテキストの並列実行

1.4 の Fixed Issues より:

> "Enabled running multiple inference contexts in parallel with CUDA
> graph capture. Set a unique stream per context using the `enqueueV3` API."

つまり複数コンテキストを並列に走らせること自体は 1.4 で有効化された。
ただし次の 2.7 の既知の問題があるため、v1 では使わない。

### 2.7 既知の問題: 複数コンテキストの並列実行はクラッシュすることがある (1.4)

1.4 Release Notes の Known Issues (Functional) より、原文のまま:

> "Running multiple inference contexts in parallel (for example, in a
> thread pool) may cause occasional crashes. Affects all GPU
> architectures; a fix is planned for the next TensorRT-RTX update."

**注意:** これは 1.4 時点の話である。1.6 の Stability Fixes には
"Resolved a JIT initialization crash ... and a Windows race condition
affecting concurrent inference across multiple engines." とあり、
このプロジェクトが使う 1.7.1.116 では修正済みのはずである。
ただし修正済みでも、**このプロジェクトの推奨設計は並列実行ではなく
単一ストリームでの直列実行**なので、この問題を踏む設計にはなっていない。
並列化を試すのは、直列で測って足りないと分かった後でいい。

### 2.8 検証の結論 (GPT-6 free の話について)

| GPT-6 free の言い方 | 実際の公式ドキュメント |
|---|---|
| アンロードせずに切り替える公式機能がある | その名前の機能は見つからなかった |
| (おそらく指していたもの) | 複数エンジン常駐 + 複数コンテキストで enqueue を切り替える標準パターン。VRAM に入る限りアンロードは発生しない |
| 同上 | Deferred weights loading (1.6+): 複数エンジンを常駐させるための仕組み |
| 同上 | IWeightsManager offload/restore: 同じエンジンの重みの一時退避 |
| 同上 | Refit: 同じアーキテクチャの重み差し替え (Body/Hand 間では使えない) |

**daisuke への答え: 「切り替え」は特別な機能を呼ぶのではなく、
Body と Hand のエンジンを両方 VRAM に載せておいて、enqueue する
コンテキストを変えるだけ、という設計で正しい。** SCG とは別の話で、
SCG はあくまで推論と描画の同居の話である。

---

## 3. SCG の本当の役割 (切り替え機能ではない)

SCG (Simultaneous Compute and Graphics) は、TensorRT-RTX が
グラフィックスワークロードと同時に推論を走らせるための機能。
典型例はゲームで NPC の生成 AI を 3D 描画と並行して動かすこと。
このプロジェクトでは SteamVR / VRChat の描画と GEM-X 推論の同居に
あたる。

公式ドキュメント (Simultaneous Compute and Graphics) より:

- Ampere / Ada (CC 8.x) では、グラフィックスと同時に走る CUDA
  カーネルの共有メモリは **48 KiB 以下**に制限される。
  Blackwell (CC 10.x 以降) では 84 KiB まで上げられる (L1 キャッシュ
  とトレードオフ)。低優先キューだと Blackwell でも 48 KiB のまま。
- Turing (CC 7.5 以下) では SCG は非対応。
- 共有メモリ制限により、SCG 時の推論は単独実行より遅くなるのが普通
  ("will generally result in reduced performance compared to running
  standalone inference")。
- エンジンビルド: Ampere/Ada では `--memPoolSize=tacticSharedMem:48K`。
  C++ API では
  `config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 * 1024)`。
- CiG モード: CUDA 12.6 以上、ドライバ 555 以上が必要。
  TensorRT-RTX は DirectX のみの対応で、Vulkan は将来予定。

**この PC での確認:** RTX 4070 Ti SUPER は Ada (CC 8.9) なので
48 KiB が上限で、84 KiB は使えない。ドライバ 32.0.16.1692
(2026/09/04) は 555+ を満たす。CUDA 13.4 も満たす。
アイドル時 VRAM 1.3/16.0 GB なので、推論用に約 14.7 GB が空いている。

SCG を「モデル切り替え」と混同しないこと。SCG が決めるのは
「推論カーネルが描画と同時に走れるか・どの共有メモリ制限で
ビルドするか」であって、「Body と Hand をどう切り替えるか」ではない。
切り替えは第2章の常駐パターンの話である。

---

## 4. このプロジェクトの推奨設計 (Body + Hand)

```
起動時 (1回だけ):
  runtime = createInferRuntime()
  bodyEngine = runtime->deserializeCudaEngine(body_fp16.trt)   # ~1.68 GB
  handEngine = runtime->deserializeCudaEngine(hand_fp16.trt)   # ~1.77 GB
  bodyCtx = bodyEngine->createExecutionContext()  # 1コンテキスト
  handCtx = handEngine->createExecutionContext()
  両方とも SCG 48K でビルド済み。ランタイムキャッシュを保存しておく。

毎推論ティック (単一 CUDA ストリーム、直列):
  1. Body: bodyCtx->enqueueV3(stream)          # 25〜40ms (参考値)
  2. Hand: handCtx->enqueueV3(stream)          # 25〜40ms (参考値)
       crop は前フレームの手首位置から切る (Body を待たないパイプライン化)
  3. SOMA FK・座標変換 → PoseFrame → Rust EXE へ
  4. Rust 側で lerp/slerp 補間して 60Hz で VMC/OpenVR/WebSocket へ送出

切り替え = 手順 1→2 で enqueue するコンテキストを変えるだけ。
アンロード/リロードは無い。VRAM に入る限り、切り替えコストは実質ゼロ。
```

並列化しない理由:

- GPU は1枚。別ストリームで同時に走らせても SM・L2・DRAM 帯域を
  奪い合うだけで合計スループットはほぼ変わらない。
- SCG で描画とも共有しているので、推論同士の並列化は競合を増やすだけ。
- 1.4 の既知の問題 (並列コンテキストのクラッシュ) は 1.6 で修正済みと
  されているが、わざわざ踏みに行く理由がない。直列で測ってから決める。

VRAM 予算 (4070 Ti SUPER, 16 GB、FP16):

| 項目 | 概算 |
|---|---|
| Body エンジン (重み) | 1.68 GB |
| Hand エンジン (重み) | 1.77 GB |
| YOLOX-X (CPU 実行なら 0) | 0 GB |
| RT-DETR hand detector (v1 では使わない) | 0 GB |
| アクティベーション + ワークスペース (2本分) | 2〜4 GB |
| ランタイムキャッシュ等 | ~0.5 GB |
| **推論側の合計** | **約 6〜7 GB** |
| SteamVR / VRChat 等の描画側 | 数 GB |
| **残り** | 数 GB の余裕 |

Motion まで載せると ViTPose 1.69 GB + Motion 0.1 GB が増える。
v1 では Motion は見送り (前回の調査レポート第4章参照)。

VRAM に入らなくなったときだけアンロードが避けられなくなる。
そのときの選択肢は、(a) IWeightsManager で使っていないエンジンの
重みを外す、(b) weight streaming を使う、(c) Motion を諦める、
の順で検討する。v1 では不要。

---

## 5. FP16 量子化 — CPU でできること (学習画像なし)

### 5.1 結論

- **重みの FP32→FP16 キャストは CPU だけでできる。** GPU は不要。
  学習画像もキャリブレーションデータもいらない。キャストは
  テンソル単位の型変換だからだ。
- **FP16 での推論を CPU でやるのは別問題で、ほぼできない**
  (CPU カーネルの多くが FP16 未対応)。このプロジェクトでは
  CPU 推論は使わないので問題にならない。FP16 ファイルは
  Windows GPU 上の TensorRT-RTX 用である。
- **128 GB RAM のおかげで、RAM 問題は消えた。** 素直な
  `model.half()` でもピーク 8〜10 GB。EgoHand の 10.66 GB を
  丸ごと読んでも 15 GB 弱で収まる。ストリーミング方式は
  残してあるが必須ではない。
- **INT8 / FP8 は v1 ではやらない。** スケールを決めるのに
  キャリブレーションデータ (実画像) が必要になるからだ。

### 5.2 作ったツール (`tools/quantize/`)

| ファイル | 役割 |
|---|---|
| `quantize_ckpt_to_fp16.py` | `.ckpt`/`.pt`/`.safetensors` を FP16 化。オプティマイザ状態は既定で除外。出力の再読み込み検証つき |
| `inspect_ckpt.py` | チェックポイントの中身一覧。EgoHand の 10.66 GB の正体確認用 |
| `hf_download.py` | ログインなしで Hugging Face から落とす (stdlib のみ) |
| `export_onnx_fp16.py` | **テンプレート (未検証)**。FP32 ONNX → ModelOpt AutoCast |
| `make_dummy_ckpt.py` | テスト用ダミー生成 |
| `requirements.txt` | torch (CPU) / safetensors / huggingface_hub / numpy / onnx / onnxruntime |
| `README.md` | 使い方・SCG・切り替えのまとめ・テスト記録 |

使い方の詳細は `tools/quantize/README.md` を参照。

### 5.3 EgoHand の 10.66 GB について (再確認)

- `egohand_s060000.ckpt` は 10,657,169,753 B。
  EgoHand は 883.5M パラメータで、FP32 の重みだけなら約 3.53 GB。
  883.5M × 12 B (FP32 重み + Adam m + v) = 約 10.60 GB とほぼ一致するため、
  **オプティマイザ状態を含むフル学習チェックポイントの可能性が高い**
  (未検証の推測。Windows PC で `inspect_ckpt.py` をかけて確認すること)。
- `quantize_ckpt_to_fp16.py` は `exp_avg` / `exp_avg_sq` /
  `optimizer` / `param_groups` などのキーと、トップレベルの
  `optimizer` / `state` 辞書を検出して既定で除外し、除外したものを
  全部報告する。重みだけ取り出して FP16 化すれば約 1.77 GB になるはず。
- Body のほうは公式 README に「重みのみでオプティマイザ状態は含まない」と
  明記されており、ファイル 3.54 GB は 840M × 4 B ≒ 3.36 GB と一致する。

### 5.4 推奨パス (変わらず)

```
1. HF から重みを落とす (hf_download.py、ログインなしで試す)
2. inspect_ckpt.py で EgoHand の中身を確認する
3. quantize_ckpt_to_fp16.py で FP16 .safetensors 化する (CPU、学習画像なし)
4. PyTorch (CPU) で PoseModel を構築し、FP32 のまま ONNX エクスポート
   (Body と Hand は別々に。SOMA FK 等は C++ 側に残す。静的形状)
5. ModelOpt AutoCast で FP16 ONNX 化 (CPU で可)
6. ONNX Runtime で PyTorch FP32 と数値比較 (MPJPE・最大絶対誤差)
7. TensorRT-RTX でエンジンをビルド (GPU 上、SCG 48K フラグ付き)
8. C++ DLL から読み、同じ入力で TensorRT 出力と PyTorch 出力を数値比較
```

### 5.5 テスト結果 (この Linux VM、GPU なし)

- ダミーの safetensors チェックポイント (float32 テンソル + int64 +
  bool + 偽の Adam 状態) を作り、`inspect_ckpt.py` と
  `quantize_ckpt_to_fp16.py` を実行。
- 結果: float テンソルは float16 になり、int64 / bool は変わらず、
  Adam 状態 (`exp_avg` / `exp_avg_sq`) は除外され、出力の再読み込み
  検証も通った。`py_compile` も全スクリプトで通った。
- `hf_download.py --list` もこの VM から実際に動いた:
  `nvidia/GEM-X: gated=False, 19 files, 9.81 GiB total`、
  `nvidia/EgoHand-v1: gated=False, 7 files, 9.93 GiB total`。
- **注意 — 前回レポートの数字との食い違い:** 前回の調査レポートには
  GEM-X リポジトリの全容量を 39,712,614,103 B (約 37 GiB) と書いたが、
  今回の `--list` の合計は 9.81 GiB だった。ファイル単位のサイズ
  (body 3.30 GiB、vitpose.pth 3.16 GiB など) は前回と一致しているので、
  v1 のダウンロード計画に影響はないが、前回の「37 GiB」という合計値は
  信頼しないこと。おそらく前回は API の別フィールドを合計していた。
  正しいのは今回のファイル単位の合計である。
- 本物の重みはこの VM ではダウンロードしていない (GPU なし、/tmp 512 MB)。
- `export_onnx_fp16.py` は未検証のテンプレート。依存関係がない環境では
  不足パッケージを列挙して終了コード 2 で止まることを確認した。

---

## 6. 未検証のこと (正直に書く)

- **本物の重みのダウンロード・FP16 変換・ONNX エクスポート・
  TensorRT エンジンビルドは実行していない。** この VM には GPU がなく、
  RAM も 7.7 GB しかないため。Windows PC 側でやること。
- **`.pt` / `.ckpt` 経路 (torch.load) はこの VM では未実行。**
  torch が入っていないため。safetensors 経路 (numpy) はテスト済み。
  Windows PC で torch を入れてから `.pt` 経路も試すこと。
- **RoPE / Attention / `grid_sample` / `einsum` の対応は、
  TensorRT-RTX 1.7.1 の API 上の存在確認まで。** 実際の ONNX グラフでの
  パース結果は未確認。`tensorrt_rtx` / `trtexec` でのパースが必須。
- **推論速度 (Body 25〜40ms 等) は未実測。** RTX 5070 Ti 級の参考値。
- **1.4 の並列コンテキストのクラッシュが 1.7.1.116 で本当に直っているかは、
  この VM では確認できない。** 1.6 の修正記録を信じるしかないが、
  推奨設計は並列を使わないので、そもそも踏まない。
- **ドライバ 32.0.16.1692 が CiG の 555+ を満たすこと**は、バージョンが
  2026/09/04 のものであることと親エージェントからの情報による。
  Windows PC 側で `nvidia-smi` を叩いて 5xx/6xx 番号で直接確認すること。

---

## 7. 明日 (Windows PC) のチェックリスト

```powershell
# 0. 環境確認
nvidia-smi            # ドライバ番号と VRAM 空きを確認 (1.3/16GB が目安)
python --version      # 3.10 以上
# RAM 128GB あれば変換は余裕。ディスクは最低 30GB、余裕を持って 50GB 空ける

# 1. ツールを入れる
python -m pip install -r tools\quantize\requirements.txt
python -m pip install nvidia-modelopt

# 2. 落とす (ログインなしで試す。401/403 なら hf auth login)
python tools\quantize\hf_download.py --repo nvidia/GEM-X --file body_s240000.ckpt --out-dir weights
python tools\quantize\hf_download.py --repo nvidia/EgoHand-v1 --file egohand_s060000.ckpt --out-dir weights

# 3. 中身を確認する (EgoHand の 10.66GB の正体)
python tools\quantize\inspect_ckpt.py weights\egohand_s060000.ckpt --top 30
python tools\quantize\inspect_ckpt.py weights\body_s240000.ckpt --top 30

# 4. まず --dry-run で確認してから FP16 化する
python tools\quantize\quantize_ckpt_to_fp16.py --input weights\body_s240000.ckpt --dry-run
python tools\quantize\quantize_ckpt_to_fp16.py --input weights\body_s240000.ckpt --output weights\body_fp16.safetensors
python tools\quantize\quantize_ckpt_to_fp16.py --input weights\egohand_s060000.ckpt --output weights\egohand_fp16.safetensors

# 5. ONNX に出す (export_onnx_fp16.py の TODO を GEM-X ソースに対して埋めてから)
#    FP32 ONNX → ModelOpt AutoCast → tensorrt_rtx でパース確認 → SCG 48K でエンジンビルド
tensorrt_rtx.exe --onnx=body_fp16.onnx --saveEngine=body_fp16.trt --memPoolSize=tacticSharedMem:48K
tensorrt_rtx.exe --onnx=hand_fp16.onnx --saveEngine=hand_fp16.trt --memPoolSize=tacticSharedMem:48K

# 6. C++ DLL に両エンジンを常駐させ、単一ストリームで Body → Hand を直列 enqueue する
#    (切り替えはコンテキストを変えるだけ。アンロードは不要)
```

---

## 出典

- How TensorRT-RTX Works (Threading / Memory / Weight Streaming / Runtime Cache):
  https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/architecture/how-trt-rtx-works.html
- Simultaneous Compute and Graphics (SCG):
  https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/compute-graphics.html
- CPU-Only AOT and TensorRT-RTX Engines (Weightless / Refit / IWeightsManager / Deferred Weights Loading):
  https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/cpu-engines.html
- Working with Runtime Cache:
  https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-runtime-cache.html
- TensorRT-RTX 1.4 Release Notes (Fixed Issues / Known Issues — 並列コンテキストのクラッシュ):
  https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.4.html
- TensorRT-RTX 1.6 Release Notes (Windows race condition fix for concurrent inference across multiple engines):
  https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.6.html
- IExecutionContext (TensorRT-RTX Python API 1.7.1):
  https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/python-api/infer/Core/ExecutionContext.html
- Refitting an Engine (TensorRT 10.x):
  https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/refitting-engines.html
- NVlabs/GEM-X README:
  https://github.com/nvlabs/gem-x/blob/HEAD/README.md
- Hugging Face API (ログインなしで確認):
  https://huggingface.co/api/models/nvidia/GEM-X
- Hugging Face API:
  https://huggingface.co/api/models/nvidia/EgoHand-v1
- 前回の調査レポート:
  `~/workspace/gemx-vr-bridge-new/docs/research-gemx-fp16-and-multimodel-2026-11-17.md`
