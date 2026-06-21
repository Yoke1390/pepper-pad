# pepper-pad — Pepper の表現的ヌル空間制御

SoftBank **Pepper** が直径 25cm の球体を胸の前で両手に抱えたまま平面上のパスを辿るタスクに対し、
**表現的ヌル空間制御 (expressive null-space control)** を実装し、**PAD**（Pleasure / Arousal / Dominance）
を変えることでロボットの「表情」（姿勢・動きのニュアンス）を変化させる実験プロジェクトです。

シミュレーションは [qibullet](https://github.com/softbankrobotics-research/qibullet) / [PyBullet](https://pybullet.org)
上で動作します。設計の理論的背景・式の導出・設計判断の根拠は [plan.md](plan.md) に詳述しています。
このREADMEは **環境構築と実行手順** に焦点を当てます。

---

## 1. このプロジェクトが解くこと

授業課題 ([assignment.md](assignment.md)) は「冗長性 2DoF 以上のロボット＋タスクを選び、表現的ヌル空間
コントローラを実装し、PAD を変えた結果を動画で示す」ことを求めています。本プロジェクトの対応は次の通りです。

| 課題が動画で求める説明項目 | 本プロジェクトの回答 |
|---|---|
| どのロボットか | **SoftBank Pepper**（qibullet / PyBullet 上でシミュレート） |
| どのタスクか | **直径 25cm の球を胸の前で両手に抱えたまま平面パスを辿る** |
| なぜ 2DoF 以上の冗長性があるか | 制御関節 **13DoF**（両腕10＋胴体3）に対し主タスクは **5 次元** → **冗長 8DoF** |
| baseline `PAD=[0,0,0]` | ヌル空間の二次目標を 0 にし、中立姿勢で球を保持してパスを辿る |
| `PAD=[1,1,1]`,`[-1,-1,-1]`,`[-1,1,-1]`,`[-1,1,1]` | PAD → ヌル空間目標マッピングで 4 種の表情的挙動を生成 |

### 核となる考え方（要約）

- **主タスク（球保持）**：台車(base)座標で `x = [ d(3), m_y, m_z ] ∈ R⁵`。
  `d = p_Lhand − p_Rhand`（把持維持）、`m = (p_Lhand + p_Rhand)/2` の左右中心 `m_y` と高さ `m_z` を拘束し、
  **前後 `m_x` は自由**にする緩和タスク。
- **冗長性解決**：`q̇ = J⁺ ẋ_d + (I − J⁺J) q̇₀`、`J⁺` は減衰最小二乗 (DLS) 擬似逆 `Jᵀ(JJᵀ + λ²I)⁻¹`。
- **PAD → ヌル空間二次目標 `q̇₀`**：目標姿勢への引き込み（静的成分）＋覚醒で増幅される律動成分。主タスクを
  乱さずに肘・前腕・胴体・頭部だけを表情として動かす。

詳しい式と設計は [plan.md](plan.md) を参照してください。

---

## 2. 必要環境

- **Python 3.11 以上**（`.python-version` で固定。`requires-python = ">=3.11"`）
- [**uv**](https://docs.astral.sh/uv/)（依存解決・仮想環境管理）
- macOS (Apple Silicon) で検証。Linux でも動作する想定。
- 主な依存：`pybullet`, `qibullet`, `numpy`, `imageio[ffmpeg]`, `tqdm`（詳細は [pyproject.toml](pyproject.toml)）。

---

## 3. セットアップ

### 3.1 依存のインストール

```bash
uv sync          # 依存をインストールし .venv を作成
```

> 多くの環境ではこれだけで完了します。macOS (Apple Silicon) では下記 2 点の追加対応が必要になる場合があります。

### 3.2 macOS (Apple Silicon) での pybullet ビルド対応

`pybullet` は macOS 用の wheel が無くソースビルドされますが、同梱 zlib の `zutil.h` が現代 macOS で
誤って `#define fdopen(fd,mode) NULL` を有効化し（classic Mac OS 用の `TARGET_OS_MAC` ガードが
常時成立するため）、システム `stdio.h` の `fdopen` 宣言を壊してコンパイルが失敗します。
次の `CFLAGS` / `CXXFLAGS` を与えて回避してください。

```bash
export CFLAGS="-D_DARWIN_C_SOURCE -Dfdopen=fdopen"
export CXXFLAGS="-D_DARWIN_C_SOURCE -Dfdopen=fdopen"
uv sync          # または: uv pip install pybullet
```

> `-Dfdopen=fdopen` が偽の `#define fdopen NULL` を握りつぶし、`fdopen` を本来のシステム関数のまま
> 残します。`-D_DARWIN_C_SOURCE` は POSIX 拡張宣言を有効化します。

### 3.3 qibullet メッシュの展開（Python 3.10+ で必須）

Pepper の URDF/メッシュ（`meshes.zip`）は SoftBank ライセンス保護のためパスワード暗号化されており、
復号は qibullet 同梱のコンパイル済みインストーラ（`.pyc`）が担います。ところがこの installer は
**Python 3.9 までしか同梱されておらず**、3.10/3.11 では `Uncompatible version of Python 3` となって
`~/.qibullet/1.4.3/` にメッシュが展開されません。

**展開だけ Python 3.9 で 1 回**実行すれば十分です（展開物は Python 非依存で、以降は 3.11 本体から
利用できます）。

```bash
# meshes.zip を ~/.qibullet/1.4.3 に展開（ライセンス同意のうえ実行）
uv run --no-project --python 3.9 --with pybullet --with qibullet \
  python -c "from qibullet import tools; tools._install_resources(agreement=True)"
```

> `agreement=True` は SoftBank メッシュライセンスへの同意を表します。完了後、`~/.qibullet/1.4.3/` に
> `pepper.urdf` や `meshes/pepper/` などが生成されます。

### 3.4 スモークテスト（環境確認）

Pepper が DIRECT モードでロードでき、手先リンク・腕関節が取得できることを確認します（ディスプレイ不要）。

```bash
uv run python scripts/smoke_test.py
```

---

## 4. プロジェクト構成

### ライブラリ `src/pepper_pad/`

| モジュール | 役割 |
|---|---|
| [`sim.py`](src/pepper_pad/sim.py) | シーン構築（Pepper＋地面＋球）。左右対称の保持姿勢を与え、球を両手中点へ配置する `PepperScene`。 |
| [`kinematics.py`](src/pepper_pad/kinematics.py) | `pybullet.calculateJacobian` を用いた手先ヤコビアン、緩和保持タスク `x=[d, m_y, m_z]` のヤコビアン `J∈R⁵ˣ¹³`、DLS-IK。 |
| [`controller.py`](src/pepper_pad/controller.py) | ヌル空間 resolved-rate コントローラ（DLS 擬似逆 + ヌル空間射影）。 |
| [`pad.py`](src/pepper_pad/pad.py) | PAD → ヌル空間二次目標 `q̇₀` のマッピング（目標姿勢の引き込み＋律動成分＋頭部直接駆動）。 |
| [`path.py`](src/pepper_pad/path.py) | 平面パス生成（直線 / S字 / 円弧）と台車追従、地面へのパス可視化。 |
| [`record.py`](src/pepper_pad/record.py) | PAD クリップの MP4 録画（PAD 値・感情ラベルの焼き込み、地面パス表示）。 |

### スクリプト `scripts/`（マイルストーン M0〜M6 に対応）

| スクリプト | 内容 |
|---|---|
| [`smoke_test.py`](scripts/smoke_test.py) | **M0** 環境確認（Pepper ロード・関節取得）。 |
| [`build_scene.py`](scripts/build_scene.py) | **M1** シーン組み立て確認。保持姿勢と球の取り付けを PNG で可視化。 |
| [`calibrate_hold.py`](scripts/calibrate_hold.py) | 保持姿勢の較正（IK で左手を球に合わせ右腕へ鏡像化）。 |
| [`verify_jacobian.py`](scripts/verify_jacobian.py) | **M2** 緩和タスクのヤコビアン `J=∂x/∂q` を中心差分で数値検証。 |
| [`demo_nullspace.py`](scripts/demo_nullspace.py) | **M3** ヌル空間制御の動作確認（静的保持／ヌル空間運動で姿勢が変わっても拘束は不変）。 |
| [`demo_pad.py`](scripts/demo_pad.py) | **M4** 5 種の PAD で表情的挙動を生成し代表フレームを描画。 |
| [`demo_path.py`](scripts/demo_path.py) | **M5** S字パス追従＋ヌル空間 PAD 表現（移動と表現の分離を確認）。 |
| [`record_demo.py`](scripts/record_demo.py) | **M6** 要求 5 PAD の MP4 クリップを録画。 |

---

## 5. 実行

### 5.1 各マイルストーンの動作確認（PNG 出力）

確認用スクリプトは `outputs/` に PNG / ログを出力します（ディスプレイ不要のヘッドレス描画）。

```bash
uv run python scripts/verify_jacobian.py   # M2: ヤコビアン検証（PASS と誤差を表示）
uv run python scripts/demo_nullspace.py    # M3: ヌル空間運動の前後比較を描画
uv run python scripts/demo_pad.py          # M4: 5 PAD の代表フレームを描画
uv run python scripts/demo_path.py         # M5: S字パス上の数フレームを描画
```

### 5.2 デモ動画の録画（M6）

要求 5 PAD（`[0,0,0]`, `[1,1,1]`, `[-1,-1,-1]`, `[-1,1,-1]`, `[-1,1,1]`）を同一パス・同一カメラで録画し、
PAD 値と感情ラベル（Joy など）を焼き込んだ MP4 を `outputs/` に出力します。

```bash
# 5 本を並列録画（multiprocessing、単一の進捗バー）
uv run python scripts/record_demo.py

# 単発 PAD を 1 本だけ録画（例: PAD=[1,1,1]）
uv run python scripts/record_demo.py 1 1 1
```

> **描画は CPU 律速**です。フレーム取得（`getCameraImage` の `ER_TINY_RENDERER`）が ~0.5s/frame と支配的で、
> Apple Silicon ではヘッドレス GPU 描画（EGL）が使えないため、5 本を **別プロセス並列**で描画して実時間を
> 短縮しています（画質・解像度・fps は据え置き）。詳細は [`record_demo.py`](scripts/record_demo.py) の
> 冒頭コメント参照。録画には数分かかります。

出力例：

```
outputs/pad_000_baseline.mp4   # Neutral
outputs/pad_111_joy.mp4        # Joy
outputs/pad_nnn_sad.mp4        # Sadness
outputs/pad_n1n_fear.mp4       # Fear
outputs/pad_n11_anger.mp4      # Anger
```

> `outputs/` は生成物のため **git 管理外**（`.gitignore` 済み）です。

---

## 6. 参照リソース（qibullet）

`resources/qibullet` は参照用クローンで **git 管理外**です（`.gitignore` 済み。依存自体は uv で解決します）。
ソースを手元で参照したい場合のみ取得してください。

```bash
git clone https://github.com/softbankrobotics-research/qibullet resources/qibullet
```

---

## 7. 関連ドキュメント

- [plan.md](plan.md) — 設計・理論・式の導出・マイルストーン (M0〜M7) の詳細。
- [assignment.md](assignment.md) — 元の課題文。
