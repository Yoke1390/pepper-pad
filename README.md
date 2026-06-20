# pepper-pad

Pepper が直径 50cm の球体を抱えて平面パスを辿るタスクで、**表現的ヌル空間制御 (expressive null-space control)** を実装し、PAD（Pleasure / Arousal / Dominance）で表情を変える実験。設計の詳細は [plan.md](plan.md) を参照。

## セットアップ（uv）

```bash
uv sync          # 依存をインストール（.venv を作成）
```

### macOS (Apple Silicon) での pybullet ビルド注意

pybullet 3.2.7 は macOS 用の wheel が無くソースビルドされるが、同梱 zlib の `zutil.h` が
現代 macOS で誤って `#define fdopen(fd,mode) NULL` を有効化し（classic Mac OS 用の
`TARGET_OS_MAC` ガードが常時成立するため）、システム `stdio.h` の `fdopen` 宣言を壊して
コンパイルが失敗する。以下の `CFLAGS`/`CXXFLAGS` を与えて回避する:

```bash
export CFLAGS="-D_DARWIN_C_SOURCE -Dfdopen=fdopen"
export CXXFLAGS="-D_DARWIN_C_SOURCE -Dfdopen=fdopen"
uv sync          # または uv pip install pybullet
```

> `-Dfdopen=fdopen` が偽の `#define fdopen NULL` を握りつぶし、`fdopen` を本来のシステム関数のまま残す。

### qibullet メッシュの展開（Python 3.10+ で必須）

Pepper の URDF/メッシュ（`meshes.zip`）は SoftBank ライセンス保護でパスワード暗号化されており、
復号は qibullet 同梱のコンパイル済みインストーラ（`.pyc`）が担う。だがその installer は
**Python 3.9 までしか同梱されておらず**、3.10/3.11 では "Uncompatible version of Python 3" となり
`~/.qibullet/1.4.3/` にメッシュが展開されない。展開だけ Python 3.9 で 1 回行えばよい（展開物は
Python 非依存で、以降は 3.11 本体から利用できる）:

```bash
# meshes.zip を ~/.qibullet/1.4.3 に展開（--agree-license 相当で実行）
uv run --no-project --python 3.9 --with pybullet --with qibullet \
  python -c "from qibullet import tools; tools._install_resources(agreement=True)"
```

> `agreement=True` は SoftBank メッシュライセンスへの同意を表す。完了後 `~/.qibullet/1.4.3/` に
> `pepper.urdf` と `meshes/pepper/` などが生成される。

スモークテスト: `uv run python scripts/smoke_test.py`

## 参照リソース（qibullet）

`resources/qibullet` は参照用クローンで **git 管理外**（`.gitignore` 済み, plan.md §6）。
依存は uv で解決する。手元に無い場合は取得:

```bash
git clone https://github.com/softbankrobotics-research/qibullet resources/qibullet
```

## 実行

```bash
uv run python scripts/run_demo.py --pad 0 0 0   --record outputs/pad_000.mp4
uv run python scripts/run_demo.py --pad 1 1 1   --record outputs/pad_111.mp4
# ... PAD=[-1,-1,-1], [-1,1,-1], [-1,1,1]
```
