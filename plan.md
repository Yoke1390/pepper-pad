# 実装ワークフロー: Pepper による「球体運搬」表現的ヌル空間制御

> 課題 (`assignment.md`): 冗長性 2DoF 以上のロボット＋タスクを選び、**表現的ヌル空間コントローラ (expressive null-space controller)** を実装し、PAD を変えた結果を動画で示す。

---

## 1. ゴールと課題への対応

| 課題で要求される動画内の説明項目 | 本計画での回答 |
|---|---|
| どのロボットか | **SoftBank Pepper**（qibullet / PyBullet 上でシミュレーション） |
| どのタスクか | **直径 25cm の球体を胸の前で両手で抱えたまま、平面上のパスを辿る** |
| なぜ 2DoF 以上の冗長性があるか | 後述（§3）。両腕＋胴体 13DoF に対し主タスクは 6 次元 → **冗長 7DoF** |
| baseline `PAD=[0,0,0]` | ヌル空間の二次目標をゼロにし、球を保持して中立姿勢でパスを辿る |
| `PAD=[1,1,1]` / `[-1,-1,-1]` / `[-1,1,-1]` / `[-1,1,1]` | PAD→ヌル空間目標マッピング（§4）で 4 種の表情的挙動を生成 |

最終成果物は **1 本の動画**（または PAD ごとのクリップを連結したもの）で、上記 6 項目をすべて満たす。

---

## 2. ロボットとシミュレーション基盤

- **qibullet**（`resources/qibullet` に参照用クローンあり）が Pepper の URDF・メッシュ・高レベル API（`setAngles`, `getAnglesPosition`, `getLinkPosition`, `goToPosture`, `moveTo`）を提供する。
- 表現的ヌル空間制御に必要な **ヤコビアンと擬似逆行列** は qibullet には無いので、`PepperVirtual.getRobotModel()` と `getPhysicsClientId()` で生の PyBullet ハンドルを取得し、`pybullet.calculateJacobian()` を直接呼んで実装する。
- エンドエフェクタ link は URDF 上の **`l_hand` / `r_hand`**。腕関節は
  `LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw`（右も同様）の各 5DoF。
- 球体は PyBullet のプリミティブ（半径 0.125m の sphere）で生成し、両手の中点に **固定拘束 (`createConstraint`)** で取り付けるか、毎ステップ両手中点へキネマティックに配置する（動力学の落下を避ける）。

---

## 3. タスク定義と冗長性の根拠

### タスクの2層分解（重要）
「球を運ぶ」を **2 つの独立したタスク**に分けて扱う:

| 層 | 座標系 | 担当 | 役割 |
|---|---|---|---|
| **移動タスク** | 世界座標 | 移動台車 (omni base) `moveTo` | 平面パスを辿る |
| **手先固定タスク** | **台車 (base) 座標** | 両腕＋胴体 (13DoF) のヌル空間制御 | 球を台車に対し定位置で抱え続ける |

→ **課題＝「球体と台車の相対位置の固定」。PAD による表現はこの手先固定タスクのヌル空間を使う。**
台車が世界座標でどこへ動こうと、台車座標から見た球の位置は変わらない（球は台車に追従）。これにより移動と上半身の表現は分離される。

### 主タスク（手先固定タスク, primary task）
**台車 (base) 座標系**で、**左右の手先位置**を制御対象とする:

```
x = [ p_Lhand(3) , p_Rhand(3) ] ∈ R^6   （台車座標系で計測）
```

- 「球と台車の相対位置を固定」＝両手先を台車基準の保持位置（胸前、左右に半径ぶん離す ≒ 23cm 間隔）に保つことが主タスク。
- 手先の**姿勢（向き）は拘束しない**ことで、ヌル空間を広く取り表現の自由度を確保する（球は手のひら位置で支える前提）。
- **計測フレームを「台車 (base)」にする**のが今回の肝心点。腕より下の胴体関節 (HipRoll/HipPitch/KneePitch) も手先-対-台車を動かすため、**胴体も制御チェーンに入る** → 前傾などの胴体表現を「腕が補償して球を台車基準の定位置に保つ」全身ヌル空間運動として実現できる。

### 制御対象関節 q
台車座標で手先を固定するので、**腕より下の胴体関節も含める**:
- 両腕 10 関節 (`L/R ShoulderPitch, ShoulderRoll, ElbowYaw, ElbowRoll, WristYaw`)
- 胴体 3 関節 (`HipRoll, HipPitch, KneePitch`)
- → `q ∈ R^13`
- 頭部 (`HeadYaw, HeadPitch`) は手先に影響しないので task から外し、別系統で直接駆動（§4d）。

### 冗長性 DoF
```
冗長 DoF = dim(q) - rank(J) = 13 - 6 = 7 DoF   ← 課題要件「2DoF 以上」を十分満たす ✓
```
- 腕だけでも 6 次元タスクを張れる（generically rank(J)=6）。胴体 3DoF は丸ごと冗長で、傾けても腕が補償して球を台車基準の定位置に保てる。
- → 主タスク（球と台車の相対位置）を一切乱さずに、肘・前腕・胴体姿勢を変えられる。これがヌル空間表現の土台。

---

## 4. 表現的ヌル空間コントローラ

### 速度レベルの冗長性解決
```
q̇ = J⁺ ẋ_d  +  (I − J⁺J) q̇₀
```
- `J = ∂x/∂q ∈ R^{6×n}`: 左右手先位置の積み上げヤコビアン（`calculateJacobian` の並進成分から、各腕に対応する列を抽出して構成）。
- `J⁺`: **減衰最小二乗 (DLS)** 擬似逆 `Jᵀ(JJᵀ + λ²I)⁻¹` で特異点近傍を安定化。
- 主タスク指令: `ẋ_d = K_task (x_ref − x_cur)`（胴体座標系で保持姿勢 `x_ref` 一定なので、フィードバックで球を抱え続ける）。
- `N = I − J⁺J`: ヌル空間射影。`q̇₀` をここに通すことで**主タスクを乱さず**二次目標を実現する。

### PAD → 二次目標 `q̇₀` のマッピング
PAD = Pleasure（快）, Arousal（覚醒/活性）, Dominance（支配性）, 各 ∈ [−1, 1]。

```
q̇₀ = −k_post · W (q − q*_PAD)        # 目標姿勢へ引き込む（静的成分）
       + amp(A) · ω(A) · sin(ω(A)·t) · d_PAD   # 律動的成分（活性で増幅）
```

**(a) 目標姿勢 `q*_PAD`**（中立姿勢 + 各感情軸のオフセット線形和）:
```
q*_PAD = q_neutral + P·δ_P + A·δ_A + D·δ_D
```
| 軸 | 解釈 | 主な関節オフセット δ の設計方針 |
|---|---|---|
| **P** Pleasure | 開き・上向き vs 閉じ・下向き | +P: 肩を開き腕・球を高く、頭やや上 / −P: 体を縮め球を下げ頭を下げる |
| **A** Arousal | 運動の振幅・周波数 | 主に律動成分を増幅。+A: 速く大きく / −A: ほぼ静止 |
| **D** Dominance | 体を大きく・高く vs 小さく | +D: 肘を外へ張り胸を開き頭を上げ膝伸展（大きく見せる）/ −D: 肘を閉じ頭を下げ縮こまる |

**(b) 律動成分**: `ω(A) = ω₀(1 + A)`, `amp(A) = amp₀·(0.2 + 0.8·(A+1)/2)` で、覚醒が高いほど速く大きく揺れる。方向ベクトル `d_PAD` は肩・肘中心に設定。

**(c) 胴体の前傾（1パラメータ表現チャンネル）**:
感情表現として胴体の向きも使う。台車座標タスクでは **HipPitch は手先-対-台車を変える＝タスクに入る**ので、前傾は「胴体を傾けつつ腕が補償して球を台車基準の定位置に保つ」**全身ヌル空間運動**として実現する（球はぶれず、姿勢だけが変わる）。スカラー1個で指示:
```
θ_lean ∈ [−1, +1]   （+: 直立〜やや後傾,  −: 前傾・うつむき）
q*_PAD の HipPitch 成分 ← clamp( θ_lean → 可動域 )   （任意で KneePitch も協調）
θ_lean = w_D · D + w_P · P                          （支配性・快が低いほど前傾＝うなだれ）
```
- `θ_lean` は目標姿勢 `q*_PAD` の胴体成分として与え、**ヌル空間射影 `N` を通す**。腕は自動で補償し、球は台車に対し定位置のまま。高 D/P＝胸を張って直立、低 D/P＝前傾して縮こまる。
- 前傾量は腕のリーチで制限される（大きく傾けるほど腕を伸ばして補償するため）→ 可動域＋到達可能性でクランプ。
- これで「肘・前腕＋胴体（腕のヌル空間で補償）＋頭部（直接）」が PAD の表現面となる。

**(d) 表現チャンネルの内訳**（PAD が動かせる自由度＝タスクを乱さないもの）:
| チャンネル | 関節 | タスクとの関係 |
|---|---|---|
| 腕（肘・前腕） | ShoulderRoll, ElbowRoll/Yaw, WristYaw | 台車座標タスクの**ヌル空間**。`N` で射影 |
| 胴体前傾 | HipPitch, HipRoll, KneePitch | タスクに入るが冗長 → **ヌル空間**で実現（腕が補償）。`N` で射影 |
| 頭部 | HeadYaw, HeadPitch | 手先に無関係＝**タスク不変**。直接駆動 |

→ 腕と胴体は同じヌル空間射影 `N` を通り、頭部だけが直接駆動。いずれも球と台車の相対位置を乱さない。

**(e) 安全**: `q*_PAD` と最終指令を**関節可動域でクランプ**し、`isSelfColliding` で自己干渉を監視（極端な PAD での破綻防止）。

### 要求 PAD ベクトルで期待される挙動
| PAD | 期待される表情・挙動 |
|---|---|
| `[0,0,0]` baseline | 中立姿勢で球を抱え、淡々とパスを辿る（揺れなし） |
| `[1,1,1]` | 開いた・高い・速い・活発 → 喜び/興奮/自信 |
| `[-1,-1,-1]` | 縮こまり・低く・遅い/静止・うなだれる → 悲しみ/疲労/服従 |
| `[-1,1,-1]` | 縮こまり＋うなだれ つつ 高エネルギー → 不安/動揺/怯え |
| `[-1,1,1]` | やや閉じつつ 支配的＋高エネルギー → 怒り/威圧/緊張 |

---

## 5. プロジェクト構成と環境（uv 管理）

```
pepper-pad/
├── plan.md                  # 本ファイル
├── pyproject.toml           # uv プロジェクト定義
├── uv.lock
├── README.md                # セットアップ手順（qibullet/resources 取得を含む）
├── .gitignore               # resources/ と outputs/ を無視
├── src/pepper_pad/
│   ├── __init__.py
│   ├── sim.py               # Pepper・地面・25cm 球のロード／シーン構築
│   ├── kinematics.py        # calculateJacobian ラッパ・FK・手先ヤコビアン構成
│   ├── controller.py        # ヌル空間 resolved-rate コントローラ（DLS）
│   ├── pad.py               # PAD → q*_PAD・律動パラメータのマッピング
│   ├── path.py              # 平面パス生成＋台車追従
│   └── record.py            # 動画録画ユーティリティ
├── scripts/
│   └── run_demo.py          # CLI: `--pad 1 1 1 --record outputs/pad_111.mp4`
├── outputs/                 # 生成動画（gitignore）
└── resources/qibullet/      # 参照用クローン（gitignore、§7 参照）
```

### uv セットアップ
```bash
uv init --python 3.11           # pybullet の wheel が揃うバージョンを指定
uv add pybullet numpy           # コア依存
uv add "imageio[ffmpeg]"        # フレーム→mp4（または opencv-python）
# qibullet 本体（下記いずれか）:
uv add qibullet                 # ① PyPI 版をピン留め（推奨・シンプル）
# uv pip install -e resources/qibullet   # ② ローカル参照を editable 導入（PyPI が不調な場合）
```
- 初回ロボットロード時に qibullet が**メッシュの展開／EULA 同意**を要求することがある点に注意（`resources/qibullet/.../meshes.zip` が同梱済み）。
- すべて `uv run python scripts/run_demo.py ...` で実行し、環境を再現可能に保つ。

---

## 6. `resources/` の git での扱い

`resources/qibullet` は **独立した git クローン**（内部に `.git`、docs 用の `.gitmodules` サブモジュール、大きな `meshes.zip` を含む）。本リポジトリにそのまま `git add` すると、入れ子 git のため壊れる／巨大化する。

**推奨方針: 参照物として gitignore し、依存は uv で解決する。**
- `.gitignore` に `resources/` と `outputs/` を追加（参照クローンと生成物はバージョン管理しない）。
- qibullet は §5 の通り **uv の依存パッケージ**として宣言（`pyproject.toml`/`uv.lock` で再現性確保）。ソースのベンダリング不要。
- `README.md` に「参照元の取得方法」を記載（`git clone https://github.com/softbankrobotics-research/qibullet resources/qibullet` 等）。

**代替案**（必要なら）:
- *git submodule*: 上流 qibullet を特定コミットにピン留めしたい場合。`git submodule add <url> resources/qibullet`。ただし台車・メッシュの再現は uv 依存で足りるため通常は不要。
- *vendoring*: 内部 `.git` を消して取り込む案は、`meshes.zip` でリポジトリが肥大化するため**非推奨**。

> まず推奨方針（gitignore + uv 依存）で進める。実装中に上流を改変する必要が出たら submodule へ切替を検討。

---

## 7. 実装マイルストーン

- [x] **M0 環境構築**（完了）: uv プロジェクト作成（Python 3.11）、依存導入（pybullet/numpy/imageio/qibullet）、`.gitignore`/README 整備、qibullet メッシュ展開、DIRECT モードでスモークテスト成功（`scripts/smoke_test.py`）。
  - macOS の pybullet ビルドは `CFLAGS="-D_DARWIN_C_SOURCE -Dfdopen=fdopen"` で回避（README 参照）。
  - メッシュ展開は Python 3.9 インストーラで実施（3.10+ は installer 非同梱、README 参照）。
  - スモーク結果: 既定姿勢で手先間隔 ≈ 0.33m（`l_hand`/`r_hand`）。**直径 0.25m の球を抱えるには M1 で間隔 ≈ 0.23m の保持姿勢に調整が必要**。
- [x] **M1 シーン構築**（完了, `sim.py`）: `PepperScene` で Pepper＋地面＋直径 25cm 球を生成。保持姿勢を**タスク優先 DLS IK**で較正（`scripts/calibrate_hold.py`）— 主タスク=両手を球の下側へ(位置)、副タスク=掌を球へ向ける(ヌル空間)、最後に位置ポリッシュ。手は開き (`HAND_OPENING=1.0`) **球を少し下から支える**。球は**台車座標の固定点 `HOLD_SPHERE_CENTER` に配置**（両手中点ではない＝「球と台車の相対位置を固定」の主タスク定義と一致、台車が動けば球も追従）。`scripts/build_scene.py` で 正面・俯瞰・真上・真横 をレンダリング（各描画時に関節角・手先・球位置を print）。
  - 重要修正: qibullet の `launchSimulation` 既定 `auto_step=True` は背景スレッドでステップし続け、保持姿勢をモータ目標(≈0)へ引き戻す。`auto_step=False`＋モータ目標設定＋`PepperScene.step()` で解決（M3 の制御ループにも必須）。
- [ ] **M2 運動学** (`kinematics.py`): 一部先取り済み — `calculateJacobian` ラッパ (`hand_jacobian`: 並進＋角速度) と、腕限定のタスク優先 DLS IK (`solve_arm_ik`: 位置主タスク＋掌向きヌル空間＋位置ポリッシュ) を実装。**残**: 左右手先の並進ヤコビアンを積み上げ `J∈R^{6×n}` を構成し、有限差分で数値検証。
- [ ] **M3 コントローラ核** (`controller.py`): `q̇ = J⁺ẋ_d + N q̇₀`（DLS）を実装。`q̇₀=0` で球を静的保持＝**baseline 動作**を確認。ランダム `q̇₀` を入れて「手先は不動・姿勢だけ変化」＝冗長性を可視化。
- [ ] **M4 PAD マッピング** (`pad.py`): `δ_P, δ_A, δ_D` と律動パラメータを実装・調整。要求 5 PAD で**見分けのつく**、かつ球を落とさない挙動を確認。
- [ ] **M5 パス追従** (`path.py`): 平面パス（例: 直線→旋回、S 字、8 の字）を `moveTo` で台車追従。腕のヌル空間表現と統合。
- [ ] **M6 録画** (`record.py`): PAD ごとに MP4 出力。画面に「ロボット名／タスク／PAD 値」ラベルを焼き込み。
- [ ] **M7 最終動画**: baseline と 4 つの PAD クリップを連結し、課題の説明 6 項目をすべて満たす 1 本に編集。

---

## 8. 動画制作（成果物）

- PyBullet の `startStateLogging(STATE_LOGGING_VIDEO_MP4, ...)`（要 ffmpeg）か、`getCameraImage` でフレーム取得 → `imageio` で mp4 化。
- 各 PAD で同一パス・同一カメラアングルにして比較しやすくする。
- `scripts/run_demo.py --pad <P> <A> <D> --record outputs/pad_xxx.mp4` を 5 通り実行 → 連結。
- 冒頭にテキストで「Robot: Pepper / Task: carry a 0.25m sphere along a planar path (sphere fixed relative to the base) / Redundancy: 13 joints − 6D task = 7 DoF」を表示。

---

## 9. リスクと留意点

- **qibullet 初回ロード**: メッシュ展開・EULA 同意が走る可能性。M0 で潰す。
- **Python/pybullet 互換**: pybullet の wheel があるバージョン（3.10/3.11 目安）を uv で固定。
- **ヤコビアンのフレーム/関節対応**: `calculateJacobian` は全可動関節分を返すので、**腕関節インデックスのマッピング**を正確に取る（M2 の数値検証で担保）。
- **両手保持の閉ループ性**: 厳密な閉運動学ループは解かず、「手先位置を主タスク化＋球はキネマティック拘束」で回避する。
- **台車運動と腕フレーム**: タスクを胴体座標系で定義し、台車並進の影響を分離。
- **極端な PAD での自己干渉**: 可動域クランプ＋`isSelfColliding` 監視。

---

## 10. 最初の一歩

1. `uv init --python 3.11` → `uv add pybullet numpy "imageio[ffmpeg]" qibullet`
2. `.gitignore` に `resources/` `outputs/` `__pycache__/` `.venv/` を追加
3. `sim.py` で Pepper＋球を表示するスモークテスト（M0–M1）
4. 以降、M2 → M7 を順に進める
