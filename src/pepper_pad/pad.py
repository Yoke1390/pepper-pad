"""M4: PAD → ヌル空間二次目標 q̇₀ のマッピング (plan.md §4)。

PAD = Pleasure(快), Arousal(覚醒), Dominance(支配性) 各 ∈ [−1, 1]。

    q̇₀ = −k_post · (q − q*_PAD)              … 目標姿勢への引き込み（静的）
           + amp(A)·ω(A)·cos(ω(A)·t) · d_dir   … 律動成分（覚醒で増幅）
    q*_PAD = q_neutral + P·δ_P + A·δ_A + D·δ_D

この q̇₀ を NullSpaceController がヌル空間 N=I−J⁺J へ射影するので、球の把持・
正面・高さ（拘束5次元）を乱さずに姿勢だけが表情を帯びる。頭部 (HeadYaw/HeadPitch)
は手先に無関係なので別系統で直接駆動する。

符号は実機 (qibullet Pepper) で確認:
  HipPitch + = 前傾（うつむき）, − = 直立〜やや後傾。  HeadPitch + = 下向き。
  ElbowRoll |大| = 肘を外へ張る。 ShoulderRoll |大| = 上腕を左右へ開く。
"""
from __future__ import annotations

import numpy as np

from .sim import ARM_JOINTS, TORSO_JOINTS

# 制御対象 13 関節（controller と同順）
CTRL_JOINTS = ARM_JOINTS + TORSO_JOINTS


def _vec(d: dict[str, float]) -> np.ndarray:
    """関節名→値の dict を 13 次元ベクトルに展開。"""
    v = np.zeros(len(CTRL_JOINTS))
    for k, val in d.items():
        v[CTRL_JOINTS.index(k)] = val
    return v


# --- 姿勢オフセット方向（軸値 +1 あたり, rad）---
# 符号メモ: ElbowRoll は負で曲がる。+方向（0へ）= 腕を伸ばして球を前へ差し出す、
# −方向 = 肘を曲げて球を抱え込む。縮こまり側(−P/−D)で「曲げてタック」させると、
# 前傾しても腕の保持リーチが残り m_z(高さ) を維持しやすい（深い前傾での球の沈み防止）。
# Pleasure: 開いて伸び上がる(+) ↔ 縮んでうなだれる(−)
DELTA_P = _vec({
    "HipPitch": -0.22,                      # +P 直立, −P 前傾
    "LShoulderRoll": +0.10, "RShoulderRoll": -0.10,   # +P やや開く
    "LElbowRoll": +0.12, "RElbowRoll": -0.12,         # +P 腕を差し出す, −P 抱え込む
})
# Arousal: 静的姿勢へはわずかに「身構える」程度（主効果は律動成分）
DELTA_A = _vec({
    "HipPitch": -0.08,                      # +A わずかに体を起こす
})
# Dominance: 大きく高く見せる(+) ↔ 小さく縮こまる(−)
DELTA_D = _vec({
    "HipPitch": -0.25,                      # +D 直立/やや反る, −D 前傾で縮こまる
    "KneePitch": -0.15,                     # +D 膝を伸ばし背を高く
    "LShoulderRoll": +0.30, "RShoulderRoll": -0.30,   # +D 上腕を左右へ張る
    "LElbowRoll": +0.45, "RElbowRoll": -0.45,         # +D 腕を張り出す, −D 抱え込み縮む
})

# 律動成分の方向（覚醒で振れる全身のリズム）。HipRoll で左右に揺れ（m_y 拘束を腕が
# 補償して球は正面のまま）、肩で上下にバウンス。正規化しておく。
RHYTHM_DIR = _vec({
    "HipRoll": 1.0,
    "LShoulderPitch": 0.4, "RShoulderPitch": 0.4,
    "LElbowRoll": 0.3, "RElbowRoll": -0.3,
})
RHYTHM_DIR = RHYTHM_DIR / np.linalg.norm(RHYTHM_DIR)

# 頭部（直接駆動）。+P/+D で上を向き、−で下を向く。HeadPitch + = 下向き。
HEAD_P = {"HeadPitch": -0.20}
HEAD_D = {"HeadPitch": -0.25}
# 覚醒時の小刻みな頭の動き（うなずき/きょろつき）振幅
HEAD_NOD_A = 0.10


def clamp_to_limits(scene, joint_names: list[str], q: np.ndarray) -> np.ndarray:
    import pybullet as p
    out = q.copy()
    for i, nm in enumerate(joint_names):
        g = scene.pepper.getJoint(nm).getIndex()
        info = p.getJointInfo(scene.model, g, physicsClientId=scene.client)
        out[i] = min(max(out[i], info[8]), info[9])
    return out


class PADExpression:
    """PAD ベクトルから二次目標 q̇₀ と頭部角を生成する表現器。"""

    def __init__(self, scene, q_neutral: np.ndarray, pad, *,
                 k_post: float = 2.5, w0: float = 2.0 * np.pi * 0.5,
                 amp0: float = 0.16, t_ramp: float = 0.8,
                 head_neutral: dict[str, float] | None = None):
        self.scene = scene
        self.P, self.A, self.D = (float(pad[0]), float(pad[1]), float(pad[2]))
        self.k_post = k_post
        self.w0 = w0
        self.amp0 = amp0
        self.t_ramp = t_ramp          # 目標姿勢へのイーズイン時間
        self.q_neutral = np.asarray(q_neutral, dtype=float)

        # 目標姿勢 q*_PAD（可動域でクランプ）
        q_star = (self.q_neutral + self.P * DELTA_P + self.A * DELTA_A
                  + self.D * DELTA_D)
        self.q_star = clamp_to_limits(scene, CTRL_JOINTS, q_star)

        # 律動パラメータ（覚醒で速く大きく。A≤0 は静止＝baseline も揺れなし）
        self.amp = self.amp0 * max(self.A, 0.0)
        self.w = self.w0 * (1.0 + max(self.A, 0.0))

        self.head_neutral = head_neutral or {"HeadYaw": 0.0, "HeadPitch": 0.0}

    # --- 二次目標 ---
    def q_dot0(self, t: float, q: np.ndarray) -> np.ndarray:
        """時刻 t・現在姿勢 q における q̇₀（R^13）。

        目標姿勢は t_ramp 秒かけて q_neutral→q*_PAD にイーズインする。これにより
        初期に q*_PAD へ強く引かれて生じる過渡（腕が追従しきれず球がぶれる）を抑える。
        律動も同じランプで立ち上げる。"""
        r = min(t / self.t_ramp, 1.0) if self.t_ramp > 0 else 1.0
        q_star_eff = self.q_neutral + r * (self.q_star - self.q_neutral)
        static = -self.k_post * (np.asarray(q, dtype=float) - q_star_eff)
        rhythm = r * self.amp * self.w * np.cos(self.w * t) * RHYTHM_DIR
        return static + rhythm

    # --- 頭部（直接駆動）---
    def head_angles(self, t: float) -> dict[str, float]:
        h = dict(self.head_neutral)
        h["HeadPitch"] += self.P * HEAD_P["HeadPitch"] + self.D * HEAD_D["HeadPitch"]
        if self.amp > 0.0:                       # 覚醒時は小さくうなずく
            h["HeadPitch"] += HEAD_NOD_A * max(self.A, 0.0) * np.sin(self.w * t)
        return h

    def apply_head(self, t: float) -> None:
        self.scene.set_posture_instant(self.head_angles(t))
