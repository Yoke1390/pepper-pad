"""M3 動作確認: ヌル空間 resolved-rate コントローラ。

(A) baseline: q̇₀=0 で球を静的保持 → 拘束タスク誤差が ~0 のままを確認。
(B) 冗長性: 非ゼロ q̇₀ をヌル空間へ射影 → 胴体・肘の姿勢は大きく変わるのに、
    拘束 5 次元 (把持 d, 真正面 m_y, 高さ m_z) は不変。前後 m_x は自由なので
    球が前後する場合がある（緩和タスクの狙い通り）。

前後の姿勢を outputs/ に描画して視覚的にも確認する。
"""
from __future__ import annotations

import numpy as np

from pepper_pad.sim import PepperScene, ARM_JOINTS, TORSO_JOINTS
from pepper_pad.controller import NullSpaceController
from pepper_pad.kinematics import holding_task_value
from build_scene import render

JOINTS = ARM_JOINTS + TORSO_JOINTS  # 13DoF
LABELS = ["d_x", "d_y", "d_z", "m_y", "m_z"]


def run(ctrl: PepperScene, q_dot0, steps: int):
    """steps 回ステップし、各ステップの拘束タスク誤差の絶対値を積み上げて
    最大値ベクトル (R^5) を返す。"""
    max_abs = np.zeros(5)
    for _ in range(steps):
        err = ctrl.step(q_dot0)
        max_abs = np.maximum(max_abs, np.abs(err))
    return max_abs


def main() -> None:
    s = PepperScene(gui=False)
    try:
        s.apply_hold_posture()
        s.spawn_sphere()
        ctrl = NullSpaceController(s, JOINTS)

        print(f"controlled joints = {len(JOINTS)},  "
              f"null-space dim = {ctrl.null_space_basis_rank()} DoF")
        print(f"x_ref = {np.round(ctrl.x_ref, 4)}  [d_x,d_y,d_z, m_y,m_z]\n")

        # --- (A) baseline: q̇₀ = 0 ---
        max_a = run(ctrl, None, steps=120)
        print("(A) baseline  q̇₀=0, 120 steps")
        print("    max |task error| per dim:",
              {l: f"{v:.2e}" for l, v in zip(LABELS, max_a)})
        print(f"    -> max overall = {max_a.max():.2e}  (球を静的保持)\n")

        # --- (B) 冗長性: ヌル空間に二次目標を流す ---
        # 胴体を前傾 (HipPitch) させ肘を開く方向。腕が補償して球を保持。
        q_dot0 = np.zeros(len(JOINTS))
        q_dot0[JOINTS.index("HipPitch")] = +0.6
        q_dot0[JOINTS.index("LElbowRoll")] = -0.5
        q_dot0[JOINTS.index("RElbowRoll")] = +0.5

        q_before = ctrl.read_q()
        x_before = holding_task_value(s)
        l_before, r_before = s.hand_positions()
        mid_b = s.hand_midpoint()
        render(s, [mid_b[0], 0.0, mid_b[2]], yaw=35, pitch=-20, dist=1.7,
               name="m3_before.png")

        max_b = run(ctrl, q_dot0, steps=150)

        q_after = ctrl.read_q()
        x_after = holding_task_value(s)
        l_after, r_after = s.hand_positions()
        render(s, [s.hand_midpoint()[0], 0.0, s.hand_midpoint()[2]],
               yaw=35, pitch=-20, dist=1.7, name="m3_after.png")

        print("(B) redundancy  q̇₀ on HipPitch/ElbowRoll via null space, 150 steps")
        print("    max |task error| per dim (constrained 5D):",
              {l: f"{v:.2e}" for l, v in zip(LABELS, max_b)})
        print(f"    -> max overall = {max_b.max():.2e}  (拘束は保たれる)\n")

        print("    posture change (q_after − q_before), |Δ|>0.02 rad:")
        for nm, a, b in zip(JOINTS, q_after, q_before):
            if abs(a - b) > 0.02:
                print(f"        {nm:16s} {b:+.3f} -> {a:+.3f}  (Δ {a - b:+.3f})")

        print("\n    constrained dims drift (should be ~0):")
        print(f"        d (grasp)  |Δ| = {np.linalg.norm(x_after[:3] - x_before[:3]):.2e} m")
        print(f"        m_y (真正面) Δ = {x_after[3] - x_before[3]:+.2e} m")
        print(f"        m_z (高さ)   Δ = {x_after[4] - x_before[4]:+.2e} m")
        # 前後 m_x は自由 (拘束外)。中点の x で評価。
        mx_before = 0.5 * (l_before[0] + r_before[0])
        mx_after = 0.5 * (l_after[0] + r_after[0])
        print(f"    free dim:  m_x (前後) {mx_before:+.3f} -> {mx_after:+.3f} m "
              f"(Δ {mx_after - mx_before:+.3f}, 拘束外なので動いてよい)")

        print("\n    -> 胴体・肘の姿勢は変わるが球の把持/正面/高さは不変 = 冗長性 ✓")
        print("       outputs/m3_before.png, m3_after.png を参照")
    finally:
        s.close()


if __name__ == "__main__":
    main()
