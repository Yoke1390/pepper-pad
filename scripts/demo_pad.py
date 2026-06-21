"""M4 動作確認: PAD → ヌル空間二次目標で 5 種の表情的挙動を生成。

要求 PAD = [0,0,0], [1,1,1], [-1,-1,-1], [-1,1,-1], [-1,1,1] それぞれで
2.5 秒シミュレートし、(1) 球を落とさない（拘束タスク誤差が小）、(2) 姿勢が
見分けられる、ことを確認。各 PAD の代表フレームを outputs/ に描画する。
"""
from __future__ import annotations

import numpy as np

from pepper_pad.sim import PepperScene, ARM_JOINTS, TORSO_JOINTS
from pepper_pad.controller import NullSpaceController
from pepper_pad.pad import PADExpression, CTRL_JOINTS
from build_scene import render

PADS = [
    ((0, 0, 0), "pad_000_baseline", "中立・静止"),
    ((1, 1, 1), "pad_111_joy", "喜び/興奮/自信"),
    ((-1, -1, -1), "pad_nnn_sad", "悲しみ/疲労/服従"),
    ((-1, 1, -1), "pad_n1n_fear", "不安/動揺/怯え"),
    ((-1, 1, 1), "pad_n11_anger", "怒り/威圧/緊張"),
]

DT = 1.0 / 60.0
STEPS = 150  # 2.5 s
WATCH = ["HipPitch", "KneePitch", "LElbowRoll", "LShoulderRoll"]  # 制御関節のみ


def run_one(pad, name, desc):
    s = PepperScene(gui=False)
    try:
        s.apply_hold_posture()
        s.spawn_sphere()
        ctrl = NullSpaceController(s, CTRL_JOINTS, k_task=10.0)
        q_neutral = ctrl.read_q()
        expr = PADExpression(s, q_neutral, pad)

        max_err = np.zeros(5)
        final_err = np.zeros(5)
        t = 0.0
        for _ in range(STEPS):
            q = ctrl.read_q()
            err = ctrl.step(expr.q_dot0(t, q))
            expr.apply_head(t)
            max_err = np.maximum(max_err, np.abs(err))
            final_err = np.abs(err)
            t += DT

        q_final = ctrl.read_q()
        head = expr.head_angles(t)
        head_pitch = head["HeadPitch"]
        mid = s.hand_midpoint()
        render(s, [mid[0], 0.0, mid[2]], yaw=35, pitch=-20, dist=1.8,
               name=name + "_persp.png")
        render(s, [mid[0], 0.0, mid[2]], yaw=90, pitch=-8, dist=1.7,
               name=name + "_front.png")

        # 報告
        dev = {nm: q_final[CTRL_JOINTS.index(nm)] for nm in WATCH if nm in CTRL_JOINTS}
        print(f"\n=== PAD={pad}  {desc} ===")
        print(f"  task error 拘束5D: peak={max_err.max():.2e} m, "
              f"final={final_err.max():.2e} m  "
              f"-> {'OK 球保持' if max_err.max() < 0.02 else 'NG 落下リスク'}")
        print(f"  amp={expr.amp:.3f} rad, ω={expr.w:.2f} rad/s "
              f"({'律動あり' if expr.amp > 0 else '静止'})")
        print("  posture: " + ", ".join(f"{nm}={dev[nm]:+.2f}" for nm in dev))
        print(f"  head: HeadPitch={head_pitch:+.2f} "
              f"({'上向き' if head_pitch < -0.05 else '下向き' if head_pitch > 0.05 else '正面'})")
        return pad, q_final, head_pitch, max_err.max()
    finally:
        s.close()


def main() -> None:
    print("M4: PAD -> null-space expression  (5 PADs, 2.5s each)")
    results = [run_one(pad, name, desc) for pad, name, desc in PADS]

    # 姿勢が互いに見分けられるか（HipPitch/ElbowRoll など主要関節の差 + 頭部）
    print("\n--- 姿勢の判別性 (主要関節, q_final) ---")
    idx = [CTRL_JOINTS.index(nm) for nm in WATCH]
    print("       " + "".join(f"{nm[:8]:>9s}" for nm in WATCH) + f"{'HeadPitch':>10s}")
    for pad, q, hp, _ in results:
        print(f"  {str(pad):>11s} " + "".join(f"{q[i]:+9.2f}" for i in idx)
              + f"{hp:+10.2f}")
    print("\noutputs/pad_*_persp.png, *_front.png を参照")


if __name__ == "__main__":
    main()
