"""M5 動作確認: 平面パス追従 + ヌル空間 PAD 表現。

台車を S 字パスに沿って動かしつつ、上半身は PAD 表現(ヌル空間)を実行。
タスクは台車座標で定義されているので、台車が並進・旋回しても球は台車正面・
一定高さに保たれる(移動と表現が分離)。各スナップショットで base 姿勢・球の
台車座標位置・タスク誤差を出力し、パス上の数フレームを描画する。
"""
from __future__ import annotations

import numpy as np

from pepper_pad.sim import PepperScene
from pepper_pad.controller import NullSpaceController
from pepper_pad.pad import PADExpression, CTRL_JOINTS
from pepper_pad.path import SCurvePath, apply_base_pose, base_height
from pepper_pad.kinematics import base_frame_point
from build_scene import render

DT = 1.0 / 60.0
PAD = (1, 1, 1)          # 表現を載せた状態で移動


def main() -> None:
    s = PepperScene(gui=False)
    try:
        s.apply_hold_posture()
        s.spawn_sphere()
        ctrl = NullSpaceController(s, CTRL_JOINTS, k_task=10.0)
        q_neutral = ctrl.read_q()
        expr = PADExpression(s, q_neutral, PAD)

        path = SCurvePath(duration=8.0, length=1.6, amp=0.5, cycles=1.0)
        z0 = base_height(s)
        steps = int(path.duration / DT)
        snap_at = {int(steps * f) for f in (0.0, 0.2, 0.4, 0.6, 0.8, 0.999)}

        max_err = 0.0
        base_y_dev = 0.0   # 球の台車座標 y(真正面=0) からのズレの最大
        base_z_dev = 0.0   # 球の台車座標 z(一定高さ) からのズレの最大
        z_ref = None
        snap = 0

        print(f"S-curve path, {path.duration}s, PAD={PAD}\n")
        print(f"{'t':>5} {'base x':>7} {'base y':>7} {'yaw°':>6} "
              f"{'ball_base(x,y,z)':>22} {'taskErr':>9}")

        for i in range(steps + 1):
            t = i * DT
            x, y, yaw = path.pose(t)
            apply_base_pose(s, x, y, yaw, z0)        # 移動タスク(世界座標)
            q = ctrl.read_q()
            err = ctrl.step(expr.q_dot0(t, q))        # 保持タスク+表現(台車座標)
            expr.apply_head(t)
            max_err = max(max_err, float(np.abs(err).max()))

            # 球の台車座標位置（正面・一定高さの確認）
            ball_w = s.sphere_world_center()
            ball_b = base_frame_point(s.model, s.client, ball_w)
            if z_ref is None:
                z_ref = ball_b[2]
            base_y_dev = max(base_y_dev, abs(ball_b[1]))
            base_z_dev = max(base_z_dev, abs(ball_b[2] - z_ref))

            if i in snap_at:
                render(s, [x, y, s.hand_midpoint()[2]], yaw=55, pitch=-18,
                       dist=2.0, name=f"m5_path_{snap}.png")
                print(f"{t:5.2f} {x:7.2f} {y:7.2f} {np.degrees(yaw):6.1f} "
                      f"({ball_b[0]:+.2f},{ball_b[1]:+.2f},{ball_b[2]:+.2f}) "
                      f"{np.abs(err).max():9.1e}")
                snap += 1

        print(f"\nmax task error (拘束5D, 移動中) = {max_err:.2e} m")
        print(f"球の台車座標 y(真正面) 最大ズレ = {base_y_dev:.2e} m")
        print(f"球の台車座標 z(一定高さ) 最大ズレ = {base_z_dev:.2e} m")
        ok = max_err < 0.02 and base_y_dev < 0.01 and base_z_dev < 0.01
        print("\n移動中も球は台車正面・一定高さを保持:",
              "PASS ✓" if ok else "要調整")
        print("outputs/m5_path_0..5.png を参照")
    finally:
        s.close()


if __name__ == "__main__":
    main()
