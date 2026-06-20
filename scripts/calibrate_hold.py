"""保持姿勢の較正: 球の左側に左手を IK で合わせ、右腕へ鏡像化する。

得られた HOLD_POSTURE を表示し、確認用 PNG を outputs/ に描画する。
数値が良ければ sim.py の HOLD_POSTURE に貼り付ける。
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from pepper_pad.sim import PepperScene, ARM_JOINTS_L, SPHERE_RADIUS, HAND_OPENING
from pepper_pad.kinematics import solve_arm_ik
from build_scene import render

# 球の中心 (台車正面) と、手を当てる左右オフセット。
# x は腕の前方リーチ (肩から ~0.33m) 内に収める。
# 直径25cm (半径0.125m)。球の「少し下」から支えるため手を赤道より下げる。
# 下側は球断面が細くなるので y も詰める。手リンクは半径より内側を狙う
# (位置ドリフト＋開いた指のぶんを見込む)。
SPHERE_CENTER = np.array([0.25, 0.0, 0.82])
HAND_Y = 0.13          # 中心から左右へ (separation = 2*HAND_Y)
HAND_DX, HAND_DZ = -0.05, -0.06  # 手を球の少し下・外に当てて下から支える
                               # (親指の食い込みを抑えつつ指は接地)

# 掌の向き合わせ。PALM_LOCAL_L は l_hand ローカル座標での掌法線 (実測, 姿勢不変)。
# PALM_TARGET_L は左手の掌を向けたい world 方向 = 内(-y)＋上(+z) で球を下から支える。
PALM_LOCAL_L = np.array([-0.668, 0.0, -0.744])
PALM_TARGET_L = np.array([0.0, -0.6, 0.6])

# 左→右の鏡像 (Pitch は同符号, Roll/Yaw は反転)
MIRROR = {
    "LShoulderPitch": ("RShoulderPitch", +1),
    "LShoulderRoll": ("RShoulderRoll", -1),
    "LElbowYaw": ("RElbowYaw", -1),
    "LElbowRoll": ("RElbowRoll", -1),
    "LWristYaw": ("RWristYaw", -1),
}


def main() -> None:
    s = PepperScene(gui=False)
    try:
        l_target = SPHERE_CENTER + np.array([HAND_DX, HAND_Y, HAND_DZ])

        # DLS IK で左腕だけ動かして左手を target へ。掌も球の方へ向ける。
        left = solve_arm_ik(s, "l_hand", ARM_JOINTS_L, l_target,
                            palm_local=PALM_LOCAL_L, palm_target=PALM_TARGET_L)

        posture: dict[str, float] = {}
        for lname in ARM_JOINTS_L:
            lval = left[lname]
            posture[lname] = lval
            rname, sign = MIRROR[lname]
            posture[rname] = sign * lval

        s.set_posture_instant(posture)
        s.set_hand_opening(HAND_OPENING)   # 開いた手で隙間を確認
        s.spawn_sphere()

        l, r = s.hand_positions()
        sep = s.hand_separation()
        print("HOLD_POSTURE = {")
        for k in ("LShoulderPitch", "RShoulderPitch", "LShoulderRoll", "RShoulderRoll",
                  "LElbowYaw", "RElbowYaw", "LElbowRoll", "RElbowRoll",
                  "LWristYaw", "RWristYaw"):
            print(f'    "{k}": {posture[k]:+.4f},')
        print("}")
        print(f"L hand = ({l[0]:+.3f}, {l[1]:+.3f}, {l[2]:+.3f})  target {l_target}")
        print(f"R hand = ({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f})")
        print(f"separation = {sep:.3f} m (target {2*HAND_Y:.2f}), "
              f"sphere d = {2*SPHERE_RADIUS:.2f}")

        # 達成した左手掌法線 (world) を確認
        st = p.getLinkState(s.model, s.pepper.getLink("l_hand").getIndex(),
                            computeForwardKinematics=True, physicsClientId=s.client)
        R = np.asarray(p.getMatrixFromQuaternion(st[5])).reshape(3, 3)
        n = R @ PALM_LOCAL_L
        print(f"L palm normal (world) = ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f})  "
              f"target {PALM_TARGET_L/np.linalg.norm(PALM_TARGET_L)}")

        # 開いた左手の指先・掌が球面からどれだけ離れているか (負=接触/めり込み)。
        # 球は台車基準の固定中心にあるので、その中心で測る。
        center = np.asarray(SPHERE_CENTER)
        def lpos(nm):
            return np.array(p.getLinkState(
                s.model, s.pepper.getLink(nm).getIndex(),
                computeForwardKinematics=True, physicsClientId=s.client)[4])
        gaps = {nm: float(np.linalg.norm(lpos(nm) - center) - SPHERE_RADIUS)
                for nm in ("LFinger13_link", "LFinger33_link", "LThumb2_link", "l_hand")}
        print("gap to sphere surface:", {k: round(v, 3) for k, v in gaps.items()},
              " min =", round(min(gaps.values()), 3))

        cam_t = [SPHERE_CENTER[0], 0.0, SPHERE_CENTER[2]]
        render(s, cam_t, yaw=35, pitch=-20, dist=1.7, name="hold_persp.png")
        render(s, cam_t, yaw=0, pitch=-89, dist=1.3, name="hold_top.png")
        render(s, cam_t, yaw=0, pitch=89, dist=0.5, name="hold_bottom.png")
        render(s, cam_t, yaw=50, pitch=-25, dist=0.55, name="hold_closeup.png")
        render(s, cam_t, yaw=90, pitch=-10, dist=0.6, name="hold_front.png")
        render(s, cam_t, yaw=0, pitch=0, dist=0.5, name="hold_side.png")
    finally:
        s.close()


if __name__ == "__main__":
    main()
