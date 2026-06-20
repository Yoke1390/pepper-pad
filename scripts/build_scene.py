"""M1: シーンを組み立て、保持姿勢と球の取り付けを PNG で確認する。

ヘッドレス環境向けに getCameraImage で 正面・俯瞰・真上・真横 の4枚を
レンダリングして outputs/ に保存する。レンダリングのたびに、その時点の
関節角・手先位置・球体位置を print する (各画像が同一姿勢か検証できる)。
"""
from __future__ import annotations

import os

import numpy as np
import pybullet as p

from pepper_pad.sim import PepperScene, SPHERE_RADIUS, ARM_JOINTS, TORSO_JOINTS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def dump_state(scene: PepperScene, label: str) -> None:
    """現在の関節角・手先位置・球体位置を出力。"""
    angles = scene.pepper.getAnglesPosition(ARM_JOINTS + TORSO_JOINTS)
    print(f"[{label}] state:")
    for nm, a in zip(ARM_JOINTS + TORSO_JOINTS, angles):
        print(f"    {nm:16s} = {a:+.4f} rad")
    l, r = scene.hand_positions()
    print(f"    L hand = ({l[0]:+.4f}, {l[1]:+.4f}, {l[2]:+.4f})")
    print(f"    R hand = ({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f})")
    print(f"    hand separation = {scene.hand_separation():.4f} m")
    if scene.sphere_id is not None:
        sp, _ = p.getBasePositionAndOrientation(
            scene.sphere_id, physicsClientId=scene.client)
        print(f"    sphere center  = ({sp[0]:+.4f}, {sp[1]:+.4f}, {sp[2]:+.4f})")


def render(scene: PepperScene, target, yaw, pitch, dist, name, w=640, h=480):
    dump_state(scene, name)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target, distance=dist, yaw=yaw, pitch=pitch,
        roll=0, upAxisIndex=2, physicsClientId=scene.client)
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=w / h, nearVal=0.1, farVal=5.0,
        physicsClientId=scene.client)
    _, _, rgb, _, _ = p.getCameraImage(
        w, h, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER, physicsClientId=scene.client)
    rgb = np.reshape(np.asarray(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]
    import imageio.v2 as imageio
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    imageio.imwrite(path, rgb)
    print(f"    -> rendered {path}")


def main() -> None:
    scene = PepperScene(gui=False)
    try:
        scene.apply_hold_posture()
        scene.spawn_sphere()

        print(f"sphere diameter = {2 * SPHERE_RADIUS:.2f} m\n")
        mid = scene.hand_midpoint()
        cam_t = [mid[0], 0.0, mid[2]]

        render(scene, cam_t, yaw=90, pitch=-10, dist=1.6, name="scene_front.png")
        render(scene, cam_t, yaw=35, pitch=-20, dist=1.7, name="scene_persp.png")
        render(scene, cam_t, yaw=0, pitch=-89, dist=1.3, name="scene_top.png")
        render(scene, cam_t, yaw=0, pitch=0, dist=1.5, name="scene_side.png")
    finally:
        scene.close()


if __name__ == "__main__":
    main()
