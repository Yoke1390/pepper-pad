"""M6: PAD クリップの MP4 録画。

パス追従＋ヌル空間 PAD 表現を回しながら getCameraImage でフレームを取得し、
「ロボット名／タスク／PAD 値・感情ラベル／冗長性」を焼き込んで mp4 に書き出す。
カメラは台車を追従するのでロボットは中央に大きく映り、表情が見やすい。
"""
from __future__ import annotations

import os

import numpy as np
import pybullet as p
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from .sim import PepperScene
from .controller import NullSpaceController
from .pad import PADExpression, CTRL_JOINTS
from .path import SCurvePath, apply_base_pose, base_height

# 要求 PAD → 感情ラベル
EMOTION = {
    (0, 0, 0): "baseline (neutral)",
    (1, 1, 1): "joy / excitement / confidence",
    (-1, -1, -1): "sadness / fatigue / submission",
    (-1, 1, -1): "anxiety / agitation / fear",
    (-1, 1, 1): "anger / intimidation / tension",
}

ROBOT_LINE = "Robot: SoftBank Pepper  (qibullet / PyBullet)"
TASK_LINE = "Task: hold a 0.25 m sphere in front at constant height; follow a planar path"
REDUN_LINE = "Null-space control: 13 joints - 5D task = 8 DoF redundancy"


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)     # Pillow>=10: TrueType
    except TypeError:
        return ImageFont.load_default()


def grab_frame(scene, target, yaw: float, pitch: float, dist: float,
               w: int = 640, h: int = 480) -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target, distance=dist, yaw=yaw, pitch=pitch,
        roll=0, upAxisIndex=2, physicsClientId=scene.client)
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=w / h, nearVal=0.1, farVal=6.0, physicsClientId=scene.client)
    _, _, rgb, _, _ = p.getCameraImage(
        w, h, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER, physicsClientId=scene.client)
    return np.reshape(np.asarray(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]


def overlay_labels(frame: np.ndarray, pad, t: float, duration: float) -> np.ndarray:
    """上部に説明帯、下部に PAD・感情ラベルを焼き込む。"""
    img = Image.fromarray(frame).convert("RGBA")
    w, h = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    small = _font(15)
    big = _font(24)

    # 上部の説明帯
    top_lines = [ROBOT_LINE, TASK_LINE, REDUN_LINE]
    d.rectangle([0, 0, w, 4 + 19 * len(top_lines) + 4], fill=(0, 0, 0, 150))
    for i, line in enumerate(top_lines):
        d.text((8, 5 + 19 * i), line, font=small, fill=(255, 255, 255, 255))

    # 下部の PAD・感情
    pad_t = tuple(int(v) for v in pad)
    emo = EMOTION.get(pad_t, "")
    pad_str = f"PAD = [{pad_t[0]:+d}, {pad_t[1]:+d}, {pad_t[2]:+d}]"
    d.rectangle([0, h - 40, w, h], fill=(0, 0, 0, 150))
    d.text((8, h - 35), pad_str, font=big, fill=(255, 230, 120, 255))
    tw = d.textlength(pad_str, font=big)
    d.text((8 + tw + 16, h - 31), emo, font=small, fill=(220, 220, 220, 255))

    # 進捗バー
    frac = min(max(t / duration, 0.0), 1.0)
    d.rectangle([0, h - 3, int(w * frac), h], fill=(255, 230, 120, 220))

    out = Image.alpha_composite(img, layer).convert("RGB")
    return np.asarray(out)


def record_clip(pad, out_path: str, *, path=None, duration: float = 8.0,
                fps: int = 30, k_task: float = 10.0,
                cam=(55.0, -18.0, 2.0), w: int = 640, h: int = 480,
                verbose: bool = True) -> float:
    """1 つの PAD クリップを mp4 に録画。最大タスク誤差を返す。"""
    control_dt = 1.0 / 60.0
    stride = max(1, round(1.0 / (fps * control_dt)))   # 何制御ステップごとに1フレーム
    path = path or SCurvePath(duration=duration, length=1.6, amp=0.5, cycles=1.0)

    s = PepperScene(gui=False)
    try:
        s.apply_hold_posture()
        s.spawn_sphere()
        ctrl = NullSpaceController(s, CTRL_JOINTS, k_task=k_task)
        expr = PADExpression(s, ctrl.read_q(), pad)
        z0 = base_height(s)
        steps = int(duration / control_dt)
        yaw, pitch, dist = cam

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                    quality=8, macro_block_size=16)
        max_err = 0.0
        try:
            for i in range(steps + 1):
                t = i * control_dt
                x, y, yawb = path.pose(t)
                apply_base_pose(s, x, y, yawb, z0)
                err = ctrl.step(expr.q_dot0(t, ctrl.read_q()))
                expr.apply_head(t)
                max_err = max(max_err, float(np.abs(err).max()))
                if i % stride == 0:
                    mid = s.hand_midpoint()
                    frame = grab_frame(s, [x, y, mid[2]], yaw, pitch, dist, w, h)
                    writer.append_data(overlay_labels(frame, pad, t, duration))
        finally:
            writer.close()
        if verbose:
            print(f"  wrote {out_path}  ({EMOTION.get(tuple(int(v) for v in pad), '')}), "
                  f"max task err = {max_err:.2e} m")
        return max_err
    finally:
        s.close()
