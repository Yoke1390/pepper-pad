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
from .path import SCurvePath, apply_base_pose, base_height, draw_path_on_ground

from tqdm import tqdm

# 要求 PAD → 感情ラベル（短い主名 + 補足）
EMOTION_SHORT = {
    (0, 0, 0): "Neutral",
    (1, 1, 1): "Joy",
    (-1, -1, -1): "Sadness",
    (-1, 1, -1): "Fear",
    (-1, 1, 1): "Anger",
}
EMOTION = {
    (0, 0, 0): "baseline",
    (1, 1, 1): "excitement / confidence",
    (-1, -1, -1): "fatigue / submission",
    (-1, 1, -1): "anxiety / agitation",
    (-1, 1, 1): "intimidation / tension",
}


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
    """下部に PAD 値と感情ラベル(Joy など)だけを焼き込む。
    ロボット/タスク等の説明は動画外で行う前提。"""
    img = Image.fromarray(frame).convert("RGBA")
    w, h = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    emo_font = _font(30)
    pad_font = _font(24)

    pad_t = tuple(int(v) for v in pad)
    emo_short = EMOTION_SHORT.get(pad_t, "")
    pad_str = f"PAD = [{pad_t[0]:+d}, {pad_t[1]:+d}, {pad_t[2]:+d}]"

    band = 56
    d.rectangle([0, h - band, w, h], fill=(0, 0, 0, 160))
    # 左: 感情ラベル(大)
    d.text((12, h - band + 13), emo_short, font=emo_font, fill=(255, 235, 130, 255))
    # 右: PAD 値
    pw = d.textlength(pad_str, font=pad_font)
    d.text((w - pw - 12, h - band + 16), pad_str, font=pad_font,
           fill=(255, 255, 255, 255))

    # 進捗バー
    frac = min(max(t / duration, 0.0), 1.0)
    d.rectangle([0, h - 3, int(w * frac), h], fill=(255, 235, 130, 220))

    out = Image.alpha_composite(img, layer).convert("RGB")
    return np.asarray(out)


def record_clip(pad, out_path: str, *, path=None, duration: float = 8.0,
                fps: int = 30, k_task: float = 10.0,
                cam=(55.0, -18.0, 2.0), w: int = 640, h: int = 480,
                verbose: bool = True, progress: bool = True,
                position: int = 0, desc: str | None = None,
                on_step=None) -> float:
    """1 つの PAD クリップを mp4 に録画。最大タスク誤差を返す。"""
    control_dt = 1.0 / 60.0
    stride = max(1, round(1.0 / (fps * control_dt)))   # 何制御ステップごとに1フレーム
    path = path or SCurvePath(duration=duration, length=1.6, amp=0.5, cycles=1.0)

    s = PepperScene(gui=False)
    try:
        s.apply_hold_posture()
        s.spawn_sphere()
        draw_path_on_ground(s, path)              # 軌跡を地面に表示
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
            rng = range(steps + 1)
            if progress:
                rng = tqdm(rng, total=steps + 1, position=position,
                           desc=(desc or os.path.basename(out_path)),
                           leave=True, ncols=90)
            for i in rng:
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
                if on_step is not None:    # 親プロセスへ進捗通知 (並列時)
                    on_step()
        finally:
            writer.close()
        if verbose:
            print(f"  wrote {out_path}  ({EMOTION.get(tuple(int(v) for v in pad), '')}), "
                  f"max task err = {max_err:.2e} m")
        return max_err
    finally:
        s.close()
