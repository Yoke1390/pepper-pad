"""M5: 平面パス生成と台車追従。

タスクの2層分解 (plan.md §3): 移動タスク(世界座標) と 球保持タスク(台車座標) は
独立。球保持タスクのヤコビアンは台車座標で定義 (kinematics.holding_task_jacobian の
R^T) されているので、**台車を剛体的に動かしても球の台車相対位置は不変** = ロボット
全体が一緒に動くだけで保持タスク誤差は出ない。よって移動は base 姿勢を毎ステップ
パスへ設定するだけでよく、上半身の PAD 表現(ヌル空間)と完全に分離して並走できる。

キネマティック運用 (auto_step=False) なので qibullet の moveTo(物理ステップ依存)は
使わず、resetBasePositionAndOrientation で base 姿勢を直接与える。
"""
from __future__ import annotations

import numpy as np
import pybullet as p


class PlanarPath:
    """平面パスの基底。pose(t) -> (x, y, yaw[rad]) を返す。"""

    def __init__(self, duration: float):
        self.duration = duration

    def pose(self, t: float) -> tuple[float, float, float]:
        raise NotImplementedError

    def _s(self, t: float) -> float:
        return min(max(t / self.duration, 0.0), 1.0)


class LinePath(PlanarPath):
    """直進（回転なし）。最も単純な移動。"""

    def __init__(self, duration: float = 6.0, length: float = 1.5,
                 heading: float = 0.0):
        super().__init__(duration)
        self.length = length
        self.heading = heading

    def pose(self, t):
        s = self._s(t)
        return self.length * s * np.cos(self.heading), \
            self.length * s * np.sin(self.heading), self.heading


class SCurvePath(PlanarPath):
    """前進しつつ左右に蛇行する S 字。台車が進行方向を向く(yaw=接線)ので
    並進＋回転の両方を含み、『曲がっても球は正面』を示せる。"""

    def __init__(self, duration: float = 8.0, length: float = 1.6,
                 amp: float = 0.5, cycles: float = 1.0):
        super().__init__(duration)
        self.length = length
        self.amp = amp
        self.cycles = cycles

    def pose(self, t):
        s = self._s(t)
        w = 2.0 * np.pi * self.cycles
        x = self.length * s
        y = self.amp * np.sin(w * s)
        dx = self.length
        dy = self.amp * w * np.cos(w * s)
        yaw = float(np.arctan2(dy, dx))
        return float(x), float(y), yaw


class CirclePath(PlanarPath):
    """円弧を周回。常に旋回し続けるので回転の追従が際立つ。"""

    def __init__(self, duration: float = 10.0, radius: float = 0.8,
                 turns: float = 1.0):
        super().__init__(duration)
        self.radius = radius
        self.turns = turns

    def pose(self, t):
        s = self._s(t)
        ang = 2.0 * np.pi * self.turns * s
        x = self.radius * np.sin(ang)
        y = self.radius * (1.0 - np.cos(ang))   # 原点接線方向に発進
        yaw = float(ang)
        return float(x), float(y), yaw


def apply_base_pose(scene, x: float, y: float, yaw: float, z0: float) -> None:
    """台車(base)を world 姿勢 (x, y, yaw) へ即時設定。z は接地高 z0 を保つ。"""
    orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    p.resetBasePositionAndOrientation(
        scene.model, [x, y, z0], orn, physicsClientId=scene.client)


def base_height(scene) -> float:
    pos, _ = p.getBasePositionAndOrientation(scene.model, physicsClientId=scene.client)
    return float(pos[2])
