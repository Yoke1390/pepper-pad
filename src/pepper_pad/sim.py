"""シーン構築: Pepper + 地面 + 直径25cmの球。

球を両手で抱える左右対称の保持姿勢を与え、球を両手中点にキネマティックに
配置する。後続のヌル空間制御 (M2〜) はこの `PepperScene` を土台に使う。
"""
from __future__ import annotations

import numpy as np
import pybullet as p
from qibullet import SimulationManager

# --- 定数 ---
SPHERE_RADIUS = 0.125  # 直径 0.25m

ARM_JOINTS_L = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
ARM_JOINTS_R = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
ARM_JOINTS = ARM_JOINTS_L + ARM_JOINTS_R
TORSO_JOINTS = ["HipRoll", "HipPitch", "KneePitch"]
HEAD_JOINTS = ["HeadYaw", "HeadPitch"]

HAND_LINKS = ("l_hand", "r_hand")

# 手の開き具合 (0=握る, 1=完全に開く)。球を支える「程よく開いた手」。
HAND_OPENING = 1.0

# 球は「台車座標に固定された点」に置く (両手中点ではない)。両手はその少し下を
# 支えるので、球中心は手の中点より上にある。これは「球と台車の相対位置を固定」
# という主タスク定義 (plan.md §3) とも一致する。
HOLD_SPHERE_CENTER = (0.25, 0.0, 0.82)  # 台車座標での球の中心 (固定)

# 球を下から支える左右対称の保持姿勢 (rad)。
# scripts/calibrate_hold.py のタスク優先 DLS IK で算出:
#   主タスク = 両手を球の下側ヘ (位置)、副タスク = 掌を球へ向ける (ヌル空間)。
# 手は球中心より下・内側で、開いた指 (HAND_OPENING) が球面に接して支える。
# 左右で符号が反転する関節 (Roll/Yaw) は鏡像。
HOLD_POSTURE: dict[str, float] = {
    "LShoulderPitch": +1.0759, "RShoulderPitch": +1.0759,
    "LShoulderRoll": +0.0087, "RShoulderRoll": -0.0087,
    "LElbowYaw": -1.3248, "RElbowYaw": +1.3248,
    "LElbowRoll": -1.0069, "RElbowRoll": +1.0069,
    "LWristYaw": -0.8706, "RWristYaw": +0.8706,
}


class PepperScene:
    """Pepper + 地面 + 抱える球のシーン。"""

    def __init__(self, gui: bool = False):
        self.sim = SimulationManager()
        # auto_step=False: バックグラウンドの実時間ステップ用スレッドを起動しない。
        # これを True にすると resetJointState で設定した姿勢を、モータ目標
        # (既定 ~0) へ引き戻すスレッドが走り、姿勢が崩れる。制御ループでは
        # step() で明示的にステップする。
        self.client = self.sim.launchSimulation(gui=gui, auto_step=False)
        self.pepper = self.sim.spawnPepper(self.client, spawn_ground_plane=True)
        self.model = self.pepper.getRobotModel()
        self.sphere_id: int | None = None

    # --- 姿勢 ---
    def set_posture_instant(self, posture: dict[str, float]) -> None:
        """関節を即座に目標角へ (resetJointState)。さらにモータ目標も同じ角度に
        設定し、後で step() してもこの姿勢が保持されるようにする。"""
        for name, value in posture.items():
            idx = self.pepper.getJoint(name).getIndex()
            p.resetJointState(self.model, idx, value, physicsClientId=self.client)
        self.pepper.setAngles(list(posture.keys()), list(posture.values()), 1.0)

    def set_hand_opening(self, value: float) -> None:
        """両手の指を value (0=握る, 1=開く) に開く。mimic 指関節を即座に設定し、
        モータ目標も合わせる。"""
        for hand in ("LHand", "RHand"):
            names, vals = self.pepper._mimicHand(hand, value)
            for nm, v in zip(names, vals):
                idx = self.pepper.getJoint(nm).getIndex()
                p.resetJointState(self.model, idx, v, physicsClientId=self.client)
        self.pepper.setAngles(["LHand", "RHand"], [value, value], 1.0)

    def apply_hold_posture(self) -> None:
        self.set_posture_instant(HOLD_POSTURE)
        self.set_hand_opening(HAND_OPENING)

    def step(self) -> None:
        """シミュレーションを1ステップ進め、球を両手中点へ追従させる。"""
        self.sim.stepSimulation(self.client)
        self.update_sphere()

    # --- 計測 ---
    def hand_positions(self) -> tuple[np.ndarray, np.ndarray]:
        l, _ = self.pepper.getLinkPosition(HAND_LINKS[0])
        r, _ = self.pepper.getLinkPosition(HAND_LINKS[1])
        return np.asarray(l), np.asarray(r)

    def hand_midpoint(self) -> np.ndarray:
        l, r = self.hand_positions()
        return 0.5 * (l + r)

    def hand_separation(self) -> float:
        l, r = self.hand_positions()
        return float(np.linalg.norm(l - r))

    # --- 球 ---
    def sphere_world_center(self) -> np.ndarray:
        """台車座標で固定した球中心を、現在の台車(ベース)姿勢で world に変換。
        台車が動いても球は台車に対し固定される。"""
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.model, physicsClientId=self.client)
        world, _ = p.multiplyTransforms(base_pos, base_orn,
                                        HOLD_SPHERE_CENTER, [0, 0, 0, 1])
        return np.asarray(world)

    def spawn_sphere(self, rgba=(0.9, 0.3, 0.2, 1.0)) -> int:
        """台車基準の固定点 (HOLD_SPHERE_CENTER) に直径25cmの球を生成
        (mass=0, キネマティック)。両手はその少し下を支える。"""
        col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=SPHERE_RADIUS, physicsClientId=self.client)
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=SPHERE_RADIUS, rgbaColor=rgba,
            physicsClientId=self.client)
        self.sphere_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=self.sphere_world_center().tolist(),
            physicsClientId=self.client)
        return self.sphere_id

    def update_sphere(self) -> None:
        """球を台車基準の固定点へ保つ (台車が動けば一緒に動く)。"""
        if self.sphere_id is None:
            return
        p.resetBasePositionAndOrientation(
            self.sphere_id, self.sphere_world_center().tolist(), [0, 0, 0, 1],
            physicsClientId=self.client)

    # --- 後始末 ---
    def close(self) -> None:
        self.sim.stopSimulation(self.client)
