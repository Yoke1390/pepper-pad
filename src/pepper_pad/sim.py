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

# 球は両手で支えられ手と一緒に動く。球中心は両手中点から「少し上・前」へずらした点
# (台車座標オフセット)。主タスク (plan.md §3) は球の左右中心(y)と高さ(z)だけを拘束し
# 前後(x)は自由なので、球は手に追従して前後する。
HOLD_SPHERE_OFFSET = (0.05, 0.0, 0.06)  # 両手中点 → 球中心 (台車座標)

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
        """両手中点から台車座標オフセット (HOLD_SPHERE_OFFSET) だけずらした点を
        球中心とする。球は手に追従する（主タスクが y,z を拘束、前後は自由）。"""
        mid = self.hand_midpoint()  # world
        _, base_orn = p.getBasePositionAndOrientation(
            self.model, physicsClientId=self.client)
        off_world, _ = p.multiplyTransforms(
            [0, 0, 0], base_orn, HOLD_SPHERE_OFFSET, [0, 0, 0, 1])
        return mid + np.asarray(off_world)

    def spawn_sphere(self, rgba=(0.9, 0.3, 0.2, 1.0)) -> int:
        """両手中点の少し上・前 (HOLD_SPHERE_OFFSET) に直径25cmの球を生成
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
