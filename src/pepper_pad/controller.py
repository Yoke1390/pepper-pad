"""M3: ヌル空間 resolved-rate コントローラ (DLS)。

緩和保持タスク x = [d(3), m_y, m_z] ∈ R^5 (kinematics.py) を主タスクとし、

    q̇ = J⁺ ẋ_d  +  (I − J⁺J) q̇₀
    ẋ_d = K_task (x_ref − x_cur)
    J⁺  = Jᵀ (J Jᵀ + λ²I)⁻¹      … 減衰最小二乗 (DLS)

で 13DoF (両腕10＋胴体3) を速度レベルで解く。主タスクが球を台車正面・一定高さ
(d, m_y, m_z) に保ち続け、二次目標 q̇₀ はヌル空間射影 N=I−J⁺J を通すので主タスク
を乱さない。q̇₀ が PAD 由来の表現（M4）になる。

シーンは auto_step=False のキネマティック運用なので、積分した関節角を
resetJointState で直接書き戻す（モータ目標も合わせる）。
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from .kinematics import holding_task_value, holding_task_jacobian, HOLD_TASK_DIM


def dls_pinv(J: np.ndarray, lam: float) -> np.ndarray:
    """減衰最小二乗擬似逆 Jᵀ(JJᵀ + λ²I)⁻¹。特異点近傍を安定化。"""
    m = J.shape[0]
    return J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(m), np.eye(m))


class NullSpaceController:
    """緩和保持タスクの resolved-rate ヌル空間コントローラ。"""

    def __init__(self, scene, joint_names: list[str], *, lam: float = 0.05,
                 k_task: float = 4.0, dt: float = 1.0 / 60.0):
        self.scene = scene
        self.joints = list(joint_names)
        self.g_idx = [scene.pepper.getJoint(nm).getIndex() for nm in self.joints]
        self.lam = lam
        self.k_task = k_task
        self.dt = dt
        self.lims = []
        for g in self.g_idx:
            info = p.getJointInfo(scene.model, g, physicsClientId=scene.client)
            self.lims.append((info[8], info[9]))
        # 現在姿勢のタスク値を保持目標として固定。
        self.x_ref = holding_task_value(scene)

    # --- 関節 I/O ---
    def read_q(self) -> np.ndarray:
        return np.array([p.getJointState(self.scene.model, g,
                                         physicsClientId=self.scene.client)[0]
                         for g in self.g_idx])

    def write_q(self, q: np.ndarray) -> None:
        """積分した関節角を即時反映 (resetJointState) し、モータ目標も合わせる。"""
        for g, qi in zip(self.g_idx, q):
            p.resetJointState(self.scene.model, g, float(qi),
                              physicsClientId=self.scene.client)
        self.scene.pepper.setAngles(self.joints, [float(v) for v in q], 1.0)

    # --- 制御 ---
    def task_error(self) -> np.ndarray:
        """x_ref − x_cur (拘束 5 次元の誤差)。"""
        return self.x_ref - holding_task_value(self.scene)

    def step(self, q_dot0: np.ndarray | None = None) -> np.ndarray:
        """1 制御ステップ。q_dot0 (R^13) を二次目標としてヌル空間へ射影。
        反映後のタスク誤差ベクトル (R^5) を返す。"""
        q = self.read_q()
        J = holding_task_jacobian(self.scene, self.joints)        # 5x13
        x_err = self.x_ref - holding_task_value(self.scene)       # 5
        x_dot_d = self.k_task * x_err

        Jp = dls_pinv(J, self.lam)                                # 13x5
        dq = Jp @ x_dot_d
        if q_dot0 is not None:
            N = np.eye(len(self.joints)) - Jp @ J                 # 13x13
            dq = dq + N @ np.asarray(q_dot0, dtype=float)

        q_new = q + dq * self.dt
        for i, (lo, hi) in enumerate(self.lims):
            q_new[i] = min(max(q_new[i], lo), hi)
        self.write_q(q_new)
        self.scene.update_sphere()      # 球を手に追従させる（前後は自由）
        return self.x_ref - holding_task_value(self.scene)

    def null_space_basis_rank(self) -> int:
        """現在配置でのヌル空間次元 = 13 − rank(J)。冗長 DoF の確認用。"""
        J = holding_task_jacobian(self.scene, self.joints)
        return len(self.joints) - int(np.linalg.matrix_rank(J))
