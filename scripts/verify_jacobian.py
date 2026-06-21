"""M2 検証: 緩和保持タスク x=[d(3), m_y, m_z]∈R^5 のヤコビアン J=∂x/∂q を、
中心差分による有限差分ヤコビアンと突き合わせて数値検証する。

制御関節 q は両腕10＋胴体3 の 13DoF (plan.md §3)。J は 5×13 で、冗長 8DoF。
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from pepper_pad.sim import PepperScene, ARM_JOINTS, TORSO_JOINTS
from pepper_pad.kinematics import (
    holding_task_value, holding_task_jacobian, HOLD_TASK_DIM,
)

ROW_LABELS = ["d_x", "d_y", "d_z", "m_y", "m_z"]


def main() -> None:
    s = PepperScene(gui=False)
    try:
        s.apply_hold_posture()
        joints = ARM_JOINTS + TORSO_JOINTS  # 13DoF
        g_idx = [s.pepper.getJoint(nm).getIndex() for nm in joints]

        J = holding_task_jacobian(s, joints)
        assert J.shape == (HOLD_TASK_DIM, len(joints)), J.shape

        # 中心差分で数値ヤコビアン
        eps = 1e-5
        q0 = [p.getJointState(s.model, g, physicsClientId=s.client)[0] for g in g_idx]
        Jfd = np.zeros_like(J)
        for k, g in enumerate(g_idx):
            p.resetJointState(s.model, g, q0[k] + eps, physicsClientId=s.client)
            xp = holding_task_value(s)
            p.resetJointState(s.model, g, q0[k] - eps, physicsClientId=s.client)
            xm = holding_task_value(s)
            p.resetJointState(s.model, g, q0[k], physicsClientId=s.client)
            Jfd[:, k] = (xp - xm) / (2 * eps)

        diff = np.abs(J - Jfd)
        denom = np.maximum(np.abs(Jfd), 1e-6)
        rel = diff / denom

        print(f"J shape = {J.shape}  (task {HOLD_TASK_DIM}D, joints {len(joints)})")
        print(f"rank(J) = {np.linalg.matrix_rank(J)}  -> redundancy "
              f"{len(joints) - np.linalg.matrix_rank(J)} DoF")
        print(f"max |J - Jfd|        = {diff.max():.3e}")
        print(f"max relative err     = {rel.max():.3e}")
        print("per-row max |J - Jfd|:")
        for i, lab in enumerate(ROW_LABELS):
            print(f"    {lab}: {diff[i].max():.3e}")

        ok = diff.max() < 1e-4
        print("\nFINITE-DIFFERENCE CHECK:", "PASS" if ok else "FAIL")
        if not ok:
            worst = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"  worst at row {ROW_LABELS[worst[0]]}, joint {joints[worst[1]]}: "
                  f"J={J[worst]:+.5f} Jfd={Jfd[worst]:+.5f}")
    finally:
        s.close()


if __name__ == "__main__":
    main()
