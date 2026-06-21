"""運動学ユーティリティ: pybullet.calculateJacobian を使った手先ヤコビアンと、
腕関節に限定した減衰最小二乗 (DLS) IK。

M2 で構築する「左右手先の積み上げヤコビアン」もここに集約していく。
"""
from __future__ import annotations

import numpy as np
import pybullet as p


def movable_joint_indices(model: int, client: int) -> list[int]:
    """可動 (非 fixed) 関節のグローバル index を昇順で返す。
    calculateJacobian の objPositions 並びと一致する。"""
    n = p.getNumJoints(model, physicsClientId=client)
    return [j for j in range(n)
            if p.getJointInfo(model, j, physicsClientId=client)[2] != p.JOINT_FIXED]


def link_position(model: int, link_index: int, client: int) -> np.ndarray:
    state = p.getLinkState(model, link_index, computeForwardKinematics=True,
                           physicsClientId=client)
    return np.asarray(state[4])  # worldLinkFramePosition


def link_com_position(model: int, link_index: int, client: int) -> np.ndarray:
    """リンク重心(CoM)の world 位置 (getLinkState[0])。calculateJacobian の
    localPosition=[0,0,0] が指す点と一致するので、ヤコビアン検証はこちらを使う。"""
    state = p.getLinkState(model, link_index, computeForwardKinematics=True,
                           physicsClientId=client)
    return np.asarray(state[0])  # linkWorldPosition (CoM)


def hand_jacobian(model: int, client: int, link_index: int,
                  movable: list[int], col_of: dict[int, int],
                  joint_global_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """指定リンクの並進・角速度ヤコビアン (各 3 x len(joint_global_idx)) を、
    movable 全体から該当列だけ抜き出して返す。"""
    positions = [p.getJointState(model, j, physicsClientId=client)[0] for j in movable]
    zeros = [0.0] * len(movable)
    lin, ang = p.calculateJacobian(model, link_index, [0.0, 0.0, 0.0],
                                   positions, zeros, zeros, physicsClientId=client)
    cols = [col_of[g] for g in joint_global_idx]
    return np.asarray(lin)[:, cols], np.asarray(ang)[:, cols]


def hand_linear_jacobian(model: int, client: int, link_index: int,
                         movable: list[int], col_of: dict[int, int],
                         joint_global_idx: list[int]) -> np.ndarray:
    """並進ヤコビアンのみ (3 x len(joint_global_idx))。"""
    return hand_jacobian(model, client, link_index, movable, col_of, joint_global_idx)[0]


# --- 緩和した球保持タスク x = [d(3), m_y, m_z] ∈ R^5 (台車座標系) ---
# plan.md §3/§4 参照。d = p_L - p_R (把持維持3), m = (p_L+p_R)/2 の y(真正面),
# z(一定高さ) を拘束。m_x(前後) は自由なので落とす。計測フレームは台車(base)。

HOLD_TASK_DIM = 5


def base_frame_point(model: int, client: int, p_world: np.ndarray) -> np.ndarray:
    """world 座標の点を台車(base)座標へ変換する: R^T (p_world - base_pos)。
    numpy で計算 (pybullet の invert/multiplyTransforms の量子化往復誤差 ~1e-8 を
    避ける。これは有限差分検証で 2*eps による増幅で 1e-3 の見かけ誤差を生む)。
    単位姿勢では R=I が厳密に出るので恒等変換になる。"""
    base_pos, base_orn = p.getBasePositionAndOrientation(model, physicsClientId=client)
    R = np.asarray(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
    return R.T @ (np.asarray(p_world) - np.asarray(base_pos))


def holding_task_value(scene, hand_links: tuple[str, str] = ("l_hand", "r_hand"),
                       ) -> np.ndarray:
    """現在姿勢での緩和保持タスク x = [d_x, d_y, d_z, m_y, m_z] ∈ R^5 (台車座標)。"""
    model, client = scene.model, scene.client
    li_l = scene.pepper.getLink(hand_links[0]).getIndex()
    li_r = scene.pepper.getLink(hand_links[1]).getIndex()
    # CoM 基準 (calculateJacobian の localPosition=[0,0,0] と同じ点)。
    pL = base_frame_point(model, client, link_com_position(model, li_l, client))
    pR = base_frame_point(model, client, link_com_position(model, li_r, client))
    d = pL - pR
    m = 0.5 * (pL + pR)
    return np.array([d[0], d[1], d[2], m[1], m[2]])


def holding_task_jacobian(scene, joint_names: list[str],
                          hand_links: tuple[str, str] = ("l_hand", "r_hand"),
                          ) -> np.ndarray:
    """緩和保持タスクのヤコビアン J = ∂x/∂q ∈ R^{5 x len(joint_names)} (台車座標)。

    左右手先の world 並進ヤコビアン (calculateJacobian) を台車座標へ回転し、
    d = p_L - p_R の3行と中点 m の y,z 行を積み上げる。台車(根リンク)は
    制御関節で動かないので base 位置・姿勢の偏微分は 0、回転 R^T を掛けるだけ。"""
    model, client = scene.model, scene.client
    movable = movable_joint_indices(model, client)
    col_of = {g: k for k, g in enumerate(movable)}
    g_idx = [scene.pepper.getJoint(nm).getIndex() for nm in joint_names]
    li_l = scene.pepper.getLink(hand_links[0]).getIndex()
    li_r = scene.pepper.getLink(hand_links[1]).getIndex()

    JL = hand_linear_jacobian(model, client, li_l, movable, col_of, g_idx)  # 3xn world
    JR = hand_linear_jacobian(model, client, li_r, movable, col_of, g_idx)

    _, base_orn = p.getBasePositionAndOrientation(model, physicsClientId=client)
    R = np.asarray(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
    JLb = R.T @ JL  # 台車座標へ
    JRb = R.T @ JR

    Jd = JLb - JRb              # 把持維持 d の3行
    Jm = 0.5 * (JLb + JRb)      # 中点 m の3行
    return np.vstack([Jd, Jm[1:3, :]])  # [d_x,d_y,d_z, m_y,m_z] = 5xn


def solve_arm_ik(scene, hand_link: str, arm_joint_names: list[str],
                 target: np.ndarray, *, palm_local: np.ndarray | None = None,
                 palm_target: np.ndarray | None = None, ori_gain: float = 1.0,
                 iters: int = 400, lam: float = 0.05,
                 step_clip: float = 0.2, tol: float = 1e-3) -> dict[str, float]:
    """1 本の腕の関節だけを動かして hand_link を target に合わせる DLS IK。
    腕以外の関節は現在値のまま。収束後の腕関節角を dict で返す。

    palm_local / palm_target を渡すと、**タスク優先**で掌の向きも合わせる:
    位置 (3DoF) を主タスクで厳密に解き、手リンク・ローカルの掌法線 (palm_local)
    が world の palm_target を向くよう、残りのヌル空間 (2DoF) で best-effort 調整。
    位置を犠牲にしないので保持点はずれない。"""
    model, client = scene.model, scene.client
    movable = movable_joint_indices(model, client)
    col_of = {g: k for k, g in enumerate(movable)}
    link_index = scene.pepper.getLink(hand_link).getIndex()
    g_idx = [scene.pepper.getJoint(nm).getIndex() for nm in arm_joint_names]
    n_dof = len(g_idx)

    lims = []
    for g in g_idx:
        info = p.getJointInfo(model, g, physicsClientId=client)
        lims.append((info[8], info[9]))

    q = np.array([p.getJointState(model, g, physicsClientId=client)[0] for g in g_idx])
    target = np.asarray(target, dtype=float)
    align = palm_local is not None and palm_target is not None
    if align:
        palm_local = np.asarray(palm_local, dtype=float)
        d = np.asarray(palm_target, dtype=float)
        d = d / np.linalg.norm(d)

    for _ in range(iters):
        for g, qi in zip(g_idx, q):
            p.resetJointState(model, g, float(qi), physicsClientId=client)
        pos = link_position(model, link_index, client)
        err_p = target - pos
        Jv, Jw = hand_jacobian(model, client, link_index, movable, col_of, g_idx)

        # 主タスク = 位置 (DLS 擬似逆)
        Jv_pinv = Jv.T @ np.linalg.solve(Jv @ Jv.T + (lam ** 2) * np.eye(3), np.eye(3))
        dq = Jv_pinv @ err_p

        if align:
            # 副タスク = 掌の向き合わせ。ヌル空間 N に射影 (位置を乱さない)。
            # 関節限界への張り付き=位置ドリフトを防ぐため、副タスク量は小さくクリップ。
            st = p.getLinkState(model, link_index, computeForwardKinematics=True,
                                physicsClientId=client)
            R = np.asarray(p.getMatrixFromQuaternion(st[5])).reshape(3, 3)
            n = R @ palm_local
            err_o = np.cross(n, d)             # n を d へ向ける回転ベクトル
            N = np.eye(n_dof) - Jv_pinv @ Jv
            sec = np.clip(N @ (ori_gain * (Jw.T @ err_o)), -0.03, 0.03)
            dq = dq + sec

        if np.linalg.norm(dq) < tol:
            break
        dq = np.clip(dq, -step_clip, step_clip)
        q = q + dq
        for i, (lo, hi) in enumerate(lims):
            q[i] = min(max(q[i], lo), hi)

    if align:
        # 位置ポリッシュ: 副タスクを切り位置誤差だけを詰める。掌向きは主に
        # WristYaw が担い位置とほぼ分離しているので、概ね保たれたまま手先が target へ。
        for _ in range(120):
            for g, qi in zip(g_idx, q):
                p.resetJointState(model, g, float(qi), physicsClientId=client)
            err_p = target - link_position(model, link_index, client)
            if np.linalg.norm(err_p) < tol:
                break
            Jv, _ = hand_jacobian(model, client, link_index, movable, col_of, g_idx)
            Jv_pinv = Jv.T @ np.linalg.solve(Jv @ Jv.T + (lam ** 2) * np.eye(3), np.eye(3))
            dq = np.clip(Jv_pinv @ err_p, -step_clip, step_clip)
            q = q + dq
            for i, (lo, hi) in enumerate(lims):
                q[i] = min(max(q[i], lo), hi)

    return {nm: float(v) for nm, v in zip(arm_joint_names, q)}
