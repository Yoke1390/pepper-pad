"""M0 smoke test: Pepper が DIRECT モードでロードでき、手先リンク・腕関節が
取得できることを確認する。GUI/ディスプレイ不要。"""
import pybullet as p
from qibullet import SimulationManager

ARM_JOINTS = [
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw",
]
TORSO_JOINTS = ["HipRoll", "HipPitch", "KneePitch"]


def main() -> None:
    sim = SimulationManager()
    client = sim.launchSimulation(gui=False)
    try:
        pepper = sim.spawnPepper(client, spawn_ground_plane=True)
        print("Pepper loaded. robot model id:", pepper.getRobotModel())

        angles = pepper.getAnglesPosition(ARM_JOINTS + TORSO_JOINTS)
        for name, a in zip(ARM_JOINTS + TORSO_JOINTS, angles):
            print(f"  {name:16s} = {a:+.3f} rad")

        for link in ("l_hand", "r_hand", "torso"):
            try:
                pos, _ = pepper.getLinkPosition(link)
                print(f"  link {link:8s} world pos = "
                      f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
            except KeyError:
                print(f"  link {link}: NOT FOUND")

        # 生 pybullet ハンドルが取れる（ヤコビアン計算に必要）ことを確認
        n_joints = p.getNumJoints(pepper.getRobotModel(),
                                  physicsClientId=pepper.getPhysicsClientId())
        print(f"pybullet getNumJoints = {n_joints}")
        print("SMOKE TEST OK")
    finally:
        sim.stopSimulation(client)


if __name__ == "__main__":
    main()
