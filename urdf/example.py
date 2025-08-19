import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

model = mujoco.MjModel.from_xml_path("robot_1.xml")
data = mujoco.MjData(model)

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

# PD 控制参数
Kp, Kd = 2, 0.2

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # === 1. 获取 base_link 四元数姿态 ===
        quat = data.xquat[base_id]  # [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # 转换成 [x,y,z,w]
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        # === 2. 角速度（世界坐标系下） ===
        ang_vel = data.cvel[base_id, :3]  # [wx, wy, wz]

        # === 3. 计算扶正力矩（只纠正 roll 和 pitch）===
        torque = np.zeros(3)
        torque[0] = 0  # 绕x轴
        torque[1] = -Kp * pitch - Kd * ang_vel[1]  # 绕y轴
        torque[2] = 0  # 不控制 yaw，自由转
        torque = np.clip(torque, -5, 5)

        # === 4. 写入外力矩 ===
        data.xfrc_applied[base_id, 3:6] = torque

        # 推进仿真
        mujoco.mj_step(model, data)
        viewer.sync()
