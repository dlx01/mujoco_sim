import mujoco
import mujoco.viewer
import time
import numpy as np

# -----------------------------
# 1. 加载模型
# -----------------------------
model = mujoco.MjModel.from_xml_path("robot_1.xml")  # 替换成你的模型文件
data = mujoco.MjData(model)
model.opt.timestep = 0.001
# -----------------------------
# Get actuator IDs for act_joint_chilun_1 and act_joint_chilun_2
act_chilun_1_id = model.actuator('act_joint_chilun_1').id
act_chilun_2_id = model.actuator('act_joint_chilun_2').id

# Set fixed control values (e.g., set to 0 as a constant value; adjust as needed)
fixed_value = -0.012  # Example fixed value; change to your desired constant
data.ctrl[act_chilun_1_id] = fixed_value
data.ctrl[act_chilun_2_id] = fixed_value

# Assume the root body or chassis is the one to apply force to (e.g., body_chassis)
# Get body ID (adjust name if different)
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")  # Replace with actual body name if needed

# PD controller constants for balance
kp = 10.0  # Proportional gain
kd = 1.0   # Derivative gain

# Simulation loop with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    while time.time() - start_time < 10:  # Run for 10 seconds; adjust as needed
        xmat = data.xmat[base_id].reshape(3, 3)
        
        # Body z-axis in global frame
        body_z_global = xmat @ np.array([0., 0., 1.])
        
        # Approximate tilt angles (small angle approximation)
        theta_x = body_z_global[1]   # Tilt around x (sign may need adjustment based on convention)
        theta_y = -body_z_global[0]  # Tilt around y
        
        # Angular velocity (assuming free joint, indices 3:6 for rotation)
        omega = data.qvel[3:6]
        
        # Compute torques
        tx = -kp * theta_x - kd * omega[0]
        ty = -kp * theta_y - kd * omega[1]
        tx = np.clip(tx, -5, 5)
        ty = np.clip(ty, -5, 5)
        tz = 0.0
        
        # External force/torque: no linear force, only torque for balance
        # If needed, add [0, 0, model.body_mass[body_id] * 9.81, tx, ty, tz] to counter gravity fully
        external_force = np.array([0., 0., 0., tx, ty, tz])
        
        data.xfrc_applied[base_id] = external_force
        

        
        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Sync viewer
        viewer.sync()
        time.sleep(0.01)  # Slow down for visualization

print("Simulation ended.")