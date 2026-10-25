import sys, mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(sys.argv[1])
print("nu", m.nu, "njnt", m.njnt, "nq", m.nq)
for i in range(m.nu):
    j = m.actuator_trnid[i, 0]
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i), "joint", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
          "qposadr", m.jnt_qposadr[j], "obs_q_index", m.jnt_qposadr[j] - 7, "ctrlrange", m.actuator_ctrlrange[i], "kp", m.actuator_gainprm[i, 0])
