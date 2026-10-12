"""Ideal onboard signals, deliberately excluding privileged simulator state.

Camera pixels are obtained with Renderer.update_scene(camera="head_camera").
This is a sensor adapter, not a policy observation or calibrated hardware API.
"""

import numpy as np


def onboard_readings(model, data, *, include_foot_contacts=False):
    """Copy SI-unit readings in actuator order; gyro/accel use the IMU frame.

    Velocity encoders represent a bus reading or differentiated position estimate.
    Touch is opt-in because physical force-sensing pads have not been selected.
    No world velocity, perfect orientation, root pose, or target pose is returned.
    """
    names = tuple(model.joint(int(j)).name for j in model.actuator_trnid[:, 0])
    readings = {
        "joint_names": names,
        "joint_position_rad": np.array([data.sensor(f"{n}_position").data[0] for n in names]),
        "joint_velocity_rad_s": np.array([data.sensor(f"{n}_velocity").data[0] for n in names]),
        "gyro_rad_s": data.sensor("pelvis_gyro").data.copy(),
        "specific_force_m_s2": data.sensor("pelvis_accel").data.copy(),
    }
    if include_foot_contacts:
        readings["foot_normal_force_N_right_left"] = np.array(
            [data.sensor(f"{side}_foot_touch").data[0] for side in ("r", "l")]
        )
    return readings
