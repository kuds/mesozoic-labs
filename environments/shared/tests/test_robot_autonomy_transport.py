from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("cv2")

from environments.shared.autonomy import CameraGeometry, Command, OnboardFollower
from environments.shared.autonomy.ros2_node import (
    MAX_REQUEST_AGE_S,
    SensorFrames,
    SensorService,
    calibrated_camera,
    frame_from_messages,
)


def header(frame, ns=1_000_000_000):
    sec, nanosec = divmod(ns, 10**9)
    return SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def sensors():
    frames = SensorFrames()
    img = dict(height=2, width=3, header=header(frames.camera), is_bigendian=True)
    rgb_bytes = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 255, 255, 255])
    rgb = SimpleNamespace(**img, encoding="bgr8", step=12, data=rgb_bytes * 2)
    z = np.array([[1234, 0, 3111, 65535], [1523, 1789, 2123, 65535]], dtype=">u2")
    depth = SimpleNamespace(**img, encoding="16UC1", step=8, data=z.tobytes())
    left = SimpleNamespace(header=header(frames.left), range=0.695, min_range=0.04, max_range=4.0)
    right = SimpleNamespace(header=header(frames.right), range=0.700, min_range=0.04, max_range=4.0)
    imu = SimpleNamespace(
        header=header(frames.imu, 991_000_000),
        orientation_covariance=[0.001] * 9,
        orientation=SimpleNamespace(x=0.0, y=np.sin(0.11 / 2), z=0.0, w=np.cos(0.11 / 2)),
    )
    return rgb, depth, left, right, imu


def convert(messages):
    return frame_from_messages(
        messages, ros_now_ns=1_050_000_000, mono_now=51.5, calibration=(2, 3), frames=SensorFrames()
    )


def test_padded_big_endian_and_clock_alignment():
    packet = convert(sensors())
    np.testing.assert_array_equal(packet.rgb[1, 0], [3, 2, 1])
    np.testing.assert_allclose(packet.depth[1], [1.523, 1.789, 2.123])
    assert np.isnan(packet.depth[0, 1])
    assert packet.timestamp == pytest.approx(51.441)
    assert packet.pitch_rad == pytest.approx(0.11)
    # Feed a decoded frame through the real robot-side pipeline with local OpenCV.
    assert OnboardFollower().update(packet, 51.5).forward_mps == 0


def test_float_meter_depth_and_invalid_range():
    messages = sensors()
    depth = messages[1]
    depth.step, depth.encoding = 16, "32FC1"
    depth.data = np.array([[1.25, -1, np.inf, 91], [0, np.nan, 2.5, 84]], dtype=">f4").tobytes()
    messages[2].range = np.inf
    packet = convert(messages)
    assert np.isnan(packet.tof[0])
    np.testing.assert_allclose(packet.depth, [[1.25, np.nan, np.nan], [np.nan, np.nan, 2.5]], equal_nan=True)


@pytest.mark.parametrize("bad", ["stale", "skew", "future", "optical", "tof_sides", "imu_cov", "quaternion", "stride"])
def test_reject_mismatched_or_unavailable_packets(bad):
    messages = list(sensors())
    if bad == "stale":
        for m in messages:
            m.header.stamp = header("unused", 790_000_000).stamp
    elif bad == "skew":
        messages[3].header.stamp = header("unused", 920_000_000).stamp
    elif bad == "future":
        messages[4].header.stamp = header("unused", 1_051_000_000).stamp
    elif bad == "optical":
        messages[1].header = header("depth_optical_frame")
    elif bad == "tof_sides":
        messages[2], messages[3] = messages[3], messages[2]
    elif bad == "imu_cov":
        messages[4].orientation_covariance[0] = -1
    elif bad == "quaternion":
        messages[4].orientation.x = 29.0
    elif bad == "stride":
        messages[0].step = 6
    with pytest.raises(ValueError):
        convert(messages)


def test_wrong_calibrated_center_fails():
    focal = 400 / (2 * np.tan(np.deg2rad(46.5) / 2))
    info = SimpleNamespace(
        header=header("lens"), width=640, height=400, p=[focal, 0, 319.5, 0, 0, focal, 199.5, 0, 0, 0, 1, 0]
    )
    assert calibrated_camera(info, CameraGeometry(), "lens") == (400, 640)
    info.p[2] += 12.0
    with pytest.raises(ValueError):
        calibrated_camera(info, CameraGeometry(), "lens")


def test_stalled_inference_cannot_renew_old_velocity_or_rearm():
    now = [10.0]
    follower, server = Mock(), Mock()
    service = SensorService(follower, server, clock=lambda: now[0])
    try:
        service.stamp = 10.0
        service.completed_stamp = 9.9
        service.work = Future()
        assert service.tick() is None
        now[0] = 10.11
        assert service.tick() == Command()
        now[0] = 10.0 + MAX_REQUEST_AGE_S + 0.01
        assert service.tick() == Command()
        service.work.set_result(None)
        assert service.tick() == Command()
        follower.stop.assert_called_once()
        follower.watchdog.assert_not_called()
    finally:
        service.close()
