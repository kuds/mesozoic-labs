"""ROS 2 sensor transport and local operator service for a fixed torso RGB-D camera.

Run as python -m environments.shared.autonomy.ros2_node --detector opencv_hog.
ROS imports stay inside main so packet handling can be tested on development machines.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import CameraGeometry, Command, OnboardFollower, OperatorServer, RGBDFrame

MAX_SKEW_S = 0.060
MAX_REQUEST_AGE_S = 0.200
WATCHDOG_PERIOD_S = 0.025


@dataclass(frozen=True)
class SensorFrames:
    camera: str = "oak_rgb_camera_optical_frame"
    imu: str = "base_link"
    left: str = "tof_left"
    right: str = "tof_right"


def image_array(msg, *, color):
    """Read registered ROS images with explicit row padding, endian and depth units."""
    h, w, stride = msg.height, msg.width, msg.step
    if h < 1 or w < 1 or h > 4096 or w > 4096:
        raise ValueError("unsupported image size")
    if color:
        if msg.encoding not in ("rgb8", "bgr8"):
            raise ValueError("RGB input requires rgb8 or bgr8")
        dtype, channels = np.dtype("u1"), 3
    else:
        if msg.encoding not in ("16UC1", "32FC1"):
            raise ValueError("optical depth requires 16UC1 millimeters or 32FC1 meters")
        dtype = np.dtype((">" if msg.is_bigendian else "<") + ("u2" if msg.encoding == "16UC1" else "f4"))
        channels = 1
    step = dtype.itemsize * channels
    if stride < w * step or len(msg.data) != h * stride:
        raise ValueError("image stride or byte count mismatch")
    if color:
        rgb = np.ndarray((h, w, 3), dtype, buffer=msg.data, strides=(stride, step, 1))
        return np.ascontiguousarray(rgb[:, :, ::-1] if msg.encoding == "bgr8" else rgb)
    depth = np.ndarray((h, w), dtype, buffer=msg.data, strides=(stride, step)).astype(np.float32)
    if msg.encoding == "16UC1":
        depth *= 0.001
    depth[(depth <= 0) | ~np.isfinite(depth)] = np.nan
    return depth


def calibrated_camera(info, geometry, optical_frame):
    """Validate the centered, rectified square-pixel projection the follower implements."""
    if info.header.frame_id != optical_frame or info.width <= 0 or info.height <= 0:
        raise ValueError("invalid rectified camera_info frame or size")
    projection = np.asarray(info.p, dtype=float).reshape(3, 4)
    f = info.height / (2 * np.tan(np.deg2rad(geometry.fovy_deg) / 2))
    ideal = np.array([[f, 0, (info.width - 1) / 2, 0], [0, f, (info.height - 1) / 2, 0], [0, 0, 1, 0]])
    # Permit calibration/cropping residuals only within explicit per-parameter tolerance.
    tolerance = np.array(
        [[0.02 * f, 0.01, 2.0, 1.0e-4], [0.01, 0.02 * f, 2.0, 1.0e-4], [1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6]]
    )
    if not np.isfinite(projection).all() or np.any(np.abs(projection - ideal) > tolerance):
        raise ValueError("calibration does not match centered rectified follower geometry")
    return info.height, info.width


def orientation(imu):
    if len(imu.orientation_covariance) != 9 or imu.orientation_covariance[0] < 0:
        raise ValueError("IMU has no orientation estimate")
    est = imu.orientation
    quat = np.array([est.x, est.y, est.z, est.w], dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or abs(norm - 1) > 0.02:
        raise ValueError("invalid torso orientation quaternion")
    x, y, z, w = quat / norm
    return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), math.asin(np.clip(2 * (w * y - z * x), -1, 1))


def range_m(msg):
    if not all(math.isfinite(k) for k in (msg.min_range, msg.max_range, msg.range)):
        return math.nan
    if msg.min_range <= 0 or not msg.min_range <= msg.range <= msg.max_range:
        return math.nan
    return float(msg.range)


def frame_from_messages(messages, *, ros_now_ns, mono_now, calibration, frames):
    rgb, depth, left, right, imu = messages
    expected = (frames.camera, frames.camera, frames.left, frames.right, frames.imu)
    if tuple(x.header.frame_id for x in messages) != expected:
        raise ValueError("unexpected sensor frame ID or unregistered depth")
    stamps = [m.header.stamp.sec * 10**9 + m.header.stamp.nanosec for m in messages]
    oldest, newest = min(stamps), max(stamps)
    age = (ros_now_ns - oldest) / 1.0e9
    if oldest <= 0 or newest > ros_now_ns or (newest - oldest) * 1.0e-9 > MAX_SKEW_S or age > MAX_REQUEST_AGE_S:
        raise ValueError("stale, future-dated or unsynchronized sensors")
    if any((m.height, m.width) != calibration for m in (rgb, depth)):
        raise ValueError("registered images must match CameraInfo resolution")
    roll, pitch = orientation(imu)
    # Keep true acquisition age across the ROS wall-clock -> monotonic boundary.
    # Stamp to the oldest contributing measurement for a conservative deadline.
    return RGBDFrame(
        image_array(rgb, color=True),
        image_array(depth, color=False),
        (range_m(left), range_m(right)),
        mono_now - age,
        roll,
        pitch,
    )


class SensorService:
    """Runs detection off the ROS executor. Only timer ticks may publish a velocity."""

    def __init__(self, follower, server, *, clock=time.monotonic):
        self.follower = follower
        self.server = server
        self.clock = clock
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.work = None
        self.pending = None
        self.stamp = -math.inf
        self.completed_stamp = -math.inf
        self.cancelled = False
        self.rejected = False
        self.error = None

    def offer(self, frame):
        if not self.rejected:
            self.pending = frame  # bounded queue: latest synchronized pair wins

    def reject(self, error):
        self.pending = None
        self.cancelled = True
        self.rejected = True
        self.error = str(error)

    def _process(self, frame):
        self.follower.update(frame, self.clock())
        self.server.publish(frame.rgb)

    def tick(self):
        """None leaves the previous stamped request to expire at the receiver."""
        now = self.clock()
        if self.work is not None:
            if now - self.stamp > MAX_REQUEST_AGE_S:
                self.reject("inference exceeded the sensor-to-command deadline")
            if not self.work.done():
                # Never block the watchdog on detector/UI locks. A fresh request
                # is never re-stamped during inference.
                return Command() if self.cancelled or now - self.completed_stamp > MAX_REQUEST_AGE_S else None
            try:
                self.work.result()
                self.completed_stamp = self.stamp
            except Exception as exc:
                self.reject(exc)
            self.work = None
        if self.rejected:
            self.follower.stop()
            self.rejected = self.cancelled = False
            return Command()
        cmd = self.follower.watchdog(now)
        if now - self.completed_stamp > MAX_REQUEST_AGE_S:
            # A stale armed command requires a fresh selection. Disarmed idle
            # status is left visible to allow the first operator selection.
            if self.follower.snapshot()["armed"]:
                self.follower.stop()
            cmd = Command()
        frame, self.pending = self.pending, None
        if frame is not None:
            if now - frame.timestamp > MAX_REQUEST_AGE_S:
                self.follower.stop()
                return Command()
            self.stamp = frame.timestamp
            self.work = self.pool.submit(self._process, frame)
        return cmd

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.server.close()


def main():
    import message_filters
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.utilities import remove_ros_args
    from sensor_msgs.msg import CameraInfo, Image, Imu, Range

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=["opencv_hog"], required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fovy", type=float, default=46.5)
    parser.add_argument("--camera-frame", default=SensorFrames.camera)
    parser.add_argument("--imu-frame", default=SensorFrames.imu)
    parser.add_argument("--left-frame", default=SensorFrames.left)
    parser.add_argument("--right-frame", default=SensorFrames.right)
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    if not 5 < args.fovy < 150:
        parser.error("fovy must be the calibrated rectified vertical field of view in degrees")
    frames = SensorFrames(args.camera_frame, args.imu_frame, args.left_frame, args.right_frame)
    geometry = CameraGeometry(fovy_deg=args.fovy)
    import cv2

    if not hasattr(cv2, "HOGDescriptor") or not hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
        parser.error("opencv_hog requires OpenCV 4.x with the HOG pedestrian classifier in objdetect")
    follower = OnboardFollower(geometry, detector=args.detector)
    rclpy.init()
    node = Node("rex_follow")
    server = None
    service = None
    publisher = None
    try:
        if node.get_parameter("use_sim_time").value:
            raise ValueError("Jetson transport requires hardware timestamps aligned to ROS wall time")
        server = OperatorServer(follower, port=args.port).start()
        service = SensorService(follower, server)
        publisher = node.create_publisher(TwistStamped, "follow/cmd_vel", 1)
        calibration = None

        def set_camera(info):
            nonlocal calibration
            try:
                calibration = calibrated_camera(info, geometry, frames.camera)
            except ValueError as exc:
                calibration = None
                service.reject(exc)

        def receive(*messages):
            if calibration is None:
                service.reject("waiting for calibrated camera_info")
                return
            try:
                frame = frame_from_messages(
                    messages,
                    ros_now_ns=node.get_clock().now().nanoseconds,
                    mono_now=time.monotonic(),
                    calibration=calibration,
                    frames=frames,
                )
                service.offer(frame)
            except (ValueError, TypeError, BufferError) as exc:
                service.reject(exc)

        def send(cmd):
            out = TwistStamped()
            out.header.frame_id = "base_link"
            out.header.stamp = node.get_clock().now().to_msg()
            out.twist.linear.x = float(cmd.forward_mps)
            out.twist.angular.z = float(cmd.yaw_rps)
            publisher.publish(out)

        def timer():
            cmd = service.tick()
            if cmd is not None:
                send(cmd)
            if service.error:
                node.get_logger().warn(service.error, throttle_duration_sec=2.0)
                service.error = None

        node.create_subscription(CameraInfo, "camera/info", set_camera, qos_profile_sensor_data)
        subscriptions = [
            message_filters.Subscriber(node, cls, topic, qos_profile=qos_profile_sensor_data)
            for cls, topic in (
                (Image, "camera/rgb"),
                (Image, "camera/aligned_depth"),
                (Range, "tof/left"),
                (Range, "tof/right"),
                (Imu, "imu/data"),
            )
        ]
        sync = message_filters.ApproximateTimeSynchronizer(subscriptions, 8, MAX_SKEW_S)
        sync.registerCallback(receive)
        node.create_timer(WATCHDOG_PERIOD_S, timer)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if publisher is not None and rclpy.ok():
            send(Command())
        if service is not None:
            service.close()
        elif server is not None:
            server.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
