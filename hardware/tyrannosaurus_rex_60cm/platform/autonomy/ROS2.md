# Running person follow on a ROS 2 Jetson

The robot can serve the selection page, run an OpenCV pedestrian detector on registered RGB-D data, track the selected person, and issue velocity **requests** to a separate locomotion bridge. The adapter publishes only forward speed and yaw rate in `base_link`. Leave the requester disconnected from actuators for the first bench test.

## Run the service

Use a Jetson image with ROS 2, sensor drivers, NumPy and **OpenCV 4.x with objdetect** in the same Python interpreter. The detector uses OpenCV's bundled pedestrian SVM. The simulator's learned depth-shape classifier is separately tested against simulated people. The transport requires hardware support for the fixed torso camera and two forward/down ToF sensors. Use the ROS system Python, or a system-site-packages environment retaining the ROS Python ABI. From the checkout root on a provisioned Ubuntu ROS system:

```sh
source /opt/ros/"$ROS_DISTRO"/setup.bash
sudo apt install "ros-${ROS_DISTRO}-message-filters" "ros-${ROS_DISTRO}-sensor-msgs" "ros-${ROS_DISTRO}-geometry-msgs" python3-opencv python3-numpy
python3 -c 'import rclpy, message_filters, cv2; print(cv2.__version__); h=cv2.HOGDescriptor(); h.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())'
python3 -m environments.shared.autonomy.ros2_node --detector opencv_hog --port 8765 \
  --ros-args -r camera/rgb:=/rex/rectified/rgb -r camera/aligned_depth:=/rex/rectified/depth \
  -r camera/info:=/rex/rectified/camera_info -r tof/left:=/rex/tof_left/range \
  -r tof/right:=/rex/tof_right/range -r imu/data:=/rex/base_imu/filtered \
  -r follow/cmd_vel:=/bench/follow_request
```

Set `ROS_DISTRO` to the installed distribution before sourcing if the shell has no ROS environment. The example topic names are remap targets. Configure the actual camera driver to publish the same rectified image grid and transformed optical depth, and the IMU node to publish world-from-torso orientation with ROS FLU body axes and ENU gravity. Print topic types/frame IDs on the robot first. Adjust `--camera-frame`, `--imu-frame`, `--left-frame`, `--right-frame` to match validated frames. Overriding a name does not transform measurements. Stop and recalibrate if the sensor mounting differs. The default camera positions, ToF angles and head occlusion limitations are recorded in the [engineering handoff](../engineering/README.md).

| Relative topic | ROS 2 type | Contract |
|---|---|---|
| camera/rgb | sensor_msgs/msg/Image | Rectified rgb8 or bgr8, registered color optical frame. |
| camera/aligned_depth | sensor_msgs/msg/Image | Same optical frame and resolution. 16UC1 mm or 32FC1 meters along optical Z. Invalid pixels propagate as unknown. |
| camera/info | sensor_msgs/msg/CameraInfo | Projection matrix for the registered grid. Square pixels, centered principal point within 2 px, focal lengths within 2% of configured VFOV. Use `--fovy` for the calibrated rectified crop. |
| tof/left, tof/right | sensor_msgs/msg/Range | Distinct physical frame IDs, acquisition timestamps and meter limits. No-return and saturated ranges lead to a stop. |
| imu/data | sensor_msgs/msg/Imu | Unit quaternion, orientation covariance available. Rotate into the torso axes upstream. |
| follow/cmd_vel | geometry_msgs/msg/TwistStamped | Forward m/s and yaw rad/s only. Enforce a short expiry (≤200 ms) and command arbitration in the receiving bridge. |

The node requires synchronized clocks at acquisition. Approximate synchronization permits at most 60 ms between the five sensor messages and a maximum 200 ms from the oldest measurement to a motion request. A worker handles OpenCV and JPEG encoding. A separate 25 ms control timer emits zero when the worker or the incoming stream exceeds the deadline. Old detections are discarded after a stall and the operator must reselect. Downstream protection must stop on ROS node death or loss of these messages. A hardware traction disconnect and an independent embedded watchdog remain physical commissioning tasks. [ROS Image](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/Image.msg) · [CameraInfo](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CameraInfo.msg) · [Imu](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/Imu.msg) · [Range](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/Range.msg)

On the operator machine, forward the loopback service using `ssh -N -L 8765:127.0.0.1:8765 USER@JETSON`, then visit **http://127.0.0.1:8765**. Select a visible person and press Start. UI ownership and heartbeat, stop, ambiguity and reacquisition follow the same controller as the simulation. The external page sends discrete selection/control events and owner heartbeats.

## Bench checks

1. With output remapped to `/bench/follow_request`, inspect `ros2 topic echo /bench/follow_request` and record/bag the inputs, capture stamps and output. Verify calibrated orientation and distances using surveyed targets. Measure RGB/depth skew, per-frame acquisition-to-publish age, lost frames, USB load, Jetson temperature and latency under full compute load.
2. Select the second of two people, switch their crossing order, occlude the selected person, block the path, cover each range channel, and close the operator page. Verify the published zero and fresh-selection rules from the recorded timestamps. Stop the process and measure the receiver's independent command expiry.
3. Run without actuator power on a fixed stand before controlled locomotion testing. A selection-page and transport pass alone cannot qualify a walking policy or prove avoidance in the camera/ToF blind zones.

On the development host the standalone packet, calibration and stalled-worker tests ran with NumPy and OpenCV 5.0.0. This OpenCV build lacks the HOG descriptor entry point. ROS 2 DDS, sensor drivers, pretrained HOG inference and Jetson timing remain unverified on target. Reproduce the packet tests with `python -m pytest -q environments/shared/tests/test_robot_autonomy_transport.py`.

