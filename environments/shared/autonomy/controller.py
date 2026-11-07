"""Local person selection, identity checks, watchdog and range-based steering."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from .perception import CameraGeometry, DepthPersonDetector, Detection, HOGPersonDetector, RGBDFrame, overlap


@dataclass(frozen=True)
class Command:
    forward_mps: float = 0.0
    yaw_rps: float = 0.0
    state: str = "stopped"


@dataclass
class Track:
    id: int
    detection: Detection
    last_seen: float
    frames: int = 1


def color_distance(a, b):
    return 1.0 - float(np.minimum(a, b).sum())


def steer(points, tof, target: Detection, shape, geometry=None, roll=0.0, pitch=0.0, previous=None, dt=0.1):
    forward, left, height = points
    h, w = shape
    geometry = geometry or CameraGeometry()
    if len(tof) != 2 or any(not np.isfinite(z) or not 0.32 <= z <= 2.0 for z in tof):
        return Command(state="near_range_stop")
    # Compare two measured hits against the level indoor floor using the torso
    # IMU. A raw fixed range threshold misclassifies safe floor during pitch-up.
    if np.any(np.abs(geometry.tof_floor_residuals(tof, roll, pitch)) > 0.085):
        return Command(state="near_range_stop")
    valid = np.isfinite(forward) & (forward > 0.39)
    # Test depth only where the calibrated camera should see floor within 5 m,
    # and keep the stop if a sector loses that view under large body pitch/roll.
    lens_z = geometry.lens_height(roll, pitch)
    unit_z = geometry.points(np.ones(shape), roll, pitch)[2] - lens_z
    expected = unit_z < -lens_z / 5.0
    sectors = np.array_split(np.arange(w), 3)
    for indices in sectors:
        mask = expected[:, indices]
        if mask.sum() < max(8, 0.001 * h * w) or valid[:, indices][mask].mean() < 0.70:
            return Command(state="depth_unknown")
    # Sample measured points across body height. Floor returns and points above
    # the crowned head may be traversable; sensor uncertainty expands this band.
    close = valid & (forward < 2.0) & (height > 0.045) & (height < 0.71)
    x, y = forward[close], left[close]
    # Plan pelvis motion against a padded hull using the visible returns.
    # Nominal measured tip and snout are -0.575/+0.435 m. The +/-0.29 m
    # hull also includes reaction/foot placement, grid and sample allowances.
    x = x + geometry.lens_x_m
    if x.size:
        # Compress the measured obstacle surface into 25 mm cells.
        grid = np.unique(np.rint(np.column_stack((x, y)) / 0.025).astype(int), axis=0) * 0.025
        x, y = grid.T
    goal = math.atan2(target.left, target.forward + geometry.lens_x_m)
    distance = math.hypot(target.forward, target.left)
    if distance <= 1.95:
        return Command(state="at_distance")
    # Match the candidate gait's tested curved-walk command range.
    speed = min(0.12, max(0.08, 0.32 * (distance - 1.95)))
    previous = previous or Command()
    # Score a 0.27 m arc joining a straight look-ahead segment. Immediate
    # stops override start smoothing. A rate-limited speed scales yaw by
    # the same factor so the commanded path retains the checked curvature.
    speed = min(speed, previous.forward_mps + 0.45 * min(max(dt, 0), 0.2))
    horizon = min(1.30, max(0.65, distance - 0.80))
    feasible = []
    arc = 0.27
    for angle in np.deg2rad([-31, -24, -16, -8, 0, 8, 16, 24, 31]):
        curvature = angle / arc
        if x.size:
            safe = True
            for travel in np.arange(0, horizon + 0.05, 0.05):
                bent = min(travel, arc)
                yaw = curvature * bent
                dx = bent * np.sinc(yaw / np.pi) + (travel - bent) * np.cos(yaw)
                dy = 0.5 * bent * yaw * np.sinc(yaw / (2 * np.pi)) ** 2 + (travel - bent) * np.sin(yaw)
                local_x = (x - dx) * np.cos(yaw) + (y - dy) * np.sin(yaw)
                local_y = -(x - dx) * np.sin(yaw) + (y - dy) * np.cos(yaw)
                if np.any((local_x > -0.62) & (local_x < 0.49) & (np.abs(local_y) < 0.29)):
                    safe = False
                    break
            if not safe:
                continue
        rate = speed * curvature
        feasible.append((abs(angle - goal) + 0.12 * abs(angle) + 0.12 * abs(rate - previous.yaw_rps), angle, rate))
    if not feasible:
        return Command(state="blocked")
    _, chosen, turn = min(feasible)
    return Command(speed, float(turn), "avoiding" if abs(chosen - goal) > 0.20 else "following")


class OnboardFollower:
    def __init__(self, geometry=None, detector="depth_shape"):
        self.geometry = geometry or CameraGeometry()
        if detector not in {"depth_shape", "opencv_hog"}:
            raise ValueError("Unknown detector")
        self.detector = DepthPersonDetector() if detector == "depth_shape" else HOGPersonDetector()
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._chosen: int | None = None
        self._reference: np.ndarray | None = None
        self._armed = False
        self._heartbeat = -np.inf
        self._stamp = -np.inf
        self._shape = (0, 0)
        self._missing = False
        self._mono_last = -np.inf
        self.command = Command()
        self._lock = threading.RLock()

    def heartbeat(self, now_s):
        with self._lock:
            if self._armed and now_s - self._heartbeat > 0.80:
                self._halt("operator_disconnected")
            self._heartbeat = now_s

    def select(self, track_id, now_s):
        with self._lock:
            track = self.tracks.get(track_id)
            if track is None or now_s - track.last_seen > 0.30 or track.frames < 2:
                raise ValueError("Person is no longer visible; refresh the camera view.")
            self._chosen = track.id
            self._reference = track.detection.appearance.copy()
            self._armed = False
            self._missing = False
            self.command = Command(state="selected")

    def start(self, now_s):
        with self._lock:
            track = self.tracks.get(self._chosen) if self._chosen is not None else None
            if track is None or now_s - track.last_seen > 0.30 or now_s - self._heartbeat > 0.80:
                raise ValueError("A live operator and a visible selected person are required.")
            self._armed = True
            self.command = Command(state="starting")

    def _halt(self, reason):
        self._armed = False
        self._chosen = None
        self._reference = None
        self._missing = False
        self.command = Command(state=reason)

    def stop(self):
        with self._lock:
            self._halt("stopped")

    def watchdog(self, now_s):
        """Call at the motor-loop rate even when no camera frame arrives."""
        with self._lock:
            if self._armed and (now_s < self._mono_last or now_s - self._heartbeat > 0.80):
                self._halt("operator_disconnected")
            elif self._armed and now_s - self._stamp > 0.35:
                self._halt("sensor_stale")
            return self.command

    def _associate(self, detections, stamp):
        self.tracks = {i: t for i, t in self.tracks.items() if stamp - t.last_seen < 0.9}
        options = {}
        for i, track in self.tracks.items():
            prior = track.detection
            signature = self._reference if i == self._chosen else prior.appearance
            for j, det in enumerate(detections):
                dist = math.hypot(det.left - prior.left, det.forward - prior.forward)
                color = color_distance(det.appearance, signature)
                if dist < 0.24 + 1.45 * (stamp - track.last_seen) and color < 0.46:
                    options[i, j] = dist + 0.95 * color
        # Never continue through competing plausible associations. Two identical
        # people may be indistinguishable even with perfect instance boxes.
        ambiguous = False
        if self._chosen in self.tracks:
            scores = sorted((v, j) for (i, j), v in options.items() if i == self._chosen)
            if scores:
                cost, best = scores[0]
                ambiguous = (len(scores) > 1 and scores[1][0] - cost < 0.20) or any(
                    i != self._chosen and j == best and v < cost + 0.20 for (i, j), v in options.items()
                )
            prior = self.tracks[self._chosen]
            ambiguous |= any(
                other.id != prior.id
                and overlap(other.detection.box, prior.detection.box) > 0.16
                and color_distance(other.detection.appearance, self._reference) < 0.40
                for other in self.tracks.values()
            )
        matched_ids: set[int] = set()
        used: set[int] = set()
        for (i, j), _ in sorted(options.items(), key=lambda item: item[1]):
            if i in matched_ids or j in used:
                continue
            track = self.tracks[i]
            track.frames = track.frames + 1 if track.last_seen == self._stamp and stamp - track.last_seen < 0.28 else 1
            track.detection, track.last_seen = detections[j], stamp
            matched_ids.add(i)
            used.add(j)
        for j, det in enumerate(detections):
            if j not in used:
                self.tracks[self._next_id] = Track(self._next_id, det, stamp)
                self._next_id += 1
        return ambiguous

    def update(self, frame: RGBDFrame, now_s: float):
        with self._lock:
            if (
                not np.isfinite([frame.timestamp, now_s, frame.roll_rad, frame.pitch_rad]).all()
                or not 0 <= now_s - frame.timestamp <= 0.35
                or frame.timestamp <= self._stamp
                or now_s < self._mono_last
            ):
                self._halt("sensor_stale")
                return self.command
            if frame.rgb.ndim != 3 or frame.rgb.shape != (*frame.depth.shape, 3) or frame.depth.ndim != 2:
                self._halt("sensor_invalid")
                return self.command
            if self._armed and frame.timestamp - self._stamp > 0.35:
                self._halt("sensor_stale")
            self._mono_last = now_s
            self._shape = frame.depth.shape
            points = self.geometry.points(frame.depth, frame.roll_rad, frame.pitch_rad)
            detections = self.detector.detect(frame, points, self.geometry)
            ambiguous = self._associate(detections, frame.timestamp)
            period = frame.timestamp - self._stamp
            self._stamp = frame.timestamp
            if ambiguous:
                self._halt("ambiguous_reselect")
            self.watchdog(now_s)
            if not self._armed:
                return self.command
            track = self.tracks.get(self._chosen) if self._chosen is not None else None
            if track is None or frame.timestamp - track.last_seen > 0.75:
                self._halt("lost_reselect")
            elif track.last_seen != frame.timestamp:
                self._missing = True
                self.command = Command(state="occluded")
            elif self._missing and track.frames < 3:
                self.command = Command(state="confirming")
            else:
                self._missing = False
                self.command = steer(
                    points,
                    frame.tof,
                    track.detection,
                    self._shape,
                    self.geometry,
                    frame.roll_rad,
                    frame.pitch_rad,
                    self.command,
                    period,
                )
            return self.command

    def snapshot(self):
        with self._lock:
            return {
                "state": self.command.state,
                "selected": self._chosen,
                "armed": self._armed,
                "timestamp": float(self._stamp) if np.isfinite(self._stamp) else None,
                "width": self._shape[1],
                "height": self._shape[0],
                "forward_mps": self.command.forward_mps,
                "yaw_rps": self.command.yaw_rps,
                "tracks": [
                    {
                        "id": i,
                        "box": list(t.detection.box),
                        "forward_m": round(t.detection.forward, 2),
                        "confidence": round(t.detection.confidence, 3),
                    }
                    for i, t in self.tracks.items()
                    if t.last_seen == self._stamp and t.frames >= 2
                ],
            }
