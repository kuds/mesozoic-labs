"""Robot-side RGB-D observations, human proposals and appearance association.

Coordinates are rectified optical depth in metres (+z forward, +x right, +y down).
The camera is fixed to the torso. No simulator, render object IDs or world poses
are available to this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraGeometry:
    fovy_deg: float = 46.5
    pitch_down_rad: float = -0.2094395102
    height_m: float = 0.366

    # Measured in the design pelvis frame. Recalibrate on the assembled hardware.
    lens_x_m: float = 0.220
    lens_z_m: float = 0.006
    tof_x_m: float = 0.205
    tof_abs_y_m: float = 0.065
    tof_z_m: float = -0.100
    tof_down_rad: float = np.deg2rad(22.0)
    tof_out_rad: float = np.deg2rad(8.0)

    def lens_height(self, roll, pitch):
        return (
            self.height_m - self.lens_z_m - np.sin(pitch) * self.lens_x_m + np.cos(pitch) * np.cos(roll) * self.lens_z_m
        )

    def tof_floor_residuals(self, tof, roll, pitch):
        """Height of each return over the assumed level floor; NaN propagates."""
        signs = np.array([1.0, -1.0])
        gravity = np.array([-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)])
        sensor_z = (
            self.height_m
            - self.lens_z_m
            + gravity[0] * self.tof_x_m
            + gravity[1] * signs * self.tof_abs_y_m
            + gravity[2] * self.tof_z_m
        )
        beam_z = (
            gravity[0] * np.cos(self.tof_down_rad) * np.cos(self.tof_out_rad)
            + gravity[1] * signs * np.cos(self.tof_down_rad) * np.sin(self.tof_out_rad)
            - gravity[2] * np.sin(self.tof_down_rad)
        )
        return sensor_z + np.asarray(tof) * beam_z

    def points(self, depth: np.ndarray, roll: float = 0.0, pitch: float = 0.0):
        """Back-project into gravity-aligned forward, left, height coordinates.

        roll/pitch are current torso IMU orientation (positive roll left side up,
        positive pitch nose down), radians. Neglects short-period heave.
        """
        h, w = depth.shape
        f = h / (2 * np.tan(np.deg2rad(self.fovy_deg) / 2))
        right = depth * ((np.arange(w) + 0.5 - w / 2) / f)
        down = depth * ((np.arange(h)[:, None] + 0.5 - h / 2) / f)
        cp, sp = np.cos(self.pitch_down_rad), np.sin(self.pitch_down_rad)
        forward = cp * depth - sp * down
        up = -sp * depth - cp * down
        left = -right * np.cos(roll) - up * np.sin(roll)
        up = -right * np.sin(roll) + up * np.cos(roll)
        return (
            np.cos(pitch) * forward + np.sin(pitch) * up,
            left,
            self.lens_height(roll, pitch) - np.sin(pitch) * forward + np.cos(pitch) * up,
        )


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    tof: tuple[float, float]
    timestamp: float
    roll_rad: float = 0.0
    pitch_rad: float = 0.0


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    forward: float
    left: float
    appearance: np.ndarray
    confidence: float


def mask_features(mask: np.ndarray, width_m: float, height_m: float) -> np.ndarray:
    occupancy = cv2.resize(mask.astype(np.float32), (8, 14), interpolation=cv2.INTER_AREA)
    return np.concatenate((occupancy.ravel(), np.array([width_m, height_m])))


def appearance(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clothing color distributions in two height bands; exposure-normalized RGB."""
    histograms = []
    for pixels, selected in zip(np.array_split(rgb, 2), np.array_split(mask, 2), strict=True):
        colors = pixels[selected].astype(float)
        chroma = colors / np.maximum(1.0, colors.sum(axis=1, keepdims=True))
        keys = np.clip((chroma[:, :2] * 10).astype(int), 0, 9)
        hist = np.bincount(10 * keys[:, 0] + keys[:, 1], minlength=100)
        histograms.append(hist / max(1, int(hist.sum())))
    return np.concatenate(histograms) / 2


def overlap(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - x * y
    return x * y / max(1, union)


class DepthPersonDetector:
    """Logistic classifier on metric depth silhouettes, trained on synthetic crops."""

    def __init__(self):
        path = Path(__file__).parent / "data" / "silhouette_v1.json"
        model = json.loads(path.read_text())
        self.weights = np.asarray(model["weights"])
        self.bias = float(model["bias"])

    def detect(self, frame: RGBDFrame, points, geometry: CameraGeometry):
        forward, left, height = points
        rows, cols = frame.depth.shape
        f = rows / (2 * np.tan(np.deg2rad(geometry.fovy_deg) / 2))
        valid = (height > 0.06) & (height < 2.35) & (forward > 0.5) & (forward < 5.5)
        # Only real returns count. Overlapping metric slabs separate people at different
        # depths and tolerate variation across a torso. NMS removes duplicate proposals.
        detections = []
        for near in np.arange(0.4, 5.5, 0.30):
            mask = (valid & (forward >= near) & (forward < near + 0.64)).astype(np.uint8)
            if not mask.any():
                continue
            kernel = np.ones((3, max(3, int(0.17 * f / (near + 0.30)) | 1)), np.uint8)
            joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
            for i in range(1, count):
                x, y, w, h, area = stats[i]
                if h < 12 or w < 4 or area < 40:
                    continue
                ys, xs = slice(y, y + h), slice(x, x + w)
                silhouette = (labels[ys, xs] == i) & (mask[ys, xs] != 0)
                ranges = forward[ys, xs][silhouette]
                if not ranges.size:
                    continue
                z = float(np.median(ranges))
                width_m, height_m = w * z / f, h * z / f
                sides = left[ys, xs][silhouette]
                # A flat side wall cut by a depth slab has apparent image width
                # but negligible installed-frame lateral extent.
                if np.ptp(sides) < 0.16:
                    continue
                if not (0.16 < width_m < 1.08 and 0.59 < height_m < 2.35):
                    continue
                features = mask_features(silhouette, width_m, height_m)
                logit = float(features @ self.weights + self.bias)
                confidence = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
                if confidence < 0.70:
                    continue
                detections.append(
                    Detection(
                        (int(x), int(y), int(x + w), int(y + h)),
                        z,
                        float(np.median(left[ys, xs][silhouette])),
                        appearance(frame.rgb[ys, xs], silhouette),
                        float(confidence),
                    )
                )

        def area(det):
            a, b, c, d = det.box
            return (c - a) * (d - b)

        # Neighboring slabs can also propose an entire torso nested inside a
        # whole person at the SAME range. Deduplicate that fragment before the
        # tracker sees it as a second human. Different ranges stay separate.
        kept: list[Detection] = []
        for det in sorted(detections, key=area, reverse=True):
            a, b, c, d = det.box
            for other in kept:
                if abs(other.forward - det.forward) > 0.43:
                    continue
                x, y, u, v = other.box
                shared = max(0, min(c, u) - max(x, a)) * max(0, min(d, v) - max(y, b))
                if shared / area(det) > 0.75 or overlap(det.box, other.box) > 0.35:
                    break
            else:
                kept.append(det)
        return kept


class HOGPersonDetector:
    """Optional upstream OpenCV pretrained 64x128 pedestrian SVM for real RGB."""

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: RGBDFrame, points, geometry: CameraGeometry):
        forward, left, _ = points
        if frame.rgb.shape[0] < 128 or frame.rgb.shape[1] < 64:
            return []
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        boxes, scores = self.hog.detectMultiScale(bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
        results = []
        for (x, y, w, h), score in zip(boxes, scores, strict=True):
            y0, y1 = max(0, y), min(y + h, frame.rgb.shape[0])
            x0, x1 = max(0, x), min(x + w, frame.rgb.shape[1])
            roi = forward[y0:y1, x0:x1]
            center = roi[h // 4 : 3 * h // 4, w // 3 : 2 * w // 3]
            good = center[np.isfinite(center) & (center > 0.45) & (center < 5.5)]
            if len(good) < max(8, center.size // 4):
                continue
            z = float(np.median(good))
            mask = np.abs(roi - z) < 0.38
            results.append(
                Detection(
                    (int(x0), int(y0), int(x1), int(y1)),
                    z,
                    float(np.median(left[y0:y1, x0:x1][mask])),
                    appearance(frame.rgb[y0:y1, x0:x1], mask),
                    float(1 / (1 + np.exp(-float(score)))),
                )
            )
        return results
