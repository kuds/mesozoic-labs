"""Fit the small RGB-D silhouette classifier without external weights/data downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from environments.shared.autonomy.perception import mask_features

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "environments/shared/autonomy/data/silhouette_v1.json"


def sample(rng, person):
    # Draw metric front-view shapes, then project and crop to simulate the low camera.
    h = rng.uniform(1.38, 1.96) if person else rng.uniform(0.62, 2.2)
    w = rng.uniform(0.35, 0.69) if person else rng.uniform(0.19, 1.03)
    mask = np.zeros((180, 112), np.uint8)
    cx = 56
    if person:
        lean = int(rng.uniform(-9, 9))
        swing = int(rng.uniform(1, 15))
        cv2.ellipse(mask, (cx + lean, 20), (rng.integers(8, 14), 17), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (cx + lean // 2, 60), (rng.integers(17, 25), 37), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (cx, 98), (19, 15), 0, 0, 360, 1, -1)
        for sign in (-1, 1):
            cv2.line(
                mask,
                (cx + lean // 2 + sign * 17, 42),
                (cx + sign * rng.integers(20, 34), rng.integers(75, 104)),
                1,
                rng.integers(6, 13),
            )
            knee = (cx + sign * rng.integers(6, 21), 130)
            toe = (cx + sign * (8 + swing), rng.integers(162, 177))
            cv2.line(mask, (cx + sign * 9, 94), knee, 1, rng.integers(11, 19))
            cv2.line(mask, knee, toe, 1, rng.integers(7, 14))
        if rng.random() < 0.68:
            # Alternate mannequin family. Project anatomical volumes through random
            # azimuths to include side-on bodies and apparent overlap of the legs.
            mask.fill(0)
            yaw = rng.uniform(-np.pi, np.pi)
            shoulder = rng.uniform(0.175, 0.24)
            hip = rng.uniform(0.080, 0.116)
            radii = rng.uniform(0.89, 1.12)

            def volume(x, y, z, rx, ry, rz):
                projected_radius = 100 * radii * np.hypot(rx * np.sin(yaw), ry * np.cos(yaw))
                cx = int(56 + 100 * (-x * np.sin(yaw) + y * np.cos(yaw)))
                cy = int(177 - 100 * z)
                cv2.ellipse(
                    mask, (cx, cy), (max(1, round(projected_radius)), max(1, round(100 * rz))), 0, 0, 360, 1, -1
                )

            volume(0, 0, 1.60, 0.103, 0.087, 0.135)
            volume(0, 0, 1.46, 0.06, 0.06, 0.069)
            volume(0, 0, 1.18, 0.15, shoulder, 0.31)
            volume(0, 0, 0.92, 0.14, 0.17, 0.15)
            for sign in (-1, 1):
                # Cylindrical trousers often form an almost continuous central column.
                for z in np.linspace(0.49, 0.87, 8):
                    volume(0, sign * hip, z, 0.087, 0.08, 0.088)
                for z in np.linspace(0.12, 0.46, 7):
                    volume(sign * 0.017, sign * hip, z, 0.064, 0.061, 0.065)
                volume(0.042 + sign * 0.018, sign * hip, 0.07, 0.135, 0.063, 0.064)
                for z in np.linspace(1.15, 1.33, 4):
                    volume(0, sign * shoulder, z, 0.07, 0.067, 0.069)
                for z in np.linspace(0.93, 1.14, 4):
                    volume(0.01, sign * (shoulder + 0.024), z, 0.049, 0.047, 0.049)
            w = 2 * np.hypot(0.17 * np.sin(yaw), (shoulder + 0.072) * np.cos(yaw)) * radii
    else:
        kind = rng.integers(6)
        if kind == 0:
            cv2.rectangle(mask, (20, 8), (93, 175), 1, -1)
        elif kind == 1:
            cv2.ellipse(mask, (55, 89), (rng.integers(10, 40), 84), 0, 0, 360, 1, -1)
        elif kind == 2:
            cv2.rectangle(mask, (18, 28), (95, 75), 1, -1)
            for x in (23, 88):
                cv2.line(mask, (x, 70), (x, 175), 1, 5)
        elif kind == 3:
            cv2.line(mask, (55, 7), (55, 174), 1, 6)
            for y in (42, 78, 125):
                cv2.line(mask, (18, y), (95, y), 1, 7)
        elif kind == 4:
            cv2.rectangle(mask, (31, 95), (79, 175), 1, -1)
            for _ in range(7):
                cv2.ellipse(mask, (rng.integers(25, 85), rng.integers(20, 93)), (15, 23), 0, 0, 360, 1, -1)
        else:
            for x in (29, 80):
                cv2.rectangle(mask, (x - 4, 8), (x + 4, 175), 1, -1)
    if rng.random() < 0.36:  # Pedestal obscuring the lower body.
        mask = mask[: rng.integers(84, 155)]
    else:
        mask = mask[int(rng.uniform(0, 0.52) * 180) : rng.integers(160, 181)]
    y, x = np.where(mask)
    mask = mask[y.min() : y.max() + 1, x.min() : x.max() + 1]
    mask = cv2.resize(mask, (rng.integers(10, 60), rng.integers(32, 100)), interpolation=cv2.INTER_NEAREST)
    mask[rng.random(mask.shape) < rng.uniform(0, 0.07)] = 0
    return mask_features(mask, w, h * (y.max() - y.min() + 1) / 175)


def dataset(seed, n):
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 2
    return np.asarray([sample(rng, bool(y)) for y in labels]), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    x, y = dataset(141, 3600)
    x = np.column_stack((x, np.ones(len(y))))

    def loss(w):
        v = x @ w
        err = 1 / (1 + np.exp(-np.clip(v, -40, 40))) - y
        return (np.sum(np.logaddexp(0, v) - y * v) + 0.2 * np.sum(w[:-1] ** 2), x.T @ err + np.r_[0.4 * w[:-1], 0])

    w = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 800, "ftol": 1e-11}).x
    x2, y2 = dataset(819, 1200)
    pred = x2 @ w[:-1] + w[-1] > np.log(0.7 / 0.3)
    model = {
        "format": 1,
        "method": "L2 logistic regression on metric silhouettes",
        "train_seed": 141,
        "train_n": len(y),
        "holdout_seed": 819,
        "holdout_n": len(y2),
        "holdout_tp": int(sum(pred & (y2 == 1))),
        "holdout_fp": int(sum(pred & (y2 == 0))),
        "holdout_positives": int(sum(y2)),
        "weights": w[:-1].round(7).tolist(),
        "bias": float(w[-1]),
    }
    args.output.write_text(json.dumps(model, indent=2) + "\n")
    print({k: v for k, v in model.items() if k != "weights"})


if __name__ == "__main__":
    main()
