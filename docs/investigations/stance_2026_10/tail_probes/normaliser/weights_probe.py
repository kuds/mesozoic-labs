"""First-layer weights on the tail_3 angle input, start and every checkpoint.  Read only; no simulation, no training.

PYTHONPATH=<policies/code_copy>:<this folder> python -B weights_probe.py > weights.txt

The actor's first linear layer sees z = (obs - mu) / sqrt(var + eps).  Its column for obs 6 (tail_3 angle), w6, is what
the weights contribute; in raw units the input enters as w6 * scale * theta, so ||w6|| * scale is the raw-unit input
gain of the first layer.  Printed per lineage and step (medians over the 12 continuations): ||w6||, scale (per deg),
||w6|| x scale, and the same for the tail_3 velocity (obs 27) and, for comparison, the median column norm over all 63
inputs.  Also the end/start ratio of ||w6|| per run.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from probe_common import NAMES, SE, START_RUN, ckpt, steps_of
from environments.shared.policy_loading import load_sb3_model

HERE = Path(__file__).resolve().parent


def first_layer(run, step):
    model = load_sb3_model(str(ckpt(run, step)[0]), device="cpu")
    pol = model.policy
    net = pol.mlp_extractor.policy_net
    lin = [m for m in net if hasattr(m, "weight")][0]
    return lin.weight.detach().numpy().astype(np.float64), str(net)


def main() -> int:
    st = json.load(open(HERE / "stats.json"))["runs"]
    eps = 1e-8
    arch_printed = False
    for lin in ("s42", "s80"):
        s0 = SE[lin][0]
        W0, arch = first_layer(START_RUN[lin], s0)
        if not arch_printed:
            print(f"actor MLP: {arch}; first layer {W0.shape}")
            arch_printed = True
        print(f"\n=== {lin}")
        print(f"  {'step':>9s} {'|w6|':>8s} {'scale6/deg':>10s} {'|w6|*scale':>10s} {'|w27|':>8s} {'scale27':>9s} "
              f"{'|w27|*sc':>9s} {'med |w_i|':>9s} {'cos(w6,w6_S)':>12s}")
        per_run_end = []
        for t in steps_of(lin):
            vals = []
            runs = [r for r in NAMES if r.startswith(lin)]
            for r in runs:
                W = W0 if t == s0 else first_layer(r, t)[0]
                var = st[r][str(t)]["var"]
                sc6 = math.radians(1) / math.sqrt(var[6] + eps)
                sc27 = math.radians(1) / math.sqrt(var[27] + eps)
                n6 = np.linalg.norm(W[:, 6])
                n27 = np.linalg.norm(W[:, 27])
                cos = float(W[:, 6] @ W0[:, 6] / (n6 * np.linalg.norm(W0[:, 6])))
                vals.append((n6, sc6, n6 * sc6, n27, sc27, n27 * sc27, np.median(np.linalg.norm(W, axis=0)), cos))
                if t == SE[lin][1]:
                    per_run_end.append((r, n6 / np.linalg.norm(W0[:, 6]), cos))
                if t == s0:
                    break
            v = np.median(np.array(vals), axis=0)
            print(f"  {t:9d} {v[0]:8.3f} {v[1]:10.3f} {v[2]:10.3f} {v[3]:8.3f} {v[4]:9.4f} {v[5]:9.4f} {v[6]:9.3f} {v[7]:12.3f}")
        print("  end / start ||w6|| per run, cos(w6_end, w6_start): " +
              ", ".join(f"{r} {q:.2f} {c:.2f}" for r, q, c in per_run_end))
    return 0


if __name__ == "__main__":
    sys.exit(main())
