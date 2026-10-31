"""Per-step raw reward terms by episode-step bucket in the why-probe's existing rollouts (read only, no simulation).

Purpose: size the 'second hidden variable' of PR #655 (the unobserved action / filter history, which the rate
penalties smoothness and action_jerk depend on) against the step-200 gated terms, by bucket; and the pop's own
reward deficit in steps 0-200 in trained-policy rollouts (run branch config). Output: stdout and argv[1] JSON.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

S2 = Path(os.environ["XREF_S2"])  # the earlier session scratchpad (the run log's S2), read only
PP = S2 / "com_revisit/notes/phase_probe/rollouts"
LABELS = ("P1", "P2", "C11E", "C31E", "B11E")
BUCKETS = [(0, 50), (50, 150), (150, 200), (200, 250), (250, 1000)]
TERMS = ("reward_smoothness", "reward_action_jerk", "reward_action_saturation", "reward_energy",
         "reward_stance_width", "reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band",
         "reward_bilateral_support", "reward_alive", "reward_foot_load_balance", "reward_foot_flatness",
         "reward_head_clearance", "reward_tail_home_pose", "reward_total")


def main():
    res = {}
    for lab in LABELS:
        z = np.load(PP / f"{lab}.npz", allow_pickle=False)
        k = z["k"].astype(np.int64)
        terms = z["terms"].astype(np.float64)
        keys = [str(x) for x in z["reward_keys"]]
        out = {}
        for t in TERMS:
            col = terms[:, keys.index(t)]
            out[t] = {f"{lo}-{hi}": float(col[(k >= lo) & (k < hi)].mean()) for lo, hi in BUCKETS}
        rate = terms[:, keys.index("reward_smoothness")] + terms[:, keys.index("reward_action_jerk")]
        out["rate_sum"] = {f"{lo}-{hi}": float(rate[(k >= lo) & (k < hi)].mean()) for lo, hi in BUCKETS}
        res[lab] = out
        print(f"=== {lab} (raw reward per step, mean over 48 episodes' states)")
        for t in ("rate_sum",) + TERMS:
            print(f"  {t:28s} " + "  ".join(f"{b}:{v:+.4f}" for b, v in out[t].items()))
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
