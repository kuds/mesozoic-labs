"""The frozen model-based trot (../walk/walk60.py) on one robot model over seeds 101-110, without rendering: the
conditions of the museum clip (0.25 m/s, 10 ms servo delay, sensor noise on, no randomisation). Simulation only.

Usage: python trot_seed_check.py MODEL_FILE_NAME OUT.json   (MODEL_FILE_NAME in ../sim/generated/)
"""

import json
import sys
from pathlib import Path

B = Path(__file__).resolve().parents[1]
for p in (B / "walk", B / "rl"):
    sys.path.insert(0, str(p))
import gait_metrics  # noqa: E402
import walk60  # noqa: E402


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    model, out = str(B / "sim" / "generated" / argv[0]), argv[1]
    rows = []
    for seed in range(101, 111):
        r = walk60.run_episode(walk60.WalkParams(), seed=seed, command=0.25, delay_steps=10, model_path=model,
                               domain_randomization=False, observation_noise=True)
        s = gait_metrics.summarise([r["measures"]])
        g = s["drive_groups_worst"]
        lim = {nm: d.get("at_limit_episode") for nm, d in s["drives_worst"].items()}
        rows.append({"seed": seed, "fell": r["fell"], "speed_mps": r["speed_mps"],
                     "speed_from_2s_mps": s["forward_speed_mps"], "ik_err_max_mm": 1000 * r["ik_err_max_m"],
                     "at_episode_limit_max": max(v for v in lim.values() if v is not None),
                     "at_episode_limit_drive": max((k for k in lim if lim[k] is not None), key=lambda k: lim[k]),
                     "worst_rms_over_continuous": max(v["rms_over_continuous_worst"] for v in g.values()),
                     "worst_drive": max(g.values(), key=lambda v: v["rms_over_continuous_worst"])["worst_joint"],
                     "at_nominal_limit_max": max(v["at_limit_nominal"] for v in g.values()),
                     "sideways_drift_m": s["sideways_drift_m_worst"], "heading_change_deg": s["heading_change_deg_worst"],
                     "touchdowns_per_s": s["touchdowns_per_s"]})
        print(seed, gait_metrics.one_line(s), "| fell", r["fell"], f"speed {r['speed_mps']:.3f}", flush=True)
    Path(out).write_text(json.dumps({"model": argv[0], "conditions": "0.25 m/s, 10 ms servo delay, sensor noise on, "
                                     "no randomisation, seeds 101-110", "simulation_only": True, "episodes": rows},
                                    indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
