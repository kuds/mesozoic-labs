"""Post-hoc harness check (decided after the 60 cm held-out result was seen; changes no result): the push in E5's push
episodes is applied. Reruns stored held-out push episodes with run_e5's own job code into a separate folder, reads the
sweep record's push measures (applied steps, impulse, lateral peak) through run_e5's pass-through wrapper, and checks
that each rerun's record equals the stored one field for field."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e5 as R  # noqa: E402

STORED = R.EPI
R.EPI = R.SCR / "check_push_applied"


def main():
    out = []
    for size_key in ("0.60", "0.50"):
        cp = R.OUT / f"e5_choice_{size_key}.json"
        if not cp.exists():
            continue
        ch = json.loads(cp.read_text())
        st = {**ch["chosen_values"], "id": ch["chosen_id"]}
        for cond in ("push_back_0.5", "push_right_0.5"):
            seed = 301
            stored = json.loads((STORED / size_key / "heldout" / f"{st['id']}__{cond}__{seed}.json").read_text())
            R._init()
            R.run_job((size_key, "heldout", st, cond, seed))
            new = json.loads(R.ep_file(size_key, "heldout", st["id"], cond, seed).read_text())
            push = (R._STASH.get("rec") or {}).get("push")
            same = all(json.dumps(stored[k], sort_keys=True) == json.dumps(new[k], sort_keys=True)
                       for k in ("record", "rpm_peak", "rpm_p99", "slip", "fell", "slow_gait", "worst_hind_knee"))
            row = {"size": size_key, "setting": st["id"], "condition": cond, "seed": seed, "push": push,
                   "rerun_equals_stored": same}
            print(json.dumps(row))
            out.append(row)
    (R.SCR / "check_push_applied.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
