"""E5 run record (prereg section 6: seeds, wall time and the code's hashes for every stage), from the scratch logs."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e5 as R  # noqa: E402

C = R.C
L = R.SCR / "logs"
out = {"experiment": "E5", "date": C.DATE, "stages": [], "workers": 2}
for size in ("0.60", "0.50"):
    for kind in ("tune", "heldout"):
        log = (L / f"{kind}_{size}.log").read_text().splitlines()
        pools, cur = [], None
        for ln in log:
            m = re.match(r"\[(\d+)/(\d+) (\d+)s\] \('([\d.]+)', '(\w+)'", ln)
            if m:
                n, N, t, _, stage = int(m[1]), int(m[2]), int(m[3]), m[4], m[5]
                if n == N:
                    pools.append({"stage": stage, "episodes": N, "wall_s": t})
        shaf = L / f"sha256_before_{kind}_{size}.txt"
        out["stages"].append({"size": size, "command": f"run_e5.py {kind} --size {size} --workers 2",
                              "pools": pools, "sha256_before": shaf.read_text().split("\n")[:2]})
out["seeds"] = {"s1": list(C.E5_SCREEN_SEEDS), "s2": list(C.E5_TUNING_SEEDS), "s3": "not run (a stage-2 setting met "
                "the rule at both sizes)", "heldout": C.E5_HELDOUT, "heldout_frozen": C.E5_HELDOUT}
out["choice_sha256_at_write"] = {s: (R.SCR / f"e5_choice_{s}.sha256").read_text().split("\n")[:2]
                                 for s in ("0.60", "0.50")}
out["choice_sha256_now"] = {s: R.sha(R.OUT / f"e5_choice_{s}.json") for s in ("0.60", "0.50")}
out["model_50cm"] = {"path": str(R.model_path("0.50")), "sha256": R.sha(R.model_path("0.50")),
                     "built_by": "sim/build_scaled_trike.py build_xml(0.50, 'D0', 'adopted', layout='L1')"}
out["provenance"] = C.provenance(Path(__file__), extra={"model_50cm": R.model_path("0.50")})
(R.OUT / "e5_run_record.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in ("stages", "choice_sha256_at_write", "choice_sha256_now")}, indent=0)[:3000])
