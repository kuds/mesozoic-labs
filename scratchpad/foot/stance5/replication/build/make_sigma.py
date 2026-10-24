"""Write SP/foot/stance5/replication/sigma_s70_s80.json: the joint-space control sigma of the two replication seeds.

    python3 -I make_sigma.py OUT.json        (refuses to overwrite OUT.json)
    python3 -I make_sigma.py --check FILE    (recomputes and compares; writes nothing)

The rule (stated as the package default on 2026-10-22 at 15:11 UTC and confirmed by the maintainer at 16:32 UTC, run_log;
the "decision" string below is kept as it was written into the sidecar, so it carries the run-log entry's pre-correction
time; the run log's times were corrected at 2026-10-22 17:47 UTC): the element-wise mean of frozen_params.json
criteria.H4.control_sigma[42], [52] and [60] .vector, the same vector for seeds 70 and 80.  frozen_params.json is
pinned by its sha256 (58b2c8fb..., the r9 verdict's params_sha256, the same pin as judge_stance5.py's
FROZEN_PARAMS_SHA256).  The arithmetic, per element i: math.fsum((v42[i], v52[i], v60[i])) / 3.0 in IEEE double (the
correctly rounded sum of the three, then one division).  The JSON is written with Python's json module, whose float
repr round-trips every double exactly, so the judge reads back the very doubles computed here.  Plain python: no
numpy, no repository import, nothing unpickled.  numpy's mean is compared as a cross-check only (recorded).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
FROZEN = SCRATCH + "/foot/symmetry/design/frozen_params.json"
FROZEN_SHA256 = "58b2c8fbb135ecddf1bf2212d5e243514d90ff1069f5f3e2cb1c3ece9c3e7af8"
SOURCE_SEEDS = ("42", "52", "60")
NEW_SEEDS = ("70", "80")
SCHEMA = "stance5-replication/control-sigma/v1"
RULE = "element-wise mean of frozen_params criteria.H4.control_sigma 42, 52, 60"
#: What the judge records as noise_matched_control_joint.sigma_source for seeds 70 and 80 (the scorer checks it).
SOURCE = ("stance5 replication sidecar (SP/foot/stance5/replication/sigma_s70_s80.json): element-wise mean of "
          "frozen_params.json criteria.H4.control_sigma 42, 52, 60 (frozen_params sha256 " + FROZEN_SHA256 + ")")


def build() -> dict:
    with open(FROZEN, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != FROZEN_SHA256:
        raise SystemExit(f"{FROZEN} sha256 {sha} != the pinned {FROZEN_SHA256}")
    h4 = json.loads(raw.decode("utf-8"))["criteria"]["H4"]
    vecs = {s: h4["control_sigma"][s]["vector"] for s in SOURCE_SEEDS}
    n = {len(v) for v in vecs.values()}
    if len(n) != 1:
        raise SystemExit(f"the three vectors differ in length: { {s: len(v) for s, v in vecs.items()} }")
    (dim,) = n
    if not all(isinstance(x, float) and math.isfinite(x) and x > 0 for v in vecs.values() for x in v):
        raise SystemExit("a source element is not a positive finite float")
    mean = [math.fsum(vecs[s][i] for s in SOURCE_SEEDS) / 3.0 for i in range(dim)]
    try:  # cross-check only (recorded): numpy's mean over axis 0
        import numpy as np

        npm = np.mean(np.array([vecs[s] for s in SOURCE_SEEDS], dtype=np.float64), axis=0).tolist()
        cross = {"numpy_version": np.__version__, "max_abs_difference": max(abs(a - b) for a, b in zip(mean, npm)),
                 "elements_bit_identical": sum(a == b for a, b in zip(mean, npm)), "of": dim}
    except ImportError:
        cross = {"numpy": "not available"}
    within = all(min(vecs[s][i] for s in SOURCE_SEEDS) <= mean[i] <= max(vecs[s][i] for s in SOURCE_SEEDS)
                 for i in range(dim))
    return {
        "schema": SCHEMA,
        "rule": RULE,
        "source": FROZEN,
        "source_sha256": FROZEN_SHA256,
        "source_path_in_file": "criteria.H4.control_sigma[<seed>].vector",
        "source_vectors": {s: {"source": h4["control_sigma"][s]["source"], "vector": vecs[s]} for s in SOURCE_SEEDS},
        "control_sigma": {s: {"source": SOURCE, "vector": list(mean)} for s in NEW_SEEDS},
        "computation": {
            "script": os.path.realpath(__file__),
            "expression": "mean[i] = math.fsum((v42[i], v52[i], v60[i])) / 3.0, IEEE double, i = 0..%d" % (dim - 1),
            "dimension": dim,
            "space": h4["space"], "episodes_per_seed": h4["episodes_per_seed"], "spawn_range": h4["spawn_range"],
            "mean_of_vector": math.fsum(mean) / dim,
            "each_element_within_the_three_sources": within,
            "numpy_cross_check": cross,
            "decision": "maintainer decision 4 of 2026-10-22 (run_log 06:45 UTC): the new seeds' joint-noise sigma is "
                        "the element-wise mean of control_sigma for seeds 42, 52, 60",
            "note": "inferred, not measured: a mean of three per-seed sigmas is not a per-seed control sigma (those were "
                    "each read from a control run of that seed); see replication/criteria.txt KNOWN LIMITS",
        },
    }


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--check":
        with open(argv[1]) as fh:
            got = json.load(fh)
        want = build()
        same = {k: got.get(k) == want[k] for k in ("schema", "rule", "source_sha256", "control_sigma", "source_vectors")}
        print(json.dumps({"file": argv[1], "matches": same, "ok": all(same.values())}))
        return 0 if all(same.values()) else 1
    if len(argv) != 1 or argv[0].startswith("-"):
        print(__doc__)
        return 2
    out = os.path.abspath(argv[0])
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    rec = build()
    with open(out, "x") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")
    with open(out, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(out) as fh:
        back = json.load(fh)
    assert back["control_sigma"] == rec["control_sigma"], "the JSON did not round-trip"
    print(json.dumps({"wrote": out, "sha256": sha, "dimension": rec["computation"]["dimension"],
                      "mean_of_vector": rec["computation"]["mean_of_vector"],
                      "numpy_cross_check": rec["computation"]["numpy_cross_check"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
