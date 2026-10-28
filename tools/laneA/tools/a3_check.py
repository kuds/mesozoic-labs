"""A-3 on a real r10 judge file (validation only): score_stance3.validity_issues as kscore_r10ltsb.py loads it for M4
(criteria lean v5: the judge identity rebound and the r9 plant check reading the pinned r10 plant in place of r9's),
and, for the record, the same file under score_stance3's own r9 plant check (identity rebound only).

    PY -I -B a3_check.py JUDGE_JSON SEED STEPS OUT.json EXPECT_HEAD

EXPECT_HEAD: the HEAD the judge file was judged on (a validation names it; no tool holds the branch HEAD).
PASS: M4's issues are none, and the r9 check alone gives exactly its one plant issue (the rebinding changes that one
check and nothing else).
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
VERSION = "a3_check 1.1 (2026-10-26; v5: A-3 decided)"
print(VERSION, flush=True)
jp, seed, steps, out, head = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
spec = importlib.util.spec_from_file_location("k", Path(__file__).resolve().parent / "kscore_r10ltsb.py")
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)
d = json.loads(Path(jp).read_text())
frozen = K.frozen_with_sidecar(("80",))
kw = dict(arm="LTSB", seed=seed, steps=steps, label=str(steps), expect_head=head, frozen=frozen, validation=False,
          needs_noise=True)
m4 = K.make_instance("r10ltsb", head).validity_issues(d, **kw)
r9_only = K.load_stance3("score_stance3_r9_check_only")
r9_only.JUDGE_SCHEMA = K.IDENTITIES["r10ltsb"]["schema"]
r9_only.ARM_ENV = {**r9_only.ARM_ENV, "LTSB": dict(K.ARM_ENV_LTSB)}
r9_only.HEADS = {**r9_only.HEADS, "LTSB": head}
strict = r9_only.validity_issues(d, **kw)
res = {"tool": VERSION, "judge_json": jp, "expect_head": head, "m4": {"bad": m4[0], "soft": m4[1]},
       "identity_rebound_r9_plant_check": {"bad": strict[0], "soft": strict[1]},
       "expected_r9_issue": K.r9_plant_issue(d)}
res["PASS"] = m4[0] == [] and strict[0] == [res["expected_r9_issue"]]
Path(out).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
sys.exit(0 if res["PASS"] else 1)
