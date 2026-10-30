"""LB V4 (the resume test) analysis, files only (nothing simulated): attempt 1 (killed, renamed <name>_crashed1) and
attempt 2 (--resume-from it) of the no-update run critic_blind_tail3 9311.

    PY -I -B v4_compare.py V4_DIR

Reports, each [measured] from the two folders:
  (a) the resume point attempt 2 chose (run_meta resume: candidates and problems, P, the links and copies);
  (b) the reloaded state: attempt 2's load digests (params, Adam, VecNormalize, num_timesteps, n_updates) against the P
      pair's own (torch.load weights_only, the pkl's arrays; computed here) and, for params / Adam, the start pair's (no
      update was ever made);
  (c) record continuity in attempt 2's finished folder: every record's rows up to P equal attempt 1's (the probe's every
      column, the evidence lines' text, both ledgers' rows' text, evaluations.npz, gate_progress.npz), its rows after P
      are attempt 2's own, one probe row per update step and one evidence line per rollout end at S + 8,192 k, one
      evaluation of 10 episodes at every 131,072, no duplicate;
  (d) the arm in force on every rollout of both attempts (the critic's four value paths exactly 0);
  (e) not bitwise the uninterrupted run: attempt 1's rows in (P, its kill] against attempt 2's at the same steps;
  (f) CHECK (check.json).
Exit 0 when (a)-(d) and (f) hold.
"""
import csv
import hashlib
import importlib.util
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location("bs_common", str(Path(__file__).resolve().parents[1] / "tools" /
                                                                "bs_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
V4 = Path(sys.argv[1])
A1, A2 = V4 / "critic_blind_tail3_9311_crashed1", V4 / "critic_blind_tail3_9311"
ST = C.STAGE_DIR
ok = True


def check(cond, msg):
    global ok
    print(("  ok    " if cond else "  FAIL  ") + msg)
    ok &= bool(cond)


def h(arrays):
    d = hashlib.sha256()
    for name, a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        d.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return "sha256:" + d.hexdigest()


def pair_digests(zp, pp):
    with zipfile.ZipFile(zp) as z:
        pol = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
        opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
        data = json.loads(z.read("data"))
    params = h([(k, pol[k].numpy()) for k in sorted(pol)])
    adam = []
    for i, v in sorted(opt["state"].items()):
        for n in sorted(v):
            x = v[n]
            adam.append((f"{i}.{n}", np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)))
    with open(pp, "rb") as fh:
        vn = pickle.load(fh)  # noqa: S301 - the run's own pkl
    vec = h([("obs.mean", vn.obs_rms.mean), ("obs.var", vn.obs_rms.var), ("obs.count", np.float64(vn.obs_rms.count)),
             ("ret.mean", vn.ret_rms.mean), ("ret.var", vn.ret_rms.var), ("ret.count", np.float64(vn.ret_rms.count))])
    return {"params": params, "adam": h(adam), "vecnormalize": vec, "num_timesteps": data["num_timesteps"],
            "n_updates": data["_n_updates"]}


def same(a, b):
    """Array equality; NaN equals NaN in float arrays (the gate_progress fall-term keys are NaN where no fall)."""
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a, b, equal_nan=a.dtype.kind in "fc" and b.dtype.kind in "fc")


def probe(run):
    with np.load(run / ST / "policy_probe.npz", allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def ev(run):
    return (run / "arm_evidence.jsonl").read_text().splitlines(keepends=True)


def csv_rows(path):
    with open(path, newline="") as fh:
        r = list(csv.reader(fh))
    return r[0], r[1:]


m1 = json.loads((A1 / "run_meta.json").read_text())
m2 = json.loads((A2 / "run_meta.json").read_text())
rs = m2["resume"]
P, T = m2["resumed_from"], m2["T"]
kill = (m1.get("heartbeat") or {}).get("rollout_end")
print(f"attempt 1: status {m1['status']} (killed: SIGKILL leaves 'running'), last heartbeat {kill:,}, "
      f"checkpoints on disk: {sorted(p.name for p in (A1 / ST / 'models').iterdir())}")
print(f"attempt 2: status {m2['status']} exit {m2['exit']} stop {m2['stop_reason']} T {T:,} attempt {m2['attempt']} "
      f"attempts {[(a['folder'], a['status'], a['last_step']) for a in m2['attempts']]}")
print("(a) THE RESUME POINT")
for c in rs["candidates"]:
    print(f"      candidate {c['steps']:,}: {c['problem'] or 'complete'}")
check(P == 2_359_296 and rs["P"] == P and m2["load_step"] == P, f"resumed_from = load_step = {P:,} (the first pair)")
check(rs["candidates"][0]["steps"] == 2_621_440 and rs["candidates"][0]["problem"],
      "the newest pair 2,621,440 refused (its probe row and the training ledger after it never reached the disk)")
links = sum(1 for v in rs["pairs"].values() if v["how"] == "link")
print(f"      pairs: {sorted(rs['pairs'])} ({links} hard links); records copied: {sorted(rs['copied'])}; "
      f"evidence lines kept {rs['evidence_lines']}")
check(links == len(rs["pairs"]) == 4 and rs["evidence_lines"] == (P - C.S) // C.ROLLOUT,
      "the start pair and the P pair linked (4 files), the evidence lines up to P kept")
print("(b) THE RELOADED STATE")
zp, pp = (A2 / ST / "models" / n for n in C.periodic_names(P))
own = pair_digests(zp, pp)
start = pair_digests(*(A2 / ST / "models" / n for n in C.periodic_names(C.S)))
dl = m2["digest_load"]
for k in ("params", "adam", "vecnormalize", "num_timesteps", "n_updates"):
    check(dl[k] == own[k], f"loaded {k} {str(dl[k])[:23]} == the P pair's {str(own[k])[:23]}")
check(own["params"] == start["params"] and own["adam"] == start["adam"] and own["n_updates"] == start["n_updates"],
      "the P pair's params / Adam / n_updates == the start pair's (no optimizer step in attempt 1)")
check(own["vecnormalize"] != start["vecnormalize"],
      "the P pair's VecNormalize moved from S's (attempt 1 ran 32 rollouts)")
check(m2["digest_end"]["params"] == dl["params"] and m2["digest_end"]["adam"] == dl["adam"],
      "attempt 2 made no optimizer step either (params / Adam at its end == at its load)")
check(m2["digest_start"]["params"] == start["params"] and m2["digest_start"]["num_timesteps"] == C.S,
      "run_meta digest_start = the start pair's (S)")
print("(c) RECORD CONTINUITY")
p1, p2 = probe(A1), probe(A2)
u1, u2 = p1["update_timesteps"].astype(np.int64), p2["update_timesteps"].astype(np.int64)
want = [C.ROLLOUT * k for k in range(1, T // C.ROLLOUT)]
check(u2.tolist() == want, f"probe: {len(u2)} rows, one at every 8,192 up to T - 8,192 ({len(want)}), none twice")
k1 = u1 <= P
k2 = u2 <= P
same_cols = [c for c in p2 if p2[c].shape[:1] == u2.shape and not same(p2[c][k2], p1[c][k1])]
check(int(k1.sum()) == int(k2.sum()) and not same_cols,
      f"probe rows <= P ({int(k2.sum())}) equal attempt 1's in every per-update column (differing: {same_cols})")
e1, e2 = ev(A1), ev(A2)
n_le = (P - C.S) // C.ROLLOUT
steps2 = [json.loads(x)["num_timesteps"] for x in e2]
check(e2[:n_le] == e1[:n_le], f"evidence lines up to P ({n_le}) equal attempt 1's, byte for byte")
check(steps2 == [C.S + C.ROLLOUT * k for k in range(1, (T - C.S) // C.ROLLOUT)],
      f"evidence: {len(steps2)} lines, one per rollout end S + 8,192 k, none twice")
for name in ("eval_episodes.csv", "train_episodes.csv"):
    h1, r1 = csv_rows(A1 / ST / name)
    h2, r2 = csv_rows(A2 / ST / name)
    ti1, ti2 = h1.index("timesteps"), h2.index("timesteps")
    a = [[r[h1.index(c)] for c in h1] for r in r1 if float(r[ti1]) <= P]
    b = [[r[h2.index(c)] if c in h2 else None for c in h1] for r in r2 if float(r[ti2]) <= P]
    after = [r for r in r2 if float(r[ti2]) > P]
    check(a == b and h1 == h2, f"{name}: rows <= P ({len(b)}) equal attempt 1's, text for text; {len(after)} rows "
          f"after P are attempt 2's")
hh, rr = csv_rows(A2 / ST / "eval_episodes.csv")
ti, ei = hh.index("timesteps"), hh.index("episode")
cnt = {}
for r in rr:
    cnt[int(float(r[ti]))] = cnt.get(int(float(r[ti])), 0) + 1
evs = [C.EVAL_FREQ * k for k in range(1, T // C.EVAL_FREQ + 1)]
check(sorted(cnt) == evs and set(cnt.values()) == {10} and len({(r[ti], r[ei]) for r in rr}) == len(rr),
      f"eval ledger: {len(cnt)} evaluations at every 131,072 up to T, 10 episodes each, no (timesteps, episode) twice")
for name in ("evaluations.npz", "gate_progress.npz"):
    with np.load(A1 / ST / name) as z1, np.load(A2 / ST / name) as z2:
        t1, t2 = np.asarray(z1["timesteps"]), np.asarray(z2["timesteps"])
        eq = all(same(np.asarray(z1[k])[t1 <= P], np.asarray(z2[k])[t2 <= P]) for k in z2.files
                 if k in z1.files and np.asarray(z2[k]).shape[:1] == t2.shape)
        check(t2.tolist() == evs and eq, f"{name}: one row per evaluation up to T ({len(t2)}); rows <= P equal "
              f"attempt 1's in every per-evaluation key")
print("(d) THE ARM ON EVERY ROLLOUT OF BOTH ATTEMPTS")
lines = [json.loads(x) for x in e2]
blind = sum(all(e[k] == 0.0 for k in ("dv_predict_values", "dv_forward", "dv_evaluate_actions", "dv_mirrored"))
            for e in lines)
actor = sum(e["dmu_actor"] > 0 for e in lines)
check(blind == len(lines) == actor, f"critic blind on all 4 value paths {blind}/{len(lines)}; actor sees obs 6 "
      f"{actor}/{len(lines)} (rollouts {n_le} from attempt 1, {len(lines) - n_le} from attempt 2)")
check(m2["arm_applied"]["mlp_extractor_class"] == "CriticBlindTail3MlpExtractor" and m2["arm_state_end"] == "in force",
      "attempt 2 re-installed the hook at its load; in force at its end")
print("(e) NOT BITWISE THE UNINTERRUPTED RUN (attempt 1's rows after P against attempt 2's, same steps)")
common = sorted(set(u1[u1 > P].tolist()) & set(u2[u2 > P].tolist()))
i1 = [int(np.flatnonzero(u1 == s)[0]) for s in common]
i2 = [int(np.flatnonzero(u2 == s)[0]) for s in common]
for col in ("obs_rms_count", "obs_rms_mean", "bank_value_now", "bank_action_now", "rollout_mu_mean"):
    d = [not same(p1[col][a], p2[col][b]) for a, b in zip(i1, i2, strict=True)]
    print(f"      probe {col}: differs on {sum(d)}/{len(d)} common rows")
ev1 = {json.loads(x)["num_timesteps"]: json.loads(x) for x in e1}
ev2 = {s: json.loads(x) for s, x in zip(steps2, e2, strict=True)}
cs = sorted(s for s in ev1 if s > P and s in ev2)
for k in ("digest_env_side", "digest_rewards", "digest_values", "obs_rms_digest"):
    print(f"      evidence {k}: differs on {sum(ev1[s][k] != ev2[s][k] for s in cs)}/{len(cs)} common rollouts")
r1 = {(int(float(r[ti])), r[ei]): r for r in csv_rows(A1 / ST / "eval_episodes.csv")[1]}
r2 = {(int(float(r[ti])), r[ei]): r for r in rr}
ce = sorted(k for k in r1 if k[0] > P and k in r2)
print(f"      eval ledger rows after P: {sum(r1[k] != r2[k] for k in ce)}/{len(ce)} common rows differ")
same_steps = [s for s in cs if ev1[s]["digest_env_side"] == ev2[s]["digest_env_side"]]
print(f"      [the first common rollout ({cs[0] if cs else None}): env side "
      f"{'equal' if cs and cs[0] in same_steps else 'differs'}]")
print("(f) CHECK")
c = json.loads((A2 / "check.json").read_text())
check(c["passed"] is True and c["failures"] == [], f"check.json passed ({c['failures']}); items "
      f"{sorted(c['items'])}; resume_stitch {c['items'].get('resume_stitch')}")
print("V4:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
