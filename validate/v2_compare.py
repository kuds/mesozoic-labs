"""V2 pairing (no simulation; reads files only): the four no-update runs of one seed against each other.

    PY -I -B v2_compare.py V2_DIR [SEED]      -> prints the comparison; exit 0 when every expectation holds

With train() a no-op and the same seed, the policy is fixed, the actions are drawn from the same torch generator and
the envs are seeded alike, so:
  critic_blind_tail3 vs control: the env side of every rollout (observations, actions, episode starts) is identical,
      the values differ (the blinded critic), the evaluation ledger and the probe's bank actions are identical, the
      probe's bank values differ;
  gae_lambda_1 vs control: the env side, the rewards and the values are identical, the advantages / returns differ;
      the ledger and the probe identical;
  frozen_obs_norm vs control: the observation statistics stay at the start pair's (so the normalised observations,
      hence the actions and the env side, differ from the first rollout on), the return statistics update.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ("control", "critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm")
TIME_COLS = ("seconds", "wall", "utc", "time_")


def evidence(run):
    return [json.loads(x) for x in (run / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]


def ledger_rows(run):
    with open(run / "01_stance" / "eval_episodes.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    keep = [k for k in rows[0] if not any(t in k for t in TIME_COLS)]
    return keep, [[r[k] for k in keep] for r in rows]


def probe(run):
    with np.load(run / "01_stance" / "policy_probe.npz", allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ("update_timesteps", "bank_action_now", "bank_value_now",
                                               "obs_rms_mean", "obs_rms_var", "obs_rms_count", "ret_rms_var")}


def main():
    root = Path(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 9101
    runs = {a: root / f"{a}_{seed}" for a in ARMS if (root / f"{a}_{seed}").is_dir()}
    ev = {a: evidence(r) for a, r in runs.items()}
    out, ok = [], True

    def expect(cond, text):
        nonlocal ok
        ok &= bool(cond)
        out.append(f"  [{'ok' if cond else 'FAIL'}] {text}")
    c = ev.get("control")
    if c is None:
        print("no control run")
        return 2
    n = len(c)
    out.append(f"control: {n} rollout ends; arms present {list(runs)}")
    for arm, e in ev.items():
        if arm == "control":
            continue
        same = {k: [x[k] == y[k] for x, y in zip(c, e)] for k in ("digest_env_side", "digest_rewards", "digest_values",
                                                                   "digest_advantages")}
        cnt = {k: f"{sum(v)}/{len(v)}" for k, v in same.items()}
        out.append(f"{arm} vs control, rollouts equal: {cnt}")
        expect(len(e) == n, f"{arm}: {len(e)} rollout ends = control's {n}")
        if arm == "critic_blind_tail3":
            expect(all(same["digest_env_side"]), "critic_blind: env side identical on every rollout")
            expect(not any(same["digest_values"]), "critic_blind: values differ on every rollout")
        if arm == "gae_lambda_1":
            expect(all(same["digest_env_side"]) and all(same["digest_rewards"]) and all(same["digest_values"]),
                   "gae_lambda_1: env side, rewards and values identical on every rollout")
            expect(not any(same["digest_advantages"]), "gae_lambda_1: advantages differ on every rollout")
        if arm == "frozen_obs_norm":
            expect(not any(same["digest_env_side"]), "frozen_obs_norm: env side differs on every rollout")
    lk, lc = ledger_rows(runs["control"])
    pc = probe(runs["control"])
    for arm in ("critic_blind_tail3", "gae_lambda_1"):
        if arm not in runs:
            continue
        k, rows = ledger_rows(runs[arm])
        expect(k == lk and rows == lc, f"{arm}: evaluation ledger identical to control's ({len(rows)} rows, "
                                       f"{len(k)} columns compared; time columns dropped)")
        p = probe(runs[arm])
        expect(np.array_equal(p["update_timesteps"], pc["update_timesteps"])
               and np.array_equal(p["bank_action_now"], pc["bank_action_now"]),
               f"{arm}: probe bank actions identical to control's on every row")
        same_v = np.array_equal(p["bank_value_now"], pc["bank_value_now"])
        expect(same_v == (arm == "gae_lambda_1"), f"{arm}: probe bank values {'identical' if same_v else 'differ'}")
    if "frozen_obs_norm" in runs:
        p = probe(runs["frozen_obs_norm"])
        S = 1048576
        w = p["update_timesteps"] > S
        m, v, cnt = p["obs_rms_mean"][w], p["obs_rms_var"][w], p["obs_rms_count"][w]
        expect(all(np.array_equal(m[i], m[0]) and np.array_equal(v[i], v[0]) and cnt[i] == cnt[0]
                   for i in range(len(m))), f"frozen_obs_norm: probe obs statistics constant over {len(m)} rows")
        r = p["ret_rms_var"][w]
        expect(len(set(r.tolist())) > 1, "frozen_obs_norm: return statistics (ret_rms_var) move")
        wc = pc["update_timesteps"] > S
        expect(len({tuple(x) for x in pc["obs_rms_mean"][wc].tolist()}) == int(wc.sum()),
               "control: probe obs statistics move every row")
    print("\n".join(out))
    print("V2 PAIRING: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
