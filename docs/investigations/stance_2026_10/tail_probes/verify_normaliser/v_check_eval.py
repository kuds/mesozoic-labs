"""Verifier: (1) my own evaluator (v_common.Net/Norm, read straight from policy.pth and my own unpickler) against the
harness loader (load_sb3_checkpoint, SB3 objects); (2) the probe's statistics swap, inspected inside the loop.

PYTHONPATH=<verify_probe/code>:<verify_probe>:<verify_probe/probe_copy> python -B v_check_eval.py > v_check_eval.txt

(1) For 6 checkpoints: statistics equality (harness obs_rms vs my reader), and the actor mean on 300 normalised
    inputs (real states under own and swapped statistics + wide random z), max |SB3 - mine|; deterministic action
    model.predict(normalize_obs(obs)) vs clip(mine).
(2) probe_copy/probe_common.py is a byte-identical copy of the probe's file (sha256 checked by the caller).  Its Agent
    is built for s42_A_1111's end and set_stats(...) is called exactly as run_swaps.py does for endW_startN (all 63),
    endW_startN_t3q (dims (6,)) and for startW_endN (start Agent, the end's statistics).  model.predict is wrapped to
    capture the normalised observation it actually receives during 60 steps of a rollout; that input is compared with
    (a) my formula under the DONOR statistics (should match) and (b) under the agent's OWN statistics (should differ on
    the swapped dims only).  Network evaluation and a short simulation only.  No training.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

import probe_common as PC  # the probe's own module (copy)
from v_common import SE, START, Net, Norm, make_env, paths

torch.set_num_threads(1)
from environments.shared.policy_loading import load_sb3_checkpoint  # noqa: E402


def factory():
    return make_env()


def sb3_mean(model, z):
    with torch.no_grad():
        return model.policy.get_distribution(torch.as_tensor(np.asarray(z, dtype=np.float32))).distribution.mean.numpy()


def part1():
    print("=== (1) my evaluator vs the harness loader")
    ck = [("s42_A_1101", SE["s42"][0]), ("s42_A_1111", SE["s42"][1]), ("s42_A_1151", SE["s42"][1]),
          ("s42_B_1251", SE["s42"][1]), ("s80_A_1301", SE["s80"][0]), ("s80_B_1421", SE["s80"][1])]
    env = make_env()
    obs, _ = env.reset(seed=11)
    real = []
    for k in range(400):
        obs, *_ = env.step(np.zeros(15, dtype=np.float32) + 0.05 * np.sin(0.01 * k))
        if k % 4 == 0:
            real.append(np.array(obs, copy=True))
    env.close()
    real = np.asarray(real, dtype=np.float32)
    rng = np.random.default_rng(0)
    worst = 0.0
    for run, st in ck:
        z_p, p_p = paths(run, st)
        model, vn, _ = load_sb3_checkpoint(str(z_p), str(p_p), factory)
        try:
            vn.venv.close()
        except Exception:  # noqa: BLE001
            pass
        net, nm = Net(run, st), Norm(run, st)
        same = (np.array_equal(np.asarray(vn.obs_rms.mean, dtype=np.float64), nm.mean)
                and np.array_equal(np.asarray(vn.obs_rms.var, dtype=np.float64), nm.var)
                and vn.epsilon == nm.eps and vn.clip_obs == nm.clip)
        zr = np.stack([nm(o) for o in real])
        zh = np.stack([vn.normalize_obs(o) for o in real]).astype(np.float32)
        dz = float(np.max(np.abs(zr - zh)))
        zrand = rng.normal(0.0, 3.0, size=(100, 63)).astype(np.float32)
        Z = np.concatenate([zr, zrand])
        dm = float(np.max(np.abs(sb3_mean(model, Z) - net.mean(Z))))
        da = max(float(np.max(np.abs(model.predict(vn.normalize_obs(o), deterministic=True)[0]
                                      - np.clip(net.mean(nm(o)[None, :])[0], -1, 1)))) for o in real[:50])
        worst = max(worst, dm, da, dz)
        print(f"  {run}@{st}: stats identical {same}; max|z diff| {dz:.2e}; max|mean diff| {dm:.2e} over {len(Z)} inputs; "
              f"max|action diff| {da:.2e}")
    print(f"  worst difference {worst:.2e}")


def part2():
    print("\n=== (2) the probe's swap, inspected inside the loop (probe_common copy)")
    s0, e0 = SE["s42"]
    run = "s42_A_1111"
    fac = PC.make_factory()
    mS, vS = PC.stats(START["s42"], s0)
    mE, vE = PC.stats(run, e0)
    own_end, own_start = Norm(run, e0), Norm(START["s42"], s0)
    cases = [("endW_startN (all 63)", (run, e0), (mS, vS, None), own_end, own_end.mixed(own_start)),
             ("endW_startN_t3q (obs 6)", (run, e0), (mS, vS, (6,)), own_end, own_end.mixed(own_start, dims=(6,))),
             ("endW_startN_tail (tail 8)", (run, e0), (mS, vS, PC.TAIL_DIMS), own_end, own_end.mixed(own_start, dims=PC.TAIL_DIMS)),
             ("startW_endN (all 63)", (START["s42"], s0), (mE, vE, None), own_start, own_start.mixed(own_end)),
             ("own (end)", (run, e0), None, own_end, own_end)]
    for name, wts, donor, own_norm, expect in cases:
        ag = PC.Agent(*wts, fac)
        if donor is None:
            ag.set_stats(None, tag="own")
        else:
            ag.set_stats(donor[0], donor[1], dims=donor[2], tag=name)
        seen = []
        orig = ag.model.predict

        def spy(x, *a, _orig=orig, **k):
            seen.append(np.array(x, copy=True))
            return _orig(x, *a, **k)

        ag.model.predict = spy
        env = fac()
        obs, _ = env.reset(seed=3042)
        raw = []
        for _ in range(60):
            raw.append(np.array(obs, copy=True))
            obs, *_ = env.step(ag.act(obs))
        env.close()
        seen, raw = np.asarray(seen), np.asarray(raw)
        z_exp = np.stack([expect(o) for o in raw])
        z_own = np.stack([own_norm(o) for o in raw])
        d_exp = float(np.max(np.abs(seen - z_exp)))
        diff_dims = sorted(set(np.nonzero(np.max(np.abs(seen - z_own), axis=0) > 1e-6)[0].tolist()))
        k = 59
        print(f"  {name}: predict saw {len(seen)} inputs; max|seen - donor-formula z| {d_exp:.2e}; dims differing from "
              f"the agent's own-statistics z: {diff_dims if len(diff_dims) <= 10 else str(len(diff_dims)) + ' dims'}")
        print(f"     step {k + 1}: raw obs6 {math.degrees(raw[k, 6]):+.3f} deg -> z6 seen {seen[k, 6]:+.4f}; "
              f"own-stats z6 {z_own[k, 6]:+.4f}; donor-formula z6 {z_exp[k, 6]:+.4f}")
        ag.set_stats(None)


if __name__ == "__main__":
    part1()
    part2()
    sys.exit(0)
