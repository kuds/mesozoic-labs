"""Verifier: independent closed-loop rollouts (my evaluator, v_common) for the loop-gain and swap checks.

PYTHONPATH=<verify_probe/code>:<verify_probe> python -B v_loop.py MODE OUT.jsonl
  MODE swaps : s42 start (own) and 3 s42 ends (s42_A_1111 highest gain, s42_A_1151 the end that does not rail,
               s42_B_1251 lowest gain) under own / endW_startN / endW_startN_t3q / startW_endN / startW_endN_tail,
               plus s80_B_1421 and s80_A_1331 under own / endW_startN_tail / startW_endN_tail; seeds 3042-3061.
  MODE ckpts : every s42 checkpoint (start + 12 runs x 6 later steps), own statistics, seed 3042.
  MODE s80ends: the s80 start and its 12 ends, own statistics, seed 3042 (states for the clip check, v_s80clip.py).
Per episode: return, own tail_3 command (mean, share >= 0.99, steps 500-1000), tail_3 angle, the loop gain (my
re-implementation of the verifier's sweep on that episode's own 6 states under the condition's statistics), and an
in-loop check of the normalised observation: max over steps |z6 - (obs6 - mean6)/sqrt(var6 + eps)| with the
condition's (mean, var), and the mean z6 under the condition vs under the weights' own statistics.
Resumable: rows already in OUT are skipped.  States (6 x 63 float32 per episode) go to OUT's folder as one .npz at the
end.  No training.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from v_common import SE, START, TAIL8, Net, Norm, gain, rollout

torch.set_num_threads(1)
S42 = [f"s42_A_{n}" for n in range(1101, 1152, 10)] + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
SEEDS = list(range(3042, 3062))


def plan(mode):
    out = []
    if mode == "swaps":
        s0, e0 = SE["s42"]
        st = (START["s42"], s0)
        out.append(("own", "start_s42", st, st, None, None))
        for r in ("s42_A_1111", "s42_A_1151", "s42_B_1251"):
            en = (r, e0)
            out += [("own", r, en, en, None, None),
                    ("endW_startN", r, en, en, st, None),
                    ("endW_startN_t3q", r, en, en, st, (6,)),
                    ("startW_endN", r, st, st, en, None),
                    ("startW_endN_tail", r, st, st, en, TAIL8)]
        s0, e0 = SE["s80"]
        st = (START["s80"], s0)
        out.append(("own", "start_s80", st, st, None, None))
        for r in ("s80_B_1421", "s80_A_1331"):
            en = (r, e0)
            out += [("own", r, en, en, None, None),
                    ("endW_startN_tail", r, en, en, st, TAIL8),
                    ("startW_endN_tail", r, st, st, en, TAIL8)]
    elif mode == "s80ends":
        s0, e0 = SE["s80"]
        out.append(("ck", "start_s80", (START["s80"], s0), (START["s80"], s0), None, None))
        for r in [f"s80_A_{n}" for n in range(1301, 1352, 10)] + [f"s80_B_{n}" for n in range(1401, 1452, 10)]:
            out.append(("ck", f"{r}@{e0}", (r, e0), (r, e0), None, None))
    else:
        s0, e0 = SE["s42"]
        out.append(("ck", "start_s42", (START["s42"], s0), (START["s42"], s0), None, None))
        for r in S42:
            for t in range(s0 + 262_144, e0 + 1, 262_144):
                out.append(("ck", f"{r}@{t}", (r, t), (r, t), None, None))
    return out


def main():
    mode, outp = sys.argv[1], Path(sys.argv[2])
    seeds = SEEDS if mode == "swaps" else [3042]
    done = set()
    if outp.exists():
        for line in open(outp):
            r = json.loads(line)
            done.add((r["cond"], r["who"], r["seed"]))
    fh = open(outp, "a")
    states_all = {}
    for cond, who, wts, base, donor, dims in plan(mode):
        todo = [s for s in seeds if (cond, who, s) not in done]
        if not todo:
            continue
        t0 = time.time()
        net = Net(*wts)
        own = Norm(*base)
        norm = own if donor is None else own.mixed(Norm(*donor), dims=dims)
        for seed in todo:
            row, states, zlog = rollout(net, norm, seed, record_z=True)
            ob = np.array([a for a, _ in zlog])
            zz = np.array([b for _, b in zlog])
            zf = np.clip((ob - norm.mean[6]) / math.sqrt(norm.var[6] + norm.eps), -norm.clip, norm.clip)
            zo = np.clip((ob - own.mean[6]) / math.sqrt(own.var[6] + own.eps), -own.clip, own.clip)
            g = gain(net, norm, states) if len(states) == 6 else float("nan")
            row.update({"cond": cond, "who": who, "weights": list(wts), "donor": list(donor) if donor else None,
                        "dims": list(dims) if dims else None, "gain": g, "z6_formula_maxdiff": float(np.max(np.abs(zz - zf))),
                        "z6_mean": float(np.mean(zz[500:])), "z6_mean_ownstats": float(np.mean(zo[500:])),
                        "scale6_per_deg": norm.scale_deg(6)})
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            states_all[f"{cond}__{who}__{seed}"] = states
        print(f"{cond} {who}: {len(todo)} ep {time.time() - t0:.1f} s", flush=True)
    fh.close()
    if states_all:
        np.savez_compressed(outp.with_suffix(f".states_{int(time.time())}.npz"), **states_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
