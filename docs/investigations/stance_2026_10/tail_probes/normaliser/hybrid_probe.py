"""Where in the actor did the weight change that raised the s42 gain happen?  Network evaluation only; no simulation.

PYTHONPATH=<policies/code_copy>:<this folder> python -B hybrid_probe.py > hybrid.txt

The actor mean is action_net(tanh(L2(tanh(L0(z))))), L0 63->512, L2 512->256, action_net 256->15 (weights.txt).
For each s42 (and s80) end network, every one of the 8 combinations of {L0, L2, action_net} taken from the end or
from the start checkpoint is evaluated with the verifier's tail_3 angle sweep on the end's stored seed-3042 states
(ckpts/states), once with the end's statistics and once with the start's.  The loop gain of each hybrid shows which
layer's change carries the gain growth (and that the first layer's tail_3 column did not change: weights.txt).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

from probe_common import NAMES, SE, START_RUN, Agent, label, make_factory, stats, sweep_states

HERE = Path(__file__).resolve().parent
PARTS = ("L0", "L2", "OUT")


def modules(agent):
    p = agent.model.policy
    return {"L0": p.mlp_extractor.policy_net[0], "L2": p.mlp_extractor.policy_net[2], "OUT": p.action_net}


def snapshot(agent):
    return {k: {n: t.detach().clone() for n, t in m.state_dict().items()} for k, m in modules(agent).items()}


def load_parts(agent, snap):
    for k, m in modules(agent).items():
        m.load_state_dict(snap[k])


def main() -> int:
    factory = make_factory()
    for lin in ("s42", "s80"):
        s0, e0 = SE[lin]
        start = Agent(START_RUN[lin], s0, factory)
        sS = snapshot(start)
        mS, vS = stats(START_RUN[lin], s0)
        print(f"\n=== {lin}: loop gain of hybrids on each end's own states (S = part from the start, E = from the end)")
        combos = list(itertools.product("ES", repeat=3))
        print(f"  {'stats':6s} {'run':11s} " + " ".join(f"{''.join(c):>7s}" for c in combos) + "   (order L0 L2 OUT)")
        table = {"end": [], "start": []}
        for run in [r for r in NAMES if r.startswith(lin)]:
            ag = Agent(run, e0, factory)
            sE = snapshot(ag)
            X = np.load(HERE / "ckpts" / "states" / f"{label(run, e0)}_3042.npy")
            for stag in ("end", "start"):
                if stag == "end":
                    ag.set_stats(None)
                else:
                    ag.set_stats(mS, vS)
                gains = []
                for c in combos:
                    mix = {p: (sE if src == "E" else sS)[p] for p, src in zip(PARTS, c)}
                    load_parts(ag, mix)
                    with torch.no_grad():
                        gains.append(sweep_states(ag, X)["loop_gain"])
                load_parts(ag, sE)
                table[stag].append(gains)
                print(f"  {stag:6s} {run:11s} " + " ".join(f"{g:+7.3f}" for g in gains))
        for stag in ("end", "start"):
            med = np.median(np.array(table[stag]), axis=0)
            print(f"  {stag:6s} {'MEDIAN':11s} " + " ".join(f"{g:+7.3f}" for g in med))
            # one-part swaps from the all-end network, and one-part grafts onto the all-start network
            idx = {"".join(c): i for i, c in enumerate(combos)}
            T = np.array(table[stag])
            print(f"    stats={stag}: all-end {med[idx['EEE']]:+.3f}, all-start {med[idx['SSS']]:+.3f}; "
                  f"start's L0 into end {med[idx['SEE']]:+.3f}, start's L2 into end {med[idx['ESE']]:+.3f}, "
                  f"start's OUT into end {med[idx['EES']]:+.3f}; end's L0 alone {med[idx['ESS']]:+.3f}, "
                  f"end's L2 alone {med[idx['SES']]:+.3f}, end's OUT alone {med[idx['SSE']]:+.3f}")
            for name, col in (("SEE", "start L0 in end"), ("ESE", "start L2 in end"), ("EES", "start OUT in end")):
                d = T[:, idx[name]] - T[:, idx["EEE"]]
                print(f"      {col:18s}: change vs all-end median {np.median(d):+.3f} ({int((d < 0).sum())}/12 lower)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
