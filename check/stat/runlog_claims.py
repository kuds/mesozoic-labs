"""Check the RUNLOG 09:03 UTC report-only claims against reads/<name>/markers.json (no simulation, no training)."""
import json
BS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/bias_screen"
names = [f"{a}_{s}" for s in (2101, 2111, 2121, 2131) for a in ("control", "critic_blind_tail3")]
par = json.load(open(f"{BS}/reads/parent_s42/markers.json"))
for L in par["loop"]:
    print("parent", L["num_timesteps"], "OWN G", round(L["OWN"]["P"]["G"], 4), "c0", round(L["OWN"]["P"]["c0"], 4),
          "REF c0", round(L["REF"]["P"]["c0"], 4))
pst = par["probe"]["steps"]; pmu = par["probe"]["rollout_mu14_mean"]
B_S = -0.03157372563146055
plast8 = [v for s, v in zip(pst, pmu) if 2424832 <= s <= 2482176]
print("parent last-8 mean (markers)", sum(plast8) / 8, "rise", sum(plast8) / 8 - B_S)
print("parent CMD_L last", par["ledger"]["direct"]["cmd_l_last"], "statue_settled trailing8 last",
      par["probe"]["trailing8_mean"][-1], "first_above", par["probe"]["trailing8_first_above"])
for n in names:
    m = json.load(open(f"{BS}/reads/{n}/markers.json"))
    d = m["ledger"]["direct"]
    p = m["probe"]
    mu = dict(zip(p["steps"], p["rollout_mu14_mean"]))
    prim = sum(mu[s] for s in range(2424832, 2482177, 8192)) / 8
    rows = []
    for L in m["loop"]:
        o, r = L["OWN"], L["REF"]
        rows.append((L["num_timesteps"], o["P"]["G"], o["P"]["c0"], r["P"]["G"], r["P"]["c0"], o["L"]["G"]))
    print(f"{n:24s} primary rise {prim - B_S:+.4f}  CMD_L last {d['cmd_l_last']:+.4f} max {d['cmd_l_max']:+.4f} "
          f"first>=0.30 {d['firsts']['cmd_l_ge_0.30']}  statue_settled trailing8 last {p['trailing8_mean'][-1]:+.4f} "
          f"max {p['trailing8_max']:+.4f} first_above {p['trailing8_first_above']}")
    print("   (step, OWN G, OWN c0, REF G, REF c0, OWN G_L):",
          [(s, f"{g:+.3f}", f"{c:+.4f}", f"{rg:+.3f}", f"{rc:+.4f}", f"{gl:+.3f}") for s, g, c, rg, rc, gl in rows])
    print(f"   OWN c0 max {max(x[2] for x in rows):+.6f}  REF c0 max {max(x[4] for x in rows):+.6f}  OWN G at T "
          f"{rows[-1][1]:+.6f}  first OWN G>0 at {next((x[0] for x in rows if x[1] > 0), None)}")
