"""Lint-only edits to judge_stance3.py (no behaviour change), applied after the first validation queue."""
p = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance3/judge/judge_stance3.py"
s = open(p).read()
rep = [
    ('def _heading_xy(quat) -> "np.ndarray":', "def _heading_xy(quat):"),
    ("    for ev, pad, (tr, readings) in zip(evidence, pads, traces):",
     "    for ev, pad, (tr, readings) in zip(evidence, pads, traces, strict=True):"),
    ("for a, b in zip(xs, cm) if a is not None and b is not None),",
     "for a, b in zip(xs, cm, strict=True) if a is not None and b is not None),"),
    ("        def p(r, key):\n            return (r.get(\"pad\") or {}).get(f\"{foot}_{key}\")",
     "        def p(r, key, foot=foot):\n            return (r.get(\"pad\") or {}).get(f\"{foot}_{key}\")"),
    ("    for row, pad, (tr, readings) in zip(rows, pad_rows(ctx, TRACES), TRACES):",
     "    for row, pad, (tr, readings) in zip(rows, pad_rows(ctx, TRACES), TRACES, strict=True):"),
]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
old_print = s[s.index('        print(json.dumps({"panel": p["verdict"]'):s.index('                          "seconds": round(p["panel_seconds"], 1)}), flush=True)')]
new_print = '''        def r_mm(v):
            return None if v is None else round(1000 * v, 2)

        print(json.dumps({"panel": p["verdict"], "clean": p["n_clean"], "lcb": round(p["clean_lcb"], 4), "full": p["n_full_horizon"],
                          "hop_or_fall": p["hop_or_fall_episodes"], "k2a": s["k2a_count"], "k2b": s["k2b_count"],
                          "k2b_seeds": s["k2b_seeds"], "xc_med_mm": r_mm(s["xc_median_full_horizon_m"]),
                          "statue_xc_med_mm": r_mm(st["xc_median_full_horizon_m"]),
                          "roll_rail": s["roll_rail_full_horizon"], "reward": round(p["mean_reward"], 1),
                          "statue": round(p["statue_mean_reward"] or float("nan"), 1), "statue_falls": st["fall_seeds"],
'''
s = s.replace(old_print, new_print)
open(p, "w").write(s)
print("patched")
