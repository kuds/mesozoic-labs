"""Fix lane: write text_corrections.txt, the exact replacement text for v3 (criteria_pilot4_draft_v3.txt, never
edited here) keyed to its line numbers. The OLD block of each correction is copied from v3 programmatically (so it is
verbatim); the numbers in the NEW blocks marked [measured: fix stage] are read from this lane's final validation
records (val_final/baseline_s*.json, written by the pinned pilot4_straight.py). Plain Python, standard library only.

usage: python -I -B make_text_corrections.py V3.txt VAL_FINAL_DIR OUT.txt
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

V3, VAL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
lines = V3.read_text().splitlines()
B = {s: json.loads((VAL / f"baseline_s{s}.json").read_text()) for s in (42, 52, 60)}
TOOL_SHA = {s: B[s]["tool_sha256"] for s in B}
assert len(set(TOOL_SHA.values())) == 1, TOOL_SHA


def e1(x):
    return f"{float(x):.2e}".replace("e-0", "e-")


def f1(x):
    return "inf" if x == "inf" else f"{float(x):.1f}"


def tw(s):
    return B[s]["checks"]["turned_twin"]


def per_seed(fn, sep=" / "):
    return sep.join(fn(s) for s in (42, 52, 60))


c = {s: B[s]["parts"]["c"] for s in B}
twin_max = max(tw(s)["max_abs_dev_deg"] for s in B)
twin_eps = sum(tw(s)["episodes"] for s in B)
twin_bad = {s: [e1(abs(x["dev_deg"])) for x in sorted(tw(s)["bad"], key=lambda x: -abs(x["dev_deg"]))] for s in B}
act_max = max(tw(s)["max_action_diff_after_651"] for s in B)
prefix = per_seed(lambda s: f"{B[s]['checks']['prefix']['equal']}/{B[s]['checks']['prefix']['episodes']}")
obs651 = per_seed(lambda s: f"{B[s]['checks']['obs651']['only_command_channels']}/{B[s]['checks']['obs651']['episodes']}")
pip = {s: B[s]["recovery"]["prey_in_place"]["summary"] for s in B}
rst = {s: B[s]["recovery"]["turned_at_reset"]["summary"] for s in B}
crab = per_seed(lambda s: f"{B[s]['parts']['b']['report_crab_median_deg']:+.1f}")
cross = per_seed(lambda s: f"{B[s]['parts']['b']['report_end_cross_track_median_m']:+.2f}")
REC = "SP/work/pilot4/tools/out/fix/val_final/baseline_s42.json, _s52, _s60"

C: list[dict] = []


def add(cid, source, start, end, new, why, insert_after=False, rewrap=None):
    """rewrap (first indent, continuation indent): the NEW block is one paragraph, re-wrapped at 118 columns."""
    new = new.rstrip("\n")
    if rewrap:
        new = textwrap.fill(" ".join(new.split()), width=118, initial_indent=rewrap[0], subsequent_indent=rewrap[1],
                            break_long_words=False, break_on_hyphens=False)
    C.append({"id": cid, "source": source, "start": start, "end": end, "new": new, "why": why,
              "insert_after": insert_after})


# ---------------------------------------------------------------- the review findings that belong in the text
add("C1", "validity review, should-fix 1 (anchors) and 2 (env-kwargs drift); review_code: the judge's env binding",
    704, 707, """\
    - New env binding: the recorded env kwargs (stage_config.json's reward_weights) carry command_mode "heading" and
      the committed command table key for key (same keys, same values of the same JSON types), prey_lateral_range
      [0, 0], the price 0.2 / 0.5 and R6's four keys, and equal, key for key, the env the judging tree builds (its
      constructor defaults and committed [env]: the gait probe's own drift record, computed by the judge before the
      probe and read again from the probe's JSON after it, must be empty); the recorded task_sha256 equals the one
      the tree derives (914a1cae... at 0bcb0cf, 7700f28 and 0e6a57b), as hyperparameters_sha256 already does
      (fbc94bef...), and at the pilot budget the tree's two derivations and its committed table must also be this
      text's own (the full digests under Run handling, the table under What is trained), so a record and a tree
      that moved together are refused.""",
    "The judge compared the run's record only with whatever the judging tree derives, and the probe only warns about "
    "an env-kwargs drift; both are now refusals (pilot4_judge.py: anchor_problems, env_kwargs_drift, drift_problem).")
add("C2", "validity review, minor (wording); judge lane text issues 1-3",
    710, 713, """\
    - Kept: the clean tree on a pilot-walk-* head (checked out, or detached at the tip of the local pilot-walk-*
      branch; --expect-head, default the head the runs record, 0e6a57b), stage 2, initialize_next_stage, the seed,
      git_commit = HEAD, the plant digests, intact pairs, --final-fallback, --check-only, one run per seed in OUT,
      the probe's fixed arguments. Added: the cadence 65536 / 10 / 250000 / 0 (Run handling), checked when the
      budget is the pilot's 2,000,000 (a validation-only smoke copy records it unchecked: the smoke overrides it in
      memory). Unit tests: pilot 3's adapted, plus refusals of pilot 3's stance parent, another seed's 8M parent, a
      -2.0 record, a 0.2 ramp, prey [-2, 2], the price off, a rewritten command table, an env-kwargs drift, and a
      tree whose digests or table moved with the record.""",
    "What the judge does: the shared tree is detached at the branch tip; the cadence is checked at the pilot budget "
    "only; --expect-head now has a default (fix stage).")
add("C3", "validity review, should-fix 2 (env-kwargs drift): the first-hour list",
    523, 525, """\
  sha256:914a1caee2c086a09ef66e9e6d4ad234dd085b69c4da687cc29af82b110fcd7a, env command_mode heading with the
  committed table, prey_lateral_range [0, 0], walk_heading_price_weight 0.2 and _scale_rad 0.5, R6's four keys, no
  other env kwarg different from the committed TOML (stage_config.json's reward_weights against the env the tree
  builds: the judge refuses a run whose record differs, as the probe's own drift record would show), the
  seed. A run whose record differs is a launch fault (stopped, restarted under _b), not a readout [P3 L670-681]. The""",
    "A run whose other env weights (forward_vel_weight, fall_penalty, ...) differ from the committed TOML was judged "
    "and scored before the fix; the first-hour list now names the check the judge makes.")
add("C4", "validity review, should-fix 3; review_code, should-fix 2 (the scorer trusted a record's verdicts)",
    717, 720, """\
    - New, in new functions with tests: --straight (the straightness tool's JSON per seed) and --straight-review (the
      look's JSON); the four seed outcomes; the pilot outcome over them; the report-only count of seeds passing (a),
      (b) and the look; the existence reading; PENDING_REVIEW;
      (c)'s NOT READ printed with its counts. A straightness record is read only when it is the seed-mode record of
      that very cell (the sha256s of its binding.json and probe.json, its checkpoint and sidecar sha256s, head and
      final-model flag), written by the pinned pilot4_straight.py (its sha256 in SHA256SUMS.txt) with this text's
      bars and windows (the record's constants), and only when each part's verdict follows from the record's own
      values under those bars; otherwise the seed waits for a record. A cell whose probe.json records an env-kwargs
      drift is not scored.""",
    "A record from an edited tool, with looser bars, or with PASS verdicts over failing values decided the outcome "
    "(the review's forged record read STRAIGHT WALKS); pilot4_score.py now refuses each (recomputed_verdicts, "
    "STRAIGHT_TOOL_SHA256, STRAIGHT_CONSTANTS, score_cell).")
add("C5", "validity review, minor (the crab is unquantified on the look's figure); review_code: seed-mode drift",
    744, 744, f"""\
    - The parts' verdicts with their bars as named constants; the top-down path plot and heading curves for the look;
      report only, the medians over the full-length panel episodes of the 1-s crab (course minus heading) and of the
      end cross-track, in the record, the printout and on the path plot (entry: {crab} deg and {cross} m
      [measured: fix stage, {REC}]). Seed mode also refuses a cell whose probe.json records an env-kwargs drift.""",
    "v3 relies on the look alone against a crab; the tool computed the per-episode crab but showed it nowhere.", rewrap=("    ", "      "))
add("C6", "review_code and validity review, minor: the turned-twin tolerance (a maintainer decision; proposed: keep "
    "1e-5 and record the flag as expected)",
    740, 743, f"""\
      only). Tests: states 0..650 equal the panel episode's bitwise; a 0-deg nudge reproduces the panel episode
      bitwise; with the command columns zeroed (the entry baseline), the nudged episode equals the panel episode
      turned by the angle (heading + angle to 1e-5 deg at 10 s; on the {twin_eps} nudged entry episodes the largest
      deviation is {e1(twin_max)} deg: s42 flags {len(twin_bad[42])} of {tw(42)['episodes']} beyond 1e-5
      ({', '.join(twin_bad[42])} deg), s52 {len(twin_bad[52])} of {tw(52)['episodes']}, s60 {len(twin_bad[60])} of
      {tw(60)['episodes']}, so the tool flags them and its s42 baseline run exits 1; that flag is expected and holds
      nothing, every deviation being six orders below the 15-deg offset [measured: fix stage, {REC}]) and the
      state-651 observation differs from the un-nudged one only in obs[60:63]; the pass check at state 650; NOT READ
      below 30 counting episodes.""",
    "Applied literally the 1e-5 deg tolerance (from 15 episodes) flags 3 s42 episodes on real dynamics; the tool keeps "
    "v3's 1e-5. ALTERNATIVE (a), not applied: write '1e-4 deg' here, set pilot4_straight.TWIN_HEADING_TOL_DEG = 1e-4 "
    "(the scorer's STRAIGHT_CONSTANTS and STRAIGHT_TOOL_SHA256 follow), change the record test's expectation to 0 "
    "flagged on every seed, re-run the three baselines and re-pin.", rewrap=("      ", "      "))
add("C7", "validity review, minor; straightness lane text issue 2 (a nan at an integer position)",
    355, 357, """\
  inf, and also when only the upper one is and the interpolation fraction is 0.5 or more, as at n = 30 with 2 inf
  [measured: postfix/text/out/pct_inf.txt]; the tool maps any nan to inf, also at an integer position with an inf
  above it, e.g. the median of 31 values with 15 inf, where p95 is inf too, so no (c) verdict changes); inf fails
  every bar, so with 40 values one inf passes a p95 bar and two fail it [measured: review_text/out/review_calc.txt
  P4].""",
    "numpy returns nan at an integer position with an inf above it although the order statistic there is finite; the "
    "tool reads it inf (pinned by a test); verdict-neutral at 30-40 counting episodes.")
add("C8", "validity review, minor: what (c) admits (insert after line 623)",
    623, 623, """\
  - Likewise (c) reads one 1-s mean (states 900-1000): a weave or an overshoot after the nudge that is back by 9 s
    passes it (a synthetic 1.01-s, 20-deg zig-zag after the nudge reads median r 0.2 deg [measured as recorded:
    SP/work/pilot4/tools/out/review_validity/logs/adversarial.txt]); only the look's recovery heading curves see it (Q25 names the
    alternative, the worst of m_900..m_950).""",
    "A Known limit v3 did not list.", insert_after=True, rewrap=("  ", "    "))
add("C9a", "validity review, minor: the (c) baseline is now measured on every nudge",
    442, 447, f"""\
    prey turns with them, the command-blind walkers pilot 4 starts from keep the offset (heading within
    {e1(twin_max)} deg of the un-nudged episode plus the angle at 10 s on {twin_eps} of {twin_eps} nudged entry
    episodes, actions within {e1(act_max)}; states 0..650 equal {prefix}; the state-651 observation beyond 1e-3 only
    in obs[60:63] on {obs651}): median r {per_seed(lambda s: f1(c[s]['median_r_deg']))} deg, p95
    {per_seed(lambda s: f1(c[s]['p95_r_deg']))}, within 11.5 deg on {c[42]['within_11_5']}/{c[42]['n_counting']},
    {c[52]['within_11_5']}/{c[52]['n_counting']} and {c[60]['within_11_5']}/{c[60]['n_counting']}, share removed 0,
    so (c) fails on all three [measured: pilot4_straight.py baseline mode on every nudge, fix stage, {REC};
    reproduced by an independent rollout within 6.1e-8 deg, SP/work/pilot4/tools/out/review_validity/data/cmp_s*.json]. With""",
    "The property was measured on 15 episodes and inferred for the rest; the tool's baseline mode now measures it on "
    "all 118.", rewrap=("    ", "    "))
add("C9b", "validity review, minor: the recovery table's proposed (c) row",
    882, 886, f"""\
    proposed (c), prey turned with the walker (the fixed mechanics): r median
      {per_seed(lambda s: f1(c[s]['median_r_deg']))}, p95 {per_seed(lambda s: f1(c[s]['p95_r_deg']))}, within 11.5 on
      {c[42]['within_11_5']}/{c[42]['n_counting']}, {c[52]['within_11_5']}/{c[52]['n_counting']},
      {c[60]['within_11_5']}/{c[60]['n_counting']}, share 0: FAIL on all three [measured: pilot4_straight.py
      baseline mode on every nudge, fix stage, {REC}; the command-blind walker keeps the offset on {twin_eps} of
      {twin_eps} nudged entry episodes]""",
    "As C9a, in the table.", rewrap=("    ", "      "))
add("C10", "validity review, minor; judge lane text issue 6: the resume floor",
    533, 533, """\
  periodic pair (250,000 or later; at the fixed cadence there is no earlier pair, so the judge does not check the
  floor) and is never off-schedule; the RESUME cell's own refusal stays: a run whose final""",
    "The judge accepts a resume from any periodic pair; moot at checkpoint_freq 250,000, which the judge enforces.")
add("C11", "review_code, should-fix 1: the smoke runner could reach a full training run",
    766, 768, """\
  The smoke runner: nb_pilot_runner4.py from nb_pilot_runner3.py (7f533128...): the pilot-4 cells, the local parent
    pair, MANUAL_VECNORM_PATH explicit, the cadence override 65,536 / 250,000 -> 32,768 in memory, the expected entry
    lines as measured; it refuses, before any cell, a launch or a resume whose config QUICK_TEST is not true unless
    --stop-at-learn (learn() stopped: nothing trains) or --plan-only (nothing runs), so the only training it can run
    is the pre-registered QUICK_TEST CPU smoke.""",
    "Inherited from pilot 3's runner; now refused (pinned by tests and by the e2e on the real notebook).")

# ---------------------------------------------------------------- the build lanes' text issues, settled the same way
add("C12a", "straightness lane text issues 3-4: the report-only variants under the fixed mechanics",
    448, 451, f"""\
    the prey left in place, under the first draft's mechanics (mj_forward and a rebuilt observation at state 650),
    they read median r 21.6 / 34.9 / 28.2 deg (under the fixed mechanics {per_seed(lambda s: f1(pip[s]['median_r_deg']))}
    [measured: fix stage, {REC}]), share removed -0.11 to +0.05 (medians per seed and sign), and the
    nudge made no walker fall (118 nudged episodes, all counting under the pass rule; s52's two left out are the
    panel fall at reset 3050, 90 steps) [measured: out/recov_rule.txt, out/recov_summary.txt]. A turn at the reset""",
    "v3's variant numbers come from the first draft's mechanics; the tool applies the fixed mechanics.", rewrap=("    ", "    "))
add("C12b", "straightness lane text issue 3: the variant turned at the reset",
    452, 453, f"""\
    instead is not discriminating: s60 passes it as it is (read at 5 s, first draft's mechanics: median 1.8, p95 6.8,
    40 of 40 within the band, share removed 0.91; under the fixed mechanics, the turn after state 0 with the prey
    left in place, median r {per_seed(lambda s: f1(rst[s]['median_r_deg']))}, within 11.5 deg
    {per_seed(lambda s: str(rst[s]['within_11_5']))} of 40 [measured: fix stage, {REC}]) by steering at the prey ahead
    [measured: out/recov_summary.txt]; so it is report only.""",
    "The same for the turned-at-reset variant (report only either way).", rewrap=("    ", "    "))
add("C13", "readouts lane text issues 1-3: the smoke's own entry lines",
    665, 668, """\
  - The notebook smoke on CPU with a smoke runner rebuilt for pilot 4 (QUICK_TEST, the cadence override to 32,768 in
    memory, RUN_ID pilot4_walk_s42 under the quick-test tree; the only training permitted then is the pre-registered
    65,536-step CPU smoke, as in pilot 3 [P3 L1040-1049]), the RESUME test from its 32,768 pair, and the judge's
    validation steps on its copies (refusals included). Locally the setup cell prints "Running locally; using this
    checkout." in place of the Colab "Repository ref ... commit:" line (the runner checks the clone's HEAD instead),
    the manual cell prints "Timesteps: 50,000" under QUICK_TEST, and the first evaluation line reads
    "Eval num_timesteps=32768" under the override [measured as recorded: the readouts lane, learn() stopped]. The
    RESUME test also shows that the resumed run's""",
    "v3's launch lines are the Colab run's; the smoke's differ in three lines, which the runner pins.")
add("C14", "adapted-readouts lane text issues 1-3: the noise readout",
    756, 758, """\
  pilot4_noise.py from pilot3_noise.py (c191c72a...): the warm-start reference becomes each parent's exp(log_std)
    (mean, min, max per seed, the seed read from the RUN_ID), no warm-up section (WARMUP_STEPS 0) but a section over
    the first 65,536 steps (the fresh ret_rms, as Report only asks), BLOCK 250,000, and the forward references pilot
    3's own last evaluation at or before 8M (7,995,392) on R6's scale (1943.495 / 1912.660 / 1897.846).""",
    "The plan and the Report-only list disagreed on a first-65,536 section; the forward reference's evaluation was "
    "not named.")
add("C15", "readouts lane text issues 4-6: the mechanics and mirror references",
    579, 587, """\
    left/right asymmetry ratio; pilot 3's 8M values for comparison: s42 knee 12 vs 28 deg per stride, right landing
    3.1 BW vs left 1.6, left foot 0.24 m behind; s52 early scuff in 56% of right swings (the per-episode median;
    384/702 = 54.7% pooled); s60 a 29-deg reversal and a 3.8-deg right lean; asymmetry 2.06x / 1.97x / 1.80x (the
    gait ledger's leg work: the median over the last 6 evaluations' full-horizon episodes of r_leg_work_abs_j /
    l_leg_work_abs_j; the panel's leg power reads 2.38 / 2.03 / 2.24); duty L-R median -0.09 / -0.01 / -0.19
    [measured as recorded: RL 1079, 1084, 1071].
  - The habit readout (pilot 3's definitions [P3 L797-847], on this run's ledger, 30 evaluations from 65,536; the
    trace at the deciding checkpoint); it picks nothing.
  - Mirror-equivariance: the mirrored policy pi_m = M_a pi(M_o o) on 10 panel resets (does the turn flip?), its
    handedness (yaw(pi) - yaw(pi_m)) / 2 against pilot 3's -26 / -47 / -33 deg, and the mirror-KL loss on the walker's
    own states (resets 3042-3061) against pilot 3's 42-79 (the checkpoints' own statistics: 78.508 / 41.671 / 54.005;
    symmetrised 70.724 / 41.119 / 54.832) [measured as recorded: VEER DIAGNOSIS findings (2) and (5); the screen
    lane's lk_check.txt].""",
    "The ratio, the scuff share and the loss's statistics were not defined; pilot4_mech.py and pilot4_mirror.py follow "
    "these sources literally.")
add("C16", "score lane text issues 1 and 5: pending and the existence reading's finality",
    467, 468, """\
  A WALKS seed whose video review, straightness record or (when (a) and (b) pass) look is still missing and could
  change its outcome is "pending" (its possible outcomes are printed). The three parts are printed for every WALKS
  seed, with their values, whether they pass or not; for a seed that is not WALKS they are printed report only.""",
    "v3 named only the review and the look; a missing record is pending too (never INCOMPLETE).")

# ---------------------------------------------------------------- write
digest = hashlib.sha256(V3.read_bytes()).hexdigest()
out = [
    "TEXT CORRECTIONS for criteria_pilot4_draft_v3.txt (fix stage, 2026-11-06). v3 itself is NOT edited.",
    f"v3: {V3} (sha256 {digest}, {len(lines)} lines).",
    "Each correction: the v3 line range, the OLD text copied from v3 verbatim (line-numbered), and the NEW text that",
    "replaces it (INSERT AFTER: the NEW text goes after the named line, nothing is removed). Numbers marked [measured:",
    f"fix stage] are read from this lane's validation records, written by the pinned pilot4_straight.py "
    f"({TOOL_SHA[42]}).",
    "Apply bottom-up (highest line first) so the line numbers stay valid. Every correction follows a fix made in the",
    "pinned tools or states what they do; none changes a bar, a seed, a budget or an outcome rule.",
    "",
]
for e in C:
    old = "\n".join(f"  {n:4d} | {lines[n - 1]}" for n in range(e["start"], e["end"] + 1))
    rng = f"line {e['start']}" if e["start"] == e["end"] else f"lines {e['start']}-{e['end']}"
    out += ["=" * 118, f"{e['id']}  v3 {rng}{'  (INSERT AFTER)' if e['insert_after'] else ''}",
            textwrap.fill(f"source: {e['source']}", width=118, subsequent_indent="        "),
            textwrap.fill(f"why: {e['why']}", width=118, subsequent_indent="     "),
            "OLD:" if not e["insert_after"] else "AFTER:", old,
            "NEW:" if not e["insert_after"] else "INSERT:", e["new"], ""]
    assert all(len(ln) <= 118 for ln in e["new"].splitlines()), (e["id"], [len(ln) for ln in e["new"].splitlines()])
notes = ["The lineage's place in the archive (the SB3 'data' JSON member) is still to be read off the notebook smoke's "
         "pair (v3 L526-527 already says so).",
         "The habit readout's head-tip reference labels (pilot 3's stance parent 0.904 m, the statue 0.959 m): v3 names "
         "none for pilot 4; report only.",
         "pilot3_vg.py's docstring keeps pilot 3's ramp remark (a byte copy); with pilot 4's inert shaping V and G share "
         "one reward at every checkpoint, as v3 L599-600 says."]
out += ["=" * 118, "Not changed in the text (recorded for the freeze):"]
out += [textwrap.fill(n, width=118, initial_indent="  - ", subsequent_indent="    ") for n in notes] + [""]
OUT.write_text("\n".join(out))
print(f"wrote {OUT}: {len(C)} corrections")
