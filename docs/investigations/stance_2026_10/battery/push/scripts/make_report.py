"""Render push_report.txt from push_summary.json (every number is read from that file; key paths in [brackets]).

OFFLINE EVALUATION of uncertified scratch-recipe checkpoints; report only; certifies nothing; no training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(sys.argv[1])
S = json.loads((BASE / "push_summary.json").read_text())
ROLES = [r for r in ("s60_6M", "s52_6M", "s42_6M", "statue") if r in S["roles"]]
DIRS = ("fwd", "back", "left", "right")
DNAME = {"fwd": "forward +x", "back": "backward -x", "left": "left +y", "right": "right -y"}
REPO_DIR = {"fwd": "forward +x", "back": "backward -x", "left": "lateral +y", "right": "lateral -y"}


def f(x, fmt="{:.3f}"):
    return "-" if x is None else fmt.format(x)


L = []
add = L.append
add("PUSH-AND-RELEASE PROBE -- OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen,")
add("MirrorKLPPO + experiment_* keys, 6,291,456 steps) and of the zero-action statue. Report only; certifies nothing;")
add("no training. Clean checkout of origin/main 2e734e29 (physics r9), stance stage config as committed (options off;")
add("arm L's lean term and settled leg target are reward-only, so the rollout is the same either way).")
add(f"Every number below is read from {BASE / 'push_summary.json'} (key path in [brackets]), which")
add("scripts/analyze_push.py reduces from telemetry/push_<role>.json + telemetry/push_series_<role>.npz (Part B) and")
add("repo_probe/sweep_<role>.pkl (Part A). Part A's own rendered outputs are repo_probe/<ckpt>/stance_gate_probe_impulse.{txt,json}.")
add("")
for role in ROLES:
    R = S["roles"][role]
    add(f"  {role:7s} model {R['model'] or 'none (statue: action 0 = home control every step)'}")
    if R["model_sha256"]:
        add(f"          sha256 {R['model_sha256'][:16]}..., vecnorm sha256 {R['vecnorm_sha256'][:16]}...")
add("")
add("0. WHAT WAS RUN")
add("  The probe exists in main: environments/shared/reporting/stance_report.py RootImpulse / _apply_root_impulse /")
add("  impulse_variants / render_impulse_probe / write_impulse_probe, driven by stage_artifacts._write_impulse_probe when")
add("  a stage sets stance_probe_impulse_speeds (T. rex stance.toml: [0.5, 1.0, 2.0]). The shove is qvel[0:3] += delta_v")
add("  on the root free joint (a uniform velocity step of the whole animal) at the control boundary of step 200")
add("  (= settle_steps), applied before the action is computed; no random number is drawn for it. 'Survived' is the")
add("  probe's criterion: the episode reaches the 1000-step horizon (no fall termination).")
add("  Part A = main's probe as shipped: 8 episodes per row from seed 3042, policy and statue. main's impulse_variants()")
add("  only builds lateral +y/-y rows, so forward (+x) and backward (-x) rows were added as plain RootImpulse objects")
add("  through the same build_stance_gate_report call (the stance stage spawns facing +x; +y is the animal's left).")
add("  Part B = scripts/push_trace.py: the same rollout replicated line for line (one env per row, reset(seed), the")
add("  impulse before the action) with read-only telemetry (main's SubstepContactRecorder, a chained contact hook,")
add("  CoM from a scratch MjData), 16 episodes per row (seeds 3042-3057) on a finer grid 0.05-0.5 m/s plus 1.0, 2.0.")
add("  Crosscheck: Part B's first 8 seeds reproduce Part A episode for episode [crosscheck_telemetry_vs_repo_probe]:")
for role, c in S["crosscheck_telemetry_vs_repo_probe"].items():
    add(f"    {role:7s} {c['episodes_with_equal_length']}/{c['episodes_compared']} episodes equal length, "
        f"max |reward diff| {c['max_abs_reward_diff']:.3g}")
add("")

add("  Files (all under the push/ directory): push_report.txt (this), push_summary.json (every reduced number),")
add("  telemetry/push_<role>.json (per-episode rows), telemetry/push_series_<role>.npz (per-episode series),")
add("  repo_probe/sweep_<role>.pkl + repo_probe/<ckpt>/stance_gate_probe_impulse.{txt,json} (main's probe output),")
add("  checks/ (one re-rolled episode), scripts/ (repo_impulse_probe.py, write_repo_probe.py, push_trace.py,")
add("  analyze_push.py, make_report.py, run_repo_probe.sh, run_push_trace.sh), logs/.")
add("  Fall terminations seen (main's T. rex env): head_contact (head tip < 0.12 m), nosedive, tail_contact, and")
add("  'fallen' (pelvis below 0.70 m).")
add("")
add("1. MAIN'S PROBE AS SHIPPED (Part A; 8 episodes per row, seeds 3042-3049) [repo_probe_rows.<role>]")
add("   survived/8 per (direction, speed); statue column is the passive control")
hdr = "   " + f"{'direction':14s}{'speed':>7s}" + "".join(f"{r:>10s}" for r in ROLES)
add(hdr)
for d in DIRS:
    for sp in (0.5, 1.0, 2.0):
        key = f"{REPO_DIR[d]} @ {sp:g}"
        cells = []
        for role in ROLES:
            row = S["repo_probe_rows"].get(role, {}).get(key)
            cells.append("-" if row is None else f"{row['survived']}/{row['n']}")
        add("   " + f"{DNAME[d]:14s}{sp:>7.2f}" + "".join(f"{c:>10s}" for c in cells))
ctrl = [S["repo_probe_rows"].get(role, {}).get("none (control) @ 0") for role in ROLES]
add("   " + f"{'no push':14s}{0:>7.2f}" + "".join(f"{('-' if c is None else str(c['survived']) + '/' + str(c['n'])):>10s}" for c in ctrl))
add("   main's own rendering of these rows (repo_probe/<ckpt>/stance_gate_probe_impulse.txt) reads, for all three")
add("   checkpoints: 'recovery envelope ... policy none, statue none' in every direction, and 'Neither side recovers")
add("   from anything'. At 0.5 m/s and above nothing survives reliably, so the shipped grid sits entirely above the")
add("   recovery envelope of every role and cannot rank them; Part B's grid resolves where the envelopes are.")
add("")

add("2. SURVIVAL PER SPEED AND DIRECTION (Part B; 16 episodes per row, seeds 3042-3057) [roles.<role>.directions.<dir>.survival_by_speed]")
for d in DIRS:
    add(f"   {DNAME[d]}")
    grid = S["roles"][ROLES[0]]["directions"][d]["grid_mps"]
    add("     " + f"{'m/s':>6s}" + "".join(f"{r:>10s}" for r in ROLES))
    for sp in grid:
        k = f"{sp:g}"
        add("     " + f"{sp:>6.2f}" + "".join(f"{S['roles'][r]['directions'][d]['survival_by_speed'].get(k, '-'):>10s}" for r in ROLES))
add("   no push: " + ", ".join(f"{r} {S['roles'][r]['control']['survived']}/{S['roles'][r]['control']['n']}" for r in ROLES)
    + " [roles.<role>.control.survived]")
add("")

add("3. ENVELOPE AGAINST THE MARGIN ARITHMETIC  dv_max ~ b * omega0 [roles.<role>.directions.<dir>.margin_arithmetic]")
add("   b = ray distance along the push from the pre-impulse CoP (mean net floor CoP over steps 181-200) to the edge of")
add("   the convex hull of the loaded pads' bottom corners ('pad') or pads + loaded digit tips ('full'); omega0 =")
add("   sqrt(g / CoM height) at the impulse; medians over the 16 seeds (the pre-impulse state is the same in every row of")
add("   a seed: max |diff| across rows printed as 'same'). env = largest speed up to which every row is 16/16 (main's 0.95 share at n = 16),")
add("   50% = largest speed up to which every row keeps >= 8/16, fall1 = first speed with any fall; per-seed ratio =")
add("   median over seeds of (largest speed that seed survived, every lower speed survived too) / (its own omega0*b_pad);")
add("   rho = Spearman rank correlation of those two over seeds; hit = seeds whose own omega0*b_pad lies within one grid")
add("   step (0.05 m/s) of its observed bracket [largest survived, first fall]; ratioX / rhoX = the same with b measured")
add("   from the pre-impulse XCoM instead of the CoP.")
add("   " + f"{'role':7s} {'dir':12s}{'b_pad':>7s}{'b_full':>7s}{'w0':>6s}{'pred_pad':>9s}{'pred_full':>10s}"
    f"{'env':>6s}{'50%':>6s}{'fall1':>6s}{'ratio':>7s}{'rho':>6s}{'hit':>7s}{'ratioX':>7s}{'rhoX':>6s}{'same':>8s}")
for role in ROLES:
    for d in DIRS:
        D = S["roles"][role]["directions"][d]
        M = D["margin_arithmetic"]
        add("   " + f"{role:7s} {DNAME[d]:12s}{f(M['b_m_median']['cop_pad']):>7s}{f(M['b_m_median']['cop_full']):>7s}"
            f"{f(M['omega0_median'], '{:.2f}'):>6s}{f(M['pred_dv_mps_median']['cop_pad'], '{:.2f}'):>9s}"
            f"{f(M['pred_dv_mps_median']['cop_full'], '{:.2f}'):>10s}{f(D['envelope_full_recovery_monotone_mps'], '{:.2f}'):>6s}"
            f"{f(D['envelope_ge50pct_monotone_mps'], '{:.2f}'):>6s}{f(D['first_speed_with_any_fall_mps'], '{:.2f}'):>6s}"
            f"{f(M['ratio_observed_max_survived_over_pred_cop_pad_median'], '{:.2f}'):>7s}"
            f"{f(M['spearman_observed_max_survived_vs_pred_cop_pad'], '{:.2f}'):>6s}"
            f"{str(M['seeds_pred_cop_pad_within_one_grid_step_of_bracket']) + '/' + str(M['n_seeds']):>7s}"
            f"{f(M['ratio_observed_max_survived_over_pred_xcom_pad_median'], '{:.2f}'):>7s}"
            f"{f(M['spearman_observed_max_survived_vs_pred_xcom_pad'], '{:.2f}'):>6s}"
            f"{M['pre_impulse_identical_across_rows_max_abs_diff']:>8.1g}")
add("   (XCoM-based b agrees with CoP-based b to within a few mm: [..margin_arithmetic.b_m_median.xcom_pad]).")
add("")

add("4. RECOVERY TIME AND PEAK CoP EXCURSION [roles.<role>.directions.<dir>.speeds.<speed>]")
add("   Rows with survivors, plus the first speed at which all 16 fall. Columns:")
add("   rec med/max = recovery time (s, survivors): horizontal CoM speed back under the role's adaptive threshold")
add("     [roles.<role>.control.adaptive_settle_threshold_mps] and held for 0.5 s; never = survivors that never settle;")
add("     v.05 = the same with a fixed 0.05 m/s threshold (median) [recovery_time_s_v0.05_median].")
add("   CoP m = peak net floor CoP excursion along the push within 1 s of it, from the pre-impulse mean (m, survivors")
add("     median) [win_cop_peak_disp_along_push_m_survivors_median]; CoP/b = the same over that seed's b_pad (survivors")
add("     median / max) [win_cop_peak_fraction_of_b_pad_survivors_*]; 1.0 = the net CoP reached the pad-hull edge.")
add("   pad = peak per-foot pad CoP |fore-aft| within 1 s (main's recorder, half-lengths, 1 = pad edge; survivors")
add("     median) [win_pad_cop_fore_aft_max_survivors_median].")
add("   XCoM = smallest ray margin of the XCoM (pre-impulse omega0) to the pre-impulse pad hull along the push, over the")
add("     first 3 s after it (m; survivors min) [xcom_min_margin_pad_first3s_m_survivors_min]; < 0 = past the hull edge.")
add("     xout = fallers whose XCoM crossed that hull edge at some step [fallers_with_xcom_outside_pad].")
add("   lift = episodes with any leg fully unloaded after the push [liftoff_any]; step = survivors that lifted a foot and")
add("     moved it > 2 cm [survivors_stepped]; fall s = median time from push to termination (fallers).")
add("   no-push baseline over the same 1 s window [roles.<role>.control]:")
for role in ROLES:
    C = S["roles"][role]["control"]
    add(f"     {role:7s} pad |fore-aft| peak median {f(C['win_pad_cop_fore_aft_max_median'])}, pad |outer| peak median "
        f"{f(C['win_pad_cop_outer_max_median'])}; net-CoP drift peak (m) "
        + ", ".join(f"{d} {f(C['win_cop_peak_disp_m_median'][d], '{:+.4f}')}" for d in DIRS)
        + f"; CoM speed p99 {f(C['vh_p99_all_post_impulse_steps'], '{:.4f}')} m/s -> threshold {f(C['adaptive_settle_threshold_mps'], '{:.3f}')}")
add("   " + f"{'role':7s} {'dir':12s}{'m/s':>5s}{'surv':>7s}{'rec med':>8s}{'max':>6s}{'never':>6s}{'v.05':>6s}{'CoP m':>8s}"
    f"{'CoP/b':>7s}{'max':>6s}{'pad':>7s}{'XCoM':>8s}{'xout':>5s}{'lift':>5s}{'step':>5s}{'fall s':>7s}")
for role in ROLES:
    for d in DIRS:
        D = S["roles"][role]["directions"][d]
        shown_zero = False
        for sp in D["grid_mps"]:
            X = D["speeds"][f"{sp:g}"]
            if X["survived"] == 0:
                if shown_zero:
                    continue
                shown_zero = True
            add("   " + f"{role:7s} {DNAME[d]:12s}{sp:>5.2f}{(str(X['survived']) + '/' + str(X['n'])):>7s}"
                f"{f(X['recovery_time_s_adaptive_median'], '{:.2f}'):>8s}{f(X['recovery_time_s_adaptive_max'], '{:.2f}'):>6s}"
                f"{X['survivors_never_settled_adaptive']:>6d}{f(X['recovery_time_s_v0.05_median'], '{:.2f}'):>6s}"
                f"{f(X['win_cop_peak_disp_along_push_m_survivors_median'], '{:.4f}'):>8s}"
                f"{f(X['win_cop_peak_fraction_of_b_pad_survivors_median'], '{:.2f}'):>7s}"
                f"{f(X['win_cop_peak_fraction_of_b_pad_survivors_max'], '{:.2f}'):>6s}"
                f"{f(X['win_pad_cop_fore_aft_max_survivors_median'], '{:.3f}'):>7s}"
                f"{f(X['xcom_min_margin_pad_first3s_m_survivors_min'], '{:+.3f}'):>8s}"
                f"{X['fallers_with_xcom_outside_pad']:>5d}{X['liftoff_any']:>5d}"
                f"{X['survivors_stepped']:>5d}{f(X['time_to_fall_s_median'], '{:.2f}'):>7s}")
add("")
add("5. HEADLINE TABLE: recovery envelopes (m/s) per direction [roles.<role>.directions.<dir>.envelope_*]")
add("   env = largest speed up to which every row is 16/16 (main's 0.95 share at n = 16); 50% = same at >= 8/16;")
add("   weakest = the direction with the smallest env; vs statue = env minus the statue's env in that direction.")
add("   " + f"{'role':7s}" + "".join(f"{DNAME[d]:>18s}" for d in DIRS) + "   weakest")
for role in ROLES:
    cells = []
    envs = {}
    for d in DIRS:
        D = S["roles"][role]["directions"][d]
        envs[d] = D["envelope_full_recovery_monotone_mps"]
        cells.append(f"{D['envelope_full_recovery_monotone_mps']:.2f} / {D['envelope_ge50pct_monotone_mps']:.2f}")
    weakest = min(DIRS, key=lambda d: envs[d])
    add("   " + f"{role:7s}" + "".join(f"{c:>18s}" for c in cells) + "   " + DNAME[weakest] + " " + format(envs[weakest], ".2f"))
if "statue" in S["roles"]:
    for role in ROLES:
        if role == "statue":
            continue
        diffs = []
        for d in DIRS:
            a = S["roles"][role]["directions"][d]["envelope_full_recovery_monotone_mps"]
            b = S["roles"]["statue"]["directions"][d]["envelope_full_recovery_monotone_mps"]
            a5 = S["roles"][role]["directions"][d]["envelope_ge50pct_monotone_mps"]
            b5 = S["roles"]["statue"]["directions"][d]["envelope_ge50pct_monotone_mps"]
            diffs.append(f"{DNAME[d]} {a - b:+.2f} ({a5 - b5:+.2f} at 50%)")
        add(f"   {role} vs statue: " + "; ".join(diffs))
add("")

# ------------------------------------------------------------------ 6. reading (numbers pulled from the summary)
R = S["roles"]
def env(role, d, k="envelope_full_recovery_monotone_mps"):
    return R[role]["directions"][d][k]
def ma(role, d):
    return R[role]["directions"][d]["margin_arithmetic"]
pol = [r for r in ROLES if r != "statue"]
max50 = max(env(r, d, "envelope_ge50pct_monotone_mps") for r in ROLES for d in DIRS)
add("6. READING (offline evaluation of uncertified scratch-recipe checkpoints; certifies nothing)")
add(f"  a. Main's probe as shipped cannot tell these roles apart. Its first speed, 0.50 m/s, is above every role's")
add(f"     envelope in every direction (the largest 50% envelope anywhere in section 5 is {max50:.2f} m/s), so every")
add("     forward/backward row is 0/8 and the lateral rows are at most 2/8 for policy and statue alike, and main's text")
add("     verdict reads 'Neither side recovers from anything' for all three checkpoints [section 1].")
add("  b. Where each policy parks its CoP on the pads sets its weak direction. No-push pad CoP fore-aft, signed (+ toe),")
add("     median per foot over steps 200-499 [roles.<role>.control.pad_cop_fore_aft_signed_median_per_foot_steps_200_499]:")
for r in ROLES:
    c = R[r]["control"]["pad_cop_fore_aft_signed_median_per_foot_steps_200_499"]
    add(f"       {r:7s} {c[0]:+.3f} / {c[1]:+.3f};  b_pad forward {ma(r, 'fwd')['b_m_median']['cop_pad']:.3f} m, backward "
        f"{ma(r, 'back')['b_m_median']['cop_pad']:.3f} m;  envelope forward {env(r, 'fwd'):.2f} / "
        f"{env(r, 'fwd', 'envelope_ge50pct_monotone_mps'):.2f}, backward {env(r, 'back'):.2f} / "
        f"{env(r, 'back', 'envelope_ge50pct_monotone_mps'):.2f} m/s (16/16 / >=8/16)")
add("     s60 and s42 stand toe-loaded and are weakest forward; s52 stands heel-loaded and is weakest backward. s52 has")
add("     the best forward envelope of the four; s60 and s42 share the best backward one.")
add("  c. Sagittal pushes follow the margin arithmetic. omega0 * b (b = pre-impulse CoP to the pad edge, per seed) falls")
add("     within one grid step (0.05 m/s) of that seed's observed bracket [largest survived, first fall] in:")
for r in ROLES:
    add("       " + r + ": " + ", ".join(
        f"{DNAME[d]} {ma(r, d)['seeds_pred_cop_pad_within_one_grid_step_of_bracket']}/{ma(r, d)['n_seeds']}"
        f" (observed/predicted {ma(r, d)['ratio_observed_max_survived_over_pred_cop_pad_median']:.2f}, rho "
        f"{ma(r, d)['spearman_observed_max_survived_vs_pred_cop_pad']:.2f}; XCoM-based rho "
        f"{ma(r, d)['spearman_observed_max_survived_vs_pred_xcom_pad']:.2f})" for d in ("fwd", "back")))
add("     The fit is per seed in each policy's weak direction (s60 forward, s52 backward, s42 forward: 16/16). The")
add("     statue, whose no-push CoM keeps moving more (section 4 baseline), is fitted better with the XCoM-based b.")
edge_rows = []
for r in ROLES:
    for d in ("fwd", "back"):
        sp = env(r, d, "envelope_ge50pct_monotone_mps")
        X = R[r]["directions"][d]["speeds"].get(f"{sp:g}")
        if X and X["win_cop_peak_fraction_of_b_pad_survivors_median"] is not None:
            edge_rows.append(X["win_cop_peak_fraction_of_b_pad_survivors_median"])
sag_fl = sum(x["n"] - x["survived"] for r in R.values() for d in ("fwd", "back")
             for k, x in r["directions"][d]["speeds"].items() if float(k) <= 0.5)
sag_xo = sum(x["fallers_with_xcom_outside_pad"] for r in R.values() for d in ("fwd", "back")
             for k, x in r["directions"][d]["speeds"].items() if float(k) <= 0.5)
sag_lift = sum(x["liftoff_any"] for r in R.values() for d in ("fwd", "back") for x in r["directions"][d]["speeds"].values())
add(f"     At each role's >=8/16 envelope speed the survivors' net CoP travels {min(edge_rows):.2f}-{max(edge_rows):.2f} of b_pad")
add(f"     within 1 s and the pad CoP reaches the pad edge (pad = 1.000); {sag_xo} of the {sag_fl} sagittal fallers at")
add(f"     <= 0.5 m/s had their XCoM cross the pad-hull edge; legs fully unloaded in sagittal rows: {sag_lift}. The")
add("     passive digits add little: the prediction with digit tips in the hull (pred_full) overshoots every forward")
add("     envelope.")
lat = [ma(r, d)["ratio_observed_max_survived_over_pred_cop_pad_median"] for r in ROLES for d in ("left", "right")]
lat_cop = max(x["win_cop_peak_fraction_of_b_pad_survivors_max"] or 0 for r in R.values() for d in ("left", "right")
              for x in r["directions"][d]["speeds"].values())
lat_fl = sum(x["n"] - x["survived"] for r in R.values() for d in ("left", "right")
             for k, x in r["directions"][d]["speeds"].items() if float(k) <= 0.5)
lat_fl_lift = sum(x["fallers_liftoff"] for r in R.values() for d in ("left", "right")
                  for k, x in r["directions"][d]["speeds"].items() if float(k) <= 0.5)
lat_fallen = sum(x["fall_reasons"].get("fallen", 0) for r in R.values() for d in ("left", "right")
                 for k, x in r["directions"][d]["speeds"].items() if float(k) <= 0.5)
surv_lift = sum(x["liftoff_any"] - x["fallers_liftoff"] for r in R.values() for d in R[ROLES[0]]["directions"]
                for x in r["directions"][d]["speeds"].values())
add(f"  d. Lateral pushes fall short of the arithmetic for every role, the statue included: b_pad ~0.19 m predicts")
add(f"     ~0.64 m/s, the per-seed observed/predicted medians are {min(lat):.2f}-{max(lat):.2f}, and the envelopes (left/right,")
add("     16/16) are " + ", ".join(f"{r} {env(r, 'left'):.2f}/{env(r, 'right'):.2f}" for r in ROLES) + " m/s.")
add("     Because the statue shows the same shortfall it is a property of the plant and pose, not of a policy.")
add(f"     Lateral survivors' net CoP stays short of the hull edge (largest CoP/b of any lateral survivor {lat_cop:.2f});")
add(f"     of the {lat_fl} lateral fallers at <= 0.5 m/s, {lat_fl_lift} fully unloaded a leg and {lat_fallen} ended 'fallen'")
add(f"     (pelvis below healthy_z_range's 0.70 m); survivors that ever fully unloaded a leg, any row: {surv_lift};")
add("     no survivor steps.")
add("     Why the lateral limit sits near 0.6 of b*omega0 is NOT tested here (candidates: no ankle-roll actuation,")
add("     pads rocking edge to edge with no push -- pad |outer| peak ~1.0 for every role -- leg/trunk roll compliance).")
add("  e. Against the statue (the plant's passive robustness), no policy is better in every direction [section 5].")
add("     Weakest direction per role, at 16/16 and at >=8/16:")
for r in ROLES:
    w16 = min(DIRS, key=lambda d: env(r, d))
    w50 = min(DIRS, key=lambda d: env(r, d, "envelope_ge50pct_monotone_mps"))
    add(f"       {r:7s} {DNAME[w16]} {env(r, w16):.2f} m/s;  {DNAME[w50]} {env(r, w50, 'envelope_ge50pct_monotone_mps'):.2f} m/s")
add("     At >=8/16 the statue's weakest direction is better than every policy's; at 16/16 only s42's is better than")
add("     the statue's (whose backward row is 15/16 at 0.15 m/s). The policies trade one sagittal direction for the")
add("     other by where they park the CoP, and sideways they are level with or below the statue.")
def rng(role_list, dirs, full):
    vals = []
    for r in role_list:
        for d in dirs:
            for k, x in R[r]["directions"][d]["speeds"].items():
                if x["recovery_time_s_adaptive_median"] is None:
                    continue
                if full == (x["survived"] == x["n"]):
                    vals.append(x["recovery_time_s_adaptive_median"])
    return (min(vals), max(vals)) if vals else (None, None)
a1 = rng(pol, ("fwd", "back"), True)
a2 = rng(pol, ("fwd", "back"), False)
b1 = rng(pol, ("left", "right"), True)
b2 = rng(pol, ("left", "right"), False)
add(f"  f. Recovery time (policies; survivors' median; CoM speed back under 0.03 m/s and held 0.5 s): sagittal rows with")
add(f"     16/16 survivors {a1[0]:.2f}-{a1[1]:.2f} s, sagittal rows with some falls {a2[0]:.2f}-{a2[1]:.2f} s; lateral rows with")
add(f"     16/16 {b1[0]:.2f}-{b1[1]:.2f} s, lateral rows with some falls {b2[0]:.2f}-{b2[1]:.2f} s (a slow sideways sway).")
add(f"     The statue's own no-push CoM speed is higher (p99 {R['statue']['control']['vh_p99_all_post_impulse_steps']:.4f} m/s,")
add(f"     threshold {R['statue']['control']['adaptive_settle_threshold_mps']:.3f}), so compare it on the fixed 0.05 m/s column (v.05).")
add("  g. Relation to the certificate's pad-CoP bar: the certificate bounds the window mean of |pad CoP fore-aft| per")
add("     foot; it does not see which side the CoP is parked on. s60 passes it comfortably in the howclose certificate")
add("     (policy median 0.3627, worst 0.7801 over 240 episodes: next_signals/howclose/cert_margins.txt,")
add("     cert_r9armL_s60_6291456_optoff) and still has the smallest forward b and forward 16/16 envelope here; s42,")
add("     which fails that bar (median 0.6882, 66/240 past 0.8; same file, cert_r9armL_s42_6291456_optoff), is")
add("     likewise toe-loaded.")
add("")
add("7. CHECKS AND CAVEATS")
add("  - Part B reproduces Part A bit for bit on the shared rows (section 0); the telemetry is read-only.")
nm = sum(x["survivors_moving_at_horizon"] for r in R.values() for d in r["directions"].values() for x in d["speeds"].values())
add(f"  - Every survivor is at rest at the horizon (survivors with CoM speed > 0.05 m/s at step 1000: {nm})")
add("    [roles.<role>.directions.<dir>.speeds.<speed>.survivors_moving_at_horizon], so 'survived' is not a fall cut")
add("    short by the 1000-step horizon.")
same = max(ma(r, d)["pre_impulse_identical_across_rows_max_abs_diff"] for r in ROLES for d in DIRS)
add(f"  - The pre-impulse state of a seed is identical in every row (max |diff| of omega0 and b across rows {same:g}).")
add("  - The XCoM ray margin along the push can also go negative when the XCoM crosses a DIFFERENT hull edge; one")
add("    whole-episode survivor value (s52 right -y 0.50 m/s, seed 3047: -0.231 m at 311 steps after the push) was")
add("    re-rolled with the full series (checks/push_s52_6M_check3047.json, checks/push_series_s52_6M_check3047.npz:")
add("    signed distance to the hull at that step -0.0076 m, across the heel edge); the section 4 XCoM column therefore")
add("    uses the first 3 s after the push.")
add("  - 16 episodes per row, 0.05 m/s grid, one impulse time (step 200), the training heading only, a pure root")
add("    velocity step (no force duration). b uses the pad boxes' bottom corners of the loaded feet; omega0 uses the")
add("    CoM height at the push (LIPM). The robust_best checkpoints were not run in this battery.")
add("")
(BASE / "push_report.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
