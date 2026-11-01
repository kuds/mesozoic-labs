"""PC fixer (2026-11-01): apply the confirmed review findings to ST/criteria.txt and ST/COLAB_STEPS.txt (each change an
exact replacement that must match once, starting from the REVIEWED bytes in ST/review/fixer/orig/), mark the builder's
draft ST/build/COLAB_STEPS.txt superseded (I7), correct DESIGN_NOTES 2.4 (I2), and refill the criteria's PINS block
from files (never typed): WT's HEAD, the switch-ON digests (ST/tools/dry/pins_sc.out), the pins file, every PC tool,
COLAB_STEPS, the comparators' inputs.  Unified diffs against the reviewed bytes go to ST/review/fixer/*.diff.

    PY -I -B fix_texts.py RESULTS.json
RESULTS.json holds the fixer's measured validation sentences (written by the fixer from its result files)."""
from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ST = Path("/home/user/work/pc_study")
FX = ST / "review" / "fixer"
ORIG = FX / "orig"
T = ST / "tools"
WT = "/home/user/work/pc_branch"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wrap_long(text: str, width: int = 120) -> str:
    """Re-wrap only lines longer than width at spaces, continuation lines at the line's own indentation (two more
    after a "- " or "N. " item marker); every other line byte for byte."""
    out = []
    for line in text.split("\n"):
        if len(line) <= width:
            out.append(line)
            continue
        body = line.lstrip(" ")
        ind = " " * (len(line) - len(body))
        cont = ind + ("  " if body.startswith("- ") else "   " if body[:1].isdigit() and body[1:3] == ". " else "")
        out.extend(textwrap.wrap(body, width=width, initial_indent=ind, subsequent_indent=cont,
                                 break_long_words=False, break_on_hyphens=False))
    return "\n".join(out)


def apply(src: str, edits, name: str) -> str:
    for i, (old, new) in enumerate(edits, 1):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"STOP: {name} edit {i}: the old text occurs {n} times:\n{old[:300]}")
        src = src.replace(old, new)
    return src


def criteria_edits(R: dict) -> list:
    return [
        # ---- status
        ("ST/adapt/criteria_v1_to_installed.diff and ST/adapt/ADAPT_NOTES.txt; PINS filled from files. Still a draft for\n"
         "the independent review: nothing here is approved.\n",
         "ST/adapt/criteria_v1_to_installed.diff and ST/adapt/ADAPT_NOTES.txt; PINS filled from files. Still a draft for\n"
         "the independent review: nothing here is approved.\n"
         "FIXED 2026-11-01 (date -u) by the fixer after the independent review (code, design and process lenses): every\n"
         "finding's verdict and change in ST/review/FIXES.txt; this text against the reviewed bytes (sha256 6132d853...) in\n"
         "ST/review/fixer/criteria_reviewed_to_fixed.diff. The code now ends at the fixer's commit (m1, typing only).\n"
         "Still a draft for the re-review and the maintainer's confirmation: nothing here is approved.\n"),
        # ---- Q1: any tail actuator (I6)
        ("  Q1 With the phase critic from step 0, does any of the 3 runs develop the tail habit by 12M (ROUND's TAIL CLASS: RAIL or\n"
         "     PARKED; with the ledger's early signs printed:",
         "  Q1 With the phase critic from step 0, does any of the 3 runs develop the tail habit by 12M (ROUND's TAIL CLASS: RAIL or\n"
         "     PARKED; a RAIL on any tail actuator 11-14 counts, as ROUND; with the ledger's early signs printed:"),
        # ---- code line
        ("  panel per CHOICE K3), with ONE change: the switch below ON. Code: HEAD = 8737cd3d plus the builder's reviewed\n"
         "  commits (no header commit: APPROVED and RUNLOG pin HEAD and this text; CHOICE HDR).\n",
         "  panel per CHOICE K3), with ONE change: the switch below ON. Code: HEAD = 8737cd3d plus the builder's reviewed\n"
         "  commits and the fixer's typing commit (no header commit: APPROVED and RUNLOG pin HEAD and this text; CHOICE HDR).\n"),
        # ---- A5 / A6 measured
        ("       of ROUND l.780-789 include the values, so they match only in the bitwise case).\n",
         "       of ROUND l.780-789 include the values, so they match only in the bitwise case). " + R["A5"] + "\n"),
        ("       bitwise 8737cd3d's: ROUND's V0 (h) digests (ROUND l.780-793) reproduced for seeds 60, 42, 80 and s42 rollouts\n"
         "       17 and 33.\n",
         "       bitwise 8737cd3d's: ROUND's V0 (h) digests (ROUND l.780-793) reproduced for seeds 60, 42, 80 and s42 rollouts\n"
         "       17 and 33. " + R["A6"] + "\n"),
        # ---- SUBST (I6)
        ("  THE READING below; the measuring copy -> a git archive of HEAD (ROUND l.116-118), the runner's commit binding\n"
         "  re-pointed to HEAD from <PCR>/APPROVED (MEASUREMENT).\n",
         "  THE READING below; the measuring copy -> a git archive of HEAD (ROUND l.116-118), the runner's commit binding\n"
         "  re-pointed to HEAD from <PCR>/APPROVED (MEASUREMENT). THE READING's line 4 requires g = 3 and hs = 0, stricter\n"
         "  than ROUND's line 4 (G+ >= 2, ROUND l.323), a deliberate departure.\n"),
        # ---- manipulation check (iii) (I5)
        ("  by exactly 0.0 on every row. (iii) The critic responds to observation 6 (largest |dV| > 0 under a +1 shift of the\n"
         "  normalised value). Printed per pair with the values read.",
         "  by exactly 0.0 on every row. (iii) The critic's input from observation 6, an interface check (A2's \"not hidden\"):\n"
         "  the critic's first-layer weights from observation 6 are not all zero. The critic's response to observation 6\n"
         "  (largest |dV| under a +1 shift of the normalised value) is printed, report only: it reads the mechanism this study\n"
         "  is about (PS: the critic's tail slope went to about 0) and decides nothing. Printed per pair with the values read."),
        # ---- HABIT ELSEWHERE A (D3)
        ("  a); A an ANY ALARM attributing a; T at a traced checkpoint (J, R1, R2, RESOLUTION) at least 2 full-horizon M1\n",
         "  a); A an ANY ALARM in one of whose two evaluations at least 5 of the 10 rows are over the BAR naming a (ROUND's\n"
         "  per-evaluation count of l.169-172 applied to a; LB's draft read \"an ANY ALARM attributing a\", which one row\n"
         "  satisfied); T at a traced checkpoint (J, R1, R2, RESOLUTION) at least 2 full-horizon M1\n"),
        # ---- comparators bracket: A site wording
        ("  47, 72 (sha256 cacffec5...), read by the pinned tools; with this text's A site (an ANY ALARM attributing a) the\n"
         "  neck reads L, A, T, D:",
         "  47, 72 (sha256 cacffec5...), read by the pinned tools; with this text's A site (an ANY ALARM with at least 5 rows\n"
         "  naming a in one of its two evaluations) the neck reads " + R["NECK_SITES"] + ":"),
        # ---- THE READING line 1 weak basis (D5 b)
        ("  1. r >= 1: \"THE TAIL RAIL FORMED IN <r> OF 3 PHASE-CRITIC RUNS (<basis as ROUND l.312-315>): THE CRITIC'S PHASE INPUT\n",
         "  1. r >= 1: \"THE TAIL RAIL FORMED IN <r> OF 3 PHASE-CRITIC RUNS (<basis as ROUND l.312-315, the weak basis's\n"
         "     figure read \"false-rail chance if the phase critic removed the habit 0.003-0.11 per round, up to 0.31 (ROUND's\n"
         "     EVID figures) [inferred, EVID]\">): THE CRITIC'S PHASE INPUT\n"),
        # ---- line 4 (D5 a, D4, I6)
        ("  4. r = 0, u = 0, p = 0, g = 3, hs = 0: \"NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, THE STANCE GATE CERTIFICATE\n"
         "     PASSED IN 3 OF 3, NO SATURATING HABIT ON ANOTHER ACTUATOR\". To the maintainer",
         "  4. r = 0, u = 0, p = 0, g = 3, hs = 0 (stricter than ROUND's G+ >= 2, SUBST): \"NO TAIL RAIL OR PARK IN 3 OF 3\n"
         "     PHASE-CRITIC RUNS, THE STANCE GATE CERTIFICATE PASSED IN 3 OF 3 (GATE CERTIFICATE: final checkpoint, measured\n"
         "     on a screen task; not certification), NO SATURATING HABIT ON ANOTHER ACTUATOR[, WITH A DRIVEN TAIL (<actuator>\n"
         "     MEAN_a <value> IN s<seed>; ...)]: CONSISTENT WITH THE PHASE CRITIC REMOVING THE HABIT, NOT CONFIRMATORY\" (the\n"
         "     bracket only when a run has a DRIVEN tail actuator, ROUND l.278-279: report only, it moves no line; a tail_3\n"
         "     held hard negative is DRIVEN, not PARKED). To the maintainer"),
        # ---- line 5 (D4)
        ("  5. r = 0, u = 0, p = 0, otherwise: \"NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, WITH: <3 - g> GATE FAIL, <hs>\n"
         "     WITH A SATURATING HABIT ON ANOTHER ACTUATOR\" (only items not 0), each GATE FAIL's criteria and each habit named.\n",
         "  5. r = 0, u = 0, p = 0, otherwise: \"NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, WITH: <3 - g> GATE FAIL, <hs>\n"
         "     WITH A SATURATING HABIT ON ANOTHER ACTUATOR[, WITH A DRIVEN TAIL (...)]\" (only items not 0; the bracket as line\n"
         "     4's), each GATE FAIL's criteria and each habit named.\n"
         "  Lines 3-5 also print \"[model, plug-in] if the phase critic changed nothing, no RAIL in 3 runs has chance (2/3)^3 =\n"
         "  0.296 at STANCE6's 1 of 3 and (3/5)^3 = 0.216 at LTSB r9's 2 of 5\" (ROUND printed such figures, TABLE E).\n"),
        ("  prints: the per-run table (status, TAIL CLASS and kind, early signs, GATE CLASS and the notebook's own\n",
         "  prints: the per-run table (status, TAIL CLASS and kind, early signs, GATE CLASS labelled \"(GATE CERTIFICATE: final\n"
         "  checkpoint, measured on a screen task; not certification)\" (ROUND l.280-283) and the notebook's own\n"),
        # ---- THE ONE TEST (I1, D5 c, I2, I3)
        ("  20/20). Sums are exact (math.fsum) and compared exactly. Printed with the six TDs, the J-window TD (evaluations from 4,194,304) beside,\n"
         "  report only (STANCE6: 0.577658 / 0.808751 / 0.217914), and the caveat line: \"historical comparison: exchangeability\n"
         "  of the 6 runs is assumed, not designed (KNOWN LIMITS)\".\n",
         "  20/20). Sums are exact: rational arithmetic on the float values (fractions.Fraction of each float), compared\n"
         "  exactly (math.fsum is correctly rounded, not exact, and could misorder a near tie). Only p = 1/20 depends on the\n"
         "  placement alone (complete separation); every other p depends on the magnitudes as well. Printed with the six TDs,\n"
         "  the line \"equivalently, every PC TD < 0.154339 (STANCE6 s80's, the lowest comparator); the statistic was chosen\n"
         "  with STANCE6's values in view (2 windows looked at: a strict bound of 2/20)\", the J-window TD (evaluations from\n"
         "  4,194,304) beside, report only (STANCE6: 0.577658 / 0.808751 / 0.217914), and the caveat line: \"historical\n"
         "  comparison: exchangeability of the 6 runs is assumed, not designed; each PC run starts from its STANCE6 seed's own\n"
         "  network (A5), so seed effects make the unpaired test conservative; same-seed pairs printed, report only (KNOWN\n"
         "  LIMITS)\". A paired sign test cannot go below p = 1/8 with 3 pairs: the unpaired test stays.\n"),
        # ---- RUN HANDLING: resume note (i4 / I6-process)
        ("and (7) (stop, LOST, relaunch _b by REPLACEMENT (ii)); the notebook's own rows",
         "and (7) (stop, LOST, relaunch _b by REPLACEMENT (ii)); if CHOICE RESUME (b) is ever taken, CRITIC_PHASE = True is\n"
         "set again before the RESUME cell and fact (8) is read only after the corrected resume rewrote stage_config.json (a\n"
         "refused load writes stage_config.json before it refuses, as at 8737cd3d); the notebook's own rows"),
        # ---- VP0 (D2, P1, P2)
        ("- VP0, before launch, the code (WT at HEAD): A1-A7 each by a named test or check with its output file; the T. rex suite\n"
         "  and the shared tests fail exactly ROUND's known 25 ids (ROUND l.511-547, known25.txt 62e01b73...) and no other, with\n"
         "  the switch at its default; the builder's own tests pass;",
         "- VP0, before launch, the code (WT at HEAD): A1-A7 each by a named test or check with its output file; the T. rex suite\n"
         "  and the shared tests, run with PYTHONPATH set to the repo (the subprocess harness of the 3\n"
         "  test_plant_contract_digest_snapshot.py::test_refusals_come_before_the_run[check-no-golden|check-skip-behaviors|\n"
         "  write-skip-behaviors] imports the package only so; without it those 3 fail at 8737cd3d and HEAD alike, an\n"
         "  environment failure), fail exactly ROUND's known 25 ids (ROUND l.511-547, known25.txt 62e01b73...) and no other,\n"
         "  with the switch at its default; the builder's own tests pass;"),
        ("  the same actor with the switch ON and OFF, so CMD_L, TD and M5 read PC and STANCE6 ledgers the same way.\n",
         "  the same actor with the switch ON and OFF, so CMD_L, TD and M5 read PC and STANCE6 ledgers the same way.\n"
         "  " + R["VP0"] + "\n"),
        # ---- VP1 (D1)
        ("  M1 on a zero-phase-column widening of that pair equals the same M1 (the loader rolls the phase-critic actor exactly).\n",
         "  M1 on a zero-phase-column widening of that pair equals the same M1 (the loader rolls the phase-critic actor exactly),\n"
         "  and M3 on that widened pair, saved as a zip by HEAD's own save path, gives a certificate equal to the round's s80\n"
         "  certificate (about 33 CPU-minutes, during the runs; M3's refusal is final, so its loader is shown before launch).\n"
         "  " + R["D1"] + "\n"),
        # ---- VP2 (P1)
        ("  widened pair equals the panel on the original (rows bitwise); if not, every run launches with K3_PANEL False.\n",
         "  widened pair equals the panel on the original (rows bitwise); if not, every run launches with K3_PANEL False.\n"
         "  " + R["VP2"] + "\n"),
        # ---- VP3 (re-run)
        ("  smoke folder); NEG faults (1), (3), (7), (8). The pin step re-runs it on the pinned files.\n",
         "  smoke folder); NEG faults (1), (3), (7), (8). " + R["VP3"] + " The pin step re-runs it on the pinned files.\n"),
        # ---- VP4 (tests count)
        ("  stock V bitwise; ST/tools/tests (42 tests) and 22 of 22 mutants killed.\n",
         "  stock V bitwise; ST/tools/tests (42 tests) and 22 of 22 mutants killed. " + R["VP4"] + "\n"),
        # ---- KNOWN LIMITS (I3, I4, D3 T asymmetry, habit rule reviewed)
        ("  form; the J window printed beside). The seeds were chosen with their earlier outcomes known.\n",
         "  form; the J window printed beside). The seeds were chosen with their earlier outcomes known. Each PC run starts\n"
         "  from its STANCE6 seed's own network (A5: the first rollout bitwise), so the 3 + 3 design is partly paired; with\n"
         "  outcomes that persist by seed (r9's s42 and r10's s42 both first over the bar at 3,670,016), the unpaired test is\n"
         "  conservative, not inflated; same-seed pairs are printed, report only. The date and the Colab image are not\n"
         "  measured (provenance.json records no torch, CUDA or GPU, and no notebook cell prints them); a concurrent switch-OFF\n"
         "  s42 run (about 15.5 L4-hours, report only) would test that confound directly: not in this design unless the\n"
         "  maintainer adds it at the confirmation.\n"),
        ("- HABIT ELSEWHERE uses LB's O6 rule from an unreviewed draft (LB's build stopped, RUNLOG line 805); it was applied to\n"
         "  STANCE6 by the pinned tools (reread_round.out), and this text's review reviews the rule itself.\n",
         "- HABIT ELSEWHERE uses LB's O6 rule from an unreviewed draft (LB's build stopped, RUNLOG line 805); it was applied to\n"
         "  STANCE6 by the pinned tools (reread_round.out); this text's review reviewed the rule itself and its A site now needs\n"
         "  at least 5 rows naming a in one evaluation (review D3). Its T site counts 2 over-bar full-horizon episodes at any\n"
         "  traced checkpoint (J, R1, R2, RESOLUTION), while the tail's TRACE RAIL reads R1 and R2 only (ROUND l.188-191):\n"
         "  intended, a habit elsewhere is looked for at more checkpoints than the tail's end-only rail.\n"),
        # ---- HOW THE COORDINATOR PINS 3(b) (P3, I5-process)
        ("   the pre-launch checks: (a) sha256sum of criteria.txt and every pinned file equals APPROVED; (b) HEAD on origin\n"
         "   equals APPROVED's branch_head once the maintainer has OK'd the push (read only: git ls-remote); (c) the launch-check\n",
         "   the pre-launch checks: (a) sha256sum of criteria.txt and every pinned file equals APPROVED; (b) HEAD on origin\n"
         "   equals APPROVED's branch_head once the maintainer has OK'd the push (read only: git ls-remote). The push itself:\n"
         "   first `git -C WT branch --unset-upstream` (the local branch tracks origin/claude/screen-trex-r10-armltsb, the\n"
         "   comparators' branch: a bare push with push.default=upstream would move it); then the real git objects with an\n"
         "   explicit refspec, never API file uploads (they make other commit SHAs): `git push origin\n"
         "   refs/heads/claude/screen-trex-r10-phasecritic:refs/heads/claude/screen-trex-r10-phasecritic`; then ls-remote\n"
         "   shows phasecritic = branch_head and armltsb still 8737cd3d, both recorded in RUNLOG. After that push the branch\n"
         "   is frozen: no push to it until all three launch checks read no-fault; (c) the launch-check\n"),
    ]


def colab_edits(R: dict) -> list:
    return [
        ("Every change against COLAB_STEPS_v1.txt: ST/adapt/COLAB_STEPS_v1_to_installed.diff.\n",
         "Every change against COLAB_STEPS_v1.txt: ST/adapt/COLAB_STEPS_v1_to_installed.diff.\n"
         "FIXED 2026-11-01 (date -u) by the fixer after the independent review: the push and freeze rule, the resume note,\n"
         "the HEAD the tools were measured on; every change in ST/review/FIXES.txt and\n"
         "ST/review/fixer/COLAB_STEPS_reviewed_to_fixed.diff. The builder's draft ST/build/COLAB_STEPS.txt is superseded by\n"
         "this file: give the maintainer only this one.\n"),
        ("   The branch: claude/screen-trex-r10-phasecritic, HEAD <HEAD> (as <PCR>/APPROVED records it). It is pushed only after\n"
         "   the maintainer's OK so that REPO_REF can install it; it is never merged.\n",
         "   The branch: claude/screen-trex-r10-phasecritic, HEAD <HEAD> (as <PCR>/APPROVED records it; the tools and dry runs\n"
         "   were last measured on " + R["HEAD"] + "). It is pushed only after the maintainer's OK so that REPO_REF can install it;\n"
         "   it is never merged. The push (HOW THE COORDINATOR PINS 3 (b)): `git -C /home/user/work/pc_branch branch\n"
         "   --unset-upstream` first (the local branch tracks origin/claude/screen-trex-r10-armltsb, the comparators' branch),\n"
         "   then `git push origin refs/heads/claude/screen-trex-r10-phasecritic:refs/heads/claude/screen-trex-r10-phasecritic`\n"
         "   (real git objects, never API file uploads); ls-remote must then show phasecritic = <HEAD> and armltsb still\n"
         "   8737cd3d. FREEZE: after that push nobody pushes to the branch until all three launch checks read no-fault\n"
         "   (REPO_REF names the branch, so a moved branch would fault fact (1) and cost a _b relaunch).\n"),
        ("   HEAD; not part of v1. The builder's smoke ran the RESUME cell's path once on HEAD with the switch ON [measured,\n"
         "   ST/build/smoke/smoke_resume.log]; the rules for a resumed run (C4, C6) are not written.)\n",
         "   HEAD; not part of v1. The builder's smoke ran the RESUME cell's path once on HEAD with the switch ON [measured,\n"
         "   ST/build/smoke/smoke_resume.log]; the rules for a resumed run (C4, C6) are not written. Such a resume must set\n"
         "   CRITIC_PHASE = True again in cell 6 before the RESUME cell: a resume with the switch off is refused, but only\n"
         "   after it has rewritten 01_stance/stage_config.json without the switch, so fact (8) is read only after the\n"
         "   corrected resume.)\n"),
    ]


def pins_block(head: str) -> str:
    sc = json.loads((T / "dry/pins_sc.out").read_text())
    pins_file = T / "pc/pins_pc.json"
    pins = json.loads(pins_file.read_text())
    tools = sorted(p for p in (T / "pc").iterdir() if p.is_file())
    tool_lines = "\n".join(f"    {p.name} {sha(p)}" for p in tools)
    on = sc["on"]
    v7 = pins.get("v7_failed_on_head")
    v7_text = ("v7_failed_on_head UNFILLED: VP2's result" if v7 == "UNFILLED" else
               f"v7_failed_on_head {json.dumps(v7)}: VP2's result on {head[:8]} [measured, ST/review/fixer/vp2d1, "
               "suites/shared]; the pin step re-checks it")
    n_commits = subprocess.run(["git", "--no-optional-locks", "-C", WT, "rev-list", "--count", "8737cd3d..HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
    return (
        "PINS (filled 2026-10-31 (date -u) by ST/tools/make_installed_texts.py from files, never by hand; refilled\n"
        "2026-11-01 (date -u) by the fixer's ST/review/fixer/tools/fix_texts.py the same way; the pin step re-reads every\n"
        "value from its file):\n"
        f"- HEAD: not held in this text (APPROVED's branch_head and the LAUNCH lines' head=). The tools, pins and dry runs\n"
        f"  were measured on WT HEAD {head} (local branch claude/screen-trex-r10-phasecritic, {n_commits} commits on 8737cd3d,\n"
        "  unpushed); if the approved HEAD differs, the pin step re-runs VP3 and the stand-in on it.\n"
        f"- The switch as stage_config.json writes it: {on['switch']['path']} = {json.dumps(on['switch']['value'])}\n"
        "  (pins_pc.json critic_phase_switch); the checkpoint record mesozoic_critic_phase = "
        f"{json.dumps({k: pins['critic_phase_spec'][k] for k in ('schema', 'settle_steps', 'gamma', 'width')})}.\n"
        f"- With the switch ON [measured, ST/tools/dry/pins_sc.out]: hyperparameters_sha256 {on['hyperparameters_sha256']}\n"
        f"  (differs from ROUND's {sc['round_pins']['hyperparameters_sha256']}: the PPO table records critic_phase);\n"
        f"  task_sha256 {on['task_sha256']} (equal to ROUND's: the option is no env keyword);\n"
        f"  stage_config_view_sha256 {on['stage_config_view_sha256']} (differs from ROUND's\n"
        f"  {sc['round_pins']['stage_config_view_sha256']}: the same hyperparameters block); plant_identity = ROUND's block\n"
        f"  (equal: {on['plant_identity_equals_round_pin']}). The switch-OFF file of the same call reproduces ROUND's three\n"
        f"  pins: {sc['off_equals_round']}.\n"
        f"- The pins file ST/tools/pc/pins_pc.json {sha(pins_file)} ({v7_text}).\n"
        "- The PC tools (ST/tools/pc; the launch-check port is complete_r10round.py, r10round_common.py,\n"
        "  ledger_r10round.py, fetch_r10round.py, fetch_agent_brief.txt and pins_pc.json):\n"
        f"{tool_lines}\n"
        f"- ST/COLAB_STEPS.txt {sha(ST / 'COLAB_STEPS.txt')}.\n"
        f"- The comparators: score.json {pins['comparators']['score_json_sha256']}, the stance6 fetch manifest\n"
        f"  {pins['comparators']['fetch_manifest_sha256']}; DESIGN/comparators.py {sha(ST / 'design/comparators.py')},\n"
        f"  comparators.out {sha(ST / 'design/comparators.out')}.\n"
        "- This text's sha256: computed at the pin into APPROVED (never inside this text).\n"
        "- Still to fill at the pin: HEAD (APPROVED); the VP0, VP2, VP3 logs' sha256 (the fixer's are in\n"
        "  ST/review/FIXES.txt). Before F2 (F2_PINS.json, HOW THE COORDINATOR PINS 5): the F2 tools' sha256 as finally\n"
        "  reviewed and the VP1, VP4 logs.\n")


def main() -> int:
    R = json.loads(Path(sys.argv[1]).read_text())
    head = subprocess.run(["git", "--no-optional-locks", "-C", WT, "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    R["HEAD"] = head
    # COLAB_STEPS first (its sha256 goes into the criteria's PINS)
    c_src = (ORIG / "COLAB_STEPS.txt").read_text(encoding="utf-8")
    c_new = wrap_long(apply(c_src, colab_edits(R), "COLAB_STEPS"))
    (ST / "COLAB_STEPS.txt").write_text(c_new, encoding="utf-8")
    (FX / "COLAB_STEPS_reviewed_to_fixed.diff").write_text("".join(difflib.unified_diff(
        c_src.splitlines(True), c_new.splitlines(True), "COLAB_STEPS.txt (reviewed)", "COLAB_STEPS.txt (fixed)")))
    k_src = (ORIG / "criteria.txt").read_text(encoding="utf-8")
    k_new = apply(k_src, criteria_edits(R), "criteria")
    i0 = k_new.index("PINS (filled 2026-10-31")
    i1 = k_new.index("HOW THE COORDINATOR PINS (after the review")
    k_new = wrap_long(k_new[:i0] + pins_block(head) + "\n" + k_new[i1:])
    (ST / "criteria.txt").write_text(k_new, encoding="utf-8")
    (FX / "criteria_reviewed_to_fixed.diff").write_text("".join(difflib.unified_diff(
        k_src.splitlines(True), k_new.splitlines(True), "criteria.txt (reviewed)", "criteria.txt (fixed)")))
    # the builder's draft: superseded (I7)
    b_src = (ORIG / "build/COLAB_STEPS.txt").read_text(encoding="utf-8")
    b_new = ("SUPERSEDED 2026-11-01 (fixer, review I7): this is the code builder's draft; the installed procedure is\n"
             "ST/COLAB_STEPS.txt. Give the maintainer only that file. Its HEAD below (da7e0b92) is also superseded: the\n"
             f"tools were last measured on {head}.\n" + b_src)
    (ST / "build/COLAB_STEPS.txt").write_text(b_new, encoding="utf-8")
    # DESIGN_NOTES 2.4 (I2)
    d_src = (ORIG / "design/DESIGN_NOTES.txt").read_text(encoding="utf-8")
    d_new = apply(d_src, [(R["I2_OLD"], R["I2_NEW"])], "DESIGN_NOTES")
    (ST / "design/DESIGN_NOTES.txt").write_text(d_new, encoding="utf-8")
    (FX / "DESIGN_NOTES_reviewed_to_fixed.diff").write_text("".join(difflib.unified_diff(
        d_src.splitlines(True), d_new.splitlines(True), "DESIGN_NOTES.txt (reviewed)", "DESIGN_NOTES.txt (fixed)")))
    for p in (ST / "COLAB_STEPS.txt", ST / "criteria.txt", ST / "build/COLAB_STEPS.txt", ST / "design/DESIGN_NOTES.txt"):
        print(f"{p} {sha(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
