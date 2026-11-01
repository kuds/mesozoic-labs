"""PC study: install ST/COLAB_STEPS.txt and ST/criteria.txt from the designer's drafts (ST/design/COLAB_STEPS_v1.txt,
ST/design/criteria_v1.txt), reconciled with the code and the tools (each change an exact replacement that must match
once; the unified diffs go to ST/adapt/), and fill the criteria's PINS section from files (never typed): the switch-ON
digests (ST/tools/dry/pins_sc.out), the pins file, every PC tool, COLAB_STEPS, the comparators' inputs, WT's HEAD.
Refuses to overwrite an installed text that differs unless --force.

    PY -I -B make_installed_texts.py [--force]
"""
from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

T = Path(__file__).resolve().parent
ST = T.parent
DESIGN = ST / "design"
ADAPT = ST / "adapt"
WT = "/home/user/work/pc_branch"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def apply(src: str, edits, name: str) -> str:
    for i, (old, new) in enumerate(edits, 1):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"STOP: {name} edit {i}: the old text occurs {n} times:\n{old[:300]}")
        src = src.replace(old, new)
    return src


COLAB_EDITS = [
    ("T. rex r10 PHASE-CRITIC ROUND (PC): THREE SEEDS FROM SCRATCH, 12M STEPS, THREE COLAB L4s -- COLAB STEPS v1 (DRAFT)\n",
     "T. rex r10 PHASE-CRITIC ROUND (PC): THREE SEEDS FROM SCRATCH, 12M STEPS, THREE COLAB L4s -- COLAB STEPS v1 (DRAFT),\n"
     "INSTALLED by the measurement adapter 2026-10-31 (date -u): <SWITCH>, <ON value> and <SWITCH_ROW> filled from the\n"
     "branch's notebook (WT notebooks/sb3_training.ipynb cells 6 and 9 at da7e0b92, as the code builder's\n"
     "ST/build/COLAB_STEPS.txt records them), the tools named by their PC paths; <HEAD> is still filled at the pin step.\n"
     "Every change against COLAB_STEPS_v1.txt: ST/adapt/COLAB_STEPS_v1_to_installed.diff.\n"),
    ("       <SWITCH>            = <ON value>                  <- THE PHASE CRITIC. Its default is OFF: set it.\n"
     "    Read SEED, RUN_ID, RUN_LABEL and K3_PANEL_SIGMA_KEY side by side (one seed), and <SWITCH> once more.\n",
     "       CRITIC_PHASE        = True                        <- THE PHASE CRITIC (the 18th setting). Its default is\n"
     "                                                         False: set it.\n"
     "    Read SEED, RUN_ID, RUN_LABEL and K3_PANEL_SIGMA_KEY side by side (one seed), and CRITIC_PHASE once more.\n"),
    ("    - cell 6:  \"Behavior recipe: stance\", \"In-run K3 panel: on\" and <SWITCH_ROW> (the phase critic on)\n",
     "    - cell 6:  \"Behavior recipe: stance\", \"In-run K3 panel: on\" and \"Critic phase: on\" (the phase critic on;\n"
     "               \"Critic phase: off\" means CRITIC_PHASE was left False: that run is not this round's\n"
     "               configuration; record the row in the CHECK line: launch-check fact (8) decides, as L3 says)\n"),
    ("    - cell 9:  \"Behavior 'stance' -> deliverable 'stance'; chain: stance\"\n",
     "    - cell 9:  \"Critic phase on: [ppo] critic_phase = true on ['stance']\" and\n"
     "               \"Behavior 'stance' -> deliverable 'stance'; chain: stance\"\n"),
    (" L1. F0: a fetch agent (the ported fetch brief, FETCH F0) copies",
     " L1. F0: a fetch agent (ST/tools/pc/fetch_agent_brief.txt, FETCH F0) copies"),
    (" L2. The ported launch-check computes the stance6 round's facts",
     " L2. The ported launch-check (PY -I -B ST/tools/pc/complete_r10round.py launch-check --run <RUN_ID> --round <PCR>;\n"
     "     pins ST/tools/pc/pins_pc.json) computes the stance6 round's facts"),
    ("                     RUN_LABEL=\"...\" RUN_ID= K3_PANEL= K3_PANEL_SIGMA_FILE= K3_PANEL_SIGMA_KEY= <SWITCH>=\n",
     "                     RUN_LABEL=\"...\" RUN_ID= K3_PANEL= K3_PANEL_SIGMA_FILE= K3_PANEL_SIGMA_KEY= CRITIC_PHASE=\n"),
    ("                     phase_row=\"<SWITCH_ROW as printed>\" new_run=<yes|no>\n",
     "                     phase_row=\"<the cell-6 'Critic phase: ...' row as printed>\" new_run=<yes|no>\n"),
    ("   then checked by the fetch tool; nothing typed by hand); at most 10 transfers per file.\n",
     "   then checked by the fetch tool; nothing typed by hand); at most 10 transfers per file. Before F2-START the\n"
     "   coordinator writes <PCR>/F2_PINS.json once (the F2 tools' sha256, VP1 and VP4) and puts its sha256 in RUNLOG.\n"),
    (" - Then, on this container (nice 10, at most 2 measurement processes): the manipulation check, M1 at J/R1/RESOLUTION,\n",
     " - Then, on this container (nice 10, at most 2 measurement processes): the manipulation check (PY -I -B\n"
     "   ST/tools/pc/manip_check.py --copy <HEAD measuring copy> --zip/--pkl the C3-verified pair --stage-config the run's\n"
     "   --run-id <RUN_ID> --label <step|final>, at every J label and final; about 7 s each [measured on CPU, smoke]),\n"
     "   M1 at J/R1/RESOLUTION,\n"),
    ("   new runtime would be written here, with its check rows, after the builder has verified the notebook's resume path on\n"
     "   HEAD; not part of v1.)\n",
     "   new runtime would be written here, with its check rows, after the builder has verified the notebook's resume path on\n"
     "   HEAD; not part of v1. The builder's smoke ran the RESUME cell's path once on HEAD with the switch ON [measured,\n"
     "   ST/build/smoke/smoke_resume.log]; the rules for a resumed run (C4, C6) are not written.)\n"),
]


def criteria_edits(fill: dict) -> list:
    return [
        ("STATUS: v1 draft, written 2026-10-31 (date -u;",
         "INSTALLED 2026-10-31 (date -u) by the measurement adapter as ST/criteria.txt: criteria_v1.txt reconciled with the\n"
         "code (WT da7e0b92) and the PC tools (ST/tools/pc); every change against criteria_v1.txt is in\n"
         "ST/adapt/criteria_v1_to_installed.diff and ST/adapt/ADAPT_NOTES.txt; PINS filled from files. Still a draft for\n"
         "the independent review: nothing here is approved.\n"
         "STATUS: v1 draft, written 2026-10-31 (date -u;"),
        ("       value as the builder writes them, pinned in PINS); its hyperparameters_sha256, task_sha256 and\n"
         "       stage_config_view_sha256 are the values the builder's smoke build gives (PINS), each stated as equal to or\n"
         "       differing from ROUND's (and why).\n",
         "       value as the builder writes them: hyperparameters.critic_phase = true, pinned in PINS); its\n"
         "       hyperparameters_sha256, task_sha256 and stage_config_view_sha256 are the values HEAD's own\n"
         "       config.save_stage_config writes for the launch configuration (seed, n_envs 4, 12,000,000; the switch set\n"
         "       as notebook cell 9 sets it), as train_base calls it (PINS; the same call with the switch OFF reproduces\n"
         "       ROUND's three pins, which checks the method; a 2-update smoke with another n_steps gives another\n"
         "       hyperparameters digest and is not the launch's), each stated as equal to or differing from ROUND's (and\n"
         "       why).\n"),
        ("  normalised value). Printed per pair with the values read.\n",
         "  normalised value). Printed per pair with the values read. Made by ST/tools/pc/manip_check.py through the lane A\n"
         "  runner's preflight (seal, commit, physics_revision 10, the run's records, the RUN_ID and label binding): the bank\n"
         "  normalised with the pair's statistics frozen exactly as M1 loads a pair (the trace tool's load_trace_policy),\n"
         "  k = the env's own _step_count at each observation (checked against the step index), the actor's deterministic\n"
         "  mean = its Gaussian mean; the record <PCR>/panels/<RUN_ID>/<label>/c8/manip_check.json is bound to the label's\n"
         "  C3-verified pair as C7's records are (checkpoint and vecnorm sha256, num_timesteps, not a validation input,\n"
         "  filed under the RUN_ID); a record of another file, or made with other seeds or another bank size, is no record\n"
         "  (C8 PENDING); a record reading NOT IN FORCE on the run's own pair fails C8.\n"),
        ("  run; a blank or non-finite cell is skipped, an evaluation with none is listed MISSING and skipped). Read only from\n",
         "  run; a blank or non-finite cell is skipped, an evaluation with none is listed MISSING and skipped; a run with no\n"
         "  finite cell at all has no TD and enters THE ONE TEST at +infinity). Read only from\n"),
        ("  HABIT ELSEWHERE        FREE                 FREE                        SATURATING neck_pitch (L, T, D);\n",
         "  HABIT ELSEWHERE        FREE                 FREE                        SATURATING neck_pitch (L, A, T, D);\n"),
        ("  47, 72 (sha256 cacffec5...), read by the pinned tools.] Under THE READING below STANCE6 reads line 1.\n",
         "  47, 72 (sha256 cacffec5...), read by the pinned tools; with this text's A site (an ANY ALARM attributing a) the\n"
         "  neck reads L, A, T, D: recomputed by ST/tools/pc/pc_common.comparators from the FMAN-checked copies, score.json\n"
         "  and the round's own M1 records [measured, ST/tools/dry/comparators_recomputed.out].] Under THE READING below\n"
         "  STANCE6 reads line 1 [measured: the PC scorer on STANCE6's own F2 data, ST/tools/dry/standin_s6].\n"),
        ("     PASSED IN 3 OF 3, NO SATURATING HABIT ON ANOTHER ACTUATOR\". To the maintainer with the best POSITIVE final named by\n"
         "     stance3's parent rule (ROUND l.321-322, 328-329); no automatic offer.\n",
         "     PASSED IN 3 OF 3, NO SATURATING HABIT ON ANOTHER ACTUATOR\". To the maintainer with the best POSITIVE final named by\n"
         "     stance3's parent rule (ROUND l.321-322, 328-329; score_r10round.parent_named, which reads M4's judge record at\n"
         "     each POSITIVE final: M4 at the finals is F2 work only when line 4 reads; without it the scorer prints \"not\n"
         "     named: an input is absent\"); no automatic offer.\n"),
        ("  \"NOT SEPARATED (p = <k>/20)\". A PC run that is not COMPLETE and FINISHED enters with TD = +infinity (above every\n"
         "  comparator): missing never helps H1. Printed",
         "  \"NOT SEPARATED (p = <k>/20)\". A PC run that is not COMPLETE and FINISHED enters with TD = +infinity (above every\n"
         "  comparator): missing never helps H1 (the PC 3-sum is then infinite, every split ties or falls below it, and p =\n"
         "  20/20). Sums are exact (math.fsum) and compared exactly. Printed"),
        ("the notebook's own \"phase critic: on\" row (Colab steps) is\nrecorded in the CHECK line.",
         "the notebook's own rows \"Critic phase: on\" (cell 6) and \"Critic phase on: [ppo]\n"
         "critic_phase = true on ['stance']\" (cell 9) (Colab steps) are recorded in the CHECK line."),
        ("- Adapted (constants only, each diff against the pinned original listed and reviewed): lane A's runner (r10m_common's\n"
         "  branch binding: the measuring copy's commit = HEAD read from <PCR>/APPROVED instead of \"one parent cf9402bc, a\n"
         "  stance.toml comment diff\"; RUN_IDs; <PCR>/panels);",
         "- Adapted (constants only, each diff against the pinned original listed and reviewed; ST/tools/adapt_diffs/):\n"
         "  lane A's runner (r10m_common's branch binding: the measuring copy's commit = HEAD read from <PCR>/APPROVED, a\n"
         "  commit of WT descending from 8737cd3d whose changes leave the M1/M3 pinned files untouched, instead of \"one\n"
         "  parent cf9402bc, a stance.toml comment diff\"; --header-parent names 8737cd3d for validation inputs; RUN_IDs;\n"
         "  <PCR>/panels);"),
        ("- VP3, before launch, the launch-check port: on the builder's smoke run files (switch ON) no fault; on STANCE6 s42's F0\n"
         "  files (S1/foot/stance6/fetch/F0/stance6_r10LTSB_s42) at least facts (1) and (8) FAULT (a negative control).\n",
         "- VP3, before launch, the launch-check port: on a switch-ON run folder written by HEAD's own writers (a 2-update\n"
         "  smoke bundled as v9_bundle.py does) no fault; on STANCE6 s42's F0 files\n"
         "  (S1/foot/stance6/fetch/F0/stance6_r10LTSB_s42) at least facts (1) and (8) FAULT (a negative control). Dry run\n"
         "  2026-10-31 [measured, ST/tools/dry/vp3/POS.txt, NEG.txt]: POS no fault ((4) not computed: no baseline file in a\n"
         "  smoke folder); NEG faults (1), (3), (7), (8). The pin step re-runs it on the pinned files.\n"),
        ("  widened zero-column pair; synthetic records give every line 1-5 and every test outcome (1/20 ... 20/20, a LOST run\n"
         "  at +infinity).\n",
         "  widened zero-column pair; synthetic records give every line 1-5 and every test outcome (1/20 ... 20/20, a LOST run\n"
         "  at +infinity). Dry runs 2026-10-31 [measured, ST/adapt/ADAPT_NOTES.txt]: 25 of 25 class keys equal for each\n"
         "  STANCE6 run and line 1 (ST/tools/dry/standin_s6); TD equal to comparators.out; IN FORCE on the 2-update smoke\n"
         "  pair; NOT IN FORCE (failing (i) only) on STANCE6 s80's final widened in memory, its phase-fed V equal to the\n"
         "  stock V bitwise; ST/tools/tests (42 tests) and 22 of 22 mutants killed.\n"),
        ("PINS (filled by the pin step from files, never by hand): HEAD; the switch key and value as stage_config.json writes it;\n"
         "hyperparameters_sha256, task_sha256, stage_config_view_sha256 with the switch ON (from the builder's smoke build);\n"
         "plant_identity = ROUND's block; this text's sha256; DESIGN/comparators.py and comparators.out sha256; the launch-check\n"
         "port's files and pins file sha256; COLAB_STEPS's sha256; VP0, VP2, VP3 logs' sha256. Before F2: the F2 tools' sha256\n"
         "and VP1, VP4 logs (into RUNLOG).\n",
         fill["pins_block"]),
        ("   HEAD, worktree clean), the PINS above, approved_by (the maintainer's answer, quoted) and approved_at (to the second).\n",
         "   HEAD, worktree clean), the PINS above, validations {\"VP0\": \"PASS\", \"VP2\": \"PASS\", \"VP3\": \"PASS\"} (the scorer\n"
         "   reads them), approved_by (the maintainer's answer, quoted) and approved_at (to the second).\n"),
        ("4. Only then the maintainer launches (DESIGN/COLAB_STEPS_v1.txt). Nothing in criteria.txt or APPROVED is edited after\n"
         "   the first training step; an error found later is recorded in RUNLOG and the reading prints it as PROVISIONAL.\n",
         "4. Only then the maintainer launches (ST/COLAB_STEPS.txt). Nothing in criteria.txt or APPROVED is edited after\n"
         "   the first training step; an error found later is recorded in RUNLOG and the reading prints it as PROVISIONAL.\n"
         "5. Before F2 (CHOICE TOOLS): <PCR>/F2_PINS.json, written once (compact JSON, chmod 0444): tool_sha256 (every\n"
         "   ST/tools/pc file score_pc.py's tool_hashes lists, as finally reviewed) and validations {\"VP1\": \"PASS\", \"VP4\":\n"
         "   \"PASS\"}; its sha256 on a RUNLOG line before the F2-START lines. The scorer reads VP1, VP4 and the tools' pins\n"
         "   there (PROVISIONAL while it is absent, differs, or its sha256 is in no RUNLOG line).\n"),
    ]


def main() -> int:
    force = "--force" in sys.argv[1:]
    ADAPT.mkdir(exist_ok=True)
    # ---- COLAB_STEPS
    colab_src = (DESIGN / "COLAB_STEPS_v1.txt").read_text(encoding="utf-8")
    colab = apply(colab_src, COLAB_EDITS, "COLAB_STEPS")
    out_c = ST / "COLAB_STEPS.txt"
    if out_c.exists() and out_c.read_text(encoding="utf-8") != colab and not force:
        raise SystemExit(f"STOP: {out_c} exists and differs (--force)")
    out_c.write_text(colab, encoding="utf-8")
    (ADAPT / "COLAB_STEPS_v1_to_installed.diff").write_text("".join(difflib.unified_diff(
        colab_src.splitlines(True), colab.splitlines(True), "design/COLAB_STEPS_v1.txt", "COLAB_STEPS.txt")))
    # ---- PINS from files
    sc = json.loads((T / "dry/pins_sc.out").read_text())
    pins_file = T / "pc/pins_pc.json"
    pins = json.loads(pins_file.read_text())
    head = subprocess.run(["git", "--no-optional-locks", "-C", WT, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    tools = sorted(p for p in (T / "pc").iterdir() if p.is_file())
    tool_lines = "\n".join(f"    {p.name} {sha(p)}" for p in tools)
    on = sc["on"]
    block = (
        "PINS (filled 2026-10-31 (date -u) by ST/tools/make_installed_texts.py from files, never by hand; the pin step\n"
        "re-reads every value from its file):\n"
        f"- HEAD: not held in this text (APPROVED's branch_head and the LAUNCH lines' head=). The tools, pins and dry runs\n"
        f"  were measured on WT HEAD {head} (local branch claude/screen-trex-r10-phasecritic, 5 commits on 8737cd3d,\n"
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
        f"- The pins file ST/tools/pc/pins_pc.json {sha(pins_file)} (v7_failed_on_head UNFILLED: VP2's result).\n"
        "- The PC tools (ST/tools/pc; the launch-check port is complete_r10round.py, r10round_common.py,\n"
        "  ledger_r10round.py, fetch_r10round.py, fetch_agent_brief.txt and pins_pc.json):\n"
        f"{tool_lines}\n"
        f"- ST/COLAB_STEPS.txt {sha(out_c)}.\n"
        f"- The comparators: score.json {pins['comparators']['score_json_sha256']}, the stance6 fetch manifest\n"
        f"  {pins['comparators']['fetch_manifest_sha256']}; DESIGN/comparators.py {sha(DESIGN / 'comparators.py')},\n"
        f"  comparators.out {sha(DESIGN / 'comparators.out')}.\n"
        "- This text's sha256: computed at the pin into APPROVED (never inside this text).\n"
        "- Still to fill at the pin: HEAD (APPROVED); v7_failed_on_head (VP2); the VP0, VP2, VP3 logs' sha256. Before F2\n"
        "  (F2_PINS.json, HOW THE COORDINATOR PINS 5): the F2 tools' sha256 as finally reviewed and the VP1, VP4 logs.\n")
    crit_src = (DESIGN / "criteria_v1.txt").read_text(encoding="utf-8")
    crit = apply(crit_src, criteria_edits(dict(pins_block=block)), "criteria")
    out_k = ST / "criteria.txt"
    if out_k.exists() and out_k.read_text(encoding="utf-8") != crit and not force:
        raise SystemExit(f"STOP: {out_k} exists and differs (--force)")
    out_k.write_text(crit, encoding="utf-8")
    (ADAPT / "criteria_v1_to_installed.diff").write_text("".join(difflib.unified_diff(
        crit_src.splitlines(True), crit.splitlines(True), "design/criteria_v1.txt", "criteria.txt")))
    print(f"COLAB_STEPS.txt {sha(out_c)}\ncriteria.txt {sha(out_k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
