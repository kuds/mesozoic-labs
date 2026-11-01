"""PC AMENDMENT (2026-11-01): refill the criteria's PINS block from the files (never by hand), after amend_texts.py,
amend_pins.py and every tool change.  Reads: ST/tools/pc/* (sha256), ST/tools/pc/pins_pc.json (the switch pins, the
off_arm block, v7_failed_on_head), ST/tools/dry/pins_sc.out (the measured digests, both switch states),
ST/COLAB_STEPS.txt, the comparator files, WT's HEAD and HEAD^{tree} (read only: git --no-optional-locks rev-parse) and
ST/new_trees.txt / ST/old_trees.txt (the reworded commits' trees).  Replaces the block from "PINS (filled" up to
"HOW THE COORDINATOR PINS (after"; writes ST/criteria.txt; prints every value.  --check: compare only (exit 1 when
the block in the file differs from what would be written).

    PY -I -B refill_pins.py [--check]
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ST = Path("/home/user/work/pc_study")
WT = "/home/user/work/pc_branch"
T = ST / "tools"
PC = T / "pc"
S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
S2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
START, END = "PINS (filled 2026-10-31", "HOW THE COORDINATOR PINS (after the review"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "--no-optional-locks", "-C", WT, *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def block() -> str:
    pins = json.loads((PC / "pins_pc.json").read_text())
    sc = json.loads((T / "dry/pins_sc.out").read_text())
    on, off, rp = sc["on"], sc["off"], sc["round_pins"]
    assert all(sc["off_equals_round"].values()) and on["plant_identity_equals_round_pin"]
    oa = pins["off_arm"]
    assert oa["hyperparameters_sha256"] == off["hyperparameters_sha256"] == rp["hyperparameters_sha256"]
    assert oa["stage_config_view_sha256"] == off["stage_config_view_sha256"] == rp["stage_config_view_sha256"]
    assert oa["task_sha256"] == off["task_sha256"] == on["task_sha256"] == pins["task_sha256"]
    assert pins["hyperparameters_sha256"] == on["hyperparameters_sha256"]
    assert pins["stage_config_view_sha256"] == on["stage_config_view_sha256"]
    head, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    old = dict(x.split() for x in (ST / "old_trees.txt").read_text().splitlines() if x.strip())
    new = dict(x.split() for x in (ST / "new_trees.txt").read_text().splitlines() if x.strip())
    assert new.get(head) == tree, (head, tree)
    reviewed = "e4d222bbde28800d7ee0851380123ff4ac9642a8"
    assert old[reviewed] == tree, "the reviewed head's tree is not WT HEAD's tree"
    same = sum(1 for a, b in zip(sorted(old.values()), sorted(new.values())) if a == b)
    tools = sorted(p for p in PC.iterdir() if p.is_file())
    tool_lines = "\n".join(f"    {p.name} {sha(p)}" for p in tools)
    for p in tools:
        if p.suffix == ".py":
            assert pins["tool_sha256"][p.name] == sha(p), f"pins_pc.json tool_sha256 {p.name} is stale"
    sw, spec = pins["critic_phase_switch"], pins["critic_phase_spec"]
    v7 = pins["v7_failed_on_head"]
    comp = pins["comparators"]
    assert sha(S2 / "r10round/f2/score.json") == comp["score_json_sha256"]
    assert sha(S1 / "foot/stance6/fetch/manifest.csv") == comp["fetch_manifest_sha256"]
    return (
        "PINS (filled 2026-10-31 (date -u) by ST/tools/make_installed_texts.py from files, never by hand; refilled\n"
        "2026-11-01 (date -u) by the fixer's ST/review/fixer/tools/fix_texts.py the same way, and again by the amendment's\n"
        "ST/amend/tools/refill_pins.py; the pin step re-reads every value from its file):\n"
        "- HEAD: not held in this text (APPROVED's branch_head and the LAUNCH lines' head=; one HEAD for all six runs).\n"
        f"  The tools, pins and dry runs were measured on WT HEAD {reviewed} (local branch\n"
        "  claude/screen-trex-r10-phasecritic, 6 commits on 8737cd3d, unpushed); its commits were reworded afterwards\n"
        f"  (trailers only): WT HEAD is now {head}, tree {tree[:8]}... equal to\n"
        f"  the reviewed head's ({same} of {len(new)} commits' trees equal [measured, ST/old_trees.txt, ST/new_trees.txt]); the\n"
        "  amendment's dry runs read it. If the approved HEAD differs from the reviewed one, the pin step re-runs VP3 (both\n"
        "  arms, ST/tools/vp3_dryrun.py) and the stand-in (ST/tools/standin_s6.py --as pc and --as off) on it.\n"
        f"- The switch as stage_config.json writes it: {sw['path']} = {json.dumps(sw['value'])} for a PHASE run\n"
        f"  (pins_pc.json critic_phase_switch); for an OFF run the key is absent (pins_pc.json off_arm.critic_phase_switch\n"
        f"  {json.dumps(oa['critic_phase_switch'], sort_keys=True)}); the checkpoint record mesozoic_critic_phase =\n"
        f"  {{\"schema\": \"{spec['schema']}\", \"settle_steps\": {spec['settle_steps']}, \"gamma\": {spec['gamma']}, "
        f"\"width\": {spec['width']}}} (PHASE runs only).\n"
        f"- With the switch ON (PHASE runs) [measured, ST/tools/dry/pins_sc.out]: hyperparameters_sha256\n"
        f"  {on['hyperparameters_sha256']}\n"
        f"  (differs from ROUND's {rp['hyperparameters_sha256']}: the PPO table records\n"
        "  critic_phase);\n"
        f"  task_sha256 {on['task_sha256']} (equal to ROUND's: the option is\n"
        "  no env keyword);\n"
        f"  stage_config_view_sha256 {on['stage_config_view_sha256']} (differs from ROUND's\n"
        f"  {rp['stage_config_view_sha256']}: the same hyperparameters block);\n"
        "  plant_identity = ROUND's block\n"
        f"  (equal: {on['plant_identity_equals_round_pin']}).\n"
        f"- With the switch OFF (OFF runs) [measured, the same file, \"off\"]: hyperparameters_sha256\n"
        f"  {off['hyperparameters_sha256']}, task_sha256\n"
        f"  {off['task_sha256']}, stage_config_view_sha256\n"
        f"  {off['stage_config_view_sha256']}: ROUND's three pins (equal:\n"
        f"  {', '.join(f'{k} {v}' for k, v in sc['off_equals_round'].items())});\n"
        f"  plant_identity = ROUND's block (equal: {off['plant_identity_equals_round_pin']}); held in pins_pc.json off_arm.\n"
        f"- The pins file ST/tools/pc/pins_pc.json {sha(PC / 'pins_pc.json')}\n"
        f"  (v7_failed_on_head {json.dumps(v7)}: VP2's result on e4d222bb, the same tree [measured, ST/review/fixer/vp2d1,\n"
        "  suites/shared]; the pin step re-checks it; off_arm and tool_sha256 filled by ST/amend/tools/amend_pins.py).\n"
        "- The PC tools (ST/tools/pc; the launch-check port is complete_r10round.py, r10round_common.py,\n"
        "  ledger_r10round.py, fetch_r10round.py, fetch_agent_brief.txt and pins_pc.json):\n"
        f"{tool_lines}\n"
        f"- ST/COLAB_STEPS.txt {sha(ST / 'COLAB_STEPS.txt')}.\n"
        f"- STANCE6 (report only): score.json {comp['score_json_sha256']}, the stance6 fetch\n"
        "  manifest\n"
        f"  {comp['fetch_manifest_sha256']}; DESIGN/comparators.py\n"
        f"  {sha(ST / 'design/comparators.py')},\n"
        f"  comparators.out {sha(ST / 'design/comparators.out')}.\n"
        "- This text's sha256: computed at the pin into APPROVED (never inside this text).\n"
        "- Still to fill at the pin: HEAD (APPROVED); the VP0, VP2, VP3 logs' sha256 (the fixer's are in\n"
        "  ST/review/FIXES.txt; the amendment's VP3 cases in ST/amend/dry/vp3.out, the pin step runs all ten). Before F2\n"
        "  (F2_PINS.json, HOW THE COORDINATOR PINS 5): the F2 tools' sha256 as finally reviewed and the VP1, VP4 logs\n"
        "  (VP4 with f2dry.py --arm pc and --arm off).\n"
        "\n")


def main() -> int:
    f = ST / "criteria.txt"
    text = f.read_text(encoding="utf-8")
    i0, i1 = text.index(START), text.index(END)
    assert text.count(START) == 1 and text.count(END) == 1 and i0 < i1
    new = text[:i0] + block() + text[i1:]
    if "--check" in sys.argv[1:]:
        same = new == text
        print(f"criteria.txt PINS block {'equals' if same else 'DIFFERS FROM'} the files")
        return 0 if same else 1
    f.write_text(new, encoding="utf-8")
    print(block())
    print(f"criteria.txt sha256 {sha(f)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
