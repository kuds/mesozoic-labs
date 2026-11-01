"""PC AMENDMENT (2026-11-01): add the switch-off arm's pins to ST/tools/pc/pins_pc.json from MEASURED files only, and
refresh tool_sha256 from the tools as they now are.  Nothing is typed by hand; every other key is kept as the reviewed
file holds it (v7_failed_on_head included: VP2's result, re-checked at the pin step).

  off_arm.hyperparameters_sha256, .stage_config_view_sha256, .task_sha256: ST/tools/dry/pins_sc.out "off" (the
      stage_config.json the HEAD measuring copy's config.save_stage_config wrote with the switch OFF, called as
      train_base calls it: seed 42, n_envs 4, 12,000,000), checked equal to the round's three pins (off_equals_round)
      and to the file ST/tools/dry/pins_sc/off/stage_config.json itself (its sha256 as pins_sc.out records it);
  off_arm.critic_phase_switch: {"path": "hyperparameters.critic_phase", "absent": true}, the path read ABSENT from that
      file (with the switch OFF the code writes no key; A6: OFF = 8737cd3d);
  plant_identity: not in off_arm (the off file's block equals the top-level pin, ROUND's: checked).

    PY -I -B amend_pins.py [--check]      (--check: compare only, write nothing; exit 1 on a difference)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ST = Path("/home/user/work/pc_study")
T = ST / "tools"
PC = T / "pc"
PINS = PC / "pins_pc.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def path_state(obj, dotted):
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return False, None
        obj = obj[part]
    return True, obj


def build() -> dict:
    pins = json.loads(PINS.read_text())
    sc = json.loads((T / "dry/pins_sc.out").read_text())
    off = sc["off"]
    f = Path(off["file"])
    assert f == T / "dry/pins_sc/off/stage_config.json", f
    assert sha(f) == off["file_sha256"], "the off stage_config.json is not the file pins_sc.out measured"
    assert all(sc["off_equals_round"].values()), sc["off_equals_round"]
    filej = json.loads(f.read_text())
    assert filej["run"]["hyperparameters_sha256"] == off["hyperparameters_sha256"]
    assert filej["task_fingerprint"]["task_sha256"] == off["task_sha256"]
    present, _v = path_state(filej, "hyperparameters.critic_phase")
    assert present is False, "the switch-OFF stage_config.json holds hyperparameters.critic_phase"
    assert off["plant_identity_equals_round_pin"] is True and filej["plant_identity"] == pins["plant_identity"]
    assert off["task_sha256"] == pins["task_sha256"], "task_sha256 differs between the arms"
    assert off["hyperparameters_sha256"] == pins["hyperparameters_sha256_round_switch_off"]
    assert off["stage_config_view_sha256"] == pins["stage_config_view_sha256_round_switch_off"]
    pins["off_arm"] = {
        "arm": "switch-off (stance7_r10LTSBoff_s60/42/80; notebook CRITIC_PHASE False, its default)",
        "hyperparameters_sha256": off["hyperparameters_sha256"],
        "stage_config_view_sha256": off["stage_config_view_sha256"],
        "task_sha256": off["task_sha256"],
        "critic_phase_switch": {"path": "hyperparameters.critic_phase", "absent": True},
    }
    pins["_sources"]["PC AMENDMENT off_arm"] = (
        f"[measured] ST/tools/dry/pins_sc.out \"off\": {f} (sha256 {off['file_sha256']}), written by the HEAD "
        "measuring copy's config.save_stage_config as train_base calls it with the switch OFF (seed 42, n_envs 4, "
        "12,000,000); its three digests equal the round's pins (off_equals_round all True); "
        "hyperparameters.critic_phase absent from the file; plant_identity equal to the top-level pin. Filled by "
        "ST/amend/tools/amend_pins.py (never by hand)")
    pins["_sources"]["PC tool_sha256"] = ("[measured] sha256 of ST/tools/pc/*.py as the amendment left them "
                                          "(ST/amend/tools/amend_pins.py)")
    st = pins["_status"]
    tag = " | PC AMENDMENT 2026-11-01 (ST/amend/AMEND_NOTES.txt)"
    if tag not in st:
        pins["_status"] = st + tag + (": off_arm (the switch-off runs' configuration pins) added and tool_sha256 "
                                      "refreshed by ST/amend/tools/amend_pins.py from files; every other key unchanged")
    pins["tool_sha256"] = {p.name: sha(p) for p in sorted(PC.glob("*.py"))}
    return pins


def main() -> int:
    new = build()
    text = json.dumps(new, indent=1, sort_keys=False) + "\n"
    if "--check" in sys.argv[1:]:
        same = PINS.read_text() == text
        print(f"pins_pc.json {'equals' if same else 'DIFFERS FROM'} what amend_pins.py would write")
        return 0 if same else 1
    PINS.write_text(text)
    print(f"wrote {PINS} sha256 {sha(PINS)}")
    print(json.dumps({"off_arm": new["off_arm"], "tool_sha256": new["tool_sha256"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
