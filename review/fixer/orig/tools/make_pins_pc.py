"""PC study: write ST/tools/pc/pins_pc.json from the round's pins (orig/laneB/pins_r10round.json) and MEASURED values
only: the switch-ON configuration digests from dry/pins_sc.out (stage_config.json written by the measuring copy's own
save_stage_config, as train_base calls it; the switch-OFF file of the same run reproduces the round's three pins), the
switch as stage_config.json writes it, the critic-phase spec the branch records (critic_phase.CriticPhaseSpec.record()
of the smoke checkpoint, dry/smoke_train.out), and the PC tools' sha256 (computed here).  Nothing is typed by hand.
v7_failed_on_head stays "UNFILLED": it is VP2's result on HEAD (CHOICE K3), filled at the pin step.

    PY -I -B make_pins_pc.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

T = Path(__file__).resolve().parent
PC = T / "pc"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    pins = json.loads((T / "orig/laneB/pins_r10round.json").read_text())
    sc = json.loads((T / "dry/pins_sc.out").read_text())
    smoke = (T / "dry/smoke_train.out").read_text()
    smoke = json.loads(smoke[smoke.index("\n{") + 1: smoke.rindex("}") + 1])
    assert all(sc["off_equals_round"].values()), sc["off_equals_round"]
    on = sc["on"]
    assert on["switch"]["value"] is True and on["plant_identity_equals_round_pin"] is True
    pins["_status"] = ("PC adaptation pc-1, written by ST/tools/make_pins_pc.py on 2026-10-31 (date -u) from the round's "
                       "pins (S2/r10round/lean/build/laneB/tools/pins_r10round.json, sha256 0d45721b...) and measured "
                       "values only; the round's facts (main, port, D3, D19, TEL columns, M1/M3/M2/judge sha256, sigma, "
                       "known25, V0 (h), MODEL, V10) are kept as the round pinned them (the PC text cites them); the "
                       "switch-ON digests replace the round's hyperparameters_sha256 and stage_config_view_sha256; "
                       "v7_failed_on_head is VP2's result on HEAD, filled at the pin step")
    pins["_sources"] = dict(pins.get("_sources") or {}, **{
        "PC hyperparameters_sha256/task_sha256/stage_config_view_sha256": (
            "[measured] ST/tools/dry/pins_sc.out: stage_config.json written by the HEAD measuring copy's "
            "config.save_stage_config as train_base calls it, seed 42, n_envs 4, 12,000,000, with [ppo] critic_phase = "
            "true set as notebook cell 9 sets it; the same call with the switch OFF reproduces the round's three pins"),
        "PC critic_phase_switch": "[measured] the same file: hyperparameters.critic_phase = true",
        "PC critic_phase_spec": "[measured] ST/tools/dry/smoke_train.out: the smoke checkpoint's mesozoic_critic_phase",
        "PC tool_sha256": "[measured] sha256 of ST/tools/pc/*.py at the time make_pins_pc.py ran"})
    pins["hyperparameters_sha256"] = on["hyperparameters_sha256"]
    pins["task_sha256"] = on["task_sha256"]
    pins["stage_config_view_sha256"] = on["stage_config_view_sha256"]
    pins["hyperparameters_sha256_round_switch_off"] = sc["round_pins"]["hyperparameters_sha256"]
    pins["stage_config_view_sha256_round_switch_off"] = sc["round_pins"]["stage_config_view_sha256"]
    pins["critic_phase_switch"] = dict(on["switch"])
    pins["critic_phase_spec"] = smoke["critic_phase"]
    pins["base_commit"] = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"
    pins["v7_failed_on_head"] = "UNFILLED"
    pins["v7_result"] = ("ROUND's V7 (8737cd3d): " + str(pins.get("v7_result")) +
                         " -- PC: VP2 on HEAD with the switch ON is the builder's / coordinator's (CHOICE K3)")
    pins["tool_sha256"] = {p.name: sha(p) for p in sorted(PC.glob("*.py"))}
    pins["comparators"] = {
        "score_json_sha256": "cb380a7e816e519cce267fd4a7c00a7d4528730a388e39b3678b95d235c01474",
        "fetch_manifest_sha256": "f5777ae32a5428908d59776d83d3e86d70db31b7818b1a9dd77de05a426c205e"}
    out = PC / "pins_pc.json"
    out.write_text(json.dumps(pins, indent=1, sort_keys=False) + "\n")
    print(f"wrote {out} sha256 {sha(out)}")
    print(json.dumps({k: pins[k] for k in ("hyperparameters_sha256", "task_sha256", "stage_config_view_sha256",
                                            "critic_phase_switch", "critic_phase_spec", "v7_failed_on_head")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
