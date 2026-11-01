"""Re-reviewer's own check: every sha256 and digest the amended criteria PINS block names, against the files."""
import hashlib
import json
import re
import sys
from pathlib import Path

ST = Path("/home/user/work/pc_study")
crit = (ST / "criteria.txt").read_text(encoding="utf-8")
pins_block = crit[crit.index("\nPINS (filled"):crit.index("\nHOW THE COORDINATOR PINS")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


bad = 0
# tool lines "    name.ext <64hex>"
for m in re.finditer(r"^\s{4}(\S+\.(?:py|txt|json)) ([0-9a-f]{64})$", pins_block, re.M):
    name, h = m.group(1), m.group(2)
    got = sha(ST / "tools/pc" / name)
    ok = got == h
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} tools/pc/{name} text {h[:16]} file {got[:16]}")
for label, path in (("pins file", "tools/pc/pins_pc.json"), ("COLAB_STEPS", "COLAB_STEPS.txt")):
    pat = {"pins file": r"pins file ST/tools/pc/pins_pc.json ([0-9a-f]{64})",
           "COLAB_STEPS": r"ST/COLAB_STEPS.txt ([0-9a-f]{64})"}[label]
    h = re.search(pat, pins_block).group(1)
    got = sha(ST / path)
    bad += got != h
    print(f"{'OK ' if got == h else 'BAD'} {path} text {h[:16]} file {got[:16]}")
pins = json.loads((ST / "tools/pc/pins_pc.json").read_text())
# pins_pc.json tool_sha256 vs files (every listed file)
for name, h in sorted(pins["tool_sha256"].items()):
    got = sha(ST / "tools/pc" / name)
    bad += got != h
    print(f"{'OK ' if got == h else 'BAD'} pins_pc.json tool_sha256 {name}")
# every ST/tools/pc file listed in the PINS text?
listed = set(re.findall(r"^\s{4}(\S+\.(?:py|txt|json)) [0-9a-f]{64}$", pins_block, re.M))
files = {p.name for p in (ST / "tools/pc").iterdir() if p.is_file()}
print("tools/pc files not in PINS list:", sorted(files - listed), "; listed not on disk:", sorted(listed - files))
# off_arm digests vs the switch-OFF stage_config.json and ROUND's pins
off = json.loads((ST / "tools/dry/pins_sc/off/stage_config.json").read_text())
on = json.loads((ST / "tools/dry/pins_sc/on/stage_config.json").read_text())
sys.path.insert(0, str(ST / "tools/pc"))
import r10round_common as C  # noqa: E402

oa = pins["off_arm"]
checks = {
    "off hyperparameters_sha256 = file": off["run"]["hyperparameters_sha256"] == oa["hyperparameters_sha256"],
    "off task_sha256 = file": off["task_fingerprint"]["task_sha256"] == oa["task_sha256"],
    "off view = file view": C.view_sha256(off) == oa["stage_config_view_sha256"],
    "off key absent in file": "critic_phase" not in (off.get("hyperparameters") or {}),
    "on key true in file": (on.get("hyperparameters") or {}).get("critic_phase") is True,
    "on hyperparameters = top pin": on["run"]["hyperparameters_sha256"] == pins["hyperparameters_sha256"],
    "on view = top pin": C.view_sha256(on) == pins["stage_config_view_sha256"],
    "on task = top pin": on["task_fingerprint"]["task_sha256"] == pins["task_sha256"],
    "plant on = pin": on.get("plant_identity") == pins["plant_identity"],
    "plant off = pin": off.get("plant_identity") == pins["plant_identity"],
}
rp = json.loads(Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/"
                     "r10round/lean/build/laneB/tools/pins_r10round.json").read_text())
for k in ("hyperparameters_sha256", "task_sha256", "stage_config_view_sha256"):
    checks[f"off_arm {k} = ROUND pins_r10round.json"] = oa[k] == rp.get(k)
checks["text off digests"] = all(oa[k] in pins_block for k in ("hyperparameters_sha256", "task_sha256",
                                                              "stage_config_view_sha256"))
for k, v in checks.items():
    bad += not v
    print(f"{'OK ' if v else 'BAD'} {k}")
# the reviewed pins file vs the amended: every key except tool_sha256/_status/_sources/off_arm equal
old = json.loads((ST / "amend/orig/tools/pc/pins_pc.json").read_text())
diffk = sorted(k for k in set(old) | set(pins) if old.get(k) != pins.get(k))
print("keys differing reviewed->amended:", diffk)
print("BAD count", bad)
