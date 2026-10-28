"""Independent check of C1-C7 facts and of the PROVISIONAL inputs for the stance6 round (skeptic check1).

Reads text files only (JSON, CSV, RUNLOG); reads each final/J zip's "data" member as JSON text (zipfile, no
extraction, no unpickling) for num_timesteps.  Writes nothing but its stdout.
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone

S6 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
RUNLOG = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance4/live/run_log.txt"
RUNS = {"stance6_r10LTSB_s60": 60, "stance6_r10LTSB_s42": 42, "stance6_r10LTSB_s80": 80}
UNPINNED = ("species", "embodiment", "stage", "library_version", "git_commit", "run", "gpu")
LOAD_KEYS = {"load_path", "load_mode", "parent_checkpoint_sha256", "resume_load_path", "resume_checkpoint_sha256"}
J = ["4194304", "5242880", "6291456", "8388608", "10485760"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(v):
    return "sha256:" + hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def keys_any_depth(o, found, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in LOAD_KEYS:
                found.append(path + "/" + k)
            keys_any_depth(v, found, path + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            keys_any_depth(v, found, path + f"[{i}]")


def ts(s):
    s = s.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def main():
    approved = json.load(open(f"{S6}/APPROVED"))
    head = approved["branch_head"]
    pins = json.load(open(f"{S2}/r10round/lean/build/laneB/tools/pins_r10round.json"))
    lines = open(RUNLOG).read().splitlines()
    stance6 = [(i + 1, l) for i, l in enumerate(lines) if re.match(r"^\S+ \S+ UTC\s+STANCE6 ", l)]
    print("STANCE6 note lines:", [(n, l.split()[4], l.split()[5]) for n, l in stance6])
    fm = list(csv.DictReader(open(f"{S6}/fetch/manifest.csv", newline="")))
    for run, seed in RUNS.items():
        print("=" * 20, run)
        base = f"{S6}/fetch/F2/{run}/copies"
        # C1
        mine = [(n, l.split()[4]) for n, l in stance6 if l.split()[5] == run]
        kinds = [k for _, k in mine]
        launch = [n for n, k in mine if k in ("LAUNCH", "RELAUNCH")]
        stops = [n for n, k in mine if k in ("STOP", "TRIGGER-d")]
        actions = [n for n, k in mine if k == "ACTION"]
        print("C1 lines", mine, "launches", launch, "stops", stops, "actions", actions,
              "->", "PASS" if len(launch) == 1 and kinds[0] == "LAUNCH" and not stops and not actions else "FAIL")
        lline = lines[launch[0] - 1]
        lt = datetime.strptime(" ".join(lline.split()[:2]), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        lhead = re.search(r"head=(\w+)", lline).group(1)
        print("   LAUNCH time", lt.isoformat(), "head=", lhead, "== APPROVED", lhead == head,
              "seed=", re.search(r" seed=(\d+)", lline).group(1), "RUN_ID=", re.search(r"RUN_ID=(\S+)", lline).group(1))
        # C2
        c1, c2 = sha(f"{base}/artifact_manifest.json.copy1"), sha(f"{base}/artifact_manifest.json.copy2")
        man = json.load(open(f"{base}/artifact_manifest.json.copy1"))
        paths = [e["path"] for e in man["files"]]
        bad = [p for p in paths if p.startswith("/") or ".." in p.split("/") or "\\" in p or os.path.normpath(p) != p]
        types_ok = all(isinstance(e.get("size_bytes"), int) and isinstance(e.get("sha256"), str) for e in man["files"])
        print("C2 copies equal", c1 == c2, "schema_version", man.get("schema_version"), "run_id", man.get("run_id"),
              "status", man.get("status"), "entries", len(paths), "unique", len(set(paths)) == len(paths),
              "non-normalized", bad, "types_ok", types_ok)
        ent = {e["path"]: e for e in man["files"]}
        # C4
        prov = json.load(open(f"{base}/provenance.json.copy1"))
        sc = json.load(open(f"{base}/01_stance/stage_config.json.copy1"))
        sess = prov.get("sessions")
        print("C4 provenance sessions", json.dumps(sess)[:400])
        st = ts(sess[0]["started_at"]) if sess else None
        print("   started_at - LAUNCH (min)", round((st - lt).total_seconds() / 60, 2) if st else None,
              "resumed_at in any session", any("resumed_at" in s for s in sess),
              "repository_commit", prov.get("repository_commit"), "dirty", prov.get("repository_dirty"),
              "run_id", prov.get("run_id"))
        found = []
        keys_any_depth(sc, found)
        run_blk = sc.get("run", {})
        print("   stage_config load keys", found, "run.seed", run_blk.get("seed"), "n_envs", run_blk.get("n_envs"),
              "git_commit", sc.get("git_commit"))
        print("   hyperparameters_sha256", run_blk.get("hyperparameters_sha256") == pins["hyperparameters_sha256"]["PPO"]
              if isinstance(pins["hyperparameters_sha256"], dict) else (run_blk.get("hyperparameters_sha256"), pins["hyperparameters_sha256"]))
        print("   task_sha256", sc.get("task_fingerprint", {}).get("task_sha256") == pins["task_sha256"],
              "plant_identity", sc.get("plant_identity") == pins["plant_identity"])
        view = canon({k: v for k, v in sc.items() if k not in UNPINNED and k not in ("plant_identity", "task_fingerprint")})
        pv = pins["stage_config_view_sha256"]
        pv = pv.get("PPO") if isinstance(pv, dict) else pv
        print("   stage_config_view_sha256 (mine)", view, "== pin", view == pv)
        print("   k3_panel.csv listed", "01_stance/k3_panel.csv" in ent, "v7_failed_on_head pin", pins.get("v7_failed_on_head"))
        # C5
        prog = json.load(open(f"{base}/01_stance/progress.json.copy1"))
        gv = json.load(open(f"{base}/01_stance/gate_verdict.json.copy1"))
        pstat = prog.get("status")
        pts = prog.get("timesteps", prog.get("num_timesteps"))
        wa = prog.get("written_at")
        ja = gv.get("judged_at")
        print("C5 progress status", pstat, "timesteps", pts, "written_at", wa, "| gate_verdict judged_at", ja,
              "delta_min", round((ts(ja) - ts(wa)).total_seconds() / 60, 1), "gate_verdict passed", gv.get("passed"),
              "selected", gv.get("selected_checkpoint", gv.get("checkpoint", ""))[:120] if isinstance(gv.get("selected_checkpoint", gv.get("checkpoint", "")), str) else "")
        for lab in J + ["final"]:
            zp = f"{base}/01_stance/models/stage1_{lab}_steps.zip.copy1" if lab != "final" else f"{base}/01_stance/models/stage1_final.zip.copy1"
            with zipfile.ZipFile(zp) as z:
                bad = z.testzip()
                data = json.loads(z.read("data").decode())
            print(f"   zip {lab}: testzip {bad}, num_timesteps {data.get('num_timesteps')}")
        # fetch manifest rows vs stored copies (every copy the reading used)
        rows = [r for r in fm if r["fetch"] == "F2" and r["run_id"] == run and r["status"] == "COPY"]
        mism = []
        used = 0
        for r in rows:
            p = r["stored_as"]
            if not p:
                mism.append(("no stored_as", r["path"]))
                continue
            if not os.path.isabs(p):
                p = os.path.join(S6, p)
            if not os.path.exists(p):
                mism.append(("missing on disk", r["stored_as"]))
                continue
            used += 1
            if sha(p) != r["sha256"]:
                mism.append(("sha differs", r["stored_as"]))
            mp = r["path"]
            if mp in ent and ent[mp]["sha256"].split(":", 1)[-1] != r["sha256"]:
                mism.append(("differs from artifact manifest", mp, r["copy"]))
        on_disk = sorted(glob.glob(f"{base}/**/*.copy*", recursive=True))
        recorded = {os.path.join(S6, r["stored_as"]) if not os.path.isabs(r["stored_as"]) else r["stored_as"] for r in rows}
        unrecorded = [p.replace(base + "/", "") for p in on_disk if p not in recorded]
        errs = [(r["path"], r["note"][:160]) for r in fm if r["fetch"] == "F2" and r["run_id"] == run and r["status"] not in ("COPY", "LISTING")]
        print("FETCH MANIFEST: COPY rows", len(rows), "checked", used, "mismatches", mism, "copies on disk not recorded", unrecorded)
        print("   non-COPY transfer rows", errs)


if __name__ == "__main__":
    sys.exit(main())
