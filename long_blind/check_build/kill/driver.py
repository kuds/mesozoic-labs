"""Independent build check (LB): crash-state snapshots of a no-update run, for the resume tests.

Runs ONE no-update run of the root-repointed tool copy (check_build/rp/tools: LB's tools byte for byte except
bs_common.BS = check_build/rp, so every file goes under check_build) and, at chosen moments, freezes it with SIGSTOP,
copies its folder exactly as a SIGKILL at that instant would leave it on disk (pair files that are complete and never
rewritten are hard-linked; everything else is copied byte for byte), then SIGCONT.  No optimizer step: --no-update
(train() is a counting no-op).  The run itself is never killed; it finishes at --until.

Moments (MOMENTS below): mid-evaluation, and at a periodic save every state the step-by-step freezing catches:
zip being written, zip done / pkl absent, pkl being written, pair done / ledgers not flushed, training ledger flushed /
probe not, probe flushed.  Each snapshot gets a manifest.json (utc, process CPU, heartbeat, file sizes and sha256).

usage: PY -I -B driver.py ARM SEED UNTIL ROOT_NAME
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

CB = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/"
          "long_blind/check_build")
PY = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python"
RP = CB / "rp"
S, ROLLOUT, EVAL_FREQ, SAVE_FREQ = 2_097_152, 8192, 131_072, 262_144

ARM, SEED, UNTIL, ROOT_NAME = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ROOT = RP / "validate" / ROOT_NAME
NAME = f"{ARM}_{SEED}"
RUN = ROOT / NAME
STAGE = RUN / "01_stance"
MODELS = STAGE / "models"
SNAPS = CB / "kill" / f"snaps_{ROOT_NAME}"
LOG = CB / "kill" / f"driver_{ROOT_NAME}.log"
# (kind, step): "eval" = freeze mid-evaluation at that evaluation step; "save" = step through the periodic save
MOMENTS = [("eval", 2_490_368), ("save", 2_621_440), ("eval", 2_752_512), ("save", 2_883_584)]


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log(msg: str) -> None:
    with open(LOG, "a") as fh:
        fh.write(f"{utc()} {msg}\n")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def proc_state(pid: int) -> tuple[str, float]:
    with open(f"/proc/{pid}/stat") as fh:
        parts = fh.read().rsplit(")", 1)[1].split()
    tck = os.sysconf("SC_CLK_TCK")
    return parts[0], (int(parts[11]) + int(parts[12])) / tck


def heartbeat() -> int:
    try:
        with open(RUN / "run_meta.json") as fh:
            m = json.load(fh)
        return int((m.get("heartbeat") or {}).get("rollout_end") or 0)
    except Exception:  # noqa: BLE001 - mid-replace or not yet written
        return 0


def evidence_cpu() -> list[tuple[int, float]]:
    out = []
    try:
        with open(RUN / "arm_evidence.jsonl") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    out.append((int(d["num_timesteps"]), float(d["clock"]["cpu_seconds"])))
    except Exception:  # noqa: BLE001
        pass
    return out


def stop(pid: int) -> None:
    os.kill(pid, signal.SIGSTOP)
    for _ in range(20000):
        if proc_state(pid)[0] == "T":
            return
        time.sleep(0.0001)
    raise RuntimeError("process did not stop")


def cont(pid: int) -> None:
    os.kill(pid, signal.SIGCONT)


def zip_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and "data" in z.namelist()
    except Exception:  # noqa: BLE001
        return False


def pkl_ok(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            pickle.load(fh)  # noqa: S301 - the run's own sidecar, as the code's own check does
        return True
    except Exception:  # noqa: BLE001
        return False


def mtime(path: Path) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def snapshot(pid: int, label: str, save_step: int | None, extra: dict) -> None:
    """The process is stopped.  Copy RUN as it is on disk now."""
    dst = SNAPS / label / NAME
    dst.mkdir(parents=True)
    files = {}
    for dirpath, _dirs, names in os.walk(RUN):
        rel_dir = Path(dirpath).relative_to(RUN)
        (dst / rel_dir).mkdir(parents=True, exist_ok=True)
        for n in names:
            src = Path(dirpath) / n
            rel = rel_dir / n
            immutable = (rel_dir == Path("01_stance/models") and n.startswith("stage1_") and "_steps." in n
                         and save_step is not None and f"_{save_step}_steps." not in n
                         and int(n.split("_steps.")[0].rsplit("_", 1)[1]) < (save_step or 0))
            if immutable:
                os.link(src, dst / rel)
                how = "link"
            else:
                shutil.copy2(src, dst / rel)
                how = "copy"
            files[str(rel)] = {"how": how, "size": os.stat(dst / rel).st_size, "sha256": sha(dst / rel)}
    st, cpu = proc_state(pid)
    man = {"label": label, "utc": utc(), "proc_state": st, "proc_cpu_s": cpu, "heartbeat": heartbeat(),
           "save_step": save_step, **extra, "files": files}
    (SNAPS / label / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    log(f"snapshot {label}: cpu {cpu:.2f} hb {man['heartbeat']} files {len(files)} extra {extra}")


def main() -> int:
    SNAPS.mkdir(parents=True, exist_ok=True)
    out = open(CB / "kill" / f"run_{ROOT_NAME}.log", "w")
    cmd = ["nice", "-n", "10", PY, "-I", "-B", str(RP / "tools" / "run_bs.py"), "--arm", ARM, "--seed", str(SEED),
           "--no-update", "--until", str(UNTIL), "--out", str(ROOT)]
    log(f"launch {' '.join(cmd)} (cwd {RP / 'cwd'})")
    p = subprocess.Popen(cmd, cwd=str(RP / "cwd"), stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    pid = p.pid
    time.sleep(1.0)
    cmdline = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace")
    if "run_bs.py" not in cmdline:
        # nice execs the python: the pid is the python after exec
        time.sleep(2.0)
        cmdline = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace")
    log(f"pid {pid} cmdline {cmdline.replace(chr(0), ' ')}")
    assert "run_bs.py" in cmdline
    for kind, step in MOMENTS:
        if kind == "eval":
            want_hb = step - ROLLOUT
            while p.poll() is None and heartbeat() < want_hb:
                time.sleep(0.5)
            if p.poll() is not None:
                break
            ev = evidence_cpu()
            plain = sorted(b - a for (s0, a), (s1, b) in zip(ev, ev[1:]) if s1 % EVAL_FREQ and s1 > S + ROLLOUT)
            r = plain[len(plain) // 2] if plain else 18.0
            cpu0 = [c for s_, c in ev if s_ == want_hb][0]
            trigger = cpu0 + r + 12.0
            log(f"eval {step}: hb {want_hb} cpu {cpu0}; median plain rollout {r:.2f} CPU-s; freeze at cpu {trigger:.2f}")
            while p.poll() is None and proc_state(pid)[1] < trigger:
                time.sleep(0.05)
            stop(pid)
            ledger_has = False
            try:
                with open(STAGE / "eval_episodes.csv") as fh:
                    ledger_has = any(line.startswith(f"{step},") or f",{step}," in line for line in fh)
            except OSError:
                pass
            snapshot(pid, f"mid_eval_{step}", None, {"kind": "mid_eval", "eval_step": step,
                                                     "eval_rows_already_in_ledger": ledger_has,
                                                     "trigger_cpu": trigger, "rollout_start_cpu": cpu0})
            cont(pid)
        else:
            z = MODELS / f"stage1_{step}_steps.zip"
            k = MODELS / f"stage1_vecnormalize_{step}_steps.pkl"
            led, prb = STAGE / "train_episodes.csv", STAGE / "policy_probe.npz"
            while p.poll() is None and heartbeat() < step - ROLLOUT:
                time.sleep(0.5)
            led0, prb0 = mtime(led), mtime(prb)
            log(f"save {step}: polling for {z.name}; ledger mtime {led0} probe mtime {prb0}")
            while p.poll() is None and not z.exists():
                time.sleep(0.0002)
            if p.poll() is not None:
                break
            stop(pid)
            done: set[str] = set()
            for it in range(200000):
                zx, kx = z.exists(), k.exists()
                zo = zip_ok(z) if zx else False
                ko = pkl_ok(k) if kx else False
                l1, p1 = mtime(led), mtime(prb)
                if not zo:
                    state = "zip_partial"
                elif not kx:
                    state = "zip_done_no_pkl"
                elif not ko:
                    state = "pkl_partial"
                elif l1 == led0 and p1 == prb0:
                    state = "pair_done_ledgers_not"
                elif l1 != led0 and p1 == prb0:
                    state = "train_ledger_done_probe_not"
                else:
                    state = "probe_done"
                if state not in done:
                    done.add(state)
                    snapshot(pid, f"save_{step}_{len(done)}_{state}", step,
                             {"kind": "save", "state": state, "iteration": it,
                              "zip_size": z.stat().st_size if zx else None,
                              "pkl_size": k.stat().st_size if kx else None})
                if state == "probe_done":
                    break
                cont(pid)
                time.sleep(0.001 if state in ("zip_partial", "pkl_partial") else 0.0003)
                stop(pid)
            log(f"save {step}: states caught {sorted(done)}")
            cont(pid)
    rc = p.wait()
    log(f"run exit {rc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
