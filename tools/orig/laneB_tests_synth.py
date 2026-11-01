"""Synthetic records for V2: a run folder as the notebook's code writes it (only the files the code had written by the
run's end or cut), its fetch stores as fetch_r10round.py records them, RUNLOG note lines, and M1/M3 outputs in the
measuring runner's layout.  Test scaffolding only (never a tool of record)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import shutil
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

TESTS = Path(__file__).resolve().parent
TOOLS = TESTS.parent / "tools"
sys.path.insert(0, str(TOOLS))
import importlib.util  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("r10round_common")
LG = _load("ledger_r10round")
CP = _load("complete_r10round")
SC = _load("score_r10round")
FT = _load("fetch_r10round")

HEAD = "a" * 40
T0 = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
LEDGER_COLS = ["evaluation", "timesteps", "episode", "env", "return", "length", "full_horizon", "cause",
               "screen_v2_clean", "screen_v2_reasons", "saturated_actuator", "saturated_actuator_fraction",
               "saturated_actuator_positive_share", "saturated_actuator_ties", "actuators_saturated", "rail_actuators"]


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def pins_for_tests(head=HEAD) -> dict:
    """The tools' pins file with the unfilled pins filled for tests.  "branch_head" is kept here as the tests' HEAD
    (written into the synthetic LAUNCH lines, records and APPROVED); v5 makes it no pin, so the tools never read it."""
    p = C.load_pins()
    p = json.loads(json.dumps(p))
    p["branch_head"] = head
    p["judge_sha256"] = "j" * 64
    p["tool_sha256"] = C.tool_hashes()
    p["v7_result"] = "PASS"
    p["v7_method"] = {"run_b2.sh": "x", "cadence_neutral_stance_toml_patch": "y"}
    p["v7_failed_on_head"] = False
    p["v0h_digests"] = {"x": "y"}
    return p


# ------------------------------------------------------------------------------------------------- ledger rows
def row(kind="clean", act=14, sign="+", frac=None, ties=0, nsat=1, length=1000, screen=None, reasons=None):
    """A ledger row by kind: clean, tail (over the bar on a tail actuator), leg (resolved non-tail), legu
    (unresolved non-tail), custom (fields given)."""
    if kind == "clean":
        d = dict(screen_v2_clean="1", screen_v2_reasons="", saturated_actuator="", saturated_actuator_fraction="0.0",
                 saturated_actuator_positive_share="nan", saturated_actuator_ties="0", actuators_saturated="0")
    elif kind in ("tail", "leg", "legu"):
        a = act if kind == "tail" else 3
        f = 0.9 if frac is None else frac
        d = dict(screen_v2_clean="0", screen_v2_reasons="max_actuator_saturation_fraction: 0.9 > 0.1",
                 saturated_actuator=C.ACTUATOR_NAMES[a], saturated_actuator_fraction=str(f),
                 saturated_actuator_positive_share="1.0" if sign == "+" else "0.0",
                 saturated_actuator_ties=str(1 if kind == "legu" else ties),
                 actuators_saturated=str(2 if kind == "legu" else nsat))
    else:
        d = {}
    d.update(length=str(length))
    if screen is not None:
        d["screen_v2_clean"] = screen
    if reasons is not None:
        d["screen_v2_reasons"] = reasons
    return d


def evaluation(spec):
    """spec: list of (row-dict, count) -> 10 rows (episodes numbered in order)."""
    out = []
    for r, k in spec:
        out += [dict(r) for _ in range(k)]
    return out


def clean_ledger(E=C.FINAL):
    return {s: evaluation([(row("clean"), 10)]) for s in C.GRID_STEPS if s <= E}


def ledger_bytes(led: dict, cols=LEDGER_COLS, token=None) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols + (["_model_token"] if token else []), lineterminator="\n",
                       extrasaction="ignore")
    w.writeheader()
    for k, s in enumerate(sorted(led, key=lambda x: (isinstance(x, str), x))):
        for i, r in enumerate(led[s]):
            d = {c: "" for c in cols}
            d.update(evaluation=str(k), timesteps=str(s), episode=str(r.get("_ep", i)), env="0",
                     **{"return": str(r.get("_ret", 4400.0 + i))})
            d.update({c: v for c, v in r.items() if not c.startswith("_")})
            if token:
                d["_model_token"] = token
            w.writerow(d)
    return buf.getvalue().encode("utf-8")


# ------------------------------------------------------------------------------------------------- checkpoints
def sb3_zip(num_timesteps, tag="") -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data", json.dumps({"num_timesteps": num_timesteps, "_tag": tag}))
        z.writestr("policy.pth", b"synthetic" + tag.encode())
    return b.getvalue()


def pkl_bytes(step, tag="") -> bytes:
    return f"synthetic vecnormalize {step} {tag}".encode()


# ------------------------------------------------------------------------------------------------- a run
class Run:
    """One launched RUN_ID's records.  Defaults: a FINISHED one-session run in the round's configuration, a clean
    ledger, every periodic pair, its bundle written ("failed" status, as the stance-only path bundles a failed gate;
    the status does not matter to C2)."""

    def __init__(self, run_id, launch=None, pins=None, **kw):
        self.run_id = run_id
        self.seed = CP.base_seed(run_id)
        self.launch = launch or T0
        self.pins = pins or pins_for_tests()
        self.head = kw.pop("head", self.pins["branch_head"])
        self.E = kw.pop("E", C.FINAL)                      # the end step written ("ended")
        self.cut_at = kw.pop("cut_at", None)                # a cut: last evaluation written (no end files)
        self.ledger = kw.pop("ledger", None)
        self.sessions = kw.pop("sessions", [("started", 5.0)])
        self.sc_over = kw.pop("sc_over", {})
        self.load_keys = kw.pop("load_keys", None)
        self.judged_after_min = kw.pop("judged_after_min", 30.0)
        self.final = kw.pop("final", "auto")                # "auto" (E), None, "corrupt", int (num_timesteps)
        self.hop_watch = kw.pop("hop_watch", False)
        self.bundle = kw.pop("bundle", "auto")              # "auto": written when the run reached its end; False
        self.bundle_status = kw.pop("bundle_status", "failed")
        self.manifest_hash_prefix = kw.pop("manifest_hash_prefix", True)
        self.zab = kw.pop("zab", 4470.4)
        self.ledger_cols = kw.pop("ledger_cols", LEDGER_COLS)
        self.prov_run_id = kw.pop("prov_run_id", run_id)
        self.dirty = kw.pop("dirty", False)
        self.k3_csv = kw.pop("k3_csv", False)
        self.token = kw.pop("token", None)
        assert not kw, kw

    def end_step(self):
        return self.cut_at if self.cut_at is not None else self.E

    def files(self) -> dict:
        """run-relative path -> bytes, as the code wrote them by the run's end (or its cut)."""
        f = {}
        start = self.launch
        sess = []
        for kind, m in self.sessions:
            sess.append({"session_token": f"t{len(sess)}", ("started_at" if kind == "started" else "resumed_at"):
                         (start + timedelta(minutes=m)).isoformat()})
        prov = {"schema_version": 1, "run_id": self.prov_run_id, "repository_commit": self.head,
                "repository_dirty": self.dirty, "sessions": sess, "plant_identity": self.pins["plant_identity"]}
        f["provenance.json"] = json.dumps(prov, indent=2, sort_keys=True).encode()
        f["zero_action_baseline.json"] = json.dumps({"schema": "zab", "results": {"tyrannosaurus_rex": {
            "anatomical": {"reward_mean": self.zab}}}}).encode()
        f["trunk_run.json"] = b'{"trunk": null}'
        sc = {"species": "tyrannosaurus_rex", "embodiment": "anatomical", "stage": 1, "name": "stance",
              "algorithm": "PPO", "library_version": "x", "git_commit": self.head, "reward_weights": {"w": 1},
              "hyperparameters": {"n_steps": 2048}, "curriculum": {"timesteps": 12000000},
              "run": {"seed": self.seed, "n_envs": 4, "timesteps": 12000000,
                      "hyperparameters_sha256": self.pins["hyperparameters_sha256"]},
              "plant_identity": self.pins["plant_identity"],
              "task_fingerprint": {"task_sha256": self.pins["task_sha256"]}}
        sc = json.loads(json.dumps(sc))                     # no aliasing of the pins' blocks
        if self.load_keys:
            for path in self.load_keys:
                node = sc
                parts = path.split(".")
                for p in parts[:-1]:
                    node = node.setdefault(p, {})
                node[parts[-1]] = "x"
        for k, v in self.sc_over.items():
            node = sc
            parts = k.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
        # the pin of the view sha256 is the synthetic file's own view unless an override moves it
        f["01_stance/stage_config.json"] = json.dumps(sc, indent=2).encode()
        led = self.ledger if self.ledger is not None else clean_ledger(self.end_step())
        f["01_stance/eval_episodes.csv"] = ledger_bytes(led, self.ledger_cols, self.token)
        end = self.end_step()
        status = "ended" if self.cut_at is None else "training"
        written = start + timedelta(minutes=20 + end / 131072 * 9.5)
        f["01_stance/progress.json"] = json.dumps({"status": status, "timesteps": end, "written_at": iso(written),
                                                   "warnings": []}).encode()
        for k in range(1, 46):
            s = k * C.CKPT
            if s <= end:
                f[f"01_stance/models/stage1_{s}_steps.zip"] = sb3_zip(s, self.run_id)
                f[f"01_stance/models/stage1_vecnormalize_{s}_steps.pkl"] = pkl_bytes(s, self.run_id)
        f["01_stance/policy_probe.npz"] = b"probe"
        if self.k3_csv:
            f["01_stance/k3_panel.csv"] = b"panel,timesteps\n"
        if self.hop_watch:
            f["01_stance/hop_watch.json"] = b'{"action": "stopped"}'
        if self.cut_at is None:
            if self.final == "auto":
                f["01_stance/models/stage1_final.zip"] = sb3_zip(self.E, self.run_id)
                f["01_stance/models/stage1_final_vecnorm.pkl"] = pkl_bytes("final", self.run_id)
            elif self.final == "corrupt":
                f["01_stance/models/stage1_final.zip"] = b"not a zip"
                f["01_stance/models/stage1_final_vecnorm.pkl"] = pkl_bytes("final", self.run_id)
            elif isinstance(self.final, int):
                f["01_stance/models/stage1_final.zip"] = sb3_zip(self.final, self.run_id)
                f["01_stance/models/stage1_final_vecnorm.pkl"] = pkl_bytes("final", self.run_id)
            if self.judged_after_min is not None:
                f["01_stance/gate_verdict.json"] = json.dumps({"judged_at": (written + timedelta(
                    minutes=self.judged_after_min)).isoformat(), "passed": False}).encode()
            if self.bundle in ("auto", True):
                f["plant_identity.json"] = json.dumps(self.pins["plant_identity"]).encode()
                f["collected_results.csv"] = b"a,b\n"
                f["summary.json"] = b"{}"
                ents = []
                for p in sorted(f):
                    h = hashlib.sha256(f[p]).hexdigest()
                    ents.append({"path": p, "size_bytes": len(f[p]),
                                 "sha256": ("sha256:" + h) if self.manifest_hash_prefix else h})
                f["artifact_manifest.json"] = json.dumps({"schema_version": 1, "run_id": self.prov_run_id,
                                                          "status": self.bundle_status, "files": ents},
                                                         indent=2, sort_keys=True).encode()
        return f


def view_pin_of(run: Run) -> str:
    return C.view_sha256(json.loads(run.files()["01_stance/stage_config.json"]))


# ------------------------------------------------------------------------------------------------- fetch stores
def listing_pages(names_sizes: dict, folders=(), per_page=5):
    """search_files-like pages (dicts with files and nextPageToken; 5 items a page, as the dry run saw)."""
    items = [{"id": f"id-{n}", "title": n, "mimeType": "application/octet-stream", "fileSize": str(s)}
             for n, s in sorted(names_sizes.items())]
    items += [{"id": f"fid-{n}", "title": n, "mimeType": "application/vnd.google-apps.folder"} for n in folders]
    pages = [items[i:i + per_page] for i in range(0, len(items), per_page)] or [[]]
    out = []
    for i, p in enumerate(pages):
        d = {"files": p}
        if i < len(pages) - 1:
            d["nextPageToken"] = f"tok{i + 1}"
        out.append(d)
    return out


def write_store(round_dir: Path, fetch: str, run: Run, drive: dict, utc: datetime, copies="auto", skip=(),
                mutate=None, listing_override=None):
    """A fetch store for one run: the listings of the run's folders (as drive holds them) and copies of files.
    copies: "auto" = every non-zip file twice and every zip once (both equal to drive); or a dict path -> list of
    bytes (each a COPY).  mutate: path -> bytes served by Drive in this fetch instead (copies then equal it).  skip:
    paths not copied.  The fetch manifest gets a COPY row for each copy."""
    rd = Path(round_dir)
    d = rd / "fetch" / fetch / run.run_id
    if d.exists():
        shutil.rmtree(d)
    served = dict(drive)
    for p, b in (mutate or {}).items():
        if b is None:
            served.pop(p, None)
        else:
            served[p] = b
    folders = {}
    for p in served:
        key = "root" if "/" not in p else p.rsplit("/", 1)[0]
        folders.setdefault(key, {})[p.rsplit("/", 1)[-1]] = len(served[p])
    sub = {"root": ["01_stance"], "01_stance": ["models", "figures", "replays"], "01_stance/figures": ["training_history"]}
    (d / "listings").mkdir(parents=True, exist_ok=True)
    keys = ["root", "01_stance", "01_stance/models", "01_stance/figures", "01_stance/figures/training_history",
            "01_stance/replays"]
    if fetch == "TRIGGER":
        keys = ["root", "01_stance", "01_stance/models"]
    for key in keys:
        files = [{"name": n, "id": f"id-{n}", "mime": "x", "size": s} for n, s in sorted(folders.get(key, {}).items())]
        if listing_override and key in listing_override:
            files = [{"name": n, "id": f"id-{n}", "mime": "x", "size": s} for n, s in listing_override[key].items()]
        fl = [{"name": n, "id": f"fid-{n}", "mime": "folder", "size": None} for n in sub.get(key, [])]
        lp = d / "listings" / f"{key}.json"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(dict(key=key, folder_id=f"fid-{key}", utc=iso(utc),
                                                                    pages=1, files=files, folders=fl, problems=[],
                                                                    complete=True)))
    rows = []
    plan = {}
    if copies == "auto":
        for p, b in served.items():
            if p in skip:
                continue
            plan[p] = [b] if p.endswith(".zip") else [b, b]
    else:
        plan = copies
    for p, bs in plan.items():
        for k, b in enumerate(bs, 1):
            f = d / "copies" / f"{p}.copy{k}"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b)
            rows.append(dict(utc=iso(utc), fetch=fetch, run_id=run.run_id, path=p, drive_id=f"id-{p}",
                             listing_size=len(served.get(p, b)), transfer=k, decoded_size=len(b), status="COPY",
                             copy=k, sha256=hashlib.sha256(b).hexdigest(), stored_as=str(f), method="synthetic"))
    fm = rd / "fetch" / "manifest.csv"
    fm.parent.mkdir(parents=True, exist_ok=True)
    new = not fm.exists()
    with open(fm, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=C.FETCH_MANIFEST_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return d


# ------------------------------------------------------------------------------------------------- RUNLOG
def note(t: datetime, kind: str, run_id: str, **fields) -> str:
    fs = " ".join(f'{k}="{v}"' if (" " in str(v) or v == "") else f"{k}={v}" for k, v in fields.items())
    return f"{t.strftime('%Y-%m-%d %H:%M')} UTC  STANCE6 {kind} {run_id}{(' ' + fs) if fs else ''}"


def launch_fields(seed, head=HEAD, **over):
    f = dict(seed=seed, runtime="L4", head=head, SPECIES="tyrannosaurus_rex", EMBODIMENT="anatomical", ALGORITHM="ppo",
             N_ENVS=4, SEED=seed, VERBOSE=0, QUICK_TEST="False", USE_GOOGLE_DRIVE="True", AUTO_DISCONNECT="True",
             BEHAVIOR="stance", TRUNK_FROM="", RETRAIN_FROM="", RUN_LABEL=C.RUN_LABELS.get(seed, ""),
             RUN_ID=f"stance6_r10LTSB_s{seed}", K3_PANEL="True", K3_PANEL_SIGMA_FILE=C.SIGMA_FILE_COLAB,
             K3_PANEL_SIGMA_KEY=C.SIGMA_KEYS.get(seed, ""))
    f.update(over)
    return f


# ------------------------------------------------------------------------------------------------- M1 / M3 outputs
def write_trace(dirp: Path, cmd=None, flags=(), head=HEAD, physics_revision=10, short=(), tiny=(), cmd_by_ep=None):
    """M1 output: 20 episodes; cmd {a: window-mean command}; flags [(a, sign), ...] one per episode (share 0.5; sign 0
    an exact tie of + and - saturated window steps); short: episode indices shorter than 1000 steps (their flags are
    FALL FLAGS); tiny: episode indices that end by step 200 (length 150: no window, no window mean; v5 O-6);
    cmd_by_ep {i: {a: command}}: an episode's own window means (over cmd)."""
    dirp.mkdir(parents=True, exist_ok=True)
    cmd = cmd or {}
    eps = []
    for i, seed in enumerate(C.TRACE_SEEDS):
        length = 150 if i in tiny else (600 if i in short else 1000)
        acts = {}
        fl = flags[i] if i < len(flags) else None
        for a in range(15):
            share = 0.5 if (fl and fl[0] == a) else 0.0
            cw = (cmd_by_ep or {}).get(i, {}).get(a, cmd.get(a, 0.0))
            acts[C.ACTUATOR_NAMES[a]] = {"index": a, "command_window_mean": None if i in tiny else cw,
                                         "saturated_fraction": None if i in tiny else share}
        eps.append({"seed": seed, "length": length, "reached_horizon": length >= 1000, "actuators": acts})
        if fl:
            a, sg = fl
            rows = ["step,in_window," + f"{C.ACTUATOR_NAMES[a]}_command"]
            for st in range(length):
                if sg == 0:                                  # an exact tie: as many +1 as -1 window steps
                    c = (1.0 if st % 4 == 0 else (-1.0 if st % 4 == 2 else 0.0)) if st >= 200 else 0.0
                else:
                    c = (sg * 1.0) if (st >= 200 and st % 2 == 0) else 0.0
                rows.append(f"{st},{int(st >= 200)},{c}")
            with gzip.open(dirp / f"episode_trace_seed{seed}.csv.gz", "wt") as fh:
                fh.write("\n".join(rows) + "\n")
    (dirp / "episode_trace.json").write_text(json.dumps({"seeds": list(C.TRACE_SEEDS), "deterministic": True,
                                                         "holds": [], "episodes": eps}))
    (dirp / "m1_summary.json").write_text(json.dumps({"physics_revision": physics_revision,
                                                      "copy_seal": {"commit": head}}))


def write_cert(dirp: Path, passed=True, head=HEAD, physics_revision=10):
    dirp.mkdir(parents=True, exist_ok=True)
    (dirp / "stance_heading_report.json").write_text(json.dumps({"passed": passed, "failures": [] if passed else
                                                                 ["heading +90: clean 30 of 40"],
                                                                 "seeds": list(C.CERT_SEEDS)}))
    (dirp / "m3_summary.json").write_text(json.dumps({"physics_revision": physics_revision,
                                                      "copy_seal": {"commit": head}}))


def write_panels(round_dir: Path, run: Run, traces=None, cert=True, skip=()):
    """M1 at R1, every J up to E, final, and the given extra labels; M3 at final for a FINISHED run."""
    pd = Path(round_dir) / "panels" / run.run_id
    E = run.end_step()
    labels = {str(s): {} for s in C.J_STEPS if s <= E}
    if E >= C.R1_STEP:
        labels[str(C.R1_STEP)] = {}
    else:
        last = max([k * C.CKPT for k in range(1, 46) if k * C.CKPT <= E], default=None)
        if last:
            labels[str(last)] = {}
    labels["final"] = {}
    for k, v in (traces or {}).items():
        labels[str(k)] = v
    for lab, spec in labels.items():
        if lab in skip:
            continue
        write_trace(pd / lab / "m1", head=run.head, **spec)
    if cert is not None and E == C.FINAL and "final-cert" not in skip:
        write_cert(pd / "final" / "m3", passed=cert, head=run.head)


# ------------------------------------------------------------------------------------------------- a round
class Round:
    def __init__(self, root: Path, pins=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pins = pins or pins_for_tests()
        self.notes = []
        self.late = []                                       # lines appended after the time-sorted notes (as written)
        self.runlog = self.root / "run_log.txt"
        self.pins_path = self.root / "pins.json"

    def set_view_pin(self, run: Run):
        self.pins["stage_config_view_sha256"] = view_pin_of(run)

    def add_note(self, t, kind, run_id, **fields):
        self.notes.append(note(t, kind, run_id, **fields))

    def write_meta(self, approved_over=None):
        self.pins_path.write_text(json.dumps(self.pins))
        # RUNLOG is written in time order, as a log is (a stable sort: same-minute lines keep their order); a test's
        # late lines (a restatement after F2's start note) are appended as written
        notes = sorted(self.notes, key=lambda x: x[:16]) + list(self.late)
        self.runlog.write_text("2026-10-26 18:00 UTC  an unrelated line\n" + "\n".join(notes) + "\n")
        for name in ("criteria.txt", "criteria_v12_cited.txt", "evidence.txt"):
            (self.root / name).write_text(name)
        ap = {"criteria_sha256": C.sha256_file(self.root / "criteria.txt"),
              "this12_sha256": C.sha256_file(self.root / "criteria_v12_cited.txt"),
              "evid_sha256": C.sha256_file(self.root / "evidence.txt"), "branch_head": self.pins["branch_head"],
              "validations": {v: "PASS" for v in ("V0", "V1", "V2", "V3", "V4", "V5", "V8 (i)", "V8 (ii)", "V9")}}
        ap.update(approved_over or {})
        (self.root / "APPROVED").write_text(json.dumps(ap))

    def score(self, bind=True, trigger_store=True):
        self.write_meta()
        if trigger_store:
            make_trigger_stores(self.root, self.runlog)
        if bind:
            bind_panel_records(self.root)
        return SC.score(self.root, self.runlog, self.pins_path, None)


# ------------------------------------------------------------------------------------------------- fixer scaffolding
def make_trigger_stores(round_dir, runlog):
    """For each LISTING note without a TRIGGER store: the fetch store's listing of 01_stance/models/ and its fetch-
    manifest LISTING row, as fetch_r10round.py writes them, holding exactly the note's names and sizes (R-m6)."""
    rd = Path(round_dir)
    notes, _o = C.parse_runlog(runlog)
    for n in notes:
        if n["kind"] != "LISTING" or n["fields"].get("folder", "").startswith("absent:"):
            continue
        lp = rd / "fetch" / "TRIGGER" / n["run"] / "listings" / "01_stance" / "models.json"
        if lp.exists():
            continue
        files = []
        for x in n["fields"].get("files", "").split(";"):
            if x.strip():
                name, size = x.split(":", 1)
                files.append(dict(name=name, id=f"id-{name}", mime="x", size=int(size)))
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(dict(key="01_stance/models", folder_id=n["fields"].get("folder", "fid"),
                                      utc=n["time"].strftime("%Y-%m-%dT%H:%M:%SZ"), pages=1, files=files, folders=[],
                                      problems=[], complete=True)))
        fm = rd / "fetch" / "manifest.csv"
        fm.parent.mkdir(parents=True, exist_ok=True)
        new = not fm.exists()
        with open(fm, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=C.FETCH_MANIFEST_COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(dict(utc=n["time"].strftime("%Y-%m-%dT%H:%M:%SZ"), fetch="TRIGGER", run_id=n["run"],
                            path="01_stance/models/", status="LISTING",
                            note=f"{len(files)} files, 0 folders, 1 pages; "
                                 + "; ".join(f"{f['name']}:{f['size']}" for f in files)))


def _verified(rd: Path, run_id):
    """(files_read, E) of the fetch the scorer reads for this run: F2, or the REFETCH when C1 holds and C2 or C3
    fails at F2 and a REFETCH store exists (score_r10round's rule, the replacement rule aside)."""
    f2 = rd / "fetch" / "F2" / run_id
    if not f2.is_dir():
        return {}, None
    notes, _o = C.parse_runlog(rd / "run_log.txt") if (rd / "run_log.txt").is_file() else ([], [])
    pins = C.load_pins(rd / "pins.json") if (rd / "pins.json").is_file() else pins_for_tests()
    fm = rd / "fetch" / "manifest.csv"
    comp = CP.completeness(run_id, C.FetchStore(f2, fm, "F2", run_id), notes, pins, CP.PanelLocator(None))
    rf = rd / "fetch" / "REFETCH" / run_id
    if comp["C1"]["ok"] is True and (comp["C2"]["ok"] is not True or comp["C3"]["ok"] is not True) and rf.is_dir():
        comp = CP.completeness(run_id, C.FetchStore(rf, fm, "REFETCH", run_id), notes, pins, CP.PanelLocator(None))
    return comp["files_read"], comp["E"]


def bind_panel_records(round_dir):
    """Every synthetic M1/M3 record without a checkpoint block gets the binding lane A's runner writes: the sha256 of
    the label's pair as the fetch the scorer reads verifies it, the checkpoint's num_timesteps (the label, or E for
    final), not a validation input (R-B4).  Records a probe wrote with its own checkpoint block are left as they are."""
    rd = Path(round_dir)
    for run_dir in sorted((rd / "panels").glob("*")) if (rd / "panels").is_dir() else []:
        files_read, E = _verified(rd, run_dir.name)
        for lab_dir in sorted(run_dir.glob("*")):
            lab = lab_dir.name
            for kind in ("m1", "m3"):
                f = lab_dir / kind / f"{kind}_summary.json"
                if not f.is_file():
                    continue
                rec = json.loads(f.read_text())
                if "checkpoint" in rec:
                    continue
                zp, pp = C.pair_paths("final" if lab == "final" else int(lab))
                n = E if lab == "final" else int(lab)
                rec["checkpoint"] = dict(sha256=files_read.get(zp), vecnorm_sha256=files_read.get(pp), num_timesteps=n)
                rec["run_records"] = dict(validation_no_run_records=False)
                f.write_text(json.dumps(rec))


def standard_run(rnd: Round, seed, launch=None, ledger=None, traces=None, cert=True, ended_note=True, f2=True,
                 f2_mutate=None, f2_skip=(), refetch=None, extra_notes=(), **runkw):
    """Launch notes, the run's files, its F2 store and its panels; returns (Run, drive files)."""
    launch = launch or T0
    run = Run(f"stance6_r10LTSB_s{seed}", launch, rnd.pins, ledger=ledger, **runkw)
    rnd.add_note(launch, "LAUNCH", run.run_id, **launch_fields(seed, rnd.pins["branch_head"]))
    end_t = launch + timedelta(hours=15)
    for (kind, minutes, fields) in extra_notes:
        rnd.add_note(launch + timedelta(minutes=minutes), kind, run.run_id, **fields)
    if ended_note:
        rnd.add_note(end_t, "TRIGGER-a", run.run_id)
        rnd.add_note(end_t, "FOUND-GONE", run.run_id)
    drive = run.files()
    if f2:
        write_store(rnd.root, "F2", run, drive, end_t + timedelta(hours=7), mutate=f2_mutate, skip=f2_skip)
        rnd.add_note(end_t + timedelta(hours=7), "F2-START", run.run_id)
    if refetch is not None:
        write_store(rnd.root, "REFETCH", run, drive, end_t + timedelta(hours=14), mutate=refetch)
        rnd.add_note(end_t + timedelta(hours=14), "REFETCH-START", run.run_id)
    write_panels(rnd.root, run, traces, cert)
    return run, drive
