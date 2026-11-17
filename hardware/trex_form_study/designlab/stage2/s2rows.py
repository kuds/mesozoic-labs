"""Track B's policy rows for the bar runner: ``R5/<seed>``, ``R6/<seed>`` and ``R5G/<seed>``, built from a
checkpoint's files (round C design section 3.11; Track B rules E1, E2, E5, K4, K9; ``STAGE2_PLAN.md`` S2-12).

* ``R5/<seed>`` is the panels' own tracker row (``panels.TrackerRow`` in R2 mode with no feedback term and no ankle
  cap, at the commanded clock) with the panels' own policy hook: the tracker, the start, the seeds, the recorder and
  the judge are those of the R0 row beside it.  ``S2Policy`` acts after every tenth physics step through the same
  ``r5core.Core`` methods, in the same order, as the training env: the observation from the tracker's estimator
  and clock, the saved normalisation on plain arrays (``s2load.normalize``), the deterministic action, the filter,
  then ``tk.residual, tk.rate_factor = core.offsets(a_f)``.
* ``R6/<seed>`` runs ``r5core.R6Tracker`` under ``control.run`` with an acting hook of its own; the policy sets
  the joint targets (``core.targets``).  Its joint spans are Stage 1's span rule over the run's own tasks' tracks on
  the episode's plant, as the env computes them.
* ``R5G/<seed>`` is the R5 row of an ``-X`` body with the bar's head script H1 (``r5core.HeadScript.fixed(episode
  seed, the episode's seconds)``; rule E5).  It is built from the files of ``R5/<seed>`` unless it is given its own.

A row is built from a job directory and a step count (``configure``, or the JSON file ``$STAGE2_POLICIES`` names:
row name -> ``{"job_dir", "steps"}``).  Building it (``build``), before any episode:

1. loads the checkpoint once through ``s2load.load_checkpoint(..., expect_run=run.json's)``: a checkpoint that is not
   intact, is another run's, or does not reproduce its fingerprint within 1e-5 is not judged (S2-12, K9);
2. compares the hashes the run's sessions recorded of ``r5core.py``, ``r5lib.py``, ``lookread.py``, of the Stage 1
   sources (``plant.sources_sha256()``: every ``designlab/stage1/*.py``, ``policy.py`` and ``env.py`` among them; a
   run records no hash of those two alone) and of the panels' sources with today's (``SOURCE_RULE``): unequal is
   refused, or, under ``STAGE2_ALLOW_SOURCE_CHANGE=1``, run with ``source_change: true`` and both hash sets in the
   row's description;
3. reads the run's set as it is today and accepts it only by rule K4 (``SET_RULE``).

A worker rebuilds the row from its ``spec()`` (paths and hashes), checks the three files' hashes again and loads
on first use.  The set a row is asked to run on must be its run's (a policy is judged on the references it
imitates); the row's library is read from the directory the bar's context reads.

``describe()`` is the record's ``controller`` and enters its key: what an episode executes, both version sets
included, and no measured value.  The measured fingerprint difference and its tolerance are in ``provenance()``,
which ``lookbar.LookRow`` writes into every episode's block (``policy_check``).

The registered factories return the row inside ``lookbar.LookRow``.  This module never imports ``r5env`` or
``train2`` at module level; ``equivalence_check`` imports ``r5env`` (through ``g8a``) inside the function.
"""

from __future__ import annotations

from designlab.stage2 import tbenv

tbenv.assign_gl("disable")

import dataclasses  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from designlab.stage1 import control, env, panels, plant, tune  # noqa: E402
from designlab.stage2 import lookbar, lookread, r5core, r5lib, s2io, s2load, setio  # noqa: E402

CONTROLLERS = ("R5", "R6", "R5G")
#: Row controller -> (the run's mode, the head).
MODES = {"R5": ("R5", "neutral"), "R6": ("R6", "neutral"), "R5G": ("R5", "H1")}
POLICIES_ENV = "STAGE2_POLICIES"
SOURCE_CHANGE_ENV = "STAGE2_ALLOW_SOURCE_CHANGE"
#: The Track B files whose hash a run's sessions recorded and a judged row compares (``SOURCE_RULE``), relative to
#: the study directory (``train2.SOURCE_FILES``' names).
SOURCE_FILES = ("designlab/stage2/r5core.py", "designlab/stage2/r5lib.py", "designlab/stage2/lookread.py")
#: The settings of an env config that an episode's ``r5core.Core`` executes.
CORE_KEYS = ("rate", "action_leg_rad", "action_other_rad", "action_rate", "filter_hz", "heading_input")
CONTROLLER = {
    "R5": "R5: the tracker on the estimate with no feedback term and no ankle cap at the commanded clock; every 20 ms "
          "a policy's clipped, low-passed action sets joint offsets on the reference's servo angles and the clock "
          "factor (rule E1)",
    "R6": "R6: no reference controller; every 20 ms a policy's clipped, low-passed action sets the joint targets "
          "clip(home + head command + action x span, range) and the clock factor (rule E2)"}
SOURCE_RULE = (
    "rule K9: a judged row is refused if the code that builds its observations and applies its actions differs from "
    "the run's recorded hashes.  Compared with today's, for every session of the run: the sha256 of r5core.py, "
    "r5lib.py and lookread.py (run.json sessions[].sources.files), the hash of the Stage 1 sources "
    "(sources.plant_sources: every designlab/stage1/*.py, policy.py and env.py among them) and the panels' sources "
    f"(sources.panel_sources).  Unequal: refused, or, with {SOURCE_CHANGE_ENV}=1, run with source_change true and "
    "both hash sets in the row's description")
SET_RULE = (
    "rule K4, for a judged row: the run's set is read as it is today.  Every file the run recorded for its tasks "
    "(run.json settings.inputs.files) must be there under the set's latest freeze with the sha256 and the entry "
    "hash the run recorded, every task the run named must still be in the table, and the plant's inputs must be the "
    "run's; a later freeze of the set is then accepted and both freezes are recorded.  Anything else is refused "
    "with what differs, and so is a run.json that records no input file, names no task or records no plant inputs.  "
    "The row runs conditions of its run's set only, and its library is read from the directory the bar's context "
    "reads, whose files of the run must again have the run's hashes")
HEADING_RULE = (
    "rule E4: a plan given as a plan keeps the integral of the commanded yaw rate as its commanded heading, unless "
    "it plays a turn in place (a periodic gait with a yaw rate and no forward speed): the row then names the "
    "reference's own yaw (r5lib.Library.from_plan's heading), the heading its policy was trained with.  No bar "
    "condition plays one")

_POLICIES: dict = {}


class RowError(s2io.S2Error):
    """A policy row cannot be built or run: its checkpoint is not judged (S2-12, K4, K9)."""


# ------------------------------------------------------------------------------------------------
# Which checkpoint a row name stands for
# ------------------------------------------------------------------------------------------------
def configure(policies: dict | None) -> None:
    """Row name (``R5/42``) -> ``{"job_dir", "steps"}`` (and, for a set outside the study's ``stage2/sets``,
    ``"root"``): the table the factories read.  It replaces the table; None or an empty one leaves the file
    ``$STAGE2_POLICIES`` names as the source."""
    _POLICIES.clear()
    for name, p in (policies or {}).items():
        _POLICIES[str(name)] = _entry(name, p)


def _entry(name, p) -> dict:
    if not isinstance(p, dict) or "job_dir" not in p or "steps" not in p:
        raise RowError(f"policy row {name!r}: {{\"job_dir\", \"steps\"}} are needed, not {p!r}")
    out = dict(job_dir=str(p["job_dir"]), steps=int(p["steps"]))
    if p.get("root") is not None:
        out["root"] = str(p["root"])
    return out


def policies() -> dict:
    """The table in force: ``configure``'s, else the JSON file ``$STAGE2_POLICIES`` names, else empty."""
    if _POLICIES:
        return dict(_POLICIES)
    path = os.environ.get(POLICIES_ENV)
    if not path:
        return {}
    doc = s2io.read_json(path)
    if not isinstance(doc, dict):
        raise RowError(f"{path}: a JSON object of row name -> {{job_dir, steps}} is needed")
    return {str(name): _entry(name, p) for name, p in doc.items()}


def policy_entry(name: str) -> dict:
    """The entry of a row name; ``R5G/<seed>`` falls back on ``R5/<seed>``'s."""
    table = policies()
    if name in table:
        return table[name]
    ctrl, _, seed = name.partition("/")
    if ctrl == "R5G" and f"R5/{seed}" in table:
        return table[f"R5/{seed}"]
    raise RowError(f"policy row {name!r}: no checkpoint is configured for it (s2rows.configure, or the JSON file "
                   f"${POLICIES_ENV} names; configured: {', '.join(sorted(table)) or 'none'})")


# ------------------------------------------------------------------------------------------------
# The three build-time checks
# ------------------------------------------------------------------------------------------------
def source_hashes() -> dict:
    """Today's hashes of what ``SOURCE_RULE`` compares: ``{files {path: sha256}, stage1, panels}``."""
    return dict(files={rel: s2io.sha256_file(s2io.STUDY / rel) for rel in SOURCE_FILES},
                stage1=plant.sources_sha256(), panels=panels.panel_sources()["sha256"])


def source_check(run: dict) -> dict:
    """``SOURCE_RULE`` on a run's ``run.json``: ``{trained [per session], judged, changed [what differs]}``."""
    today = source_hashes()
    sessions = run.get("sessions") or []
    trained, changed = [], []
    for i, s in enumerate(sessions):
        src = s.get("sources") or {}
        rec = dict(session=s.get("index", i),
                   files={rel: (src.get("files") or {}).get(rel) for rel in SOURCE_FILES},
                   stage1=src.get("plant_sources"), panels=src.get("panel_sources"))
        trained.append(rec)
        changed += [f"session {rec['session']}: {rel}" for rel in SOURCE_FILES
                    if rec["files"][rel] != today["files"][rel]]
        changed += [f"session {rec['session']}: {what}" for key, what in
                    (("stage1", "the Stage 1 sources (designlab/stage1/*.py: policy.py, env.py)"),
                     ("panels", "the panels' sources")) if rec[key] != today[key]]
    if not sessions:
        changed.append("run.json holds no session, so no recorded source hash")
    return dict(trained=trained, judged=today, changed=changed)


def set_check(run: dict, body: str, root=None) -> dict:
    """``SET_RULE`` on a run's ``run.json``, with the set read today from ``root`` (default: the study's
    ``stage2/sets``; the S1 view: the body's Stage 1 run directory): ``{started, now, later}`` (the freeze the run
    started on, the set's latest, whether they differ), or ``RowError`` with what differs."""
    settings = run.get("settings") or {}
    set_name = run.get("set")
    files = dict((settings.get("inputs") or {}).get("files") or {})
    where = f"{set_name}/{body}"
    if not files:
        raise RowError(f"{where}: run.json records no input file of its tasks (settings.inputs.files)")
    # (a run.json without its task names or its plant inputs has nothing to compare: refused, never passed)
    tasks = settings.get("task_names")
    if not isinstance(tasks, list) or not tasks:
        raise RowError(f"{where}: run.json names no task of its run (settings.task_names): rule K4 cannot be checked")
    was = (settings.get("inputs") or {}).get("plant_inputs")
    if not was:
        raise RowError(f"{where}: run.json records no plant inputs (settings.inputs.plant_inputs): rule K4 cannot be "
                       "checked")
    diff = []
    try:
        if set_name == r5lib.STOCK_SET:
            view, _ = setio.stage1_view(setio._anchored(r5lib.s1_root(body)), body)
            today, now = view["files"], None
        else:
            read, _ = setio.read_inputs(setio.set_dir(set_name, body, setio.ROOT if root is None else root),
                                        sorted(files), require_frozen=True)
            today, now = read["files"], read["freeze"]
        names = r5lib.task_names(body, set_name, **({} if root is None else dict(root=root)))
    except tune.InputError as exc:
        raise RowError(f"{where}: the run's set cannot be read as the run recorded it (rule K4): {exc}") from exc
    for f in sorted(files):
        have = today.get(f)
        if have is None:
            diff.append(f"{f} is not in the set")
        else:
            diff += [f"{f}: {k} {str(have.get(k))[:12]}..., the run recorded {str(files[f].get(k))[:12]}..."
                     for k in ("sha256", "entry_sha256") if have.get(k) != files[f].get(k)]
    diff += [f"task {t} was in the run's table and is absent now" for t in tasks if t not in names]
    if was != panels.key_context(body)["plant_inputs"]:
        diff.append("the plant's inputs (the body's round-two files) are not the run's")
    if diff:
        raise RowError(f"{where}: what the run used is not what the set holds today (rule K4): {'; '.join(diff)}")
    started = settings.get("set_freeze")
    return dict(started=started, now=now, later=bool((started or {}).get("manifest_sha256")
                                                     != (now or {}).get("manifest_sha256")))


def build(name: str, body: str, job_dir, steps, root=None) -> tuple[dict, dict]:
    """The three checks of the module docstring for the row ``name`` (``R5/42``, ``R6/42``, ``R5G/42``) on
    checkpoint ``steps`` of ``job_dir``: ``(ck, loaded)``: everything the row holds of its checkpoint, as plain
    JSON values (its ``spec``), and ``s2load.load_checkpoint``'s value.  ``RowError`` (or ``s2load.CheckpointError``)
    before any episode."""
    ctrl, _, tail = str(name).partition("/")
    if ctrl not in MODES or not tail.isdigit():
        raise RowError(f"policy row {name!r}: {', '.join(c + '/<seed>' for c in CONTROLLERS)}")
    mode, head = MODES[ctrl]
    job_dir, n = Path(job_dir).resolve(), int(steps)
    path = job_dir / "run.json"
    if not path.is_file():
        raise RowError(f"{path}: no run.json (a row is built from a run's job directory)")
    run = s2io.read_json(path)
    settings = run.get("settings") or {}
    if not run.get("settings_sha256") or s2io.sha256_json(settings) != run["settings_sha256"]:
        raise RowError(f"{path}: its settings do not hash to its settings_sha256 (an unfinished or changed record)")
    for key, want in (("mode", mode), ("body", body), ("seed", int(tail))):
        if run.get(key) != want:
            raise RowError(f"{path}: the run's {key} is {run.get(key)!r}, the row {name!r} on {body} asks {want!r}")
    cfgs = settings.get("env_configs") or []
    if not cfgs:
        raise RowError(f"{path}: the run's settings hold no env config")
    core = {k: cfgs[0].get(k) for k in CORE_KEYS}
    if any({k: c.get(k) for k in CORE_KEYS} != core for c in cfgs) or any(v is None for v in core.values()):
        raise RowError(f"{path}: the run's env configs do not agree on {', '.join(CORE_KEYS)}")
    if float(core["rate"]) != panels.RATE:
        raise RowError(f"{path}: the run's commanded clock is {core['rate']!r}, the bar's rows run at {panels.RATE:g}")
    import torch

    torch.set_num_threads(1)
    expect = dict(run_uuid=run["run_uuid"], settings_sha256=run["settings_sha256"])
    loaded = s2load.load_checkpoint(job_dir, n, expect_run=expect)
    record = loaded["record"]
    layout = record["layout"]
    if head == "H1" and not layout.get("head_x"):
        raise RowError(f"{name}: the head script H1 needs a body with a neck yaw and a jaw (an -X body), not {body}")
    src = source_check(run)
    allowed = os.environ.get(SOURCE_CHANGE_ENV) == "1"
    if src["changed"] and not allowed:
        raise RowError(f"{name}: the code that builds its observations and applies its actions is not the run's "
                       f"({'; '.join(src['changed'])}): not judged (rule K9; {SOURCE_CHANGE_ENV}=1 runs it flagged)")
    fz = set_check(run, body, root)
    scales = (dict(leg_rad=float(core["action_leg_rad"]), other_rad=float(core["action_other_rad"]),
                   clock=float(core["action_rate"])) if mode == "R5" else
              dict(span_rule=env.POLICY_ONLY_SPAN_RULE, clock=float(core["action_rate"])))
    ck = dict(
        name=str(name), mode=mode, head=head, body=body, ppo_seed=int(run["seed"]), job_dir=str(job_dir), steps=n,
        root=None if root is None else str(root),
        files={kind: dict(name=f["name"], sha256=f["sha256"], bytes=f["bytes"]) for kind, f in record["files"].items()},
        record_sha256=record.get("sha256"), expect_run=expect,
        run=dict(run_uuid=run["run_uuid"], job=run.get("job"), set=run["set"], set_freeze=fz["started"],
                 settings_sha256=run["settings_sha256"]),
        set_freeze_now=fz["now"], later_freeze=fz["later"], layout=layout, core=core,
        action=dict(scales=scales, rate=float(core["rate"]), filter_hz=float(core["filter_hz"])),
        task_names=list(settings["task_names"]), inputs_files=dict(settings["inputs"]["files"]),
        versions_trained=s2load.comparable(loaded["versions_saved"]),
        versions_judged=s2load.comparable(loaded["versions_here"]), cross_version=bool(loaded["cross_version"]),
        source_change=bool(src["changed"]),
        source_hashes=dict(trained=src["trained"], judged=src["judged"], changed=src["changed"]),
        source_judged=src["judged"])
    return s2io.plain(ck), loaded


def in_place(lib: r5lib.Library, plan) -> bool:
    """``HEADING_RULE``: whether a plan plays a turn in place: a periodic gait with a yaw rate and no forward speed
    (a periodic walking turn, as Stage 1's, walks: its commanded heading stays the integral)."""
    for item in plan:
        if item[0] != "ref":
            continue
        e = lib.refs_doc.get(item[1]) or {}
        if float(e.get("yaw_rate_rad_s") or 0.0) != 0.0 and not float(e.get("speed_m_s") or 0.0) > 0.0:
            return True
    return False


# ------------------------------------------------------------------------------------------------
# The policy
# ------------------------------------------------------------------------------------------------
class S2Policy(panels.Policy):
    """A checkpoint's deterministic policy acting as the training env acts (module docstring): ``prepare(ctx,
    cond, built)`` before the episode's reset (the library, the core and the script of the built condition: nothing
    of it touches the plant's data), ``begin`` and ``act`` at the panels' hooks.  The model loads on first use."""

    every = r5core.SUBSTEPS

    def __init__(self, ck: dict):
        self.ck = ck
        self.mode, self.head = ck["mode"], ck["head"]
        self._loaded = None
        self._parts: dict = {}  # id(plant) -> {p, lib, core}: per process
        self._episode = None
        self.core = None

    def describe(self) -> dict:
        return dict(policy=type(self).__name__, mode=self.mode, head=self.head, steps=self.ck["steps"],
                    files={k: v["sha256"] for k, v in self.ck["files"].items()})

    # ------------------------------------------------------------------ the checkpoint
    def load(self) -> dict:
        """``s2load.load_checkpoint``'s value for the row's checkpoint (once per process): the files must still be
        the ones the row was built on, and this process's sources and versions the ones it was built with."""
        if self._loaded is None:
            ck = self.ck
            import torch

            torch.set_num_threads(1)
            now = source_hashes()
            if now != ck["source_judged"]:
                raise RowError(f"{ck['name']}: the sources changed since the row was built (rule K9)")
            ld = s2load.load_checkpoint(ck["job_dir"], ck["steps"], expect_run=ck["expect_run"])
            if ld["record"].get("sha256") != ck["record_sha256"]:
                raise RowError(f"{ck['name']}: checkpoint {ck['steps']} of {ck['job_dir']} changed since the row was "
                               "built")
            if s2load.comparable(ld["versions_here"]) != ck["versions_judged"]:
                raise RowError(f"{ck['name']}: this process's library versions are not the ones the row was built "
                               f"with ({s2load.comparable(ld['versions_here'])} against {ck['versions_judged']})")
            self._loaded = ld
        return self._loaded

    def provenance(self) -> dict:
        """What is measured or read when the row is judged, for the episode's block and the companion record:
        both version sets as recorded, ``cross_version``, the largest fingerprint difference and its tolerance,
        the checkpoint, ``source_change``, the run and both freezes."""
        ld, ck = self.load(), self.ck
        return dict(
            versions_trained=dict(ld["versions_saved"]), versions_judged=dict(ld["versions_here"]),
            cross_version=bool(ld["cross_version"]), fingerprint_max_abs_diff=float(ld["fingerprint_max_abs_diff"]),
            fingerprint_tol=float(ld["tol"]),
            checkpoint=dict(steps=ck["steps"], record_sha256=ck["record_sha256"],
                            files={k: dict(name=v["name"], sha256=v["sha256"]) for k, v in ck["files"].items()}),
            source_change=bool(ck["source_change"]),
            source_hashes=ck["source_hashes"] if ck["source_change"] else None, run=dict(ck["run"]),
            set_freeze_now=ck["set_freeze_now"], later_freeze=bool(ck["later_freeze"]), mode=self.mode,
            head=self.head)

    # ------------------------------------------------------------------ an episode
    def _library(self, p, set_name: str, run_dir: Path) -> r5lib.Library:
        ck = self.ck
        if set_name == r5lib.STOCK_SET:
            lib = r5lib.Library(p, ck["body"], set_name, float(ck["core"]["rate"]))
        else:
            root = run_dir.parent.parent
            if setio.set_dir(set_name, ck["body"], root) != run_dir:
                raise RowError(f"{ck['name']}: {run_dir} is not the directory of set {set_name} and body {ck['body']}")
            lib = r5lib.Library(p, ck["body"], set_name, float(ck["core"]["rate"]), root=root, require_frozen=True)
        if Path(lib.dir).resolve() != run_dir.resolve():
            raise RowError(f"{ck['name']}: the bar reads the set from {run_dir}, the row's library from {lib.dir}")
        return lib

    def _check_files(self, lib: r5lib.Library) -> None:
        """``SET_RULE`` at the episode: the files the library read that the run recorded have the run's hashes."""
        was = self.ck["inputs_files"]
        bad = [f for f, h in lib.inputs["files"].items()
               if f in was and (h["sha256"], h["entry_sha256"]) != (was[f]["sha256"], was[f]["entry_sha256"])]
        if bad:
            raise RowError(f"{self.ck['name']}: {', '.join(bad)} in {lib.dir} are not the files the run recorded "
                           "(rule K4)")

    def prepare(self, ctx, cond, b) -> None:
        ck = self.ck
        self.load()
        set_name = getattr(cond, "set", None) or r5lib.STOCK_SET
        if set_name != ck["run"]["set"]:
            raise RowError(f"{ck['name']}: the condition {cond.key} is of set {set_name}, the policy's run of "
                           f"{ck['run']['set']}: a policy is judged on the references it imitates")
        p = b.p
        parts = self._parts.get(id(p))
        if parts is None or parts["p"] is not p:
            lib = self._library(p, set_name, Path(ctx.run_dir))
            cfg = dict(ck["core"])
            if self.mode == "R6":  # Stage 1's span rule over the run's own tasks' tracks, as the env computes it
                cfg["span"] = r5core.policy_span(p, [lib.script(t).track for t in ck["task_names"]])
            core = r5core.Core(p, cfg, None)
            lay = ck["layout"]
            dims = {k: int(core.layout[k]) for k in ("actor_dim", "obs_dim", "action_dim")}
            if dims != {k: int(lay[k]) for k in dims}:
                raise RowError(f"{ck['name']}: the checkpoint's layout {({k: lay[k] for k in dims})} is not this "
                               f"body's ({dims})")
            parts = self._parts[id(p)] = dict(p=p, lib=lib, core=core, scripts={})
        lib = parts["lib"]
        kept = parts["scripts"].get(id(b))
        if kept is None or kept[0] is not b:  # (once per built condition: its script, and the files it read)
            script = lib.from_plan(b.plan, track=b.track, heading=r5lib.HEADING_REFERENCE if in_place(lib, b.plan)
                                   else r5lib.HEADING_COMMAND)
            self._check_files(lib)
            kept = parts["scripts"][id(b)] = (b, script)
        self._episode = dict(core=parts["core"], script=kept[1], seconds=float(b.seconds), built=b)

    def begin(self, tk, built, seed: int) -> None:
        ep = self._episode
        if ep is None or ep["built"] is not built:
            raise RowError(f"{self.ck['name']}: begin without prepare (the row's episode prepares its policy)")
        head = r5core.HeadScript.fixed(int(seed), ep["seconds"]) if self.head == "H1" else None
        self.core = ep["core"]
        self.core.begin(lambda: tk.tau_ref, lambda: r5core.tracker_estimate(tk), ep["script"], head)
        self.act(tk, -1)

    def act(self, tk, k: int) -> None:
        ld, core = self._loaded, self.core
        a, _ = ld["model"].predict(s2load.normalize(ld["stats"], core.actor_obs()), deterministic=True)
        a_f = core.filter(a)
        if self.mode == "R6":
            tk.q_target, tk.rate_factor = core.targets(a_f)
        else:
            tk.residual, tk.rate_factor = core.offsets(a_f)


# ------------------------------------------------------------------------------------------------
# The rows
# ------------------------------------------------------------------------------------------------
class _PolicyRow:
    """What the R5 and the R6 row share: the description, the sources, the spec and the provenance."""

    ck: dict
    policy: S2Policy
    name: str

    def describe(self) -> dict:
        """Design 3.11's keys: what an episode executes, and no measured value."""
        ck = self.ck
        d = dict(row=self.name, controller=CONTROLLER[ck["mode"]], mode=ck["mode"], body=ck["body"],
                 ppo_seed=ck["ppo_seed"],
                 checkpoint=dict(steps=ck["steps"], files={k: dict(name=v["name"], sha256=v["sha256"])
                                                           for k, v in ck["files"].items()}),
                 run=dict(ck["run"]), layout=ck["layout"], action=ck["action"],
                 versions_trained=ck["versions_trained"], versions_judged=ck["versions_judged"],
                 cross_version=bool(ck["cross_version"]), deterministic=True, head=ck["head"],
                 source_change=bool(ck["source_change"]))
        if ck["source_change"]:
            d["source_hashes"] = ck["source_hashes"]
        return d

    def sources(self) -> dict:
        """The sha256 of the modules a policy row's episodes run beside the panels' and the bar's sources: this
        file, ``r5core.py``, ``r5lib.py``, ``s2load.py``, ``lookbar.py``, ``lookread.py`` and Stage 1's
        ``policy.py`` and ``env.py``, by path."""
        out = {}
        for mod in (S2Policy, r5core, r5lib, s2load, lookbar, lookread, env):
            out.update(panels.module_sha256(mod))
        path = plant.SOURCES / "policy.py"
        out[s2io.portable(str(path))] = s2io.sha256_file(path)
        return out

    def spec(self) -> dict:
        return dict(cls=f"{type(self).__module__}.{type(self).__qualname__}", name=self.name, ck=self.ck)

    @classmethod
    def from_spec(cls, s: dict):
        ck = s["ck"]
        files = s2load.checkpoint_files(ck["job_dir"], ck["steps"])
        for kind, f in ck["files"].items():
            if not files[kind].is_file() or s2io.sha256_file(files[kind]) != f["sha256"]:
                raise RowError(f"{files[kind]}: not the file the row {s['name']!r} was built on (its sha256 differs, "
                               "or it is gone)")
        return cls(s["name"], ck)

    def provenance(self) -> dict:
        return self.policy.provenance()

    def _check_set(self, cond) -> None:
        set_name = getattr(cond, "set", None) or r5lib.STOCK_SET
        if set_name != self.ck["run"]["set"]:
            raise RowError(f"{self.name}: the condition {cond.key} is of set {set_name}, the policy's run of "
                           f"{self.ck['run']['set']}: a policy is judged on the references it imitates")


class R5Row(_PolicyRow, panels.TrackerRow):
    """``R5/<seed>`` and ``R5G/<seed>`` (module docstring): ``panels.TrackerRow(name, "R2", gains=None, terms=(),
    ankle_cap=(), policy=S2Policy(...))``."""

    def __init__(self, name: str, ck: dict):
        if ck["mode"] != "R5":
            raise RowError(f"{name}: an R5 row on a checkpoint of mode {ck['mode']}")
        panels.TrackerRow.__init__(self, name, "R2", gains=None, gains_info=None, terms=(), ankle_cap=(),
                                   policy=S2Policy(ck), rate=float(ck["core"]["rate"]), controller=CONTROLLER["R5"])
        self.ck = ck

    def episode(self, ctx, cond, seed: int, relax=()):
        self._check_set(cond)
        self.policy.prepare(ctx, cond, ctx.built(cond))
        return panels.TrackerRow.episode(self, ctx, cond, seed, relax)


class _Acting:
    """``control.run``'s recorder hooks forwarded to a ``metrics.Recorder``, with the policy acting at the hooks
    (the R6 row's own: ``panels._Acting``'s rule)."""

    def __init__(self, rec, policy: S2Policy, built, seed: int):
        self.rec, self.policy, self.built, self.seed = rec, policy, built, seed

    def begin(self, tk, n):
        self.tk = tk
        self.rec.begin(tk, n)
        self.policy.begin(tk, self.built, self.seed)

    def step(self, k):
        self.rec.step(k)
        if (k + 1) % self.policy.every == 0:
            self.policy.act(self.tk, k)

    def update(self, *args):
        self.rec.update(*args)

    def end(self, steps, out):
        self.rec.end(steps, out)


class R6Row(_PolicyRow, panels.Row):
    """``R6/<seed>`` (module docstring): ``r5core.R6Tracker`` under ``control.run``, started as the panels' rows
    start (the settled stance, the panels' start and noise seeds, tune's start noise), with the condition's push."""

    hard_only = False

    def __init__(self, name: str, ck: dict):
        if ck["mode"] != "R6":
            raise RowError(f"{name}: an R6 row on a checkpoint of mode {ck['mode']}")
        self.name, self.ck, self.policy = name, ck, S2Policy(ck)
        self.rate = float(ck["core"]["rate"])

    def episode(self, ctx, cond, seed: int, relax=()):
        self._check_set(cond)
        b = ctx.built(cond)
        self.policy.prepare(ctx, cond, b)
        tk = r5core.R6Tracker(b.p, b.track, self.rate)
        rec = b.recorder()
        s = panels.episode_seeds(cond, seed)
        out = control.run(tk, b.seconds, start="stance", seed=s["start"], noise_seed=s["noise"],
                          recorder=_Acting(rec, self.policy, b, seed), joint_noise=tune.START_JOINT_NOISE_RAD,
                          root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S, push=b.push(seed), relax=tuple(relax))
        return b, tune.warned_fall(out, tune.sim_warnings(b.p.d)), rec


def policy_row(name: str, body: str, job_dir, steps, root=None) -> panels.Row:
    """The bare policy row ``name`` on a checkpoint (``build``, then ``R5Row`` or ``R6Row``), its policy holding the
    load of the build."""
    ck, loaded = build(name, body, job_dir, steps, root)
    row = (R6Row if ck["mode"] == "R6" else R5Row)(name, ck)
    row.policy._loaded = loaded
    return row


def _factory(body: str, run_dir, name: str) -> panels.Row:
    """``panels.register_rows``' factory of the three controllers: the policy row inside ``lookbar.LookRow``."""
    e = policy_entry(name)
    return lookbar.LookRow(policy_row(name, body, e["job_dir"], e["steps"], e.get("root")))


for _c in CONTROLLERS:
    panels.register_rows(_c, _factory)


# ------------------------------------------------------------------------------------------------
# A row episode against the env
# ------------------------------------------------------------------------------------------------
def equivalence_check(job_dir, steps, body: str, set_name: str, condition: str, seed: int, k: int, root=None,
                      row: str | None = None, seconds: float | None = None) -> dict:
    """One episode of a checkpoint's row as the bar runs it (``bar.Delayed(lookbar.LookRow(row), k)`` on the
    condition ``condition`` of the set, panel seed ``seed``) against the training env playing the same plan with
    the same policy's deterministic actions, digest for digest after every physics step (``g8a.RULE``).

    The env is built from the run's first env config with the condition's floor, the nominal plant, no random push
    or head event and no interference reading, and is reset as gate G8a resets it (the row's plan, seconds, start
    and noise seeds, friction, push; the delay ``k``; for ``R5G`` the head script H1 of the episode seed).  ``row``
    names the row (default: the run's ``<mode>/<seed>``; ``R5G/<seed>`` for the head script); ``seconds`` cuts the
    condition; ``root`` is where the set is read from.  Nothing is written.  ``r5env`` is imported here, through
    ``g8a``, and nowhere else in this module.  Which sets it may run on is the caller's to keep (rule X1).

    A run whose config has the servo option's draw (``servo="draw"``) is given the panel seed as the env's episode
    seed (the lead's fourth list of 17 Nov 2026), so the env draws ``servo.training_draw(seed)`` and the result names
    it (``servo``); the row as the bar runs it is not served, so the two then run different plant models.

    On a look set (or a version of one) the whole check runs inside ``lookbar.look_set_placement`` (the lead's
    ruling 37 (b) of 17 Nov 2026): the row plays the set's turns and stop at join rule J1's leave point, as a bar
    record made inside ``lookbar``'s context does, and the env is given that plan.  Off the look sets the context
    does nothing."""
    with lookbar.look_set_placement(set_name, body, setio.ROOT if root is None else root):
        return _equivalence_check(job_dir, steps, body, set_name, condition, seed, k, root, row, seconds)


def _equivalence_check(job_dir, steps, body: str, set_name: str, condition: str, seed: int, k: int, root,
                       row: str | None, seconds: float | None) -> dict:
    """``equivalence_check``'s episode and comparison, inside its placement."""
    from designlab.stage2 import bar, g8a, r5env  # (inside the function: no row imports the env)

    job_dir = Path(job_dir).resolve()
    run = s2io.read_json(job_dir / "run.json")
    name = row or f"{run['mode']}/{int(run['seed'])}"
    inner = policy_row(name, body, job_dir, steps, root)
    look = lookbar.LookRow(inner)
    delayed = bar.Delayed(look, int(k))
    opened = lookbar.open_set(body, set_name, [condition], **({} if root is None else dict(root=root)))
    if condition in opened["missing"]:
        raise RowError(f"{set_name}/{body}: {condition} cannot be run ({opened['missing'][condition]['reason']})")
    ctx, cond = opened["ctx"], opened["conds"][condition]
    if seconds is not None:
        cond = dataclasses.replace(cond, seconds=float(seconds))
    r = g8a.row_episode(delayed, ctx, cond, int(seed))
    b = r["built"]
    if inner.policy._episode["script"].heading_rule != r5lib.HEADING_COMMAND:
        raise RowError(f"{condition} plays a turn in place: the row names the reference's yaw as its commanded "
                       "heading and a plan given to the env by its reset options names none (rule E4): the two are "
                       "not compared")
    ld = inner.policy.load()
    mode = inner.ck["mode"]
    cfg = r5env.config_from(dict(run["settings"]["env_configs"][0], mode=mode, floor=cond.floor, randomize=False,
                                 push_prob=0.0, head_events=False, mid_start=0.0, interference="off"))
    e = (r5env.R6Env if mode == "R6" else r5env.R5Env)(cfg, **({} if root is None else dict(root=root)))
    head = r5core.HeadScript.fixed(int(seed), float(b.seconds)) if inner.ck["head"] == "H1" else None
    seeds = panels.episode_seeds(cond, int(seed))
    # a run trained with the servo option's draw (r5env.SERVO_DRAW) draws each episode's servo set from its episode
    # seed, which the env does not have with options.seeds: the panel seed is given as the episode seed (the lead's
    # fourth list of 17 Nov 2026; SERVO_MODEL_DESIGN.md section 4), so the env draws servo.training_draw(seed)
    servo_draw = cfg.servo == r5env.SERVO_DRAW
    episode_seed = dict(episode_seed=int(seed)) if servo_draw else {}
    servo_set = None
    digests: list = []
    e._step_hook = lambda p: digests.append(env._state_digest(p.d))
    term = cause = None
    a_max, factors = 0.0, []
    try:
        obs, _ = e.reset(seed=0, options=dict(
            plan=b.plan, seconds=float(b.seconds), start="stance",
            seeds=dict(start=seeds["start"], noise=seeds["noise"]),
            factors=dict(friction=cond.friction, mass=1.0, kp=1.0, joint_damping=1.0, joint_friction=1.0,
                         latency_steps=int(k), noise_scale=1.0),
            push=b.push(int(seed)), head=head, terminate_on_interference=False, **episode_seed))
        if servo_draw:
            servo_set = dict(option=cfg.servo, episode_seed=e.episode_seed, setting=e.servo_params.name)
        actor = int(e.layout["actor_dim"])
        while True:
            a, _ = ld["model"].predict(s2load.normalize(ld["stats"], obs[:actor]), deterministic=True)
            a_max = max(a_max, float(np.max(np.abs(a))))
            obs, _, terminated, truncated, info = e.step(a)
            factors.append(float(info["clock_factor"]))
            if terminated:
                term, cause = e.k_env, info["fall_cause"]
            if terminated or truncated:
                break
    finally:
        e._step_hook = None
        e.close()
    c = g8a.compare(r["digests"], r["fell_step"], digests, term)
    if r["warned"]:
        c.update(identical=False, ends_alike=False)
    return dict(rule=g8a.RULE, body=body, set=set_name, row=delayed.name, condition=cond.key, seed=int(seed),
                k=int(k), seconds=float(b.seconds), **c, env_steps=int(e.k_env),
                reference_fall_cause=r["fall_cause"], reference_warned=r["warned"], env_fall_cause=cause,
                action_abs_max=a_max, clock_factor_min=min(factors) if factors else None,
                clock_factor_max=max(factors) if factors else None, look=look.last,
                checkpoint=dict(job=run.get("job"), steps=int(steps), run_uuid=run.get("run_uuid")),
                fingerprint_max_abs_diff=float(ld["fingerprint_max_abs_diff"]), cross_version=bool(ld["cross_version"]),
                **({} if servo_set is None else dict(servo=servo_set)))


__all__ = ["CONTROLLERS", "R5Row", "R6Row", "RowError", "S2Policy", "SET_RULE", "SOURCE_RULE", "build", "configure",
           "equivalence_check", "policies", "policy_entry", "policy_row", "set_check", "source_check", "source_hashes"]
