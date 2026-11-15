"""Gate G8a: R5's env with the zero action is the bar's R0 row, physics step for physics step (round C design 3.7;
rule Z1).

One comparison for the trainer, the gates, the notebook and the tests:

* ``compare``: the pure rule on two digest lists and two ends (``RULE``).
* ``stock``: the development form on the S1 view (Stage 1's own run directory): the env on the stock condition's
  plan against the panels' R0 row inside ``speedfit.servo_delay``.  No ``bar`` import.
* ``bar_form``: the gate's form on a frozen set: the env against ``bar.rows(body, ["R0@<ms>ms"])`` on the bar's
  conditions (``bar`` is imported inside the function).  ``short`` is the trainer's and the notebook's line 3.

Alignment (design 3.7's table): the env builds its own plant; factors ``{friction: the condition's, mass 1, kp 1,
joint damping 1, joint friction 1, latency_steps k, noise_scale 1}``; the script is ``lib.from_plan`` of the row's
own plan; a stance start with the row's start and noise seeds; the row's push; the delay through the same
``speedfit.servo_delay``; interference termination off (the env step at which it would have fired is recorded).
The row runs ``round(seconds / dt)`` physics steps or to its fall; the env runs ``ceil`` of a tenth of that many
env steps.  Digests are ``env._state_digest`` (sha256 of ``qpos``, ``qvel``, ``ctrl``, ``sensordata``) after every
physics step: the row's through a wrapper lent to the built condition's ``recorder`` for the one episode (an
instance attribute, removed in a ``finally``), the env's through its step hook.

No episode here is a record; a gate's record is written by ``gates2``.  Until the lead says otherwise ``bar_form``
runs on no set of the 75 cm round (rule X1); ``stock`` reads Stage 1's run directory only.
"""

from __future__ import annotations

from designlab.stage2 import tbenv

tbenv.assign_gl("disable")

import dataclasses  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import numpy as np  # noqa: E402

from ..stage1 import env, panels, speedfit  # noqa: E402
from . import r5env, r5lib, s2io, setio  # noqa: E402

SUBSTEPS = r5env.SUBSTEPS
RULE = (
    "an episode is identical when (1) no digest (sha256 of qpos, qvel, ctrl and sensordata after a physics step) "
    "differs over the physics steps both ran, and the env ran every step the row ran; (2) the ends agree: if the "
    "row fell at physics step j, the env terminates in the env step that holds j (falls are read at env-step "
    "ends); if the row did not fall, the env does not terminate in any env step that lies wholly inside the row's "
    "steps (the env's last, partial step, up to 9 physics steps past the row's end, is not read).  The row runs "
    "round(seconds / dt) physics steps or to its fall, the env ceil of a tenth of that many env steps.  A row "
    "episode during which MuJoCo warned is not identical to anything")
#: Rule Z1: the gate's conditions beside the gated mu 0.9 ones, its panel seeds and its delays (physics steps).
EXTRA_CONDITIONS = ("walk/v0.4_mu0.54", "walk/nominal")
SEEDS = (3042, 3043)
DELAYS = (0, 5)
SHORT_CONDITION = "walk/v0.4"


def compare(row_digests: list, row_fell_step: int | None, env_digests: list, env_term_step: int | None) -> dict:
    """``RULE`` on the row's digests (one per physics step it ran) and the physics step at which it fell (1-based:
    the number of steps it ran; None: it did not fall), and on the env's digests and the env step in which it
    terminated (1-based; None: it did not)."""
    n_row, n_env = len(row_digests), len(env_digests)
    compared = min(n_row, n_env)
    first = next((i for i in range(compared) if row_digests[i] != env_digests[i]), None)
    if first is None and n_env < n_row:
        first = n_env  # the env stopped before the row did
    if row_fell_step is not None:
        ends_alike = env_term_step == -(-int(row_fell_step) // SUBSTEPS)
    else:
        ends_alike = env_term_step is None or int(env_term_step) > n_row // SUBSTEPS
    return dict(reference_steps=n_row, env_physics_steps=n_env, compared_steps=compared, first_difference=first,
                reference_fell_step=None if row_fell_step is None else int(row_fell_step),
                env_terminated_step=None if env_term_step is None else int(env_term_step),
                ends_alike=bool(ends_alike), identical=bool(first is None and ends_alike))


class _Digests:
    """A ``control.run`` recorder that keeps every physics step's digest and hands every hook on to the built
    condition's own recorder (it only reads)."""

    def __init__(self, rec, p, rows: list):
        self.rec, self.p, self.rows = rec, p, rows

    def begin(self, tk, n):
        self.rec.begin(tk, n)

    def step(self, k):
        self.rows.append(env._state_digest(self.p.d))
        self.rec.step(k)

    def update(self, *args):
        self.rec.update(*args)

    def end(self, *args):
        self.rec.end(*args)

    def __getattr__(self, name):
        return getattr(self.rec, name)


@contextmanager
def lent_recorder(b, rows: list):
    """The built condition ``b`` hands out a digest-keeping recorder for the block: an instance attribute over the
    class's ``recorder``, removed when the block ends, whatever happens in it."""
    make = type(b).recorder
    b.recorder = lambda: _Digests(make(b), b.p, rows)
    try:
        yield
    finally:
        b.__dict__.pop("recorder", None)


def row_episode(row, ctx, cond, seed: int) -> dict:
    """One episode of a row (delay and all, as the row runs it) with its digests: ``{digests, fell_step, out}``."""
    b = ctx.built(cond)
    rows: list = []
    with lent_recorder(b, rows):
        _, out, _ = row.episode(ctx, cond, int(seed), ())
    warned = bool(out.get("mujoco_warnings"))
    return dict(digests=rows, fell_step=len(rows) if out["fell"] and not warned else None, warned=warned,
                fall_cause=out.get("fall_cause"), built=b)


def env_episode(e: r5env.R5Env, plan, seconds: float, seeds: dict, friction, k: int, push) -> dict:
    """The env's episode with the zero action under 3.7's alignment: ``{digests, terminated_step, fall_cause,
    env_steps, interference_first_env_step}``."""
    rows: list = []
    e._step_hook = lambda p: rows.append(env._state_digest(p.d))
    term, cause = None, None
    try:
        e.reset(seed=0, options=dict(
            plan=plan, seconds=float(seconds), start="stance", seeds=dict(start=seeds["start"], noise=seeds["noise"]),
            factors=dict(friction=friction, mass=1.0, kp=1.0, joint_damping=1.0, joint_friction=1.0,
                         latency_steps=int(k), noise_scale=1.0),
            push=push, head=None, terminate_on_interference=False))
        a = np.zeros(e.layout["action_dim"], np.float32)
        while True:
            _, _, terminated, truncated, info = e.step(a)
            if terminated:
                term, cause = e.k_env, info["fall_cause"]
            if terminated or truncated:
                break
    finally:
        e._step_hook = None
        e.close()  # (the plant's own command and reset back)
    return dict(digests=rows, terminated_step=term, fall_cause=cause, env_steps=e.k_env,
                interference_first_env_step=e.interference_step)


def _episode(row, ctx, cond, seed: int, k: int, e: r5env.R5Env) -> dict:
    r = row_episode(row, ctx, cond, seed)
    b = r["built"]
    x = env_episode(e, b.plan, b.seconds, panels.episode_seeds(cond, seed), cond.friction, k, b.push(seed))
    c = compare(r["digests"], r["fell_step"], x["digests"], x["terminated_step"])
    if r["warned"]:
        c.update(identical=False, ends_alike=False)
    return dict(condition=cond.key, seed=int(seed), k=int(k), seconds=float(b.seconds), **c, env_steps=x["env_steps"],
                reference_fall_cause=r["fall_cause"], reference_warned=r["warned"], env_fall_cause=x["fall_cause"],
                interference_first_env_step=x["interference_first_env_step"])


def _env(body: str, set_name: str, floor: str, root=None) -> r5env.R5Env:
    """The gate's env: nominal, no random push or head event, the stand as its one built task (an episode's script
    comes with ``options["plan"]``)."""
    cfg = r5env.R5Config(body=body, set=set_name, floor=floor, tasks=("stand",), randomize=False, push_prob=0.0,
                         head_events=False, mid_start=0.0)
    return r5env.R5Env(cfg, **({} if root is None else dict(root=root)))


#: ``stock``'s plants and env per body and floor (a test calls it several times; ``release`` forgets them).
_STOCK: dict = {}


def release() -> None:
    _STOCK.clear()


def stock(body: str, condition: str = "walk/brisk", seed: int = 3042, k: int = 0, seconds: float | None = None) -> dict:
    """The development form on the S1 view: the env on ``Library(p, body, "S1").from_plan(the stock condition's
    plan)`` with the zero action against the panels' R0 row inside ``speedfit.servo_delay(b.p, k)``.  ``seconds``
    cuts the condition (a length that is no multiple of 20 ms is fine)."""
    conds = {c.key: c for c in panels.conditions()}
    if condition not in conds:
        raise ValueError(f"condition {condition!r}: choose from {', '.join(conds)}")
    cond = conds[condition]
    if seconds is not None:
        cond = dataclasses.replace(cond, seconds=float(seconds))
    if body not in _STOCK:
        run_root = setio._anchored(r5lib.s1_root(body))
        _, docs = setio.stage1_view(run_root, body)
        _STOCK[body] = dict(ctx=panels.Context(body, run_root / body, docs),
                            row=panels.tracker_rows(body, run_root / body, ("R0",))["R0"], envs={})
    parts = _STOCK[body]
    if cond.floor not in parts["envs"]:
        parts["envs"][cond.floor] = _env(body, r5lib.STOCK_SET, cond.floor)

    class _Delayed:  # the bar's ``Delayed.episode``, without the bar: the inner row inside the delay
        @staticmethod
        def episode(ctx, c, s, relax):
            with speedfit.servo_delay(ctx.built(c).p, int(k)):
                return parts["row"].episode(ctx, c, s, relax)

    out = _episode(_Delayed, parts["ctx"], cond, seed, k, parts["envs"][cond.floor])
    return dict(rule=RULE, body=body, set=r5lib.STOCK_SET, row="R0", **out)


def conditions(set_docs: dict) -> list[str]:
    """Rule Z1's conditions on a set: the bar's gated mu 0.9 conditions and ``EXTRA_CONDITIONS``, by key."""
    from . import bar

    gated = [c.key for c in bar.conditions(set_docs) if c.gated and c.friction is None]
    return gated + [c for c in EXTRA_CONDITIONS if c not in gated]


def bar_form(body: str, set_name: str, conditions, seeds, delays, cut_s: float | None = None, root=setio.ROOT) -> dict:
    """The gate's form: for every condition (the bar's keys), panel seed and delay (physics steps), the env against
    the bar's row ``R0@<2k>ms`` on the frozen set.  A condition whose reference the set does not hold is listed
    under ``not_run`` with the set's reason.  ``cut_s`` cuts every condition (a smoke check).  Returns the gate's
    document (``identical`` of ``of`` episodes; rule Z1 passes when they are equal and nothing differs)."""
    from . import bar

    d = setio.set_dir(set_name, body, root)
    set_docs = setio.load(d)
    planner = bar.BarPlanner(d, set_docs["refs.json"]["references"], set_docs["transitions.json"]["transitions"],
                             set_docs["set.json"])
    todo, not_run = [], []
    for c in bar.choose(set_docs, bar.HEADLINE_BLOCK, list(conditions)):
        why = planner.missing(c)
        if why is not None:
            not_run.append(dict(condition=c.key, **why))
        else:
            todo.append(c if cut_s is None else dataclasses.replace(c, seconds=float(cut_s)))
    files = sorted({f for c in todo for f in planner.resolve(c)["files"]})
    inputs, docs = setio.read_inputs(d, files, require_frozen=True)
    ctx = bar.BarContext(body, d, docs, set_docs["set.json"], bar.set_crouch(set_docs["set.json"]))
    rows = bar.rows(body, [f"R0@{bar.STEP_MS * int(k):d}ms" for k in delays])
    envs: dict = {}
    episodes = []
    for c in todo:
        if c.floor not in envs:
            envs[c.floor] = _env(body, set_name, c.floor, root)
        for seed in seeds:
            for k in delays:
                row = rows[f"R0@{bar.STEP_MS * int(k):d}ms"]
                episodes.append(dict(_episode(row, ctx, c, int(seed), int(k), envs[c.floor]), row=row.name))
    ctx.release()
    track_b = {s2io.portable(str(s2io.module_path(n))): s2io.sha256_file(s2io.module_path(n))
               for n in ("r5lib.py", "r5core.py", "r5env.py", "g8a.py")}
    return dict(rule=RULE, body=body, set=set_name, set_freeze=inputs["freeze"], cut_s=cut_s, episodes=episodes,
                identical=sum(1 for e in episodes if e["identical"]), of=len(episodes), not_run=not_run,
                passes=bool(episodes) and all(e["identical"] and e["compared_steps"] == e["reference_steps"]
                                             for e in episodes),
                sources=dict(panels=panels.panel_sources()["sha256"], bar=bar.bar_sources()["sha256"], track_b=track_b),
                key_context=panels.key_context(body), versions=panels.library_versions(),
                blas_threads=s2io.blas_threads(), **s2io.git_state(), at=s2io.now_utc())


def short(body: str, set_name: str) -> dict:
    """``bar_form`` on ``walk/v0.4``, seed 3042, k = 0 and 5: the trainer's own G8a before every new run and the
    notebook's line 3."""
    return bar_form(body, set_name, (SHORT_CONDITION,), SEEDS[:1], DELAYS)


__all__ = ["DELAYS", "EXTRA_CONDITIONS", "RULE", "SEEDS", "bar_form", "compare", "conditions", "release", "short",
           "stock"]
