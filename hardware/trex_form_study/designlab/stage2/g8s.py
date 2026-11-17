"""Gate G8s: the training env with the servo option and the zero action is the served R0 row of the same setting,
physics step for physics step (``stage2/design/servo_model/SERVO_MODEL_DESIGN.md`` 5.1, test T13; package S2).

G8a's form (``g8a.py``) with the servo layer on both sides.  Simulation only: the servo model's values are a model,
none is measured on this build's hardware, and no episode here is a record or a result.

* ``compare`` and the digests are G8a's (``g8a.RULE``): ``env._state_digest`` (sha256 of ``qpos``, ``qvel``,
  ``ctrl``, ``sensordata``) after every physics step, the row's through a recorder lent to the built condition, the
  env's through its step hook.
* ``ALIGN_RULE``: G8a's alignment, and on both sides the one parameter object (``servo.ServoParams``), the same
  context manager (``servo.served``) entered after the delay, and a setting's trunk mass applied before the start.
* ``stock``: the development form on the S1 view (Stage 1's own run directory), the env against the panels' R0 row
  inside ``speedfit.servo_delay`` and ``ServedRow``.  No ``bar`` import.  ``check`` runs T13's settings on it.
* ``bar_form``: the gate's form on a frozen set, the env against ``bar.Delayed(ServedRow(the bar's R0 row))``
  (``bar`` is imported inside the function).

**The served row.**  Package S3's row ``servo.Served`` and its names are not built yet.  ``ServedRow`` is this gate's
judging side, in the form design 5.1 gives that row: the setting's trunk mass applied before the inner episode and
undone after (``env.set_trunk_mass``), the inner row's episode inside ``servo.served`` on the condition's plant, and
the bar's delay outside it (``bar.rows`` wraps a registered row in ``bar.Delayed``).  When S3's row exists this gate
should compare the env against it (G8s run again then).

**Sets.**  As G8a's: ``stock`` reads Stage 1's run directory only; ``bar_form`` refuses no set, and on a look set
rule X1's limits are the caller's to keep (no closed-loop episode there before G9-ref has passed).
"""

from __future__ import annotations

from designlab.stage2 import tbenv

tbenv.assign_gl("disable")

import dataclasses  # noqa: E402
import types  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ..stage1 import env, panels, speedfit  # noqa: E402
from . import g8a, r5env, r5lib, s2io, servo, setio  # noqa: E402

SUBSTEPS = r5env.SUBSTEPS
BODY = servo.BODY
RULE = g8a.RULE
ALIGN_RULE = (
    "G8a's alignment (g8a.py), and with the servo layer: one parameter object (servo.ServoParams) is given to both "
    "sides, the env through options.servo (r5env.SERVO_RULE) and the row through ServedRow; on both sides the layer "
    "is servo.served on the plant, entered after the delay (the env: after speedfit.servo_delay at its reset's step "
    "12; the row: bar.Delayed outside ServedRow), and acts from the first physics step; a setting's trunk mass factor "
    "(M-, M+, W1, W2) is applied before the start on both sides (the env: its mass factor at step 6, before the "
    "start at step 11; the row: env.set_trunk_mass before the inner episode, which starts it, and undone after); the "
    "drawn panel D's set is drawn from the run's own bar seed (servo.panel_draw).  An episode is identical by "
    "g8a.RULE")
#: Test T13's named settings (design 5.5); the drawn panel D runs at ``D_SEEDS`` (three runs), each from its own seed.
SETTINGS = ("N", "F+", "P+", "V-", "J+", "M-", "M+")
D_SEEDS = (3042, 3043, 3044)
SEED = 3042
#: 0 and 10 ms (physics steps), as T13 has them.
DELAYS = (0, 5)
CONDITION = "walk/brisk"


def params_of(setting, seed: int) -> servo.ServoParams:
    """The parameter object of a setting for the run with bar seed ``seed``: ``"D"`` is the drawn panel's set of that
    seed; anything else as ``r5env.servo_params`` reads it."""
    if isinstance(setting, str) and setting == "D":
        return servo.setting("D", int(seed))
    return r5env.servo_params(setting)


@contextmanager
def trunk_mass(p, factor: float):
    """The trunk mass and inertia of ``p`` x ``factor`` for the block (``env.set_trunk_mass``, from the masses as they
    are at entry), undone when it ends whatever happens in it; nothing for a factor of 1."""
    factor = float(factor)
    if factor == 1.0:
        yield
        return
    m = p.m
    nominal = types.SimpleNamespace(body_mass=m.body_mass.copy(), body_inertia=m.body_inertia.copy())
    scratch = mujoco.MjData(m)
    env.set_trunk_mass(m, p.trunk, factor, nominal, scratch)
    try:
        yield
    finally:
        env.set_trunk_mass(m, p.trunk, 1.0, nominal, scratch)


class ServedRow:
    """This gate's judging side (module docstring): ``inner``'s episode with the setting's trunk mass and inside
    ``servo.served(p, params)`` on the condition's plant.  ``readings`` holds the layer's readings of the last
    episode."""

    def __init__(self, inner, params: servo.ServoParams):
        self.inner, self.params = inner, params
        self.name = f"{getattr(inner, 'name', 'row')}/servo:{params.name}"
        self.hard_only = bool(getattr(inner, "hard_only", False))
        self.readings = None

    def episode(self, ctx, cond, seed: int, relax=()):
        p = ctx.built(cond).p
        self.readings = None
        with trunk_mass(p, self.params.mass), servo.served(p, self.params) as lay:
            out = self.inner.episode(ctx, cond, seed, relax)
            self.readings = lay.readings()
        return out


class _Delayed:
    """The bar's ``Delayed.episode`` without the bar (``g8a.stock``'s): the inner row inside the delay."""

    def __init__(self, inner, k: int):
        self.inner, self.k = inner, int(k)

    def episode(self, ctx, cond, seed, relax=()):
        with speedfit.servo_delay(ctx.built(cond).p, self.k):
            return self.inner.episode(ctx, cond, seed, relax)


def env_episode(e: r5env.R5Env, plan, seconds: float, seeds: dict, friction, k: int, push,
                params: servo.ServoParams, action=None) -> dict:
    """``g8a.env_episode`` with the servo layer at ``params`` (``options.servo``) and the setting's trunk mass as the
    episode's mass factor: ``{digests, terminated_step, fall_cause, env_steps, interference_first_env_step,
    servo_readings}``."""
    rows: list = []
    e._step_hook = lambda p: rows.append(env._state_digest(p.d))
    term, cause = None, None
    try:
        e.reset(seed=0, options=dict(
            plan=plan, seconds=float(seconds), start="stance", seeds=dict(start=seeds["start"], noise=seeds["noise"]),
            factors=dict(friction=friction, mass=float(params.mass), kp=1.0, joint_damping=1.0, joint_friction=1.0,
                         latency_steps=int(k), noise_scale=1.0),
            push=push, head=None, terminate_on_interference=False, servo=params))
        if e.servo_params is not params:
            raise RuntimeError("the env does not run the parameter object it was given")
        a = np.zeros(e.layout["action_dim"], np.float32) if action is None else np.asarray(action, np.float32)
        while True:
            _, _, terminated, truncated, info = e.step(a)
            if terminated:
                term, cause = e.k_env, info["fall_cause"]
            if terminated or truncated:
                break
        readings = e.servo_layer.readings()
    finally:
        e._step_hook = None
        e.close()  # (the layer out, then the plant's own command and reset back)
    return dict(digests=rows, terminated_step=term, fall_cause=cause, env_steps=e.k_env,
                interference_first_env_step=e.interference_step, servo_readings=readings)


def episode(row, ctx, cond, seed: int, k: int, e: r5env.R5Env, params: servo.ServoParams) -> dict:
    """One comparison: ``row`` (a delayed ``ServedRow`` of ``params``) against the env given ``params``."""
    served = row.inner
    if not isinstance(served, ServedRow) or served.params is not params:
        raise ValueError("the row is not the served row of this parameter object (ALIGN_RULE)")
    r = g8a.row_episode(row, ctx, cond, seed)
    b = r["built"]
    x = env_episode(e, b.plan, b.seconds, panels.episode_seeds(cond, seed), cond.friction, k, b.push(seed), params)
    c = g8a.compare(r["digests"], r["fell_step"], x["digests"], x["terminated_step"])
    if r["warned"]:
        c.update(identical=False, ends_alike=False)
    return dict(condition=cond.key, seed=int(seed), k=int(k), seconds=float(b.seconds), setting=params.name,
                trunk_mass=float(params.mass), **c, env_steps=x["env_steps"], reference_fall_cause=r["fall_cause"],
                reference_warned=r["warned"], env_fall_cause=x["fall_cause"],
                interference_first_env_step=x["interference_first_env_step"],
                servo_readings=dict(row=served.readings, env=x["servo_readings"]))


def _env(body: str, set_name: str, floor: str, root=None) -> r5env.R5Env:
    """G8a's env (``g8a._env``) with the servo option on; every episode is given its parameter object."""
    cfg = r5env.R5Config(body=body, set=set_name, floor=floor, tasks=("stand",), randomize=False, push_prob=0.0,
                         head_events=False, mid_start=0.0, servo="N")
    return r5env.R5Env(cfg, **({} if root is None else dict(root=root)))


#: ``stock``'s plants, row and envs per body and floor (``release`` forgets them).
_STOCK: dict = {}


def release() -> None:
    _STOCK.clear()


def stock(setting, condition: str = CONDITION, seed: int = SEED, k: int = 0, seconds: float | None = None,
          body: str = BODY, params: servo.ServoParams | None = None) -> dict:
    """The development form on the S1 view: the env given the setting against ``_Delayed(ServedRow(the panels' R0
    row))`` on the stock condition (``g8a.stock``'s).  ``params`` is the parameter object (default: ``setting``'s for
    this seed, ``params_of``); ``seconds`` cuts the condition."""
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
    params = params_of(setting, seed) if params is None else params
    row = _Delayed(ServedRow(parts["row"], params), k)
    out = episode(row, parts["ctx"], cond, seed, k, parts["envs"][cond.floor], params)
    return dict(rule=RULE, align_rule=ALIGN_RULE, body=body, set=r5lib.STOCK_SET, row="R0",
                table_sha256=servo.TABLE_SHA256, **out)


def cases(settings=SETTINGS, d_seeds=D_SEEDS, seed: int = SEED) -> list[tuple]:
    """T13's (setting, bar seed) pairs: each named setting at ``seed``, and the drawn panel at each of ``d_seeds``."""
    return [(s, int(seed)) for s in settings] + [("D", int(s)) for s in d_seeds]


def check(condition: str = CONDITION, seconds: float | None = 1.0, delays=DELAYS, settings=SETTINGS,
          d_seeds=D_SEEDS, seed: int = SEED, body: str = BODY) -> dict:
    """T13 on the S1 view: every case of ``cases`` at every delay through ``stock``; ``passes`` when every episode
    is identical over every step its row ran."""
    episodes = [dict(stock(s, condition, sd, k, seconds, body), case=f"{s}@{sd}")
                for s, sd in cases(settings, d_seeds, seed) for k in delays]
    return dict(rule=RULE, align_rule=ALIGN_RULE, body=body, set=r5lib.STOCK_SET, condition=condition,
                seconds=seconds, delays=list(delays), table_sha256=servo.TABLE_SHA256, episodes=episodes,
                identical=sum(1 for e in episodes if e["identical"]), of=len(episodes),
                passes=bool(episodes) and all(e["identical"] and e["compared_steps"] == e["reference_steps"]
                                             for e in episodes))


def bar_form(body: str, set_name: str, conditions, seeds, delays, settings, cut_s: float | None = None,
             root=setio.ROOT) -> dict:
    """The gate's form (``g8a.bar_form``'s): for every condition, panel seed, setting (``params_of``: ``"D"`` draws
    from the seed) and delay, the env against ``bar.Delayed(ServedRow(the bar's R0 row))`` on the frozen set.  A
    condition whose reference the set does not hold is listed under ``not_run``.  Nothing is written."""
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
            for setting in settings:
                params = params_of(setting, seed)
                for k in delays:
                    base = rows[f"R0@{bar.STEP_MS * int(k):d}ms"]
                    row = bar.Delayed(ServedRow(base.inner, params), base.k, f"{base.name}/servo:{params.name}")
                    episodes.append(dict(episode(row, ctx, c, int(seed), int(k), envs[c.floor], params),
                                         row=row.name))
    ctx.release()
    track_b = {s2io.portable(str(s2io.module_path(n))): s2io.sha256_file(s2io.module_path(n))
               for n in ("r5lib.py", "r5core.py", "r5env.py", "g8a.py", "servo.py", "g8s.py")}
    return dict(rule=RULE, align_rule=ALIGN_RULE, body=body, set=set_name, set_freeze=inputs["freeze"], cut_s=cut_s,
                table_sha256=servo.TABLE_SHA256, episodes=episodes,
                identical=sum(1 for e in episodes if e["identical"]), of=len(episodes), not_run=not_run,
                passes=bool(episodes) and all(e["identical"] and e["compared_steps"] == e["reference_steps"]
                                             for e in episodes),
                sources=dict(panels=panels.panel_sources()["sha256"], bar=bar.bar_sources()["sha256"], track_b=track_b),
                key_context=panels.key_context(body), versions=panels.library_versions(),
                blas_threads=s2io.blas_threads(), **s2io.git_state(), at=s2io.now_utc())


__all__ = ["ALIGN_RULE", "CONDITION", "DELAYS", "D_SEEDS", "RULE", "SEED", "SETTINGS", "ServedRow", "bar_form", "cases",
           "check", "env_episode", "episode", "params_of", "release", "stock", "trunk_mass"]
