"""PROTOTYPE (design paper of 16 Nov 2026, item B; scratch, not in the repository, never run in an episode).

Floors as rows: a row wrapper that replaces the floor under the row's plant for one episode, the way ``bar.Delayed``
adds a servo delay.  Proposed home: ``designlab/stage2/floors.py`` (a new file; it edits no module that enters an
existing record's key; its own sha256 enters its own records' ``row_sources``).

Rows are named ``<R0|R2>~<floor>[/<seed>]`` and reach the stock runner through ``--row-module`` (``panels.register_rows``),
so ``bar.rows`` wraps them in ``Delayed``: ``R2~c1.25_mu0.9/42@4ms``.
"""

from __future__ import annotations

from contextlib import contextmanager

from designlab import carpet, statics
from designlab.stage1 import control, panels, plant

LABEL = "carpet range (DECISION_LOG P37): floor values without a source; reported, no verdict"
SEP = "~"
BASES = ("R0", "R2")

#: name -> the floor.  ``contact``: None (the floor as compiled), ``("linear", sink_mm)`` (a linear spring per pad
#: contact that carries the whole weight on one foot at that sink: ``carpet.soft_floor``) or ``("stage1", name)``
#: (``carpet.floor_variants``).  ``mu``: the sliding friction of every colliding geom (``control.set_friction``).
FLOORS = {
    "h_mu0.9": dict(contact=None, mu=0.9, why="the null floor: the hard floor as compiled; the identity check B0"),
    "h_mu1.25": dict(contact=None, mu=1.25, why="grippier only"),
    "c0.6_mu0.9": dict(contact=("linear", 0.6), mu=0.9, why="softer only: half the sink of c1.25"),
    "c1.25_mu0.9": dict(contact=("linear", 1.25), mu=0.9, why="softer only: the stiffness of the study's low pile"),
    "c2.4_mu0.9": dict(contact=("linear", 2.4), mu=0.9, why="softer only: the thinnest pile's whole height"),
    "c4.4_mu0.9": dict(contact=("linear", 4.4), mu=0.9, why="softer only: the thickest pile's whole height"),
    "c1.25_mu1.25": dict(contact=("linear", 1.25), mu=1.25, why="softer and grippier"),
    "c1.25_mu0.54": dict(contact=("linear", 1.25), mu=0.54, why="softer and slippery"),
    "rfc_mu0.95": dict(contact=("stage1", "robot_floor_carpet"), mu=0.95, why="Stage 1's surrogate, the anchor"),
}

_ARRAYS = ("geom_priority", "geom_condim", "geom_solref", "geom_solimp", "geom_friction")


def params(p: plant.Plant, name: str) -> dict | None:
    """The floor geom's parameters for ``carpet.apply_floor`` (None: nothing is applied)."""
    kind = FLOORS[name]["contact"]
    if kind is None:
        return None
    m = p.m
    if kind[0] == "stage1":
        return carpet.floor_variants(m, piles={})[kind[1]]
    sink = 1e-3 * float(kind[1])
    weight = float(m.body_subtreemass[p.trunk]) * carpet.G
    pads = carpet.pad_geoms(m)["left"]
    n = 4 * len(pads)
    a_hat = float(m.body_invweight0[p.feet[0], 0])
    pair = carpet.pair_params(m, statics.gid(m, "floor"), pads[0])
    prm = carpet.soft_floor(weight / sink, n, a_hat, sink, [float(v) for v in pair["friction"]],
                            timestep=float(m.opt.timestep))
    return dict(solref=prm["solref"], solimp=prm["solimp"], friction=prm["friction"], condim=int(pair["condim"]),
                K_foot_N_per_m=prm["K_foot_N_per_m"], sink_single_mm=float(kind[1]))


@contextmanager
def floor_on(p: plant.Plant, name: str, settle: bool = True):
    """``name``'s floor under ``p`` for the block, the stance settled on it; ``p`` restored exactly afterwards."""
    m = p.m
    saved = {a: getattr(m, a).copy() for a in _ARRAYS}
    q, meta = p.q_settled.copy(), p.meta.get("settle")
    prm = params(p, name)
    try:
        if prm is not None:
            carpet.apply_floor(m, prm)
        control.set_friction(p, float(FLOORS[name]["mu"]))
        if settle:
            s = p.settle(p._predicted_sinkage(prm, None), float(p.meta["H_m"]))
            if not s["settled"]:
                raise plant.PlantError(f"{p.body} does not stand on the floor {name}: {s['status']}")
        yield prm
    finally:
        for a, v in saved.items():
            getattr(m, a)[:] = v
        p.q_settled = q
        if meta is None:
            p.meta.pop("settle", None)
        else:
            p.meta["settle"] = meta
        if settle:
            p.reset(allow_unsettled=True)


class OnFloor(panels.Row):
    """A row on another floor: the inner row's episode inside ``floor_on``.  Reported, never gated; it runs the walk
    conditions of the hard floor as compiled only (the floor's own friction is the row's)."""

    def __init__(self, inner: panels.Row, floor: str, name: str | None = None):
        if floor not in FLOORS:
            raise ValueError(f"floor {floor!r}: choose from {', '.join(FLOORS)}")
        self.inner, self.floor = inner, floor
        self.name = name or f"{inner.name}{SEP}{floor}"
        self.hard_only = False

    def runs(self, cond) -> bool:
        return getattr(cond, "judge", None) == "bar_walk" and cond.floor == "hard" and cond.friction is None

    def gated(self, cond) -> bool:
        return False

    def describe(self) -> dict:
        return dict(self.inner.describe(), floor_row=self.name, floor=dict(name=self.floor, **FLOORS[self.floor]),
                    floor_label=LABEL)

    def sources(self) -> dict:
        return {**self.inner.sources(), **panels.module_sha256(OnFloor)}

    def spec(self) -> dict:
        return dict(cls=f"{type(self).__module__}.{type(self).__qualname__}", name=self.name, floor=self.floor,
                    inner=self.inner.spec())

    @classmethod
    def from_spec(cls, s: dict) -> "OnFloor":
        return cls(panels.row_from_spec(s["inner"]), s["floor"], s["name"])

    def episode(self, ctx, cond, seed: int, relax=()):
        b = ctx.built(cond)
        with floor_on(b.p, self.floor):
            return self.inner.episode(ctx, cond, seed, relax)


def _factory(body: str, run_dir, name: str) -> OnFloor:
    head, _, seed = name.partition("/")
    base, _, floor = head.partition(SEP)
    inner = base + ("/" + seed if seed else "")
    return OnFloor(panels.tracker_rows(body, run_dir, (inner,))[inner], floor, name)


for _b in BASES:
    for _f in FLOORS:
        panels.register_rows(f"{_b}{SEP}{_f}", _factory)
