"""Component catalog: actuators, electronics, batteries and materials.

``data/actuators.json`` and ``data/parts.json`` hold the sourced records (manufacturer pages,
datasheets or distributor listings, with URLs and access dates).  Each value carries a basis:

* ``sourced``   read from the cited source;
* ``derived``   computed from sourced values (the derivation is stated);
* ``estimate``  an engineering estimate, with its rule stated, to be replaced by measurement;
* ``measured``  measured on hardware (none in this study).

Planning limits (what the screens use) are derived from the sourced ratings with stated
fractions; they are assumptions pending bench thermal tests, not qualifications.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

#: Sustained planning torque as a fraction of the vendor rated torque.  PR #662 planned the
#: RMD-X4-P36-36-E at 6 N·m sustained against its 10.5 N·m rating (0.57); the same fraction is
#: applied to every actuator so sizes are compared on one convention.
SUSTAINED_FRACTION = 0.57
#: Peak planning torque as a fraction of the vendor peak torque (short transients).
PEAK_FRACTION = 0.70
#: Installed envelope allowances around catalog bodies (radial, axial), metres.
RADIAL_ALLOWANCE = 0.0025
AXIAL_ALLOWANCE = 0.0030


@dataclass
class Actuator:
    id: str
    vendor: str
    model: str
    mass: float  # kg
    diameter: float  # m (body), or box width for servos
    length: float  # m axial
    rated_torque: float  # N·m (vendor definition in notes)
    peak_torque: float  # N·m
    rated_speed_rpm: float
    no_load_speed_rpm: float
    nominal_voltage: float
    kt: float | None  # output-side N·m/A (RMS phase current convention) if known
    phase_resistance: float | None  # ohm
    price: float  # USD
    armature: float  # reflected rotor inertia, kg m^2 (estimate unless sourced)
    box: tuple | None = None  # (L, W, H) for box servos
    basis: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    notes: str = ""

    @property
    def sustained(self) -> float:
        return SUSTAINED_FRACTION * self.rated_torque

    @property
    def peak(self) -> float:
        return PEAK_FRACTION * self.peak_torque

    @property
    def rated_speed(self) -> float:
        return self.rated_speed_rpm * 2 * 3.141592653589793 / 60.0

    @property
    def no_load_speed(self) -> float:
        return self.no_load_speed_rpm * 2 * 3.141592653589793 / 60.0

    @property
    def envelope(self) -> tuple[float, float]:
        """Installed (diameter, length) including allowances."""
        return self.diameter + 2 * RADIAL_ALLOWANCE, self.length + 2 * AXIAL_ALLOWANCE


@dataclass
class Part:
    id: str
    category: str
    vendor: str
    model: str
    dims: tuple  # m (L, W, H)
    mass: float  # kg
    power_typ: float = 0.0
    power_max: float = 0.0
    capacity_wh: float = 0.0
    voltage: float = 0.0
    price: float = 0.0
    fov: str = ""
    max_current: float | None = None
    basis: str = "estimate"
    sources: list = field(default_factory=list)
    notes: str = ""


def _load(name: str) -> dict:
    p = DATA / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@lru_cache(maxsize=1)
def actuators() -> dict[str, Actuator]:
    out = {}
    for rec in _load("actuators.json").get("actuators", []):
        out[rec["id"]] = Actuator(
            id=rec["id"], vendor=rec["vendor"], model=rec["model"], mass=rec["mass_kg"],
            diameter=rec["diameter_m"], length=rec["length_m"], rated_torque=rec["rated_torque_nm"],
            peak_torque=rec["peak_torque_nm"], rated_speed_rpm=rec["rated_speed_rpm"],
            no_load_speed_rpm=rec["no_load_speed_rpm"], nominal_voltage=rec["nominal_voltage_v"],
            kt=rec.get("kt_nm_per_a"), phase_resistance=rec.get("phase_resistance_ohm"),
            price=rec["price_usd"], armature=rec.get("armature_kg_m2", 0.0),
            box=tuple(rec["box_m"]) if rec.get("box_m") else None, basis=rec.get("basis", {}),
            sources=rec.get("sources", []), notes=rec.get("notes", ""),
        )
    return out


@lru_cache(maxsize=1)
def parts() -> dict[str, Part]:
    out = {}
    for rec in _load("parts.json").get("parts", []):
        out[rec["id"]] = Part(
            id=rec["id"], category=rec["category"], vendor=rec.get("vendor", ""), model=rec.get("model", ""),
            dims=tuple(rec["dims_m"]), mass=rec["mass_kg"], power_typ=rec.get("power_typ_w", 0.0),
            power_max=rec.get("power_max_w", 0.0), capacity_wh=rec.get("capacity_wh", 0.0),
            voltage=rec.get("voltage_v", 0.0), price=rec.get("price_usd", 0.0), fov=rec.get("fov", ""),
            max_current=rec.get("max_current_a"),
            basis=rec.get("basis", "estimate"), sources=rec.get("sources", []), notes=rec.get("notes", ""),
        )
    return out


@lru_cache(maxsize=1)
def materials() -> dict:
    return _load("materials.json").get("materials", {})


def actuator(aid: str) -> Actuator:
    return actuators()[aid]


def part(pid: str) -> Part:
    return parts()[pid]


def leg_actuator_options(voltage_class: float | None = None) -> list[Actuator]:
    """Leg-capable actuators sorted by mass (lightest first)."""
    opts = [a for a in actuators().values() if a.box is None and a.rated_torque > 0]
    if voltage_class is not None:
        opts = [a for a in opts if abs(a.nominal_voltage - voltage_class) < 1e-6 or a.nominal_voltage == 0]
    return sorted(opts, key=lambda a: (a.mass, a.price))


def servo_options() -> list[Actuator]:
    return sorted([a for a in actuators().values() if a.box is not None], key=lambda a: (a.mass, a.price))
