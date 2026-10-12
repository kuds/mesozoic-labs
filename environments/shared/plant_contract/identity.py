"""The reviewed revision counters and the portable checkpoint identity.

:class:`PlantVersion` is the human-maintained side of the contract (read from
``configs/plant_versions.toml``); :class:`PlantIdentity` is the machine-derived
identity embedded in checkpoints and training artifacts.  Both name their
plant by model, the pair (species, embodiment) of decision D-D33."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..model_id import ModelId
from .constants import PLANT_IDENTITY_SCHEMA
from .digests import _validate_digest
from .errors import PlantCompatibilityError, PlantContractError

# The identity before decision D-D33 named its plant by species alone, so a
# checkpoint stamped with it cannot say which embodiment it was trained on.
_PLANT_IDENTITY_SCHEMA_V1 = "mesozoic.plant-identity/v1"


@dataclass(frozen=True)
class PlantVersion:
    """Human-reviewed revision counters and interface semantics for one model."""

    model_id: ModelId
    physics_revision: int
    policy_interface_revision: int
    visual_revision: int
    observation_schema: str


@dataclass(frozen=True)
class PlantIdentity:
    """Portable identity embedded in checkpoints and training artifacts.

    It names its plant by ``species`` and ``embodiment`` (schema v2, decision
    D-D33), and :meth:`compatibility_errors` compares both.
    """

    species: str
    embodiment: str
    model_path: str
    physics_revision: int
    policy_interface_revision: int
    visual_revision: int
    source_closure_sha256: str
    policy_interface_sha256: str
    physics_sha256: str
    visual_sha256: str
    nq: int
    nv: int
    nu: int
    observation_dim: int
    action_dim: int

    @property
    def model_id(self) -> ModelId:
        """The model this identity is the plant of."""
        return ModelId(self.species, self.embodiment)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema": PLANT_IDENTITY_SCHEMA,
            "species": self.species,
            "embodiment": self.embodiment,
            "model_path": self.model_path,
            "physics_revision": self.physics_revision,
            "policy_interface_revision": self.policy_interface_revision,
            "visual_revision": self.visual_revision,
            "source_closure_sha256": self.source_closure_sha256,
            "policy_interface_sha256": self.policy_interface_sha256,
            "physics_sha256": self.physics_sha256,
            "visual_sha256": self.visual_sha256,
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PlantIdentity:
        """Parse and validate checkpoint metadata; a v1 identity is refused by name."""
        if value.get("schema") == _PLANT_IDENTITY_SCHEMA_V1:
            raise PlantCompatibilityError(
                f"plant identity schema {_PLANT_IDENTITY_SCHEMA_V1} predates decision D-D33: it names its plant "
                "by species alone, without an embodiment, so it matches no current plant; retrain on the "
                f"current plant, whose identity is {PLANT_IDENTITY_SCHEMA}"
            )
        if value.get("schema") != PLANT_IDENTITY_SCHEMA:
            raise PlantCompatibilityError(
                f"unsupported plant identity schema: {value.get('schema')!r}; expected {PLANT_IDENTITY_SCHEMA}"
            )
        try:
            identity = cls(
                species=str(value["species"]),
                embodiment=str(value["embodiment"]),
                model_path=str(value["model_path"]),
                physics_revision=int(value["physics_revision"]),
                policy_interface_revision=int(value["policy_interface_revision"]),
                visual_revision=int(value["visual_revision"]),
                source_closure_sha256=str(value["source_closure_sha256"]),
                policy_interface_sha256=str(value["policy_interface_sha256"]),
                physics_sha256=str(value["physics_sha256"]),
                visual_sha256=str(value["visual_sha256"]),
                nq=int(value["nq"]),
                nv=int(value["nv"]),
                nu=int(value["nu"]),
                observation_dim=int(value["observation_dim"]),
                action_dim=int(value["action_dim"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlantCompatibilityError(f"invalid plant identity metadata: {exc}") from exc
        for field_name in (
            "source_closure_sha256",
            "policy_interface_sha256",
            "physics_sha256",
            "visual_sha256",
        ):
            try:
                _validate_digest(getattr(identity, field_name), field=field_name)
            except PlantContractError as exc:
                raise PlantCompatibilityError(f"invalid plant identity metadata: {exc}") from exc
        return identity

    def compatibility_errors(self, recorded: PlantIdentity) -> list[str]:
        """Describe policy- or physics-level incompatibilities.

        Source and visual revisions are intentionally excluded: a policy can be
        replayed after a render-only change when its interface and physics are
        identical.
        """
        checks = (
            "species",
            "embodiment",
            "physics_revision",
            "policy_interface_revision",
            "policy_interface_sha256",
            "physics_sha256",
            "nq",
            "nv",
            "nu",
            "observation_dim",
            "action_dim",
        )
        return [
            f"{name}: checkpoint={getattr(recorded, name)!r}, current={getattr(self, name)!r}"
            for name in checks
            if getattr(recorded, name) != getattr(self, name)
        ]
