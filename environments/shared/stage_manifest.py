"""Semantic stage identity: the per-species stage manifest (stage 1b, W3).

Why (STAGE1_SPLIT_PLAN §4, decided 2026-08-15): the T-Rex curriculum gains a
real fourth stage — recovery — between stance and locomotion.  Renumbering
(recovery becomes 2, locomotion 3, bite 4) would silently change what
"stage 2" and "stage 3" mean in every existing artifact, CSV, postmortem,
and website page.  The manifest replaces renumbering with stable semantic
IDs: a stage *is* ``stance`` / ``recovery`` / ``locomotion`` / ``behavior``,
its display number is its position in the manifest's order, and its
historical integer identity — when it has one — is carried explicitly as a
``legacy_number``.

The no-silent-renumbering guarantee, concretely: **an integer stage
reference always means the legacy number**, never the manifest position.
``resolve("trex", 2)`` is locomotion today and stays locomotion after the
manifest gives locomotion position 3; the recovery stage, having no legacy
number, is reachable only by its semantic ID.  New code should iterate
``manifest.stages`` in order (positions 1..N) instead of ``range(1, 4)``;
old code and old artifacts keep meaning without translation.

Species without a ``configs/<species>/stages.toml`` get a synthesized
legacy manifest from their existing ``stage{1,2,3}_*.toml`` files
(stance / locomotion / behavior with legacy numbers 1 / 2 / 3), which is
how "recovery is enabled for the T-Rex only at first" is expressed: a
species gains a fourth stage by gaining a manifest that declares it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

STAGE_MANIFEST_SCHEMA = "mesozoic.stage-manifest/v1"

#: The stage vocabulary (STAGE1_SPLIT_PLAN §4).  Fail-closed: an unknown ID
#: in a manifest is fatal, so a typo cannot mint a new kind of stage.
KNOWN_STAGE_IDS = ("stance", "recovery", "locomotion", "behavior")

#: Semantic identity of the historical numeric stages, used both to
#: synthesize manifests for manifest-less species and to validate that a
#: declared manifest's legacy claims agree with history.
LEGACY_STAGE_IDS = {1: "stance", 2: "locomotion", 3: "behavior"}

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
_LEGACY_STAGE_FILE_PREFIXES = {1: "stage1_", 2: "stage2_", 3: "stage3_"}


class StageManifestError(RuntimeError):
    """A stage manifest is missing, malformed, or self-contradictory."""


@dataclass(frozen=True)
class StageEntry:
    """One stage of a species' curriculum."""

    id: str
    #: 1-based order in the curriculum — the DISPLAY number.  For a species
    #: with a recovery stage this diverges from legacy_number (locomotion is
    #: position 3, legacy 2); artifacts written by manifest-aware code carry
    #: the id, and integers in old artifacts resolve through legacy_number.
    position: int
    #: TOML file name under configs/<species>/.
    config_file: str
    #: The integer this stage was known as before the manifest, or None for
    #: stages (recovery) that never had a numeric identity.
    legacy_number: int | None

    @property
    def reference(self) -> "int | str":
        """The canonical in-memory reference for this stage.

        The one spelling writers use (load_all_stages keys, stage-result
        dicts, CSV cells): the legacy number where one exists — so every
        pre-manifest artifact and consumer keeps its meaning — and the
        semantic id otherwise (recovery).
        """
        return self.legacy_number if self.legacy_number is not None else self.id

    @property
    def key(self) -> str:
        """The canonical serialized key: :attr:`reference` forced to a string.

        JSON object keys and CSV cells cannot hold integers, so ``2``
        serializes as ``"2"`` and ``"recovery"`` as itself.  Readers accept
        the wider vocabulary via :func:`resolve_stage_key`; writers emit
        only this spelling so two artifacts cannot name one stage two ways.
        """
        return str(self.reference)


@dataclass(frozen=True)
class StageManifest:
    species: str
    stages: tuple[StageEntry, ...]
    #: True when synthesized from stage{N}_*.toml files rather than read
    #: from a declared configs/<species>/stages.toml.
    synthesized: bool

    def by_id(self, stage_id: str) -> StageEntry:
        for entry in self.stages:
            if entry.id == stage_id:
                return entry
        known = [entry.id for entry in self.stages]
        raise StageManifestError(f"{self.species} has no stage {stage_id!r}; declared stages: {known}")

    def by_legacy_number(self, number: int) -> StageEntry:
        for entry in self.stages:
            if entry.legacy_number == number:
                return entry
        raise StageManifestError(
            f"{self.species} has no stage with legacy number {number}; integer references resolve "
            "through legacy numbers only (a stage without one, like recovery, must be named by ID)"
        )

    def by_position(self, position: int) -> StageEntry:
        if not 1 <= position <= len(self.stages):
            raise StageManifestError(f"{self.species} has stages at positions 1..{len(self.stages)}, not {position}")
        return self.stages[position - 1]

    def resolve(self, ref: "int | str") -> StageEntry:
        """Resolve a stage reference: str = semantic ID, int = LEGACY number.

        Integers deliberately do not mean positions — that is the entire
        no-silent-renumbering guarantee (module docstring).
        """
        if isinstance(ref, bool):
            raise StageManifestError(f"invalid stage reference {ref!r}")
        if isinstance(ref, str):
            return self.by_id(ref)
        if isinstance(ref, int):
            return self.by_legacy_number(ref)
        raise StageManifestError(f"invalid stage reference type {type(ref).__name__}")

    @property
    def advancing_stages(self) -> "tuple[StageEntry, ...]":
        """The stages whose gates advance the curriculum, in manifest order.

        Defined as the stages carrying a legacy number.  That equivalence is
        deliberate, not incidental (bundle/catalog migration, 2026-08-23):
        the historical "exactly stages 1, 2, and 3" completeness rules meant
        "a complete curriculum recorded a handoff per advancing stage", and
        the only stage without a legacy number — recovery — is non-advancing
        by design (gate_kind none/v1 until P5; its checkpoint feeds nothing
        forward until a frozen recovery_quality/v1 gate certifies one).  A
        future stage that is numbered-but-non-advancing or
        semantic-but-advancing must revisit this property, not work around
        it.
        """
        return tuple(entry for entry in self.stages if entry.legacy_number is not None)


def _synthesize_legacy_manifest(species: str, species_dir: Path) -> StageManifest:
    entries: list[StageEntry] = []
    for number in sorted(LEGACY_STAGE_IDS):
        matches = sorted(species_dir.glob(f"{_LEGACY_STAGE_FILE_PREFIXES[number]}*.toml"))
        if not matches:
            continue
        if len(matches) > 1:
            raise StageManifestError(f"multiple stage-{number} config files for {species}: {matches}")
        entries.append(
            StageEntry(
                id=LEGACY_STAGE_IDS[number],
                position=len(entries) + 1,
                config_file=matches[0].name,
                legacy_number=number,
            )
        )
    if not entries:
        raise StageManifestError(f"no stage config files found for {species} in {species_dir}")
    return StageManifest(species=species, stages=tuple(entries), synthesized=True)


def load_stage_manifest(species: str, configs_dir: "Path | str | None" = None) -> StageManifest:
    """Load the species' declared manifest, or synthesize the legacy one.

    Fail-closed on everything a declared manifest can get wrong: schema,
    unknown or duplicate IDs, missing config files, duplicate or
    historically-wrong legacy numbers, and legacy ordering that disagrees
    with the declared order.
    """
    base = Path(configs_dir) if configs_dir is not None else _CONFIGS_DIR
    species_dir = base / species
    if not species_dir.is_dir():
        raise StageManifestError(f"config directory not found: {species_dir}")
    manifest_path = species_dir / "stages.toml"
    if not manifest_path.is_file():
        return _synthesize_legacy_manifest(species, species_dir)

    with open(manifest_path, "rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema") != STAGE_MANIFEST_SCHEMA:
        raise StageManifestError(
            f"{manifest_path} declares schema {raw.get('schema')!r}; this reader requires {STAGE_MANIFEST_SCHEMA!r}"
        )
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise StageManifestError(f"{manifest_path} must declare a non-empty [[stages]] list")

    entries: list[StageEntry] = []
    for index, raw_stage in enumerate(raw_stages, start=1):
        unknown_keys = sorted(set(raw_stage) - {"id", "config", "legacy_number"})
        if unknown_keys:
            raise StageManifestError(f"{manifest_path} stage {index} has unknown keys {unknown_keys}")
        stage_id = raw_stage.get("id")
        if stage_id not in KNOWN_STAGE_IDS:
            raise StageManifestError(
                f"{manifest_path} stage {index} has unknown id {stage_id!r}; known: {KNOWN_STAGE_IDS}"
            )
        config_file = raw_stage.get("config")
        if not isinstance(config_file, str) or not (species_dir / config_file).is_file():
            raise StageManifestError(f"{manifest_path} stage {stage_id!r}: config file {config_file!r} not found")
        legacy_number = raw_stage.get("legacy_number")
        if legacy_number is not None and (isinstance(legacy_number, bool) or not isinstance(legacy_number, int)):
            raise StageManifestError(f"{manifest_path} stage {stage_id!r}: legacy_number must be an integer or absent")
        entries.append(StageEntry(id=stage_id, position=index, config_file=config_file, legacy_number=legacy_number))

    ids = [entry.id for entry in entries]
    if len(set(ids)) != len(ids):
        raise StageManifestError(f"{manifest_path} declares duplicate stage ids: {ids}")
    legacy_numbers = [entry.legacy_number for entry in entries if entry.legacy_number is not None]
    if len(set(legacy_numbers)) != len(legacy_numbers):
        raise StageManifestError(f"{manifest_path} declares duplicate legacy numbers: {legacy_numbers}")
    for entry in entries:
        if entry.legacy_number is not None:
            expected = LEGACY_STAGE_IDS.get(entry.legacy_number)
            if expected != entry.id:
                raise StageManifestError(
                    f"{manifest_path}: legacy number {entry.legacy_number} historically means "
                    f"{expected!r}, not {entry.id!r} — a legacy claim must not rewrite history"
                )
    declared_legacy = [entry.legacy_number for entry in entries if entry.legacy_number is not None]
    if declared_legacy != sorted(declared_legacy):
        raise StageManifestError(
            f"{manifest_path}: legacy stages must appear in their historical order "
            "(new stages may be inserted between them, never reorder them)"
        )
    return StageManifest(species=species, stages=tuple(entries), synthesized=False)


def stage_label(ref: "int | str") -> str:
    """Canonical artifact/directory label for a stage reference.

    Legacy integers keep their historical ``stage{N}`` form so every
    existing path, artifact, and consumer stays valid; semantic IDs are
    their own label (a recovery run writes ``recovery_final.zip`` under a
    ``recovery_*`` directory, never a fabricated number).
    """
    if isinstance(ref, bool) or not isinstance(ref, (int, str)):
        raise StageManifestError(f"invalid stage reference {ref!r}")
    if isinstance(ref, int):
        return f"stage{ref}"
    if ref not in KNOWN_STAGE_IDS:
        raise StageManifestError(f"unknown stage id {ref!r}; known: {KNOWN_STAGE_IDS}")
    return ref


def resolve_stage_key(species: str, key: "int | str") -> StageEntry:
    """Resolve a stage reference as serialized artifacts spell it.

    JSON object keys and CSV cells force every reference through a string,
    so the decimal spelling of a legacy number (``"2"``) means the same
    stage the integer ``2`` always has — the legacy number, never the
    position (module docstring).  Semantic ids and in-memory references
    pass through :meth:`StageManifest.resolve`.  Everything else —
    booleans, empty strings, non-decimal non-id strings — fails closed
    with :class:`StageManifestError`.
    """
    manifest = load_stage_manifest(species)
    if isinstance(key, str) and key.isdigit():
        return manifest.by_legacy_number(int(key))
    return manifest.resolve(key)


def stage_dirname(species: str, ref: "int | str") -> str:
    """Stage DIRECTORY name for new run artifacts: ``{position:02d}_{id}``.

    Adopted 2026-08-20 (project decision): stage directories inside a run
    are named ``01_stance``, ``02_recovery``, ``03_locomotion``,
    ``04_behavior`` so a run's folders sort in curriculum order and say
    what they trained. Two rules keep this safe against the
    no-silent-renumbering invariant:

    * The **id suffix is the key**. Anything that aggregates across runs
      joins on the id — the numeric prefix is display and provenance,
      never identity, because a stage's position may move when a future
      stage is inserted while its id never does.
    * The prefix records the stage's position **when the run happened** (a
      run directory is a snapshot); old runs keep their recorded layouts,
      and readers accept every generation (see
      ``stage_dir_candidates``). Deliberately NOT ``stage{position}``:
      ``stage2`` already means locomotion to every pre-manifest artifact,
      and a ``stage2_recovery`` directory would reintroduce through the
      filesystem exactly the renumbering hazard the manifest exists to
      prevent.

    File-level prefixes (checkpoint names, ``*_final`` artifacts, videos)
    deliberately stay on :func:`stage_label` — they live inside the stage
    directory, where the position would be redundant.
    """
    entry = load_stage_manifest(species).resolve(ref)
    return f"{entry.position:02d}_{entry.id}"


def stage_dir_candidates(species: str, ref: "int | str") -> "tuple[str, ...]":
    """Every directory name this stage has ever been written under.

    Newest first: the position-prefixed form for runs from 2026-08-20 on,
    then the :func:`stage_label` form every earlier run used
    (``stage{N}`` for legacy integers, the bare id for semantic stages).
    Readers that locate a stage inside an existing run directory should
    take the first candidate that exists.
    """
    entry = load_stage_manifest(species).resolve(ref)
    labels = [f"{entry.position:02d}_{entry.id}"]
    if entry.legacy_number is not None:
        labels.append(f"stage{entry.legacy_number}")
    labels.append(entry.id)
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return tuple(seen)


def find_stage_dir(run_path: "Path | str", stage: "int | str") -> "Path":
    """Locate a stage's directory inside an existing run, any generation.

    Species-free on purpose — bundle readers often hold only a run path
    and a stage reference. Tries the layouts in age order: the historical
    exact name (``stage{N}`` for legacy integers, the bare id for semantic
    stages), then the position-prefixed ``NN_{id}`` form new runs write
    (matched by glob, because the position prefix depends on the species'
    manifest at run time). Returns the first directory that exists; when
    none does, returns the historical name so callers' missing-file errors
    keep reading the way they always have.
    """
    root = Path(run_path)
    if isinstance(stage, int) and not isinstance(stage, bool):
        legacy_name = f"stage{stage}"
        stage_id = LEGACY_STAGE_IDS.get(stage)
    elif isinstance(stage, str):
        legacy_name = stage
        stage_id = stage
    else:
        raise StageManifestError(f"invalid stage reference {stage!r}")
    exact = root / legacy_name
    if exact.is_dir():
        return exact
    if stage_id is not None:
        matches = sorted(root.glob(f"[0-9][0-9]_{stage_id}"))
        for match in matches:
            if match.is_dir():
                return match
    return exact
