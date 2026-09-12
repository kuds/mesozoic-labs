"""Semantic stage identity: the per-species stage manifest, v2 (a DAG on ids).

Why (STAGE1_SPLIT_PLAN §4, decided 2026-08-15): the T-Rex curriculum gains a
real fourth stage — recovery — between stance and locomotion.  Renumbering
(recovery becomes 2, locomotion 3, bite 4) would silently change what
"stage 2" and "stage 3" mean in every existing artifact, CSV, postmortem,
and website page.  The manifest replaces renumbering with stable semantic
IDs: a stage *is* its id, its display number is its position in the
manifest's order, and its historical integer identity — when it has one —
is carried explicitly as a ``legacy_number``.

The no-silent-renumbering guarantee, concretely: **an integer stage
reference always means the legacy number**, never the manifest position.
``resolve("trex", 2)`` is locomotion today and stays locomotion after the
manifest gives locomotion position 3; the recovery stage, having no legacy
number, is reachable only by its semantic ID.  New code should iterate
``manifest.stages`` in order (positions 1..N) instead of ``range(1, 4)``;
old code and old artifacts keep meaning without translation.

Manifest v2 (BEHAVIOR_RECIPES_PLAN §4.1, adopted 2026-09-05) makes the
manifest a DAG on those ids.  Ids are an open vocabulary matching
:data:`STAGE_ID_PATTERN`; the four historical ids stay reserved
(:data:`RESERVED_STAGE_IDS`) and the legacy no-rewrite / no-reorder rules
are unchanged.  Each entry may declare three more keys:

* ``warm_start_from`` — the id of an EARLIER entry this node initialises
  from under ``initialize_next_stage``; absent means a root trained from
  scratch.  "Earlier" is enforced at load, so list order is a topological
  order by construction and every in-order walker stays correct.
* ``deliverable`` — this node's certified checkpoint is a published policy.
* ``recipe`` — the behavior label the node belongs to (``stand`` / ``walk``
  / ``hunt`` / ...).  A label resolves to its deepest deliverable in
  manifest order.

**Recipes are derived from the edges, never declared in a second table**: a
recipe is a deliverable plus its ``warm_start_from`` chain
(:meth:`StageManifest.chain_for`).

Two vocabularies live on one manifest (plan §7):

* **advancing** stages (:attr:`StageManifest.advancing_stages`) are the
  legacy-numbered ones — the integer-keyed vocabulary of
  ``CurriculumManager``, ``thresholds_from_configs`` and the sweeps, and the
  schema-v3 historical-bundle completeness rule;
* **deliverables** (:attr:`StageManifest.deliverables`) are the publication
  vocabulary.

They coincide for the legacy trio only by coincidence.

A v1 manifest, or one synthesized from ``stage{1,2,3}_*.toml`` files for a
species without a ``stages.toml``, reads bit-identically to before: the
same entries and advancing trio, with ``warm_start_from`` derived as the
previous advancing entry and ``deliverable`` set on the last advancing
entry only (:func:`_derive_legacy_edges`).  Stand and walk become
deliverables only where a v2 file declares them.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

STAGE_MANIFEST_SCHEMA_V1 = "mesozoic.stage-manifest/v1"
STAGE_MANIFEST_SCHEMA_V2 = "mesozoic.stage-manifest/v2"
#: Every schema this reader accepts.
STAGE_MANIFEST_SCHEMAS = (STAGE_MANIFEST_SCHEMA_V1, STAGE_MANIFEST_SCHEMA_V2)
#: The schema new manifests are written with.
STAGE_MANIFEST_SCHEMA = STAGE_MANIFEST_SCHEMA_V2

#: The four historical ids (STAGE1_SPLIT_PLAN §4).  Since manifest v2 this is
#: NO LONGER the complete vocabulary — any id matching
#: :data:`STAGE_ID_PATTERN` may be declared — but these four keep their
#: meaning everywhere: species-free readers (:func:`stage_ref_from_dirname`,
#: ``evaluation.detect_stage_from_path``, the sweep collector) recognise them
#: without a manifest in hand, and the numbered three map through
#: :data:`LEGACY_STAGE_IDS`.
RESERVED_STAGE_IDS = ("stance", "recovery", "locomotion", "behavior")

#: Semantic identity of the historical numeric stages, used both to
#: synthesize manifests for manifest-less species and to validate that a
#: declared manifest's legacy claims agree with history.
LEGACY_STAGE_IDS = {1: "stance", 2: "locomotion", 3: "behavior"}

#: The open id vocabulary.  Lower-case, digits and underscores, so an id is
#: a valid file prefix, directory suffix, TOML bare key and wandb run token.
STAGE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
#: Ids of this shape are refused: they would collide with ``stage_label``'s
#: integer form, ``find_stage_dir``'s exact ``stage{N}`` probe,
#: ``detect_stage_from_path``'s legacy token and the upload filter.
_LEGACY_LABEL_SHAPE = re.compile(r"^stage[0-9]+$")
#: The ``{position:02d}_{id}`` directory form (:func:`stage_dirname`).
_POSITION_PREFIXED_DIRNAME = re.compile(r"^[0-9]{2}_(?P<id>[a-z][a-z0-9_]*)$")

_V1_ENTRY_KEYS = frozenset({"id", "config", "legacy_number"})
_V2_ENTRY_KEYS = _V1_ENTRY_KEYS | {"warm_start_from", "deliverable", "recipe"}

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
_LEGACY_STAGE_FILE_PREFIXES = {1: "stage1_", 2: "stage2_", 3: "stage3_"}


class StageManifestError(RuntimeError):
    """A stage manifest is missing, malformed, or self-contradictory."""


@dataclass(frozen=True)
class StageEntry:
    """One stage (node) of a species' curriculum."""

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
    #: The id of the EARLIER entry this node initialises from
    #: (``initialize_next_stage``, lineage recorded), or None for a root.
    warm_start_from: str | None = None
    #: Whether this node's certified checkpoint is a published policy.
    deliverable: bool = False
    #: The behavior label this node belongs to, or None.
    recipe: str | None = None

    @property
    def reference(self) -> "int | str":
        """The canonical in-memory reference for this stage.

        The one spelling writers use (load_all_stages keys, stage-result
        dicts, CSV cells): the legacy number where one exists — so every
        pre-manifest artifact and consumer keeps its meaning — and the
        semantic id otherwise (recovery, and every open id).
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

    @property
    def has_parent(self) -> bool:
        """Whether this node has an edge — the thing entry shaping keys on."""
        return self.warm_start_from is not None


@dataclass(frozen=True)
class StageManifest:
    species: str
    stages: tuple[StageEntry, ...]
    #: True when synthesized from stage{N}_*.toml files rather than read
    #: from a declared configs/<species>/stages.toml.
    synthesized: bool
    #: The schema string the file declared; None for a synthesized manifest.
    schema: str | None = None

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

        Defined as the stages carrying a legacy number: the integer-keyed
        vocabulary of ``CurriculumManager``, ``thresholds_from_configs`` and
        the sweeps, and the schema-v3 historical-bundle completeness rule
        ("a complete curriculum recorded a handoff per advancing stage",
        bundle/catalog migration 2026-08-23).  This is NOT the publication
        vocabulary — that is :attr:`deliverables` (manifest v2) — and the two
        coincide for the legacy trio only by coincidence (plan §4.2
        "Curriculum manager", §7 risk 1).  Recovery and every open id are
        non-advancing by construction: they are judged post-stage, and a
        future stage that is numbered-but-non-advancing or
        semantic-but-advancing must revisit this property, not work around
        it.
        """
        return tuple(entry for entry in self.stages if entry.legacy_number is not None)

    @property
    def deliverables(self) -> "tuple[StageEntry, ...]":
        """The nodes whose certified checkpoints are published, in manifest order."""
        return tuple(entry for entry in self.stages if entry.deliverable)

    @property
    def recipe_labels(self) -> "tuple[str, ...]":
        """Every recipe label the manifest carries, in first-appearance order."""
        return tuple(dict.fromkeys(entry.recipe for entry in self.stages if entry.recipe is not None))

    def parent_of(self, ref: "int | str") -> "StageEntry | None":
        """The node *ref* warm-starts from, or None for a root."""
        entry = self.resolve(ref)
        if entry.warm_start_from is None:
            return None
        return self.by_id(entry.warm_start_from)

    def ancestors(self, ref: "int | str") -> "tuple[StageEntry, ...]":
        """*ref*'s ancestor chain, root first, EXCLUDING *ref* itself.

        Terminates because the loader only accepts edges to earlier entries.
        """
        chain: list[StageEntry] = []
        node = self.parent_of(ref)
        while node is not None:
            chain.append(node)
            node = self.parent_of(node.id)
        return tuple(reversed(chain))

    def chain_for(self, ref: "int | str") -> "tuple[StageEntry, ...]":
        """The recipe that ends at *ref*: its ancestors root-first, then *ref*."""
        return self.ancestors(ref) + (self.resolve(ref),)

    def resolve_behavior(self, name: str) -> StageEntry:
        """Resolve a behavior name to the deliverable node it means.

        A recipe label resolves to the LAST deliverable carrying it in
        manifest order (list order is topological, so last is deepest along
        its chain).  Otherwise *name* must be the id of a deliverable node.
        The label-before-id precedence is unambiguous because the loader
        refuses a label equal to any stage id.
        """
        labelled = [entry for entry in self.deliverables if entry.recipe == name]
        if labelled:
            return labelled[-1]
        try:
            entry = self.by_id(name)
        except StageManifestError:
            raise StageManifestError(
                f"{self.species} has no behavior {name!r}; recipe labels: {list(self.recipe_labels)}, "
                f"deliverable ids: {[entry.id for entry in self.deliverables]}"
            ) from None
        if not entry.deliverable:
            raise StageManifestError(
                f"{self.species} stage {name!r} is not a deliverable, so it is not a behavior a chain can "
                "target; train it on its own through the notebook's manual single-node cell"
            )
        return entry


def _derive_legacy_edges(entries: "list[StageEntry]") -> "list[StageEntry]":
    """The v1 / synthesized reading of the DAG: bit-identical to pre-v2 behaviour.

    ``warm_start_from`` = the last advancing (legacy-numbered) entry at a
    lower position, or None — the 2026-08-23 lineage rule — so
    ``entry.has_parent == (entry.position > 1)`` on every legacy manifest,
    which is the entry-shaping bit-identity proof (plan §8.1, §8.3).
    ``deliverable`` = the last advancing entry only; no recipe labels.
    """
    advancing = [entry for entry in entries if entry.legacy_number is not None]
    last_advancing_id = advancing[-1].id if advancing else None
    derived: list[StageEntry] = []
    for entry in entries:
        previous = [candidate for candidate in advancing if candidate.position < entry.position]
        derived.append(
            replace(
                entry,
                warm_start_from=previous[-1].id if previous else None,
                deliverable=entry.id == last_advancing_id,
                recipe=None,
            )
        )
    return derived


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
    return StageManifest(species=species, stages=tuple(_derive_legacy_edges(entries)), synthesized=True, schema=None)


def load_stage_manifest(species: str, configs_dir: "Path | str | None" = None) -> StageManifest:
    """Load the species' declared manifest, or synthesize the legacy one.

    Fail-closed on everything a declared manifest can get wrong: schema,
    malformed or duplicate IDs, missing config files, duplicate or
    historically-wrong legacy numbers, legacy ordering that disagrees with
    the declared order, and — under v2 — an edge that does not name an
    earlier entry, a non-boolean ``deliverable``, a recipe label that is
    malformed or collides with a stage id, a numbered reserved id without
    its legacy number, or a manifest with nothing to publish.  A v1 body
    carrying v2 keys is fatal rather than silently derived over.
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
    schema = raw.get("schema")
    if schema not in STAGE_MANIFEST_SCHEMAS:
        raise StageManifestError(
            f"{manifest_path} declares schema {schema!r}; this reader accepts one of {STAGE_MANIFEST_SCHEMAS}"
        )
    is_v2 = schema == STAGE_MANIFEST_SCHEMA_V2
    allowed_keys = _V2_ENTRY_KEYS if is_v2 else _V1_ENTRY_KEYS
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise StageManifestError(f"{manifest_path} must declare a non-empty [[stages]] list")

    entries: list[StageEntry] = []
    seen_ids: list[str] = []
    for index, raw_stage in enumerate(raw_stages, start=1):
        unknown_keys = sorted(set(raw_stage) - allowed_keys)
        if unknown_keys:
            message = f"{manifest_path} stage {index} has unknown keys {unknown_keys} under schema {schema!r}"
            v2_only = sorted(set(unknown_keys) & _V2_ENTRY_KEYS)
            if v2_only:
                message += f"; declare schema {STAGE_MANIFEST_SCHEMA_V2!r} to use {v2_only}"
            raise StageManifestError(message)
        stage_id = raw_stage.get("id")
        if not isinstance(stage_id, str) or STAGE_ID_PATTERN.match(stage_id) is None:
            raise StageManifestError(
                f"{manifest_path} stage {index} has id {stage_id!r}; a stage id must match {STAGE_ID_PATTERN.pattern}"
            )
        if _LEGACY_LABEL_SHAPE.match(stage_id):
            raise StageManifestError(
                f"{manifest_path} stage {index} has id {stage_id!r}, the legacy stage{{N}} label shape; "
                "that shape is reserved for integer references"
            )
        config_file = raw_stage.get("config")
        if not isinstance(config_file, str) or not (species_dir / config_file).is_file():
            raise StageManifestError(f"{manifest_path} stage {stage_id!r}: config file {config_file!r} not found")
        legacy_number = raw_stage.get("legacy_number")
        if legacy_number is not None and (isinstance(legacy_number, bool) or not isinstance(legacy_number, int)):
            raise StageManifestError(f"{manifest_path} stage {stage_id!r}: legacy_number must be an integer or absent")
        warm_start_from = raw_stage.get("warm_start_from")
        if warm_start_from is not None:
            if not isinstance(warm_start_from, str):
                raise StageManifestError(f"{manifest_path} stage {stage_id!r}: warm_start_from must be a stage id")
            if warm_start_from == stage_id:
                raise StageManifestError(f"{manifest_path} stage {stage_id!r} warm-starts from itself")
            if warm_start_from not in seen_ids:
                raise StageManifestError(
                    f"{manifest_path} stage {stage_id!r}: warm_start_from {warm_start_from!r} must name an EARLIER "
                    f"entry (declared so far: {seen_ids})"
                )
        deliverable = raw_stage.get("deliverable", False)
        if not isinstance(deliverable, bool):
            raise StageManifestError(f"{manifest_path} stage {stage_id!r}: deliverable must be a boolean")
        recipe = raw_stage.get("recipe")
        if recipe is not None and (not isinstance(recipe, str) or STAGE_ID_PATTERN.match(recipe) is None):
            raise StageManifestError(
                f"{manifest_path} stage {stage_id!r}: recipe label {recipe!r} must match {STAGE_ID_PATTERN.pattern}"
            )
        entries.append(
            StageEntry(
                id=stage_id,
                position=index,
                config_file=config_file,
                legacy_number=legacy_number,
                warm_start_from=warm_start_from,
                deliverable=deliverable,
                recipe=recipe,
            )
        )
        seen_ids.append(stage_id)

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
    labels = {entry.recipe for entry in entries if entry.recipe is not None}
    colliding = sorted(labels & set(ids))
    if colliding:
        raise StageManifestError(
            f"{manifest_path}: recipe label collides with a stage id: {colliding} "
            "(a behavior name must resolve to exactly one thing)"
        )
    if is_v2:
        # Species-free readers map these ids through LEGACY_STAGE_IDS no
        # matter what the manifest says; the manifest must agree.
        numbered_ids = {stage_id: number for number, stage_id in LEGACY_STAGE_IDS.items()}
        for entry in entries:
            if entry.id in numbered_ids and entry.legacy_number is None:
                raise StageManifestError(
                    f"{manifest_path} stage {entry.id!r} is a numbered reserved id and must declare "
                    f"legacy_number = {numbered_ids[entry.id]}"
                )
        if not any(entry.deliverable for entry in entries):
            raise StageManifestError(
                f"{manifest_path} declares no deliverable stage; a v2 manifest must have at least one node to publish"
            )
    else:
        entries = _derive_legacy_edges(entries)
    return StageManifest(species=species, stages=tuple(entries), synthesized=False, schema=schema)


def stage_label(ref: "int | str") -> str:
    """Canonical artifact/directory label for a stage reference.

    Legacy integers keep their historical ``stage{N}`` form so every
    existing path, artifact, and consumer stays valid; semantic ids are
    their own label (a recovery run writes ``recovery_final.zip`` under a
    ``recovery_*`` directory, never a fabricated number; an open id such as
    ``follow_direction`` labels ``follow_direction_final.zip`` the same way).

    Species-free, so this validates SHAPE only (:data:`STAGE_ID_PATTERN`,
    never the ``stage{N}`` shape): membership in a species' manifest is
    enforced by every caller's earlier resolution — ``train()`` and
    ``evaluate()`` resolve the stage through the manifest before any label
    is written, and the reporting, wandb and notebook call sites are
    reached only through them.
    """
    if isinstance(ref, bool) or not isinstance(ref, (int, str)):
        raise StageManifestError(f"invalid stage reference {ref!r}")
    if isinstance(ref, int):
        return f"stage{ref}"
    if STAGE_ID_PATTERN.match(ref) is None:
        raise StageManifestError(f"invalid stage reference {ref!r}: a stage id must match {STAGE_ID_PATTERN.pattern}")
    if _LEGACY_LABEL_SHAPE.match(ref):
        raise StageManifestError(
            f"invalid stage reference {ref!r}: the stage{{N}} label shape is reserved for legacy integers"
        )
    return ref


def resolve_stage_key(species: str, key: "int | str") -> StageEntry:
    """Resolve a stage reference as serialized artifacts spell it.

    JSON object keys and CSV cells force every reference through a string,
    so the decimal spelling of a legacy number (``"2"``) means the same
    stage the integer ``2`` always has — the legacy number, never the
    position (module docstring).  Semantic ids — reserved or open — and
    in-memory references pass through :meth:`StageManifest.resolve`.
    Everything else — booleans, empty strings, non-decimal non-id strings —
    fails closed with :class:`StageManifestError`.
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
    what they trained.  An open id follows the same form
    (``05_follow_direction``).  Two rules keep this safe against the
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
    (``stage{N}`` for legacy integers, the bare id for semantic stages —
    an open id, which postdates that layout, still lists its bare form so
    a reader that probes every candidate stays uniform).  Readers that
    locate a stage inside an existing run directory should take the first
    candidate that exists.
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
    manifest at run time; the glob is exact after the prefix, so
    ``follow_direction`` never matches ``06_follow_direction_speed``).
    Returns the first directory that exists; when none does, returns the
    historical name so callers' missing-file errors keep reading the way
    they always have.
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


def stage_ref_from_dirname(name: str, *, species: "str | None" = None) -> "int | str | None":
    """The stage reference a run-directory child NAME denotes, or None.

    The one place that recognises every directory generation, replacing the
    hand-rolled ``NN_{id}`` / bare-id checks that treated the reserved four
    as the complete vocabulary:

    * ``stage{N}`` with N in :data:`LEGACY_STAGE_IDS` -> N (universal: the
      legacy token means the same stage for every species; ``stage4`` is
      nothing).
    * ``NN_{id}`` (:func:`stage_dirname`) -> the id's reference.  With no
      *species* only the :data:`RESERVED_STAGE_IDS` are recognised (a
      numbered one as its legacy int, ``recovery`` as itself) — an
      ``NN_<word>`` directory on some path cannot claim to be a stage
      without a manifest to vouch for it (decision D-A12).  With *species*,
      any id that species' manifest declares (legacy int when numbered,
      else the id); an id the manifest does not declare -> None.
    * a bare id (the 20260819 layout, which predates every open id) -> only
      a reserved id, and with *species* only one that species declares —
      deliberately closed, so ``models`` / ``replays`` / ``ancestors`` are
      never read as stages.
    * anything else -> None.

    A :class:`StageManifestError` from loading *species*' manifest is not
    swallowed.
    """
    if not isinstance(name, str):
        return None
    if _LEGACY_LABEL_SHAPE.match(name):
        number = int(name[len("stage") :])
        return number if number in LEGACY_STAGE_IDS and name == f"stage{number}" else None
    prefixed = _POSITION_PREFIXED_DIRNAME.match(name)
    if prefixed:
        stage_id = prefixed.group("id")
    elif STAGE_ID_PATTERN.match(name) is not None and name in RESERVED_STAGE_IDS:
        stage_id = name
    else:
        return None
    if _LEGACY_LABEL_SHAPE.match(stage_id):
        return None
    if species is None:
        if stage_id not in RESERVED_STAGE_IDS:
            return None
        for number, legacy_id in LEGACY_STAGE_IDS.items():
            if legacy_id == stage_id:
                return number
        return stage_id
    for entry in load_stage_manifest(species).stages:
        if entry.id == stage_id:
            return entry.reference
    return None
