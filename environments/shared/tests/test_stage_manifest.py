"""Tests for environments.shared.stage_manifest.

The manifest's first promise is no silent renumbering: an integer stage
reference means the legacy number forever, a new stage is reachable only
by its semantic ID, and a manifest that tries to rewrite history is
rejected at load.

Its second promise (manifest v2, BEHAVIOR_RECIPES_PLAN §8.1-8.2): a v1
manifest reads bit-identically under the v2 reader — same entries, same
advancing trio, edges derived as "the previous advancing entry", one
deliverable — and edges are DECLARED, never inferred from position, once a
file says v2.  A model with no stages.toml is refused (decision D-D33
deleted the manifest synthesized from stage file names).
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from collections.abc import Iterator
from typing import Any

import pytest

from environments.shared.model_id import ModelId
from environments.shared.species_names import model_ids
from environments.shared.stage_manifest import (
    _CONFIGS_DIR,
    LEGACY_STAGE_IDS,
    RESERVED_STAGE_IDS,
    RETIRED_RECIPE_LABELS,
    STAGE_ID_PATTERN,
    STAGE_MANIFEST_SCHEMA_V1,
    STAGE_MANIFEST_SCHEMA_V2,
    STAGE_MANIFEST_SCHEMAS,
    StageEntry,
    StageManifestError,
    load_stage_manifest,
    retired_recipe_label_message,
    stage_ref_from_dirname,
)

from .notebook_cells import REPO_ROOT, code_cells, strip_magics

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
#: Every manifest model; each has a committed stages.toml.
MODEL_IDS = model_ids()
INTEGER_MODELS = [
    VELOCIRAPTOR_MONGOLIENSIS,
    ModelId("brachiosaurus_altithorax", "anatomical"),
    ModelId("dibothrosuchus_elaphros", "anatomical"),
]
#: The models whose manifest declares a recovery node, labelled "recover" (decision D-D35).
RECOVERY_MODELS = [TYRANNOSAURUS_REX, COMPSOGNATHUS_LONGIPES, COMPSOGNATHUS_LONGIPES_ROBOT]
#: The pointer a retired "stand" is refused with (decision D-D38).
STAND_POINTER = "'stand' was renamed to 'stance' (decision D-D38); use 'stance'"
#: A made-up model for the synthetic manifests below (the reader needs no manifest entry, only its directory).
PROBE = ModelId("probe_species", "anatomical")

V1 = f'schema = "{STAGE_MANIFEST_SCHEMA_V1}"\n'
V2 = f'schema = "{STAGE_MANIFEST_SCHEMA_V2}"\n'

#: configs/tyrannosaurus_rex/anatomical/stages.toml exactly as committed before Phase A (2026-08-20
#: through 2026-09-05): the frozen v1 body the bit-identity pins read.
TYRANNOSAURUS_REX_V1_BODY = """# T-Rex stage manifest (mesozoic.stage-manifest/v1; STAGE1_SPLIT_PLAN §4,
# adopted 2026-08-15). The curriculum has FOUR stages ordered by position;
# a stage's identity is its semantic id, not its number. legacy_number is
# the integer the stage was known as before this manifest existed — old
# artifacts and integer stage references resolve through it, which is why
# locomotion stays "stage 2" to history even though its position here is 3.
# recovery has no legacy_number: it never had a numeric identity, and is
# addressed by id ("recovery") everywhere.
#
# Config files are named by stage id (decision of 2026-08-20, reversing
# the earlier keep-historical-names stance): the manifest is the only
# place order and numbering live, so filenames carry no stage{N} token
# that could collide with legacy numbers. All references resolve through
# the `config` fields below — integer refs included.

schema = "mesozoic.stage-manifest/v1"

[[stages]]
id = "stance"
config = "stance.toml"
legacy_number = 1

[[stages]]
id = "recovery"
config = "recovery.toml"

[[stages]]
id = "locomotion"
config = "locomotion.toml"
legacy_number = 2

[[stages]]
id = "behavior"
config = "behavior.toml"
legacy_number = 3
"""

#: A v2 manifest with two open ids off the legacy trunk — the plan's
#: "follow" recipe (§4.1) — used wherever a test needs an open id.
OPEN_ID_BODY = (
    V2 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\nrecipe = "stance"\ndeliverable = true\n'
    '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nlegacy_number = 2\nwarm_start_from = "stance"\n'
    'recipe = "walk"\ndeliverable = true\n'
    '[[stages]]\nid = "follow_direction"\nconfig = "follow_direction.toml"\nwarm_start_from = "locomotion"\n'
    'recipe = "follow"\ndeliverable = true\n'
    '[[stages]]\nid = "follow_direction_speed"\nconfig = "follow_direction_speed.toml"\n'
    'warm_start_from = "follow_direction"\nrecipe = "follow"\ndeliverable = true\n'
)

_STUB_CONFIGS = (
    "stance.toml",
    "stance.toml",
    "recovery.toml",
    "locomotion.toml",
    "behavior.toml",
    "follow_direction.toml",
    "follow_direction_speed.toml",
)


def _tuples(manifest):
    return [(entry.id, entry.position, entry.config_file, entry.legacy_number) for entry in manifest.stages]


def _edges(manifest):
    return [(entry.id, entry.warm_start_from) for entry in manifest.stages]


def _write_manifest(tmp_path, model_id, body):
    """A configs root holding one model, ``<root>/<species>/<embodiment>/``, with *body* as its stages.toml."""
    model_dir = model_id.path(tmp_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in _STUB_CONFIGS:
        (model_dir / name).write_text("[stage]\nname = 'x'\n")
    (model_dir / "stages.toml").write_text(body)
    return tmp_path


class TestTyrannosaurusRexManifest:
    def test_four_stages_in_curriculum_order(self):
        manifest = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        assert manifest.schema == STAGE_MANIFEST_SCHEMA_V2
        assert [entry.id for entry in manifest.stages] == ["stance", "recovery", "locomotion", "behavior"]
        assert [entry.position for entry in manifest.stages] == [1, 2, 3, 4]

    def test_integer_references_mean_legacy_numbers_not_positions(self):
        """THE no-silent-renumbering guarantee: 2 is locomotion, forever."""
        manifest = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        assert manifest.resolve(1).id == "stance"
        assert manifest.resolve(2).id == "locomotion"
        assert manifest.resolve(2).position == 3
        assert manifest.resolve(3).id == "behavior"
        assert manifest.resolve(3).position == 4

    def test_recovery_is_reachable_only_by_id(self):
        manifest = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        recovery = manifest.resolve("recovery")
        assert recovery.position == 2
        assert recovery.legacy_number is None
        assert recovery.config_file == "recovery.toml"
        with pytest.raises(StageManifestError, match="named by ID"):
            manifest.by_legacy_number(4)

    def test_unknown_references_are_fatal(self):
        manifest = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        with pytest.raises(StageManifestError, match="no stage 'sprint'"):
            manifest.resolve("sprint")
        with pytest.raises(StageManifestError, match="invalid stage reference"):
            manifest.resolve(True)


class TestDeclaredLegacyTrio:
    """The three integer models declare, in v2, exactly the trio they used to synthesize."""

    @pytest.mark.parametrize("model_id", INTEGER_MODELS, ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_the_three_models_declare_the_legacy_three_stages(self, model_id):
        manifest = load_stage_manifest(model_id)
        assert manifest.schema == STAGE_MANIFEST_SCHEMA_V2
        assert [entry.id for entry in manifest.stages] == ["stance", "locomotion", "behavior"]
        assert [entry.legacy_number for entry in manifest.stages] == [1, 2, 3]
        # For a species without new stages, positions and legacy agree.
        assert all(entry.position == entry.legacy_number for entry in manifest.stages)


class TestSchemaVersions:
    def test_v1_and_v2_both_load_and_v3_is_fatal(self, tmp_path):
        assert STAGE_MANIFEST_SCHEMAS == (STAGE_MANIFEST_SCHEMA_V1, STAGE_MANIFEST_SCHEMA_V2)
        entry = '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
        root = _write_manifest(tmp_path / "v1", PROBE, V1 + entry)
        assert load_stage_manifest(PROBE, configs_dir=root).schema == STAGE_MANIFEST_SCHEMA_V1
        root = _write_manifest(tmp_path / "v2", PROBE, V2 + entry + "deliverable = true\n")
        assert load_stage_manifest(PROBE, configs_dir=root).schema == STAGE_MANIFEST_SCHEMA_V2
        root = _write_manifest(tmp_path / "v3", PROBE, 'schema = "mesozoic.stage-manifest/v3"\n' + entry)
        with pytest.raises(StageManifestError, match=re.escape(f"accepts one of {STAGE_MANIFEST_SCHEMAS}")):
            load_stage_manifest(PROBE, configs_dir=root)

    @pytest.mark.parametrize("key", ['warm_start_from = "stance"', "deliverable = true", 'recipe = "stance"'])
    def test_v1_body_with_v2_keys_is_fatal(self, tmp_path, key):
        """A v1 file cannot smuggle an edge past a reader that would derive one."""
        body = V1 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n' + key + "\n"
        root = _write_manifest(tmp_path, PROBE, body)
        with pytest.raises(StageManifestError, match="declare schema 'mesozoic.stage-manifest/v2'"):
            load_stage_manifest(PROBE, configs_dir=root)


class TestStageIdVocabulary:
    def test_reserved_ids_are_exactly_the_four_and_match_the_pattern(self):
        assert RESERVED_STAGE_IDS == ("stance", "recovery", "locomotion", "behavior")
        assert all(STAGE_ID_PATTERN.match(stage_id) for stage_id in RESERVED_STAGE_IDS)
        assert set(LEGACY_STAGE_IDS.values()) < set(RESERVED_STAGE_IDS)
        assert STAGE_ID_PATTERN.pattern == r"^[a-z][a-z0-9_]*$"

    @pytest.mark.parametrize("stage_id", ["stage4", "stage10"])
    def test_legacy_label_shape_is_refused_as_an_id(self, tmp_path, stage_id):
        body = V2 + f'[[stages]]\nid = "{stage_id}"\nconfig = "stance.toml"\ndeliverable = true\n'
        root = _write_manifest(tmp_path, PROBE, body)
        with pytest.raises(StageManifestError, match="label shape"):
            load_stage_manifest(PROBE, configs_dir=root)


class TestLegacyReadsAreBitIdentical:
    """Plan §8.1: the v2 reader on a v1 manifest changes nothing."""

    def test_v1_tyrannosaurus_rex_body_derives_previous_advancing_edges_and_one_deliverable(self, tmp_path):
        root = _write_manifest(tmp_path, TYRANNOSAURUS_REX, TYRANNOSAURUS_REX_V1_BODY)
        legacy = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root)
        committed = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        assert legacy.schema == STAGE_MANIFEST_SCHEMA_V1
        assert _tuples(legacy) == _tuples(committed)
        assert [entry.reference for entry in legacy.stages] == [1, "recovery", 2, 3]
        assert _edges(legacy) == [
            ("stance", None),
            ("recovery", "stance"),
            ("locomotion", "stance"),
            ("behavior", "locomotion"),
        ]
        assert [entry.id for entry in legacy.deliverables] == ["behavior"]
        assert [entry.id for entry in legacy.advancing_stages] == ["stance", "locomotion", "behavior"]
        assert legacy.recipe_labels == ()

    def test_committed_v2_files_are_the_only_place_stance_and_walk_become_deliverables(self, tmp_path):
        legacy_tyrannosaurus_rex = load_stage_manifest(
            ModelId("tyrannosaurus_rex", "anatomical"),
            configs_dir=_write_manifest(tmp_path / "t", TYRANNOSAURUS_REX, TYRANNOSAURUS_REX_V1_BODY),
        )
        assert [entry.id for entry in legacy_tyrannosaurus_rex.deliverables] == ["behavior"]
        assert [entry.id for entry in load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical")).deliverables] == [
            "stance",
            "recovery",
            "locomotion",
            "behavior",
        ]
        for model_id in INTEGER_MODELS:
            assert [entry.id for entry in load_stage_manifest(model_id).deliverables] == [
                "stance",
                "locomotion",
                "behavior",
            ]

    def test_derived_edge_is_non_none_exactly_when_position_gt_1(self, tmp_path):
        """The entry-shaping bit-identity proof (plan §8.3): has_parent == position > 1."""
        legacy_tyrannosaurus_rex = load_stage_manifest(
            ModelId("tyrannosaurus_rex", "anatomical"),
            configs_dir=_write_manifest(tmp_path / "t", TYRANNOSAURUS_REX, TYRANNOSAURUS_REX_V1_BODY),
        )
        legacy_trio = load_stage_manifest(
            PROBE,
            configs_dir=_write_manifest(
                tmp_path / "v1",
                PROBE,
                V1 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
                '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nlegacy_number = 2\n'
                '[[stages]]\nid = "behavior"\nconfig = "behavior.toml"\nlegacy_number = 3\n',
            ),
        )
        for manifest in (legacy_tyrannosaurus_rex, legacy_trio):
            assert [entry.has_parent for entry in manifest.stages] == [entry.position > 1 for entry in manifest.stages]


class TestFailClosedValidation:
    def test_wrong_schema_is_fatal(self, tmp_path):
        root = _write_manifest(tmp_path, TYRANNOSAURUS_REX, 'schema = "mesozoic.stage-manifest/v3"\n')
        with pytest.raises(StageManifestError, match=re.escape(f"accepts one of {STAGE_MANIFEST_SCHEMAS}")):
            load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root)

    @pytest.mark.parametrize("stage_id", ["Sprint", "4x", "", "stage4", "stance-2"])
    def test_malformed_stage_ids_are_fatal(self, tmp_path, stage_id):
        body = V2 + f'[[stages]]\nid = "{stage_id}"\nconfig = "stance.toml"\ndeliverable = true\n'
        root = _write_manifest(tmp_path, TYRANNOSAURUS_REX, body)
        with pytest.raises(StageManifestError, match="must match|label shape"):
            load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root)

    def test_missing_config_file_is_fatal(self, tmp_path):
        body = V1 + '[[stages]]\nid = "stance"\nconfig = "missing.toml"\n'
        root = _write_manifest(tmp_path, TYRANNOSAURUS_REX, body)
        with pytest.raises(StageManifestError, match="not found"):
            load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root)

    def test_rewriting_history_is_fatal(self, tmp_path):
        """legacy_number 2 IS locomotion; a manifest cannot reassign it."""
        body = V1 + '[[stages]]\nid = "behavior"\nconfig = "stance.toml"\nlegacy_number = 2\n'
        root = _write_manifest(tmp_path, TYRANNOSAURUS_REX, body)
        with pytest.raises(StageManifestError, match="must not rewrite history"):
            load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root)

    def test_reordering_legacy_stages_is_fatal(self, tmp_path):
        body = (
            V1 + '[[stages]]\nid = "locomotion"\nconfig = "stance.toml"\nlegacy_number = 2\n'
            '[[stages]]\nid = "stance"\nconfig = "recovery.toml"\nlegacy_number = 1\n'
        )
        root = _write_manifest(tmp_path, TYRANNOSAURUS_REX, body)
        with pytest.raises(StageManifestError, match="historical order"):
            load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root)

    def test_duplicate_ids_are_fatal(self, tmp_path):
        body = (
            V1 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
            '[[stages]]\nid = "stance"\nconfig = "recovery.toml"\n'
        )
        root = _write_manifest(tmp_path, TYRANNOSAURUS_REX, body)
        with pytest.raises(StageManifestError, match="duplicate stage ids"):
            load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root)

    def test_unknown_entry_keys_are_fatal(self, tmp_path):
        body = V1 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nnumber = 1\n'
        root = _write_manifest(tmp_path, TYRANNOSAURUS_REX, body)
        with pytest.raises(StageManifestError, match="unknown keys"):
            load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root)
        # The three v2 keys are accepted under v2 and fatal under v1.
        v2_entry = (
            '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\ndeliverable = true\n'
            'recipe = "stance"\n'
            '[[stages]]\nid = "recovery"\nconfig = "recovery.toml"\nwarm_start_from = "stance"\n'
        )
        root = _write_manifest(tmp_path / "v2", TYRANNOSAURUS_REX, V2 + v2_entry)
        assert [
            entry.id
            for entry in load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root).stages
        ] == ["stance", "recovery"]
        root = _write_manifest(tmp_path / "v1", TYRANNOSAURUS_REX, V1 + v2_entry)
        with pytest.raises(StageManifestError, match="unknown keys.*declare schema"):
            load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"), configs_dir=root)


class TestV2Validators:
    """Plan §8.2: edges name earlier entries; legacy rules unchanged."""

    def _load(self, tmp_path, body):
        return load_stage_manifest(PROBE, configs_dir=_write_manifest(tmp_path, PROBE, V2 + body))

    def test_warm_start_from_must_name_an_earlier_entry(self, tmp_path):
        body = (
            '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\nwarm_start_from = "locomotion"\n'
            '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nlegacy_number = 2\ndeliverable = true\n'
        )
        with pytest.raises(StageManifestError, match="must name an EARLIER entry"):
            self._load(tmp_path, body)

    def test_self_reference_is_fatal(self, tmp_path):
        body = '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\nwarm_start_from = "stance"\n'
        with pytest.raises(StageManifestError, match="warm-starts from itself"):
            self._load(tmp_path, body)

    def test_unknown_parent_is_fatal(self, tmp_path):
        body = (
            '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
            '[[stages]]\nid = "recovery"\nconfig = "recovery.toml"\nwarm_start_from = "balance"\ndeliverable = true\n'
        )
        with pytest.raises(StageManifestError, match="must name an EARLIER entry"):
            self._load(tmp_path, body)
        body = '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\nwarm_start_from = 1\n'
        with pytest.raises(StageManifestError, match="warm_start_from must be a stage id"):
            self._load(tmp_path, body)

    def test_deliverable_must_be_a_boolean(self, tmp_path):
        # tomllib yields an int for `deliverable = 1`; a truthy int is refused.
        body = '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\ndeliverable = 1\n'
        with pytest.raises(StageManifestError, match="deliverable must be a boolean"):
            self._load(tmp_path, body)

    def test_recipe_label_must_match_the_pattern_and_not_collide_with_an_id(self, tmp_path):
        body = '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\nrecipe = "Stand"\ndeliverable = true\n'
        with pytest.raises(StageManifestError, match="recipe label 'Stand' must match"):
            self._load(tmp_path, body)
        body = (
            '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\nrecipe = "locomotion"\n'
            "deliverable = true\n"
            '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nlegacy_number = 2\n'
        )
        with pytest.raises(
            StageManifestError, match=re.escape("recipe label collides with a stage id: ['locomotion']")
        ):
            self._load(tmp_path, body)

    def test_a_label_may_equal_a_stage_id_only_when_it_resolves_to_that_stage(self, tmp_path):
        """Decision D-D38: resolve_behavior reads a name as a label first, so a label equal to a stage id
        is accepted only when the deepest deliverable carrying it IS that stage; then the label and the id
        name one node."""
        stance = '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
        recovery = '[[stages]]\nid = "recovery"\nconfig = "recovery.toml"\nwarm_start_from = "stance"\n'
        # Its own id, on its own deliverable: accepted, and the label and the id resolve to one node.
        manifest = self._load(tmp_path / "own", stance + 'recipe = "stance"\ndeliverable = true\n')
        assert manifest.resolve_behavior("stance") is manifest.by_id("stance")
        assert manifest.recipe_labels == ("stance",)
        # A deeper deliverable sharing it would make the label mean recovery and shadow the id: refused.
        body = stance + 'recipe = "stance"\ndeliverable = true\n' + recovery + 'recipe = "stance"\ndeliverable = true\n'
        with pytest.raises(StageManifestError, match=re.escape("recipe label collides with a stage id: ['stance']")):
            self._load(tmp_path / "deeper", body)
        # Only on a non-deliverable, the label resolves to nothing while the id names a stage: refused.
        body = stance + 'recipe = "stance"\n' + recovery + "deliverable = true\n"
        with pytest.raises(StageManifestError, match=re.escape("only when it resolves to that stage, decision D-D38")):
            self._load(tmp_path / "non_deliverable", body)
        # Another stage's id on this one is refused as before.
        body = stance + 'recipe = "recovery"\ndeliverable = true\n' + recovery
        with pytest.raises(StageManifestError, match=re.escape("recipe label collides with a stage id: ['recovery']")):
            self._load(tmp_path / "other", body)

    def test_a_retired_recipe_label_is_refused_at_load_naming_its_replacement(self, tmp_path):
        """Decision D-D38: ``stand`` was the stance node's label; a manifest declaring it is refused with
        the pointer, on any node, so a custom manifest cannot bring the old name back as an alias."""
        assert retired_recipe_label_message("stand") == STAND_POINTER
        body = '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\nrecipe = "stand"\ndeliverable = true\n'
        with pytest.raises(StageManifestError) as excinfo:
            self._load(tmp_path / "stance", body)
        assert str(excinfo.value).endswith(f"stages.toml stage 'stance': recipe label {STAND_POINTER}")
        body = (
            '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\ndeliverable = true\n'
            '[[stages]]\nid = "recovery"\nconfig = "recovery.toml"\nwarm_start_from = "stance"\n'
            'recipe = "stand"\ndeliverable = true\n'
        )
        with pytest.raises(StageManifestError, match=re.escape(f"stage 'recovery': recipe label {STAND_POINTER}")):
            self._load(tmp_path / "recovery", body)

    def test_legacy_rules_are_unchanged_under_v2(self, tmp_path):
        cases = {
            "must not rewrite history": (
                '[[stages]]\nid = "behavior"\nconfig = "behavior.toml"\nlegacy_number = 2\ndeliverable = true\n'
            ),
            "historical order": (
                '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nlegacy_number = 2\ndeliverable = true\n'
                '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
            ),
            "duplicate stage ids": (
                '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\ndeliverable = true\n'
                '[[stages]]\nid = "stance"\nconfig = "recovery.toml"\n'
            ),
            "duplicate legacy numbers": (
                '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\ndeliverable = true\n'
                '[[stages]]\nid = "recovery"\nconfig = "recovery.toml"\nlegacy_number = 1\n'
            ),
        }
        for index, (message, body) in enumerate(cases.items()):
            with pytest.raises(StageManifestError, match=message):
                self._load(tmp_path / str(index), body)

    def test_numbered_reserved_id_must_carry_its_legacy_number(self, tmp_path):
        """Decision D-A4: species-free readers map these ids through LEGACY_STAGE_IDS regardless."""
        body = (
            '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\ndeliverable = true\n'
            '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nwarm_start_from = "stance"\n'
        )
        with pytest.raises(
            StageManifestError, match="'locomotion' is a numbered reserved id and must declare legacy_number = 2"
        ):
            self._load(tmp_path, body)
        # recovery is reserved but never numbered, so it needs none.
        body = (
            '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\ndeliverable = true\n'
            '[[stages]]\nid = "recovery"\nconfig = "recovery.toml"\nwarm_start_from = "stance"\n'
        )
        assert self._load(tmp_path / "ok", body).by_id("recovery").legacy_number is None

    def test_a_v2_manifest_with_no_deliverable_is_fatal(self, tmp_path):
        """Decision D-A4: nothing could ever publish from it."""
        body = '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
        with pytest.raises(StageManifestError, match="declares no deliverable stage"):
            self._load(tmp_path, body)

    def test_absent_keys_under_v2_mean_root_and_not_deliverable(self, tmp_path):
        """No derivation once a file says v2: an undeclared edge is a root."""
        body = (
            '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
            '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nlegacy_number = 2\n'
            '[[stages]]\nid = "behavior"\nconfig = "behavior.toml"\nlegacy_number = 3\ndeliverable = true\n'
        )
        manifest = self._load(tmp_path, body)
        assert _edges(manifest) == [("stance", None), ("locomotion", None), ("behavior", None)]
        assert [entry.deliverable for entry in manifest.stages] == [False, False, True]
        assert [entry.recipe for entry in manifest.stages] == [None, None, None]
        assert manifest.chain_for("behavior") == (manifest.by_id("behavior"),)


class TestManifestV2Edges:
    def test_entry_fields_default_to_root_not_deliverable_no_label(self):
        entry = StageEntry("x", 1, "x.toml", None)
        assert entry.warm_start_from is None
        assert entry.deliverable is False
        assert entry.recipe is None
        assert entry.has_parent is False
        assert entry.reference == "x" and entry.key == "x"

    def test_tyrannosaurus_rex_parents_are_the_declared_edges(self):
        manifest = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        assert manifest.parent_of("stance") is None
        assert manifest.parent_of("recovery").id == "stance"
        assert manifest.parent_of("locomotion").id == "stance"
        assert manifest.parent_of(2).id == "stance"
        assert manifest.parent_of("behavior").id == "locomotion"
        assert [entry.has_parent for entry in manifest.stages] == [False, True, True, True]

    def test_ancestors_and_chain_are_root_first(self):
        manifest = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        assert [entry.id for entry in manifest.chain_for("behavior")] == ["stance", "locomotion", "behavior"]
        assert [entry.id for entry in manifest.chain_for(3)] == ["stance", "locomotion", "behavior"]
        # A chain never visits a sibling branch.
        assert [entry.id for entry in manifest.chain_for("recovery")] == ["stance", "recovery"]
        assert [entry.id for entry in manifest.chain_for("locomotion")] == ["stance", "locomotion"]
        assert [entry.id for entry in manifest.ancestors("behavior")] == ["stance", "locomotion"]
        assert [entry.id for entry in manifest.ancestors("recovery")] == ["stance"]
        assert manifest.ancestors("stance") == ()
        assert manifest.chain_for("stance") == (manifest.by_id("stance"),)

    def test_deliverables_in_manifest_order_for_all_four_models(self):
        assert [entry.id for entry in load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical")).deliverables] == [
            "stance",
            "recovery",
            "locomotion",
            "behavior",
        ]
        for model_id in INTEGER_MODELS:
            assert [entry.id for entry in load_stage_manifest(model_id).deliverables] == [
                "stance",
                "locomotion",
                "behavior",
            ]

    def test_recipe_labels_resolve_to_the_deepest_deliverable(self, tmp_path):
        tyrannosaurus_rex = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        assert tyrannosaurus_rex.recipe_labels == ("stance", "recover", "walk", "hunt")
        assert tyrannosaurus_rex.resolve_behavior("stance").id == "stance"
        assert tyrannosaurus_rex.resolve_behavior("recover").id == "recovery"
        assert tyrannosaurus_rex.resolve_behavior("walk").id == "locomotion"
        assert tyrannosaurus_rex.resolve_behavior("hunt").id == "behavior"
        for model_id in INTEGER_MODELS:
            manifest = load_stage_manifest(model_id)
            assert manifest.resolve_behavior("stance").id == "stance"
            assert manifest.resolve_behavior("walk").id == "locomotion"
            assert manifest.resolve_behavior("hunt").id == "behavior"
        # Every committed manifest, the Compsognathus pair included: root-first chains, no unknown label.
        for model_id in MODEL_IDS:
            manifest = load_stage_manifest(model_id)
            assert manifest.resolve_behavior("hunt").id == "behavior", model_id
            assert manifest.resolve_behavior("stance").id == "stance", model_id
            assert manifest.chain_for("behavior")[0].id == "stance", model_id
            with pytest.raises(StageManifestError):
                manifest.resolve_behavior("sprint")
        pilot = load_stage_manifest(PROBE, configs_dir=_write_manifest(tmp_path, PROBE, OPEN_ID_BODY))
        assert pilot.recipe_labels == ("stance", "walk", "follow")
        follow = pilot.resolve_behavior("follow")
        assert follow.id == "follow_direction_speed"
        assert [entry.id for entry in pilot.chain_for(follow.id)] == [
            "stance",
            "locomotion",
            "follow_direction",
            "follow_direction_speed",
        ]
        assert [entry.id for entry in pilot.advancing_stages] == ["stance", "locomotion"]

    def test_resolve_behavior_accepts_a_deliverable_id_and_refuses_a_non_deliverable_or_unknown_one(self, tmp_path):
        pilot = load_stage_manifest(PROBE, configs_dir=_write_manifest(tmp_path, PROBE, OPEN_ID_BODY))
        assert pilot.resolve_behavior("follow_direction").id == "follow_direction"
        assert load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical")).resolve_behavior("stance").id == "stance"
        body = (
            V2 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
            '[[stages]]\nid = "recovery"\nconfig = "recovery.toml"\nwarm_start_from = "stance"\ndeliverable = true\n'
        )
        manifest = load_stage_manifest(PROBE, configs_dir=_write_manifest(tmp_path / "x", PROBE, body))
        with pytest.raises(StageManifestError, match="'stance' is not a deliverable.*manual single-node cell"):
            manifest.resolve_behavior("stance")
        with pytest.raises(StageManifestError, match=re.escape("recipe labels: [], deliverable ids: ['recovery']")):
            manifest.resolve_behavior("sprint")

    def test_a_second_leaf_off_one_parent_is_expressible(self, tmp_path):
        body = (
            V2 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\ndeliverable = true\n'
            '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nlegacy_number = 2\nwarm_start_from = "stance"\n'
            "deliverable = true\n"
            '[[stages]]\nid = "behavior"\nconfig = "behavior.toml"\nlegacy_number = 3\nwarm_start_from = "locomotion"\n'
            'recipe = "hunt"\ndeliverable = true\n'
            '[[stages]]\nid = "follow_direction"\nconfig = "follow_direction.toml"\nwarm_start_from = "locomotion"\n'
            'recipe = "follow"\ndeliverable = true\n'
        )
        manifest = load_stage_manifest(PROBE, configs_dir=_write_manifest(tmp_path, PROBE, body))
        assert manifest.parent_of("behavior").id == manifest.parent_of("follow_direction").id == "locomotion"
        assert [entry.id for entry in manifest.chain_for("follow_direction")] == [
            "stance",
            "locomotion",
            "follow_direction",
        ]
        assert [entry.id for entry in manifest.chain_for("behavior")] == ["stance", "locomotion", "behavior"]
        assert manifest.by_id("follow_direction").position == 4
        assert manifest.by_id("follow_direction").reference == "follow_direction"


class TestRecipeLabels:
    """The recipe labels on every committed manifest: ``stance``, ``recover`` (where a model has a recovery
    node), ``walk`` and ``hunt``.

    Decision D-D35 (the maintainer, 2026-10-16) gave the recovery node its own label, ``recover``: before it
    the recovery node was labelled ``stand`` too, so ``stand`` resolved to recovery (stance -> recovery) on
    the three models that have one and to stance elsewhere. Decision D-D38 (2026-10-18) made the stance
    behavior's name ``stance`` everywhere: the stance node's label is its own id, so the label and the id
    name one node with one chain on every model, and ``stand`` is retired, refused naming ``stance`` (no
    alias; decision D-D33's rule for retired identifiers). ``recover`` is stance -> recovery where a model
    has a recovery node and an unknown behavior elsewhere. A behavior's chain is ``chain_for`` of the node
    it resolves to, as the notebook's resolve cell and ``train_curriculum`` take it (``chain_for`` itself
    takes a stage reference, never a label).
    """

    @staticmethod
    def _chain(manifest, behavior):
        return manifest.chain_for(manifest.resolve_behavior(behavior).id)

    @pytest.mark.parametrize("model_id", MODEL_IDS, ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_stance_is_the_stance_stage_on_every_model(self, model_id):
        manifest = load_stage_manifest(model_id)
        assert manifest.by_id("stance").recipe == "stance"
        assert manifest.resolve_behavior("stance") is manifest.by_id("stance")
        assert self._chain(manifest, "stance") == manifest.chain_for("stance") == (manifest.by_id("stance"),)

    @pytest.mark.parametrize("model_id", MODEL_IDS, ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_stand_is_refused_naming_stance_on_every_model(self, model_id):
        """``stand``, the stance label until D-D38, does not resolve: the refusal names ``stance`` and the
        decision, in the "has no behavior" shape every caller wraps."""
        with pytest.raises(StageManifestError) as excinfo:
            load_stage_manifest(model_id).resolve_behavior("stand")
        assert str(excinfo.value) == f"{model_id} has no behavior 'stand': {STAND_POINTER}"

    def test_stand_is_refused_even_by_a_manifest_that_could_not_declare_it(self, tmp_path):
        """The refusal is the vocabulary's, not the manifest's: a v1 manifest, which carries no labels at
        all, and a synthetic v2 one refuse ``stand`` with the same pointer."""
        v1 = load_stage_manifest(
            PROBE,
            configs_dir=_write_manifest(
                tmp_path / "v1", PROBE, V1 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
            ),
        )
        open_ids = load_stage_manifest(PROBE, configs_dir=_write_manifest(tmp_path / "v2", PROBE, OPEN_ID_BODY))
        for manifest in (v1, open_ids):
            with pytest.raises(StageManifestError, match=re.escape(f"has no behavior 'stand': {STAND_POINTER}")):
                manifest.resolve_behavior("stand")

    @pytest.mark.parametrize("model_id", RECOVERY_MODELS, ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_recover_is_stance_then_recovery_on_the_three_recovery_models(self, model_id):
        manifest = load_stage_manifest(model_id)
        assert (
            manifest.resolve_behavior("recover") is manifest.resolve_behavior("recovery") is manifest.by_id("recovery")
        )
        assert [entry.id for entry in self._chain(manifest, "recover")] == ["stance", "recovery"]
        assert self._chain(manifest, "recover") == self._chain(manifest, "recovery")

    @pytest.mark.parametrize("model_id", INTEGER_MODELS, ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_recover_is_not_a_behavior_of_a_model_without_recovery(self, model_id):
        with pytest.raises(
            StageManifestError, match=re.escape("has no behavior 'recover'; recipe labels: ['stance', ")
        ):
            load_stage_manifest(model_id).resolve_behavior("recover")

    def test_recipe_labels_per_model(self):
        expected = {model_id: ("stance", "walk", "hunt") for model_id in INTEGER_MODELS}
        expected.update({model_id: ("stance", "recover", "walk", "hunt") for model_id in RECOVERY_MODELS})
        assert set(expected) == set(MODEL_IDS)
        assert {model_id: load_stage_manifest(model_id).recipe_labels for model_id in MODEL_IDS} == expected
        labelled = {
            model_id: {entry.id: entry.recipe for entry in load_stage_manifest(model_id).stages}
            for model_id in MODEL_IDS
        }
        for model_id, recipes in labelled.items():
            assert recipes["stance"] == "stance" and recipes["locomotion"] == "walk" and recipes["behavior"] == "hunt"
            assert recipes.get("recovery", "recover") == "recover", model_id

    @pytest.mark.parametrize("model_id", MODEL_IDS, ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_a_label_equal_to_a_stage_id_names_that_stage(self, model_id):
        """Only ``stance`` equals a stage id, and it resolves to that stage; every other label is no id.
        A label that is one of the four historical ids, which model-free readers resolve by name, resolves
        to that id's node, so the two readings agree ("recover" is not "recovery")."""
        manifest = load_stage_manifest(model_id)
        ids = {entry.id for entry in manifest.stages}
        assert set(manifest.recipe_labels) & ids == {"stance"}
        for label in set(manifest.recipe_labels) & ids:
            assert manifest.resolve_behavior(label) is manifest.by_id(label)
        assert set(manifest.recipe_labels) & set(RESERVED_STAGE_IDS) == {"stance"}
        assert not set(manifest.recipe_labels) & set(RETIRED_RECIPE_LABELS)


def _toml_strings(value: Any) -> Iterator[str]:
    """Every string value of a parsed TOML document, at any depth."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _toml_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _toml_strings(item)


class TestRetiredRecipeLabelsAreNamedNowhere:
    """Decision D-D38 under decision D-D33's rule for retired identifiers: ``stand`` was renamed to ``stance``,
    and nothing resolves or offers it. The loader and ``resolve_behavior`` refuse it (above); here, no
    committed configuration value (a stage manifest's ``recipe``, a species-manifest ``deliverable``) and no
    notebook code cell (a string constant, or an ``@param`` dropdown option) names it, so no shipped default
    or choice leads to the refusal. Prose may still say "stand"."""

    def test_no_committed_config_value_is_a_retired_recipe_label(self):
        hits = []
        for path in sorted(_CONFIGS_DIR.rglob("*.toml")):
            with path.open("rb") as handle:
                document = tomllib.load(handle)
            hits += [
                (str(path.relative_to(_CONFIGS_DIR)), value)
                for value in _toml_strings(document)
                if value in RETIRED_RECIPE_LABELS
            ]
        assert hits == []

    def test_no_notebook_code_cell_names_a_retired_recipe_label(self):
        notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
        assert notebooks
        hits = []
        for notebook in notebooks:
            for index, source in code_cells(notebook):
                constants = [
                    node.value
                    for node in ast.walk(ast.parse(strip_magics(source)))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                ]
                options: list[Any] = []
                for line in source.splitlines():
                    if "# @param [" in line:
                        found, _ = json.JSONDecoder().raw_decode(line.split("# @param ", 1)[1])
                        options += found
                hits += [
                    (notebook.name, index, value) for value in constants + options if value in RETIRED_RECIPE_LABELS
                ]
        assert hits == []


class TestCommittedManifestsAreV2:
    @pytest.mark.parametrize("model_id", MODEL_IDS, ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_every_model_manifest_is_v2_with_all_nodes_deliverable(self, model_id):
        manifest = load_stage_manifest(model_id)
        assert manifest.schema == STAGE_MANIFEST_SCHEMA_V2
        assert all(entry.deliverable for entry in manifest.stages)
        assert manifest.recipe_labels == (
            ("stance", "recover", "walk", "hunt") if model_id in RECOVERY_MODELS else ("stance", "walk", "hunt")
        )
        for entry in manifest.stages:
            assert (model_id.path(_CONFIGS_DIR) / entry.config_file).is_file()
        assert [entry.id for entry in manifest.advancing_stages] == ["stance", "locomotion", "behavior"]

    def test_tyrannosaurus_rex_declares_edges_deliverables_and_labels(self):
        manifest = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        assert [(entry.id, entry.warm_start_from, entry.deliverable, entry.recipe) for entry in manifest.stages] == [
            ("stance", None, True, "stance"),
            # locomotion's parent is stance until plan A2 flips it to recovery.
            ("recovery", "stance", True, "recover"),
            ("locomotion", "stance", True, "walk"),
            ("behavior", "locomotion", True, "hunt"),
        ]

    @pytest.mark.parametrize("model_id", INTEGER_MODELS, ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_the_integer_models_declare_no_recovery(self, model_id):
        manifest = load_stage_manifest(model_id)
        assert [(entry.id, entry.warm_start_from, entry.recipe) for entry in manifest.stages] == [
            ("stance", None, "stance"),
            ("locomotion", "stance", "walk"),
            ("behavior", "locomotion", "hunt"),
        ]
        with pytest.raises(StageManifestError, match="no stage 'recovery'"):
            manifest.resolve("recovery")

    def test_v2_manifest_does_not_move_any_dirname(self):
        from environments.shared.stage_manifest import stage_dirname

        pre_phase_a = {
            TYRANNOSAURUS_REX: ["01_stance", "02_recovery", "03_locomotion", "04_behavior"],
            VELOCIRAPTOR_MONGOLIENSIS: ["01_stance", "02_locomotion", "03_behavior"],
            ModelId("brachiosaurus_altithorax", "anatomical"): ["01_stance", "02_locomotion", "03_behavior"],
            ModelId("dibothrosuchus_elaphros", "anatomical"): ["01_stance", "02_locomotion", "03_behavior"],
            # The two Compsognathus models landed (PRs #521-#526) on the
            # T-Rex layout with v1 manifests; their v2 files keep it.
            COMPSOGNATHUS_LONGIPES: ["01_stance", "02_recovery", "03_locomotion", "04_behavior"],
            COMPSOGNATHUS_LONGIPES_ROBOT: ["01_stance", "02_recovery", "03_locomotion", "04_behavior"],
        }
        assert set(pre_phase_a) == set(MODEL_IDS)
        for model_id, names in pre_phase_a.items():
            manifest = load_stage_manifest(model_id)
            assert [stage_dirname(model_id, entry.reference) for entry in manifest.stages] == names
            assert [stage_dirname(model_id, entry.id) for entry in manifest.stages] == names


#: The tables each model's recovery.toml inherits from its stance.toml
#: through ``extends`` (CU-13).  Never ``curriculum``; never ``sac`` for the
#: T-Rex, whose recovery stage is PPO-only (giving it stance's never-run
#: ``[sac]`` would move two golden digest lines).
RECOVERY_EXTENDS = {
    TYRANNOSAURUS_REX: ["stage", "env", "ppo"],
    COMPSOGNATHUS_LONGIPES: ["stage", "env", "ppo", "sac"],
    COMPSOGNATHUS_LONGIPES_ROBOT: ["stage", "env", "ppo", "sac"],
}
#: The one ``[env]`` delta, in the order each recovery.toml declares it.
PERTURBATION_KEYS = [
    "perturbation_capture_velocity_multiple",
    "perturbation_interval",
    "perturbation_jitter",
    "perturbation_duration",
    "perturbation_direction",
]
#: The T-Rex keys recovery.toml writes out with stance's values rather than
#: inheriting them (the cleanup plan's CU-13 row).
TYRANNOSAURUS_REX_REDECLARED_ENV_KEYS = ["foot_contact_gate", "foot_contact_weight", "foot_contact_saturation_force"]
#: The T-Rex [env] keys recovery.toml overrides with ITS OWN value, in place
#: (decision D-D27): the settled stance-width capture moves before the first
#: push, which lands at 200 +- 50 steps.
TYRANNOSAURUS_REX_OVERRIDDEN_ENV_KEYS = {"stance_width_settle_steps": 140}


def _ordered(value):
    """*value* with every mapping turned into its item list, so ``==`` also compares key order."""
    if isinstance(value, dict):
        return [(key, _ordered(sub)) for key, sub in value.items()]
    return value


def _raw_stage_toml(model_id, stage_id):
    import tomllib

    path = model_id.path(_CONFIGS_DIR) / load_stage_manifest(model_id).by_id(stage_id).config_file
    with open(path, "rb") as handle:
        return tomllib.load(handle)


class TestRecoveryStageConfig:
    """The recovery stage config is stance's task plus exactly one delta, inherited through ``extends``.

    The resolved tables are pinned with their KEY ORDER.  Every golden digest
    hashes sorted JSON, so neither the digest snapshot nor ``--exact`` would
    see an order change, but ``stage_config.json`` is written unsorted and
    result bundles hash its bytes.
    """

    def test_loads_by_semantic_id_and_only_by_it(self):
        from environments.shared.config import load_stage_config

        config = load_stage_config(ModelId("tyrannosaurus_rex", "anatomical"), "recovery")
        assert config["name"] == "recovery"
        env = config["env_kwargs"]
        assert env["perturbation_capture_velocity_multiple"] == 1.5
        # Integer loading is untouched: 2 still means locomotion.
        assert (
            load_stage_config(ModelId("tyrannosaurus_rex", "anatomical"), 2)["name"]
            == load_stage_config(ModelId("tyrannosaurus_rex", "anatomical"), "locomotion")["name"]
        )

    def test_every_recovery_stage_is_covered(self):
        models_with_recovery = [
            model_id
            for model_id in MODEL_IDS
            if "recovery" in [entry.id for entry in load_stage_manifest(model_id).stages]
        ]
        assert sorted(models_with_recovery) == sorted(RECOVERY_EXTENDS)

    @pytest.mark.parametrize("model_id", sorted(RECOVERY_EXTENDS), ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_extends_stance_for_the_declared_tables_never_curriculum(self, model_id):
        extends = _raw_stage_toml(model_id, "recovery")["extends"]
        assert extends == {"stage": "stance", "tables": RECOVERY_EXTENDS[model_id]}
        assert "curriculum" not in extends["tables"]
        if model_id == TYRANNOSAURUS_REX:
            assert "sac" not in extends["tables"]

    @pytest.mark.parametrize("model_id", sorted(RECOVERY_EXTENDS), ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_the_file_declares_only_its_deltas(self, model_id):
        recovery = _raw_stage_toml(model_id, "recovery")
        stance = _raw_stage_toml(model_id, "stance")
        redeclared = TYRANNOSAURUS_REX_REDECLARED_ENV_KEYS if model_id == TYRANNOSAURUS_REX else []
        overridden = TYRANNOSAURUS_REX_OVERRIDDEN_ENV_KEYS if model_id == TYRANNOSAURUS_REX else {}
        assert list(recovery) == ["extends", "stage", "env", "ppo", "curriculum"]
        assert list(recovery["stage"]) == ["name", "description"]
        assert list(recovery["env"]) == redeclared + list(overridden) + PERTURBATION_KEYS
        assert {key: recovery["env"][key] for key in redeclared} == {key: stance["env"][key] for key in redeclared}
        assert {key: recovery["env"][key] for key in overridden} == overridden
        assert all(stance["env"][key] != value for key, value in overridden.items())
        assert recovery["ppo"] == {"ent_coef_decay_timesteps": 2_000_000}

    @pytest.mark.parametrize("model_id", sorted(RECOVERY_EXTENDS), ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_resolved_env_is_stance_plus_exactly_the_perturbation_block_in_order(self, model_id):
        from environments.shared.config import load_stage_config

        stance = load_stage_config(model_id, "stance")
        recovery = load_stage_config(model_id, "recovery")
        assert (recovery["name"], recovery["description"]) != (stance["name"], stance["description"])
        perturbation = [(key, recovery["env_kwargs"][key]) for key in PERTURBATION_KEYS]
        overridden = TYRANNOSAURUS_REX_OVERRIDDEN_ENV_KEYS if model_id == TYRANNOSAURUS_REX else {}
        expected = [(key, overridden.get(key, value)) for key, value in stance["env_kwargs"].items()]
        assert _ordered(recovery["env_kwargs"]) == expected + perturbation

    @pytest.mark.parametrize("model_id", sorted(RECOVERY_EXTENDS), ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_resolved_algorithm_tables_are_stance_but_the_entropy_horizon_in_order(self, model_id):
        from environments.shared.config import load_stage_config

        stance = load_stage_config(model_id, "stance")
        recovery = load_stage_config(model_id, "recovery")
        # [ppo] is stance's with exactly one measured delta, overridden in
        # place: ent_coef_decay_timesteps is anchored to recovery's own 3M
        # budget (2M, ~2/3 — the same ratio as stance's 7M-of-11M) because
        # the 20260821_142144 pilot trained its whole budget at ent_coef >=
        # 0.0014 when the mirrored 7M horizon never completed (2026-08
        # review §3.4).  policy_kwargs, nested, comes over whole.
        assert stance["ppo_kwargs"]["ent_coef_decay_timesteps"] == 7_000_000
        expected_ppo = {**stance["ppo_kwargs"], "ent_coef_decay_timesteps": 2_000_000}
        assert _ordered(recovery["ppo_kwargs"]) == _ordered(expected_ppo)
        assert "policy_kwargs" in recovery["ppo_kwargs"]
        if "sac" in RECOVERY_EXTENDS[model_id]:
            assert recovery["sac_kwargs"] and _ordered(recovery["sac_kwargs"]) == _ordered(stance["sac_kwargs"])
        else:
            assert stance["sac_kwargs"] and recovery["sac_kwargs"] == {}

    @pytest.mark.parametrize("model_id", sorted(RECOVERY_EXTENDS), ids=lambda m: f"{m.species}-{m.embodiment}")
    def test_curriculum_is_the_recovery_files_own(self, model_id):
        from environments.shared.config import load_stage_config

        curriculum = load_stage_config(model_id, "recovery")["curriculum_kwargs"]
        assert _ordered(curriculum) == _ordered(_raw_stage_toml(model_id, "recovery")["curriculum"])
        assert curriculum["gate_kind"] == "recovery_quality/v1"
        # The stance's [curriculum] is its own too: trex stance moved to
        # stance_quality/v2 with physics r8 (D-D24) and the anatomical
        # compsognathus with physics r2 (D-D26), and both to the six-heading
        # certificate stance_quality_heading/v1 (D-D32), while each recovery,
        # which extends the stance's [env], keeps its frozen recovery_quality/v1.
        certified = model_id in (TYRANNOSAURUS_REX, COMPSOGNATHUS_LONGIPES)
        stance_kind = "stance_quality_heading/v1" if certified else "stance_quality/v1"
        assert load_stage_config(model_id, "stance")["curriculum_kwargs"]["gate_kind"] == stance_kind

    def test_gate_declares_the_frozen_recovery_kind(self):
        import tomllib

        from environments.shared.curriculum.gate_schema import validate_gate_config

        recovery = tomllib.load(open("configs/tyrannosaurus_rex/anatomical/recovery.toml", "rb"))
        curriculum = recovery["curriculum"]
        # Frozen 2026-08-28 (plan P5): the none/v1 placeholder is gone and the
        # declared kind validates whether or not the run advances. Advancement
        # is still not implied by the config — the verdict comes only from the
        # stage directory's frozen gate_resolution.json, which
        # test_recovery_gate_config.py pins along with the thresholds.
        assert curriculum["gate_kind"] == "recovery_quality/v1"
        assert validate_gate_config("recovery", curriculum) == "recovery_quality/v1"
        assert validate_gate_config("recovery", curriculum, advancement_enabled=False) == "recovery_quality/v1"


class TestStageLabels:
    def test_legacy_integers_keep_their_historical_form(self):
        from environments.shared.stage_manifest import stage_label

        assert stage_label(1) == "stage1"
        assert stage_label(3) == "stage3"

    def test_semantic_ids_are_their_own_label(self):
        from environments.shared.stage_manifest import stage_label

        assert stage_label("recovery") == "recovery"

    def test_open_ids_are_their_own_label(self):
        from environments.shared.stage_manifest import stage_label

        # A species-free helper cannot know the manifest: shape is the contract.
        assert stage_label("follow_direction") == "follow_direction"
        assert stage_label("sprint") == "sprint"

    def test_legacy_label_shape_cannot_be_minted_as_an_id_label(self):
        from environments.shared.stage_manifest import stage_label

        with pytest.raises(StageManifestError, match="stage{N} label shape is reserved"):
            stage_label("stage4")
        assert stage_label(4) == "stage4"

    def test_malformed_references_are_fatal(self):
        from environments.shared.stage_manifest import stage_label

        for bad in ("Sprint", "stage4", "", "stance-2"):
            with pytest.raises(StageManifestError, match="invalid stage reference"):
                stage_label(bad)
        with pytest.raises(StageManifestError, match="invalid stage reference"):
            stage_label(True)


class TestLoadAllStages:
    def test_tyrannosaurus_rex_gains_recovery_without_moving_the_integers(self):
        from environments.shared.config import load_all_stages

        configs = load_all_stages(ModelId("tyrannosaurus_rex", "anatomical"))
        assert list(configs) == [1, "recovery", 2, 3]  # manifest order
        assert configs[2]["name"] == "locomotion"
        assert configs["recovery"]["env_kwargs"]["perturbation_capture_velocity_multiple"] == 1.5

    def test_the_declared_legacy_trio_is_untouched(self):
        from environments.shared.config import load_all_stages

        assert list(load_all_stages(ModelId("velociraptor_mongoliensis", "anatomical"))) == [1, 2, 3]

    def test_the_legacy_curriculum_ignores_semantic_only_stages(self):
        """thresholds_from_configs must not validate recovery's gate under
        advancement — that would refuse the whole legacy 1→2→3 curriculum the
        moment a species gains a manifest, since the in-training curriculum
        cannot evaluate recovery_quality/v1 and correctly refuses it."""
        from environments.shared.config import load_all_stages
        from environments.shared.curriculum.manager import thresholds_from_configs

        thresholds = thresholds_from_configs(load_all_stages(ModelId("tyrannosaurus_rex", "anatomical")))
        assert sorted(thresholds) == [1, 2, 3]


class TestCliStageParsing:
    def test_digits_are_legacy_numbers_and_words_are_ids(self):
        from environments.shared.cli import _parse_stage_ref

        assert _parse_stage_ref("2") == 2
        assert _parse_stage_ref("recovery") == "recovery"


class TestStageDirnames:
    """Run-directory naming (2026-08-20): 01_stance style, readers accept all."""

    def test_dirnames_sort_in_curriculum_order_and_carry_the_id(self):
        from environments.shared.stage_manifest import stage_dirname

        names = [
            stage_dirname(ModelId("tyrannosaurus_rex", "anatomical"), ref)
            for ref in ("stance", "recovery", "locomotion", "behavior")
        ]
        assert names == ["01_stance", "02_recovery", "03_locomotion", "04_behavior"]
        assert names == sorted(names)

    def test_integer_refs_keep_their_legacy_meaning(self):
        from environments.shared.stage_manifest import stage_dirname

        # stage 2 is locomotion FOREVER — at position 3 in the manifest.
        assert stage_dirname(ModelId("tyrannosaurus_rex", "anatomical"), 2) == "03_locomotion"
        assert stage_dirname(ModelId("velociraptor_mongoliensis", "anatomical"), 2) == "02_locomotion"

    def test_candidates_cover_every_generation_newest_first(self):
        from environments.shared.stage_manifest import stage_dir_candidates

        assert stage_dir_candidates(ModelId("tyrannosaurus_rex", "anatomical"), 1) == ("01_stance", "stage1", "stance")
        assert stage_dir_candidates(ModelId("tyrannosaurus_rex", "anatomical"), "recovery") == (
            "02_recovery",
            "recovery",
        )

    def test_open_id_dirname_and_candidates(self, tmp_path, monkeypatch):
        from environments.shared import stage_manifest
        from environments.shared.stage_manifest import stage_dir_candidates, stage_dirname

        monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", _write_manifest(tmp_path, PROBE, OPEN_ID_BODY))
        assert stage_dirname(PROBE, "follow_direction") == "03_follow_direction"
        assert stage_dir_candidates(PROBE, "follow_direction") == ("03_follow_direction", "follow_direction")
        assert stage_dirname(PROBE, 2) == "02_locomotion"

    def test_find_stage_dir_prefers_whatever_exists(self, tmp_path):
        from environments.shared.stage_manifest import find_stage_dir

        legacy = tmp_path / "legacy_run"
        (legacy / "stage1").mkdir(parents=True)
        assert find_stage_dir(legacy, 1).name == "stage1"

        modern = tmp_path / "modern_run"
        (modern / "01_stance").mkdir(parents=True)
        (modern / "02_recovery").mkdir()
        assert find_stage_dir(modern, 1).name == "01_stance"
        assert find_stage_dir(modern, "recovery").name == "02_recovery"

        # Missing stays legible: the historical name comes back for errors.
        assert find_stage_dir(modern, 3).name == "stage3"

    def test_find_stage_dir_is_exact_for_prefix_sharing_ids(self, tmp_path):
        from environments.shared.stage_manifest import find_stage_dir

        run = tmp_path / "run"
        (run / "06_follow_direction_speed").mkdir(parents=True)
        # Only the longer id exists: the shorter one must not match it.
        assert find_stage_dir(run, "follow_direction").name == "follow_direction"
        (run / "05_follow_direction").mkdir()
        assert find_stage_dir(run, "follow_direction").name == "05_follow_direction"
        assert find_stage_dir(run, "follow_direction_speed").name == "06_follow_direction_speed"


class TestStageRefFromDirname:
    """The one species-free reader of run-directory names (decision D-A12)."""

    def test_every_generation(self, tmp_path, monkeypatch):
        from environments.shared import stage_manifest

        assert stage_ref_from_dirname("stage1") == 1
        assert stage_ref_from_dirname("stage3") == 3
        assert stage_ref_from_dirname("03_locomotion") == 2
        assert stage_ref_from_dirname("01_stance") == 1
        assert stage_ref_from_dirname("02_recovery") == "recovery"
        assert stage_ref_from_dirname("recovery") == "recovery"
        assert stage_ref_from_dirname("stance") == 1
        for name in (
            "stage4",
            "stage01",
            "stage1b",
            "models",
            "replays",
            "ancestors",
            "figures",
            "",
            "3_stance",
            "03_stage2",
        ):
            assert stage_ref_from_dirname(name) is None, name
        # Species-free, an NN_<word> directory cannot claim to be a stage.
        assert stage_ref_from_dirname("05_follow_direction") is None
        assert stage_ref_from_dirname("01_experiments") is None
        assert stage_ref_from_dirname("01_experiments", model_id=ModelId("tyrannosaurus_rex", "anatomical")) is None
        # With a species that declares the id, the NN_id form resolves to it.
        monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", _write_manifest(tmp_path, PROBE, OPEN_ID_BODY))
        assert stage_ref_from_dirname("05_follow_direction", model_id=PROBE) == "follow_direction"
        assert stage_ref_from_dirname("02_locomotion", model_id=PROBE) == 2
        # A bare open id never reads as a stage, declared or not.
        assert stage_ref_from_dirname("follow_direction") is None
        assert stage_ref_from_dirname("follow_direction", model_id=PROBE) is None
        assert stage_ref_from_dirname("stage1", model_id=PROBE) == 1

    def test_species_check_requires_the_id_to_be_declared(self):
        assert stage_ref_from_dirname("02_recovery", model_id=ModelId("tyrannosaurus_rex", "anatomical")) == "recovery"
        assert stage_ref_from_dirname("recovery", model_id=ModelId("tyrannosaurus_rex", "anatomical")) == "recovery"
        # velociraptor declares no recovery stage: reserved is not enough.
        assert (
            stage_ref_from_dirname("02_recovery", model_id=ModelId("velociraptor_mongoliensis", "anatomical")) is None
        )
        assert stage_ref_from_dirname("recovery", model_id=ModelId("velociraptor_mongoliensis", "anatomical")) is None
        assert stage_ref_from_dirname("03_locomotion", model_id=ModelId("velociraptor_mongoliensis", "anatomical")) == 2
        # A manifest that cannot load is not swallowed.
        with pytest.raises(StageManifestError, match="config directory not found"):
            stage_ref_from_dirname("01_stance", model_id=ModelId("pterodactylus_antiquus", "anatomical"))


class TestSerializedStageKeys:
    """resolve_stage_key and the canonical reference/key spellings.

    Added with the bundle/catalog migration (2026-08-23): JSON object keys
    and CSV cells force every stage reference through a string, and this is
    the one place that decides what those strings mean.
    """

    def test_decimal_strings_are_legacy_numbers_never_positions(self):
        from environments.shared.stage_manifest import resolve_stage_key

        # "2" means locomotion (legacy) even though recovery holds
        # manifest position 2 — the no-silent-renumbering guarantee.
        assert resolve_stage_key(ModelId("tyrannosaurus_rex", "anatomical"), "2").id == "locomotion"
        assert resolve_stage_key(ModelId("tyrannosaurus_rex", "anatomical"), 2).id == "locomotion"
        assert resolve_stage_key(ModelId("tyrannosaurus_rex", "anatomical"), "recovery").id == "recovery"

    def test_unknown_and_malformed_keys_fail_closed(self):
        from environments.shared.stage_manifest import StageManifestError, resolve_stage_key

        with pytest.raises(StageManifestError):
            resolve_stage_key(ModelId("tyrannosaurus_rex", "anatomical"), "warp")
        with pytest.raises(StageManifestError):
            resolve_stage_key(ModelId("tyrannosaurus_rex", "anatomical"), "4")
        with pytest.raises(StageManifestError):
            resolve_stage_key(ModelId("velociraptor_mongoliensis", "anatomical"), "recovery")
        with pytest.raises(StageManifestError):
            resolve_stage_key(ModelId("tyrannosaurus_rex", "anatomical"), True)

    def test_canonical_reference_and_key_spellings(self, tmp_path):
        from environments.shared.stage_manifest import load_stage_manifest

        manifest = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        assert [entry.reference for entry in manifest.stages] == [1, "recovery", 2, 3]
        assert [entry.key for entry in manifest.stages] == ["1", "recovery", "2", "3"]
        # An open id serializes as itself, exactly like recovery.
        pilot = load_stage_manifest(PROBE, configs_dir=_write_manifest(tmp_path, PROBE, OPEN_ID_BODY))
        assert [entry.reference for entry in pilot.stages] == [1, 2, "follow_direction", "follow_direction_speed"]
        assert [entry.key for entry in pilot.stages] == ["1", "2", "follow_direction", "follow_direction_speed"]

    def test_advancing_stages_are_the_numbered_trio(self):
        from environments.shared.stage_manifest import load_stage_manifest

        assert [
            entry.id for entry in load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical")).advancing_stages
        ] == [
            "stance",
            "locomotion",
            "behavior",
        ]
        assert [
            entry.id
            for entry in load_stage_manifest(ModelId("velociraptor_mongoliensis", "anatomical")).advancing_stages
        ] == [
            "stance",
            "locomotion",
            "behavior",
        ]
        # Two vocabularies: advancing (integer-keyed manager) is not
        # deliverables (publication); they coincide only for the trio.
        tyrannosaurus_rex = load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical"))
        assert [entry.id for entry in tyrannosaurus_rex.advancing_stages] == ["stance", "locomotion", "behavior"]
        assert [entry.id for entry in tyrannosaurus_rex.deliverables] == [
            "stance",
            "recovery",
            "locomotion",
            "behavior",
        ]
        velociraptor_mongoliensis = load_stage_manifest(ModelId("velociraptor_mongoliensis", "anatomical"))
        assert [entry.id for entry in velociraptor_mongoliensis.advancing_stages] == [
            entry.id for entry in velociraptor_mongoliensis.deliverables
        ]
