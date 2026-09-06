"""Tests for environments.shared.stage_manifest.

The manifest's first promise is no silent renumbering: an integer stage
reference means the legacy number forever, a new stage is reachable only
by its semantic ID, and a manifest that tries to rewrite history is
rejected at load.

Its second promise (manifest v2, BEHAVIOR_RECIPES_PLAN §8.1-8.2): a v1 or
synthesized manifest reads bit-identically under the v2 reader — same
entries, same advancing trio, edges derived as "the previous advancing
entry", one deliverable — and edges are DECLARED, never inferred from
position, once a file says v2.
"""

from __future__ import annotations

import re
import shutil

import pytest

from environments.shared.stage_manifest import (
    _CONFIGS_DIR,
    LEGACY_STAGE_IDS,
    RESERVED_STAGE_IDS,
    STAGE_ID_PATTERN,
    STAGE_MANIFEST_SCHEMA,
    STAGE_MANIFEST_SCHEMA_V1,
    STAGE_MANIFEST_SCHEMA_V2,
    STAGE_MANIFEST_SCHEMAS,
    StageEntry,
    StageManifestError,
    load_stage_manifest,
    stage_ref_from_dirname,
)

COMMITTED_SPECIES = sorted(path.name for path in _CONFIGS_DIR.iterdir() if path.is_dir())
INTEGER_SPECIES = ["velociraptor", "brachiosaurus", "dibothrosuchus"]

V1 = f'schema = "{STAGE_MANIFEST_SCHEMA_V1}"\n'
V2 = f'schema = "{STAGE_MANIFEST_SCHEMA_V2}"\n'

#: configs/trex/stages.toml exactly as committed before Phase A (2026-08-20
#: through 2026-09-05): the frozen v1 body the bit-identity pins read.
TREX_V1_BODY = """# T-Rex stage manifest (mesozoic.stage-manifest/v1; STAGE1_SPLIT_PLAN §4,
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
    V2 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\nrecipe = "stand"\ndeliverable = true\n'
    '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nlegacy_number = 2\nwarm_start_from = "stance"\n'
    'recipe = "walk"\ndeliverable = true\n'
    '[[stages]]\nid = "follow_direction"\nconfig = "follow_direction.toml"\nwarm_start_from = "locomotion"\n'
    'recipe = "follow"\ndeliverable = true\n'
    '[[stages]]\nid = "follow_direction_speed"\nconfig = "follow_direction_speed.toml"\n'
    'warm_start_from = "follow_direction"\nrecipe = "follow"\ndeliverable = true\n'
)

_STUB_CONFIGS = (
    "stage1_balance.toml",
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


def _write_manifest(tmp_path, species, body, *, synthesized=False):
    """A configs root holding one species.

    With *synthesized* the species gets stage1_a/stage2_b/stage3_c.toml and
    NO stages.toml — the manifest-less layout the three integer species had
    before Phase A; otherwise *body* is written as its stages.toml.
    """
    species_dir = tmp_path / species
    species_dir.mkdir(parents=True, exist_ok=True)
    if synthesized:
        for name in ("stage1_a.toml", "stage2_b.toml", "stage3_c.toml"):
            (species_dir / name).write_text("[stage]\nname = 'x'\n")
        return tmp_path
    for name in _STUB_CONFIGS:
        (species_dir / name).write_text("[stage]\nname = 'x'\n")
    (species_dir / "stages.toml").write_text(body)
    return tmp_path


class TestTrexManifest:
    def test_four_stages_in_curriculum_order(self):
        manifest = load_stage_manifest("trex")
        assert not manifest.synthesized
        assert manifest.schema == STAGE_MANIFEST_SCHEMA_V2
        assert [entry.id for entry in manifest.stages] == ["stance", "recovery", "locomotion", "behavior"]
        assert [entry.position for entry in manifest.stages] == [1, 2, 3, 4]

    def test_integer_references_mean_legacy_numbers_not_positions(self):
        """THE no-silent-renumbering guarantee: 2 is locomotion, forever."""
        manifest = load_stage_manifest("trex")
        assert manifest.resolve(1).id == "stance"
        assert manifest.resolve(2).id == "locomotion"
        assert manifest.resolve(2).position == 3
        assert manifest.resolve(3).id == "behavior"
        assert manifest.resolve(3).position == 4

    def test_recovery_is_reachable_only_by_id(self):
        manifest = load_stage_manifest("trex")
        recovery = manifest.resolve("recovery")
        assert recovery.position == 2
        assert recovery.legacy_number is None
        assert recovery.config_file == "recovery.toml"
        with pytest.raises(StageManifestError, match="named by ID"):
            manifest.by_legacy_number(4)

    def test_unknown_references_are_fatal(self):
        manifest = load_stage_manifest("trex")
        with pytest.raises(StageManifestError, match="no stage 'sprint'"):
            manifest.resolve("sprint")
        with pytest.raises(StageManifestError, match="positions 1..4"):
            manifest.by_position(5)
        with pytest.raises(StageManifestError, match="invalid stage reference"):
            manifest.resolve(True)


class TestDeclaredLegacyTrio:
    """The three integer species declare, in v2, exactly the trio they used to synthesize."""

    @pytest.mark.parametrize("species", INTEGER_SPECIES)
    def test_the_three_species_declare_the_legacy_three_stages(self, species):
        manifest = load_stage_manifest(species)
        assert not manifest.synthesized
        assert manifest.schema == STAGE_MANIFEST_SCHEMA_V2
        assert [entry.id for entry in manifest.stages] == ["stance", "locomotion", "behavior"]
        assert [entry.legacy_number for entry in manifest.stages] == [1, 2, 3]
        # For a species without new stages, positions and legacy agree.
        assert all(entry.position == entry.legacy_number for entry in manifest.stages)


class TestSchemaVersions:
    def test_v1_and_v2_both_load_and_v3_is_fatal(self, tmp_path):
        assert STAGE_MANIFEST_SCHEMAS == (STAGE_MANIFEST_SCHEMA_V1, STAGE_MANIFEST_SCHEMA_V2)
        assert STAGE_MANIFEST_SCHEMA == STAGE_MANIFEST_SCHEMA_V2
        entry = '[[stages]]\nid = "stance"\nconfig = "stage1_balance.toml"\nlegacy_number = 1\n'
        root = _write_manifest(tmp_path / "v1", "x", V1 + entry)
        assert load_stage_manifest("x", configs_dir=root).schema == STAGE_MANIFEST_SCHEMA_V1
        root = _write_manifest(tmp_path / "v2", "x", V2 + entry + "deliverable = true\n")
        assert load_stage_manifest("x", configs_dir=root).schema == STAGE_MANIFEST_SCHEMA_V2
        root = _write_manifest(tmp_path / "v3", "x", 'schema = "mesozoic.stage-manifest/v3"\n' + entry)
        with pytest.raises(StageManifestError, match=re.escape(f"accepts one of {STAGE_MANIFEST_SCHEMAS}")):
            load_stage_manifest("x", configs_dir=root)

    @pytest.mark.parametrize("key", ['warm_start_from = "stance"', "deliverable = true", 'recipe = "stand"'])
    def test_v1_body_with_v2_keys_is_fatal(self, tmp_path, key):
        """A v1 file cannot smuggle an edge past a reader that would derive one."""
        body = V1 + '[[stages]]\nid = "stance"\nconfig = "stage1_balance.toml"\nlegacy_number = 1\n' + key + "\n"
        root = _write_manifest(tmp_path, "x", body)
        with pytest.raises(StageManifestError, match="declare schema 'mesozoic.stage-manifest/v2'"):
            load_stage_manifest("x", configs_dir=root)


class TestStageIdVocabulary:
    def test_reserved_ids_are_exactly_the_four_and_match_the_pattern(self):
        assert RESERVED_STAGE_IDS == ("stance", "recovery", "locomotion", "behavior")
        assert all(STAGE_ID_PATTERN.match(stage_id) for stage_id in RESERVED_STAGE_IDS)
        assert set(LEGACY_STAGE_IDS.values()) < set(RESERVED_STAGE_IDS)
        assert STAGE_ID_PATTERN.pattern == r"^[a-z][a-z0-9_]*$"

    @pytest.mark.parametrize("stage_id", ["stage4", "stage10"])
    def test_legacy_label_shape_is_refused_as_an_id(self, tmp_path, stage_id):
        body = V2 + f'[[stages]]\nid = "{stage_id}"\nconfig = "stage1_balance.toml"\ndeliverable = true\n'
        root = _write_manifest(tmp_path, "x", body)
        with pytest.raises(StageManifestError, match="label shape"):
            load_stage_manifest("x", configs_dir=root)


class TestLegacyReadsAreBitIdentical:
    """Plan §8.1: the v2 reader on a v1 or synthesized manifest changes nothing."""

    def test_v1_trex_body_derives_previous_advancing_edges_and_one_deliverable(self, tmp_path):
        root = _write_manifest(tmp_path, "trex", TREX_V1_BODY)
        legacy = load_stage_manifest("trex", configs_dir=root)
        committed = load_stage_manifest("trex")
        assert legacy.schema == STAGE_MANIFEST_SCHEMA_V1 and not legacy.synthesized
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

    def test_synthesized_species_derives_the_same(self, tmp_path):
        root = _write_manifest(tmp_path, "raptor", "", synthesized=True)
        manifest = load_stage_manifest("raptor", configs_dir=root)
        assert manifest.synthesized and manifest.schema is None
        assert _tuples(manifest) == [
            ("stance", 1, "stage1_a.toml", 1),
            ("locomotion", 2, "stage2_b.toml", 2),
            ("behavior", 3, "stage3_c.toml", 3),
        ]
        assert _edges(manifest) == [("stance", None), ("locomotion", "stance"), ("behavior", "locomotion")]
        assert [entry.id for entry in manifest.deliverables] == ["behavior"]
        assert [entry.id for entry in manifest.advancing_stages] == ["stance", "locomotion", "behavior"]

    def test_committed_v2_files_are_the_only_place_stand_and_walk_become_deliverables(self, tmp_path):
        legacy_trex = load_stage_manifest("trex", configs_dir=_write_manifest(tmp_path / "t", "trex", TREX_V1_BODY))
        synthesized = load_stage_manifest("x", configs_dir=_write_manifest(tmp_path / "s", "x", "", synthesized=True))
        assert [entry.id for entry in legacy_trex.deliverables] == ["behavior"]
        assert [entry.id for entry in synthesized.deliverables] == ["behavior"]
        assert [entry.id for entry in load_stage_manifest("trex").deliverables] == [
            "stance",
            "recovery",
            "locomotion",
            "behavior",
        ]
        for species in INTEGER_SPECIES:
            assert [entry.id for entry in load_stage_manifest(species).deliverables] == [
                "stance",
                "locomotion",
                "behavior",
            ]

    @pytest.mark.parametrize("species", INTEGER_SPECIES)
    def test_three_species_declare_exactly_what_the_synthesizer_derived(self, tmp_path, species):
        shutil.copytree(_CONFIGS_DIR / species, tmp_path / species)
        (tmp_path / species / "stages.toml").unlink()
        synthesized = load_stage_manifest(species, configs_dir=tmp_path)
        committed = load_stage_manifest(species)
        assert synthesized.synthesized and not committed.synthesized
        assert _tuples(synthesized) == _tuples(committed)
        assert _edges(synthesized) == _edges(committed)
        assert [entry.reference for entry in synthesized.stages] == [entry.reference for entry in committed.stages]
        # The only differences: every node is now a deliverable, and labelled.
        assert [entry.deliverable for entry in synthesized.stages] == [False, False, True]
        assert [entry.deliverable for entry in committed.stages] == [True, True, True]
        assert [entry.recipe for entry in synthesized.stages] == [None, None, None]
        assert [entry.recipe for entry in committed.stages] == ["stand", "walk", "hunt"]

    def test_derived_edge_is_non_none_exactly_when_position_gt_1(self, tmp_path):
        """The entry-shaping bit-identity proof (plan §8.3): has_parent == position > 1."""
        legacy_trex = load_stage_manifest("trex", configs_dir=_write_manifest(tmp_path / "t", "trex", TREX_V1_BODY))
        synthesized = load_stage_manifest("x", configs_dir=_write_manifest(tmp_path / "s", "x", "", synthesized=True))
        for manifest in (legacy_trex, synthesized):
            assert [entry.has_parent for entry in manifest.stages] == [entry.position > 1 for entry in manifest.stages]


class TestFailClosedValidation:
    def test_wrong_schema_is_fatal(self, tmp_path):
        root = _write_manifest(tmp_path, "trex", 'schema = "mesozoic.stage-manifest/v3"\n')
        with pytest.raises(StageManifestError, match=re.escape(f"accepts one of {STAGE_MANIFEST_SCHEMAS}")):
            load_stage_manifest("trex", configs_dir=root)

    @pytest.mark.parametrize("stage_id", ["Sprint", "4x", "", "stage4", "stance-2"])
    def test_malformed_stage_ids_are_fatal(self, tmp_path, stage_id):
        body = V2 + f'[[stages]]\nid = "{stage_id}"\nconfig = "stage1_balance.toml"\ndeliverable = true\n'
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="must match|label shape"):
            load_stage_manifest("trex", configs_dir=root)

    def test_missing_config_file_is_fatal(self, tmp_path):
        body = V1 + '[[stages]]\nid = "stance"\nconfig = "missing.toml"\n'
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="not found"):
            load_stage_manifest("trex", configs_dir=root)

    def test_rewriting_history_is_fatal(self, tmp_path):
        """legacy_number 2 IS locomotion; a manifest cannot reassign it."""
        body = V1 + '[[stages]]\nid = "behavior"\nconfig = "stage1_balance.toml"\nlegacy_number = 2\n'
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="must not rewrite history"):
            load_stage_manifest("trex", configs_dir=root)

    def test_reordering_legacy_stages_is_fatal(self, tmp_path):
        body = (
            V1 + '[[stages]]\nid = "locomotion"\nconfig = "stage1_balance.toml"\nlegacy_number = 2\n'
            '[[stages]]\nid = "stance"\nconfig = "recovery.toml"\nlegacy_number = 1\n'
        )
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="historical order"):
            load_stage_manifest("trex", configs_dir=root)

    def test_duplicate_ids_are_fatal(self, tmp_path):
        body = (
            V1 + '[[stages]]\nid = "stance"\nconfig = "stage1_balance.toml"\nlegacy_number = 1\n'
            '[[stages]]\nid = "stance"\nconfig = "recovery.toml"\n'
        )
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="duplicate stage ids"):
            load_stage_manifest("trex", configs_dir=root)

    def test_unknown_entry_keys_are_fatal(self, tmp_path):
        body = V1 + '[[stages]]\nid = "stance"\nconfig = "stage1_balance.toml"\nnumber = 1\n'
        root = _write_manifest(tmp_path, "trex", body)
        with pytest.raises(StageManifestError, match="unknown keys"):
            load_stage_manifest("trex", configs_dir=root)
        # The three v2 keys are accepted under v2 and fatal under v1.
        v2_entry = (
            '[[stages]]\nid = "stance"\nconfig = "stage1_balance.toml"\nlegacy_number = 1\ndeliverable = true\n'
            'recipe = "stand"\n'
            '[[stages]]\nid = "recovery"\nconfig = "recovery.toml"\nwarm_start_from = "stance"\n'
        )
        root = _write_manifest(tmp_path / "v2", "trex", V2 + v2_entry)
        assert [entry.id for entry in load_stage_manifest("trex", configs_dir=root).stages] == ["stance", "recovery"]
        root = _write_manifest(tmp_path / "v1", "trex", V1 + v2_entry)
        with pytest.raises(StageManifestError, match="unknown keys.*declare schema"):
            load_stage_manifest("trex", configs_dir=root)


class TestV2Validators:
    """Plan §8.2: edges name earlier entries; legacy rules unchanged."""

    def _load(self, tmp_path, body):
        return load_stage_manifest("x", configs_dir=_write_manifest(tmp_path, "x", V2 + body))

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

    def test_trex_parents_are_the_declared_edges(self):
        manifest = load_stage_manifest("trex")
        assert manifest.parent_of("stance") is None
        assert manifest.parent_of("recovery").id == "stance"
        assert manifest.parent_of("locomotion").id == "stance"
        assert manifest.parent_of(2).id == "stance"
        assert manifest.parent_of("behavior").id == "locomotion"
        assert [entry.has_parent for entry in manifest.stages] == [False, True, True, True]

    def test_ancestors_and_chain_are_root_first(self):
        manifest = load_stage_manifest("trex")
        assert [entry.id for entry in manifest.chain_for("behavior")] == ["stance", "locomotion", "behavior"]
        assert [entry.id for entry in manifest.chain_for(3)] == ["stance", "locomotion", "behavior"]
        assert [entry.id for entry in manifest.ancestors("behavior")] == ["stance", "locomotion"]
        assert [entry.id for entry in manifest.ancestors("recovery")] == ["stance"]
        assert manifest.ancestors("stance") == ()
        assert manifest.chain_for("stance") == (manifest.by_id("stance"),)

    def test_deliverables_in_manifest_order_for_all_four_species(self):
        assert [entry.id for entry in load_stage_manifest("trex").deliverables] == [
            "stance",
            "recovery",
            "locomotion",
            "behavior",
        ]
        for species in INTEGER_SPECIES:
            assert [entry.id for entry in load_stage_manifest(species).deliverables] == [
                "stance",
                "locomotion",
                "behavior",
            ]

    def test_recipe_labels_resolve_to_the_deepest_deliverable(self, tmp_path):
        trex = load_stage_manifest("trex")
        assert trex.recipe_labels == ("stand", "walk", "hunt")
        assert trex.resolve_behavior("stand").id == "recovery"
        assert trex.resolve_behavior("walk").id == "locomotion"
        assert trex.resolve_behavior("hunt").id == "behavior"
        for species in INTEGER_SPECIES:
            manifest = load_stage_manifest(species)
            assert manifest.resolve_behavior("stand").id == "stance"
            assert manifest.resolve_behavior("walk").id == "locomotion"
            assert manifest.resolve_behavior("hunt").id == "behavior"
        pilot = load_stage_manifest("pilot", configs_dir=_write_manifest(tmp_path, "pilot", OPEN_ID_BODY))
        assert pilot.recipe_labels == ("stand", "walk", "follow")
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
        pilot = load_stage_manifest("pilot", configs_dir=_write_manifest(tmp_path, "pilot", OPEN_ID_BODY))
        assert pilot.resolve_behavior("follow_direction").id == "follow_direction"
        body = (
            V2 + '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
            '[[stages]]\nid = "recovery"\nconfig = "recovery.toml"\nwarm_start_from = "stance"\ndeliverable = true\n'
        )
        manifest = load_stage_manifest("x", configs_dir=_write_manifest(tmp_path / "x", "x", body))
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
        manifest = load_stage_manifest("x", configs_dir=_write_manifest(tmp_path, "x", body))
        assert manifest.parent_of("behavior").id == manifest.parent_of("follow_direction").id == "locomotion"
        assert [entry.id for entry in manifest.chain_for("follow_direction")] == [
            "stance",
            "locomotion",
            "follow_direction",
        ]
        assert [entry.id for entry in manifest.chain_for("behavior")] == ["stance", "locomotion", "behavior"]
        assert manifest.by_id("follow_direction").position == 4
        assert manifest.by_id("follow_direction").reference == "follow_direction"


class TestCommittedManifestsAreV2:
    @pytest.mark.parametrize("species", COMMITTED_SPECIES)
    def test_every_species_manifest_is_v2_with_all_nodes_deliverable(self, species):
        manifest = load_stage_manifest(species)
        assert manifest.schema == STAGE_MANIFEST_SCHEMA_V2 and not manifest.synthesized
        assert all(entry.deliverable for entry in manifest.stages)
        assert manifest.recipe_labels == ("stand", "walk", "hunt")
        for entry in manifest.stages:
            assert (_CONFIGS_DIR / species / entry.config_file).is_file()
        assert [entry.id for entry in manifest.advancing_stages] == ["stance", "locomotion", "behavior"]

    def test_trex_declares_edges_deliverables_and_labels(self):
        manifest = load_stage_manifest("trex")
        assert [(entry.id, entry.warm_start_from, entry.deliverable, entry.recipe) for entry in manifest.stages] == [
            ("stance", None, True, "stand"),
            # locomotion's parent is stance until plan A2 flips it to recovery.
            ("recovery", "stance", True, "stand"),
            ("locomotion", "stance", True, "walk"),
            ("behavior", "locomotion", True, "hunt"),
        ]

    @pytest.mark.parametrize("species", INTEGER_SPECIES)
    def test_the_integer_species_declare_no_recovery(self, species):
        manifest = load_stage_manifest(species)
        assert [(entry.id, entry.warm_start_from, entry.recipe) for entry in manifest.stages] == [
            ("stance", None, "stand"),
            ("locomotion", "stance", "walk"),
            ("behavior", "locomotion", "hunt"),
        ]
        with pytest.raises(StageManifestError, match="no stage 'recovery'"):
            manifest.resolve("recovery")

    def test_v2_manifest_does_not_move_any_dirname(self):
        from environments.shared.stage_manifest import stage_dirname

        pre_phase_a = {
            "trex": ["01_stance", "02_recovery", "03_locomotion", "04_behavior"],
            "velociraptor": ["01_stance", "02_locomotion", "03_behavior"],
            "brachiosaurus": ["01_stance", "02_locomotion", "03_behavior"],
            "dibothrosuchus": ["01_stance", "02_locomotion", "03_behavior"],
        }
        assert sorted(pre_phase_a) == COMMITTED_SPECIES
        for species, names in pre_phase_a.items():
            manifest = load_stage_manifest(species)
            assert [stage_dirname(species, entry.reference) for entry in manifest.stages] == names
            assert [stage_dirname(species, entry.id) for entry in manifest.stages] == names


class TestRecoveryStageConfig:
    """The recovery stage config is stage 1's task plus exactly one delta."""

    def test_loads_by_semantic_id_and_only_by_it(self):
        from environments.shared.config import load_stage_config

        config = load_stage_config("trex", "recovery")
        assert config["name"] == "recovery"
        env = config["env_kwargs"]
        assert env["perturbation_capture_velocity_multiple"] == 1.5
        # Integer loading is untouched: 2 still means locomotion.
        assert load_stage_config("trex", 2)["name"] == load_stage_config("trex", "locomotion")["name"]

    def test_env_mirrors_stance_plus_exactly_the_perturbation_block(self):
        import tomllib

        from environments.shared.stage_manifest import load_stage_manifest

        stance_file = load_stage_manifest("trex").resolve("stance").config_file
        stance = tomllib.load(open(f"configs/trex/{stance_file}", "rb"))
        recovery = tomllib.load(open("configs/trex/recovery.toml", "rb"))
        perturbation_keys = {key for key in recovery["env"] if key.startswith("perturbation_")}
        assert perturbation_keys == {
            "perturbation_capture_velocity_multiple",
            "perturbation_interval",
            "perturbation_jitter",
            "perturbation_duration",
            "perturbation_direction",
        }
        mirrored = {key: value for key, value in recovery["env"].items() if key not in perturbation_keys}
        # Freshness pin: a stage-1 shaping change that forgets this file
        # fails here, with the fix being to re-mirror (and re-derive the
        # recovery rails once the W4 gate exists).
        assert mirrored == stance["env"]
        # [ppo] mirrors stance with exactly one measured delta:
        # ent_coef_decay_timesteps is anchored to recovery's own 3M budget
        # (2M, ~2/3 — the same ratio as stance's 7M-of-11M) rather than
        # mirrored, because the 20260821_142144 pilot trained its whole
        # budget at ent_coef >= 0.0014 when the mirrored 7M horizon never
        # completed (2026-08 review §3.4).
        assert (
            recovery["ppo"] | {"ent_coef_decay_timesteps": stance["ppo"]["ent_coef_decay_timesteps"]} == stance["ppo"]
        )
        assert recovery["ppo"]["ent_coef_decay_timesteps"] == 2_000_000

    def test_gate_declares_the_frozen_recovery_kind(self):
        import tomllib

        from environments.shared.curriculum.gate_schema import validate_gate_config

        recovery = tomllib.load(open("configs/trex/recovery.toml", "rb"))
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
    def test_trex_gains_recovery_without_moving_the_integers(self):
        from environments.shared.config import load_all_stages

        configs = load_all_stages("trex")
        assert list(configs) == [1, "recovery", 2, 3]  # manifest order
        assert configs[2]["name"] == "locomotion"
        assert configs["recovery"]["env_kwargs"]["perturbation_capture_velocity_multiple"] == 1.5

    def test_the_declared_legacy_trio_is_untouched(self):
        from environments.shared.config import load_all_stages

        assert list(load_all_stages("velociraptor")) == [1, 2, 3]

    def test_the_legacy_curriculum_ignores_semantic_only_stages(self):
        """thresholds_from_configs must not validate recovery's gate under
        advancement — that would refuse the whole legacy 1→2→3 curriculum the
        moment a species gains a manifest, since the in-training curriculum
        cannot evaluate recovery_quality/v1 and correctly refuses it."""
        from environments.shared.config import load_all_stages
        from environments.shared.curriculum.manager import thresholds_from_configs

        thresholds = thresholds_from_configs(load_all_stages("trex"))
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

        names = [stage_dirname("trex", ref) for ref in ("stance", "recovery", "locomotion", "behavior")]
        assert names == ["01_stance", "02_recovery", "03_locomotion", "04_behavior"]
        assert names == sorted(names)

    def test_integer_refs_keep_their_legacy_meaning(self):
        from environments.shared.stage_manifest import stage_dirname

        # stage 2 is locomotion FOREVER — at position 3 in the manifest.
        assert stage_dirname("trex", 2) == "03_locomotion"
        assert stage_dirname("velociraptor", 2) == "02_locomotion"

    def test_candidates_cover_every_generation_newest_first(self):
        from environments.shared.stage_manifest import stage_dir_candidates

        assert stage_dir_candidates("trex", 1) == ("01_stance", "stage1", "stance")
        assert stage_dir_candidates("trex", "recovery") == ("02_recovery", "recovery")

    def test_open_id_dirname_and_candidates(self, tmp_path, monkeypatch):
        from environments.shared import stage_manifest
        from environments.shared.stage_manifest import stage_dir_candidates, stage_dirname

        monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", _write_manifest(tmp_path, "pilot", OPEN_ID_BODY))
        assert stage_dirname("pilot", "follow_direction") == "03_follow_direction"
        assert stage_dir_candidates("pilot", "follow_direction") == ("03_follow_direction", "follow_direction")
        assert stage_dirname("pilot", 2) == "02_locomotion"

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
        assert stage_ref_from_dirname("01_experiments", species="trex") is None
        # With a species that declares the id, the NN_id form resolves to it.
        monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", _write_manifest(tmp_path, "pilot", OPEN_ID_BODY))
        assert stage_ref_from_dirname("05_follow_direction", species="pilot") == "follow_direction"
        assert stage_ref_from_dirname("02_locomotion", species="pilot") == 2
        # A bare open id never reads as a stage, declared or not.
        assert stage_ref_from_dirname("follow_direction") is None
        assert stage_ref_from_dirname("follow_direction", species="pilot") is None
        assert stage_ref_from_dirname("stage1", species="pilot") == 1

    def test_species_check_requires_the_id_to_be_declared(self):
        assert stage_ref_from_dirname("02_recovery", species="trex") == "recovery"
        assert stage_ref_from_dirname("recovery", species="trex") == "recovery"
        # velociraptor declares no recovery stage: reserved is not enough.
        assert stage_ref_from_dirname("02_recovery", species="velociraptor") is None
        assert stage_ref_from_dirname("recovery", species="velociraptor") is None
        assert stage_ref_from_dirname("03_locomotion", species="velociraptor") == 2
        # A manifest that cannot load is not swallowed.
        with pytest.raises(StageManifestError, match="config directory not found"):
            stage_ref_from_dirname("01_stance", species="pterodactyl")


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
        assert resolve_stage_key("trex", "2").id == "locomotion"
        assert resolve_stage_key("trex", 2).id == "locomotion"
        assert resolve_stage_key("trex", "recovery").id == "recovery"

    def test_unknown_and_malformed_keys_fail_closed(self):
        from environments.shared.stage_manifest import StageManifestError, resolve_stage_key

        with pytest.raises(StageManifestError):
            resolve_stage_key("trex", "warp")
        with pytest.raises(StageManifestError):
            resolve_stage_key("trex", "4")
        with pytest.raises(StageManifestError):
            resolve_stage_key("velociraptor", "recovery")
        with pytest.raises(StageManifestError):
            resolve_stage_key("trex", True)

    def test_canonical_reference_and_key_spellings(self, tmp_path):
        from environments.shared.stage_manifest import load_stage_manifest

        manifest = load_stage_manifest("trex")
        assert [entry.reference for entry in manifest.stages] == [1, "recovery", 2, 3]
        assert [entry.key for entry in manifest.stages] == ["1", "recovery", "2", "3"]
        # An open id serializes as itself, exactly like recovery.
        pilot = load_stage_manifest("pilot", configs_dir=_write_manifest(tmp_path, "pilot", OPEN_ID_BODY))
        assert [entry.reference for entry in pilot.stages] == [1, 2, "follow_direction", "follow_direction_speed"]
        assert [entry.key for entry in pilot.stages] == ["1", "2", "follow_direction", "follow_direction_speed"]

    def test_advancing_stages_are_the_numbered_trio(self):
        from environments.shared.stage_manifest import load_stage_manifest

        assert [entry.id for entry in load_stage_manifest("trex").advancing_stages] == [
            "stance",
            "locomotion",
            "behavior",
        ]
        assert [entry.id for entry in load_stage_manifest("velociraptor").advancing_stages] == [
            "stance",
            "locomotion",
            "behavior",
        ]
        # Two vocabularies: advancing (integer-keyed manager/sweeps) is not
        # deliverables (publication); they coincide only for the trio.
        trex = load_stage_manifest("trex")
        assert [entry.id for entry in trex.advancing_stages] == ["stance", "locomotion", "behavior"]
        assert [entry.id for entry in trex.deliverables] == ["stance", "recovery", "locomotion", "behavior"]
        velociraptor = load_stage_manifest("velociraptor")
        assert [entry.id for entry in velociraptor.advancing_stages] == [
            entry.id for entry in velociraptor.deliverables
        ]
