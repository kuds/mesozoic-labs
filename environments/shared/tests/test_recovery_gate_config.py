"""The T-Rex recovery stage's FROZEN gate declaration (plan P5, 2026-08-28).

``configs/trex/recovery.toml`` stopped being a non-advancing pilot on
2026-08-28: its ``[curriculum]`` block declares ``recovery_quality/v1`` with
thresholds taken from measurement, not from aspiration.  This module pins
that declaration in three ways, because each is a different failure:

1. **It validates as the kind it declares.**  A config that declares a gate
   kind but omits a required threshold field, or carries a field belonging to
   a different kind, fails OPEN on the SB3 path — which is the whole reason
   :mod:`environments.shared.curriculum.gate_schema` exists.
2. **The frozen numbers are exactly the P5 decisions**, and each is checked
   against the measurement it was derived from rather than against a copy of
   itself: the bounds are recomputed here from the §9 panel counts, so a
   silently edited threshold has to disagree with the measurement, not merely
   with a literal.
3. **The config alone still cannot advance anything.**  Declaring the gate
   moves no verdict into this file.  The verdict comes only from the stage
   directory's frozen ``gate_resolution.json``
   (:mod:`environments.shared.curriculum.gate_resolver`), and its absence
   refuses — exactly as the ``none/v1`` placeholder did, but now for the
   right reason: missing baselines block, they are never skipped.

Measurement source throughout:
``docs/investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md`` §4 (the P3
safe-set calibration) and §9 (the off-distribution and checkpoint panels).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from environments.shared.config import load_stage_config
from environments.shared.curriculum.gate_resolver import (
    GateResolutionError,
    require_gate_resolution,
)
from environments.shared.curriculum.gate_schema import (
    GATE_KINDS,
    GATE_SCHEMA_VERSION,
    GateSchemaError,
    validate_gate_config,
)
from environments.shared.curriculum.recovery_gate import (
    RECOVERY_GATE_KIND,
    RecoveryGateThresholds,
    RecoveryPanel,
    binomial_lcb,
    binomial_ucb,
    evaluate_recovery_gate,
)

RECOVERY_CONFIG = Path("configs/trex/recovery.toml")

#: The §9.1 training-schedule panels, as their counts.  Every bound this
#: module checks is recomputed from these rather than quoted, so the frozen
#: thresholds are tied to the measurement and not to a transcription of it.
PANEL_EPISODES = 40
POLICY_3M_SUCCESSES = 20
POLICY_5M_SUCCESSES = 19
STATUE_SUCCESSES = 0


@pytest.fixture(scope="module")
def curriculum() -> dict[str, Any]:
    """The committed ``[curriculum]`` block, read the way a stage reads it."""
    loaded = load_stage_config("trex", "recovery")
    block: dict[str, Any] = dict(loaded["curriculum_kwargs"])
    # The loader is the path that matters, but the raw TOML is what a reviewer
    # edits; a divergence between them would make every pin below vacuous.
    raw = tomllib.load(RECOVERY_CONFIG.open("rb"))["curriculum"]
    assert block == raw
    return block


def _paired_differences_against_the_statue(policy_successes: int) -> tuple[float, ...]:
    """Per-seed policy-minus-statue differences for a 0/40 statue panel.

    The statue failed every seed in §9.1, so each seed's difference is just
    the policy's own outcome there — which is why the paired margins in §9
    are the policy's success fractions and their LCBs are tighter than the
    binomial ones.
    """
    return tuple([1.0] * policy_successes + [0.0] * (PANEL_EPISODES - policy_successes))


class TestFrozenDeclaration:
    def test_the_config_declares_the_recovery_gate_at_this_schema_version(self, curriculum: dict[str, Any]) -> None:
        assert curriculum["gate_kind"] == RECOVERY_GATE_KIND == "recovery_quality/v1"
        assert curriculum["gate_schema_version"] == GATE_SCHEMA_VERSION

    def test_it_validates_as_that_kind_with_advancement_enabled(self, curriculum: dict[str, Any]) -> None:
        """The pin that would have caught the placeholder never being flipped.

        ``advancement_enabled=True`` is the strict reading: a ``none/v1``
        declaration raises here, and so does a declared kind missing any of
        its required threshold fields.
        """
        assert validate_gate_config("recovery", curriculum) == RECOVERY_GATE_KIND
        # A single-stage pilot of the same config stays legal, and resolves to
        # the same kind rather than degrading to a non-advancing placeholder.
        assert validate_gate_config("recovery", curriculum, advancement_enabled=False) == RECOVERY_GATE_KIND

    def test_no_threshold_belongs_to_a_different_gate_kind(self, curriculum: dict[str, Any]) -> None:
        """A leftover field from another kind implies a gate nobody enforces."""
        every_threshold_key: set[str] = set().union(*GATE_KINDS.values())
        assert (set(curriculum) & every_threshold_key) <= GATE_KINDS[RECOVERY_GATE_KIND]

    @pytest.mark.parametrize(
        "field",
        ["min_recovery_success_lcb", "recovery_t_recover_steps", "recovery_dwell_steps"],
    )
    def test_each_load_bearing_field_is_load_bearing_here(self, curriculum: dict[str, Any], field: str) -> None:
        """Dropping any of the three required fields must be fatal, not quiet.

        Without ``t_recover``/``dwell`` the success event is undefined and
        without the LCB the gate rests on a raw fraction — and an omission
        used to fail OPEN on the SB3 path, where the missing thresholds became
        ``StageThreshold``'s permissive defaults.  Asserted through the schema
        rather than against its private required-key table, so this pins the
        behaviour a stage actually gets.
        """
        stripped = {key: value for key, value in curriculum.items() if key != field}
        with pytest.raises(GateSchemaError, match="missing required"):
            validate_gate_config("recovery", stripped)

    def test_no_reward_rail_is_declared(self, curriculum: dict[str, Any]) -> None:
        """``min_avg_reward`` is allowed by the kind and deliberately absent.

        No rail has been derived for the PUSHED task, and the frozen
        capability spec records none, so declaring one would gate on a number
        nobody measured.  The discarded-return failure mode a rail catches is
        covered by the collapse backstop, which is anchored to the pushed
        statue (974.7, first-runs record §3/§6.5).
        """
        assert "min_avg_reward" not in curriculum
        assert curriculum["collapse_peak_floor_reference"] == 974.7


class TestFrozenThresholds:
    """The P5 decisions, and the measurements each one was taken against."""

    def test_the_thresholds_are_exactly_the_frozen_values(self, curriculum: dict[str, Any]) -> None:
        assert curriculum["min_recovery_success_lcb"] == 0.30
        assert curriculum["recovery_t_recover_steps"] == 100
        assert curriculum["recovery_dwell_steps"] == 50
        assert curriculum["min_paired_success_delta_lcb"] == 0.20
        assert curriculum["min_eval_episodes"] == PANEL_EPISODES
        assert curriculum["required_consecutive"] == 3

    def test_the_success_threshold_is_attainable_and_clears_the_null(self, curriculum: dict[str, Any]) -> None:
        """0.30 sits below what was measured and far above the null ceiling.

        §9.5's whole argument in two comparisons: the attainable LCB95 is
        0.361 (3M, 20/40) and 0.338 (5M, 19/40), while the statue's one-sided
        UCB95 at 0/40 is 0.072.  Both bounds are recomputed here.
        """
        threshold = curriculum["min_recovery_success_lcb"]
        assert binomial_lcb(POLICY_3M_SUCCESSES, PANEL_EPISODES) == pytest.approx(0.3611, abs=5e-5)
        assert binomial_lcb(POLICY_5M_SUCCESSES, PANEL_EPISODES) == pytest.approx(0.3377, abs=5e-5)
        assert binomial_ucb(STATUE_SUCCESSES, PANEL_EPISODES) == pytest.approx(0.0722, abs=5e-5)
        assert threshold < binomial_lcb(POLICY_5M_SUCCESSES, PANEL_EPISODES)
        assert threshold > binomial_ucb(STATUE_SUCCESSES, PANEL_EPISODES)
        # And it is NOT the plan's aspirational 0.725 (the LCB of 34/40),
        # which no checkpoint in the project reaches.
        assert threshold < binomial_lcb(34, PANEL_EPISODES)

    def test_a_short_panel_is_refused_on_the_episode_count(self, curriculum: dict[str, Any]) -> None:
        """40 episodes is the panel every §3/§4.3/§9 figure was measured on.

        Freezing ``min_eval_episodes`` AT the panel size is the only thing
        that notices a certified claim made on a different panel: at 30
        episodes the same 50% success rate still clears 0.30 on its own bound
        (LCB 0.339) and the paired margin still clears 0.20, so both
        statistical criteria would pass a panel nothing in §9 was measured on.
        """
        assert curriculum["min_eval_episodes"] == PANEL_EPISODES
        assert binomial_lcb(15, 30) > curriculum["min_recovery_success_lcb"]
        short = RecoveryPanel(
            episode_successes=tuple([True] * 15 + [False] * 15),
            paired_null_differences=tuple([1.0] * 15 + [0.0] * 15),
        )
        result = evaluate_recovery_gate(short, _thresholds_from(curriculum))
        assert result.passed is False
        assert result.failures == (f"n_episodes 30 < min_eval_episodes {PANEL_EPISODES}",)

    def test_the_config_agrees_with_the_freeze_producer(self, curriculum: dict[str, Any]) -> None:
        """The two places the frozen numbers are written must not drift.

        ``reporting/gates.py`` refuses when a config's declaration disagrees
        with the resolution it is judged against, so a producer frozen at
        different values would write a ``gate_resolution.json`` this stage can
        never pass — blocked by a transcription error rather than by evidence.
        Imported inside the test because the producer is hand-run tooling, not
        something a config test should require at collection time.
        """
        from environments.shared.harnesses import freeze_recovery_gate as producer

        assert curriculum["min_recovery_success_lcb"] == producer.MIN_RECOVERY_SUCCESS_LCB
        assert curriculum["min_paired_success_delta_lcb"] == producer.MIN_PAIRED_SUCCESS_DELTA_LCB
        assert curriculum["recovery_t_recover_steps"] == producer.T_RECOVER_STEPS
        assert curriculum["recovery_dwell_steps"] == producer.DWELL_STEPS
        assert curriculum["min_eval_episodes"] == producer.MIN_EVAL_EPISODES == PANEL_EPISODES
        # Not a config key: the seed block lives with the producer, and the
        # panel it names is the one every threshold above was measured on.
        assert producer.PANEL_SEED_START == 3042


class TestConfigAloneCannotAdvance:
    """Declaring the gate is not passing it — the P5 guard."""

    def test_a_stage_directory_without_a_resolution_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(GateResolutionError, match="no gate_resolution.json"):
            require_gate_resolution(tmp_path, current_task_sha256="sha256:whatever")

    def test_a_perfect_panel_still_fails_on_the_config_alone(self, curriculum: dict[str, Any]) -> None:
        """40/40 does not pass without the frozen null manifest.

        The config declares ``min_paired_success_delta_lcb``, so the paired
        criterion is live; its evidence comes from the resolver's frozen null
        panel and from nowhere else.  A panel offered without that pairing —
        which is all the config by itself can ever produce — fails closed,
        even at a success rate no measured checkpoint has come near.
        """
        thresholds = _thresholds_from(curriculum)
        result = evaluate_recovery_gate(RecoveryPanel(episode_successes=(True,) * PANEL_EPISODES), thresholds)
        assert result.passed is False
        assert any("paired null panel missing" in failure for failure in result.failures)

    def test_the_measured_checkpoints_pass_once_the_pairing_exists(self, curriculum: dict[str, Any]) -> None:
        """Attainable, not aspirational: both §9.1 panels clear both criteria.

        Paired against the frozen 0/40 statue this reproduces §9.1's margins
        exactly — +0.365 (3M) and +0.340 (5M) — which is the evidence
        ``min_paired_success_delta_lcb = 0.20`` was frozen on.
        """
        thresholds = _thresholds_from(curriculum)
        for successes, expected_paired in ((POLICY_3M_SUCCESSES, 0.3651), (POLICY_5M_SUCCESSES, 0.3403)):
            panel = RecoveryPanel(
                episode_successes=tuple([True] * successes + [False] * (PANEL_EPISODES - successes)),
                paired_null_differences=_paired_differences_against_the_statue(successes),
            )
            result = evaluate_recovery_gate(panel, thresholds)
            assert result.failures == ()
            assert result.passed is True
            assert result.paired_delta_lcb == pytest.approx(expected_paired, abs=5e-5)

    def test_the_statue_null_fails_both_criteria(self, curriculum: dict[str, Any]) -> None:
        """The instrument's point: doing nothing is refused, not merely scored.

        Zero successes and a zero paired margin against itself — the gate must
        name both failures rather than passing on the absence of evidence.
        """
        thresholds = _thresholds_from(curriculum)
        panel = RecoveryPanel(
            episode_successes=(False,) * PANEL_EPISODES,
            paired_null_differences=(0.0,) * PANEL_EPISODES,
        )
        result = evaluate_recovery_gate(panel, thresholds)
        assert result.passed is False
        assert len(result.failures) == 2


def _thresholds_from(curriculum: dict[str, Any]) -> RecoveryGateThresholds:
    """The config's declaration as the gate's own dataclass.

    Deliberately a local construction rather than a production helper: no
    production path builds recovery thresholds from a config, because the
    real ones come from the frozen ``gate_resolution.json``.  This exists so
    the tests above can ask what the declared numbers WOULD mean.
    """
    return RecoveryGateThresholds(
        min_recovery_success_lcb=float(curriculum["min_recovery_success_lcb"]),
        t_recover_steps=int(curriculum["recovery_t_recover_steps"]),
        dwell_steps=int(curriculum["recovery_dwell_steps"]),
        min_eval_episodes=int(curriculum["min_eval_episodes"]),
        min_paired_success_delta_lcb=float(curriculum["min_paired_success_delta_lcb"]),
    )
