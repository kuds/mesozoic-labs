"""Checkpoint selection, publication, and VecNormalize statistics.

Keeping the normalisation statistics beside the policy they were trained with
is what makes a checkpoint replayable; :func:`load_vecnorm_stats` refuses to
load statistics whose plant identity does not match the current plant."""

from __future__ import annotations

import logging
import pickle
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np

from . import sb3_compat
from .command_evaluation import (
    CommandEvaluationPlan,
    family_robust_score,
    family_statistics,
    recorded_command_families,
)
from .command_identity import hold_vec_normalize_command_identity
from .hop_watch import HopRegimeWatchCallback
from .posture_watch import PostureDriftWatchCallback, PostureTrendWatchCallback
from .sb3_compat import BaseCallback

if TYPE_CHECKING:
    from environments.shared.plant_contract import PlantIdentity

logger = logging.getLogger(__name__)


class RobustBestModelCallback(BaseCallback):  # type: ignore[misc]
    """Save the checkpoint with the best risk-adjusted evaluation score.

    SB3's ``EvalCallback`` saves ``best_model.zip`` by mean eval reward.
    In run 20260709_185946 that selected the 800k-step checkpoint whose
    eval distribution was already bimodal — a high mean propped up by
    27/30 good episodes while the catastrophic-fall fraction was growing
    — when the genuinely robust checkpoint (~450k steps, zero failures)
    scored a slightly lower mean. This callback scores each evaluation as
    ``mean - risk_coef * std`` and saves ``robust_best_model.zip`` (plus
    its matching VecNormalize stats) whenever the score improves, so
    next-stage training and gating can prefer consistently-good policies
    over great-on-average ones with a fat failure tail.

    On a live-command stage (its evaluation callback carries a
    :class:`~environments.shared.curriculum.command_evaluation.CommandEvaluationPlan`
    as ``command_evaluation``, decision D-D62) the score is family-aware
    instead (:func:`~environments.shared.curriculum.command_evaluation.family_robust_score`):
    ``mean - risk_coef * std`` within the cruise and the stand episodes,
    combined at the stage's family probabilities, the families read from each
    episode's command (``evaluations_command_families``).  Pooled over both
    families, the gap between a stand episode's return and a cruise
    episode's would read as risk, and a checkpoint that stood worse would
    score higher.  Under command_mode "none" the score is the pooled one.

    Place after the ``EvalCallback`` in the callback list.

    Args:
        eval_callback: The ``EvalCallback`` whose in-memory eval results
            to score.
        model_dir: Directory to save ``robust_best_model.zip`` into.
        risk_coef: Weight on the per-episode std in the score
            (default 1.0, i.e. one standard deviation below the mean).
        staging_dir: When set (``train()`` sets it on a Drive/GCS mount), the
            pair is saved there first and published into *model_dir* by
            :func:`publish_staged_pair`, so a reclaim cannot leave it
            truncated or mixed with the previous pair (CU-3).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        eval_callback: Any,
        model_dir: "str | Path",
        risk_coef: float = 1.0,
        verbose: int = 0,
        staging_dir: "str | Path | None" = None,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RobustBestModelCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.model_dir = Path(model_dir)
        self.risk_coef = risk_coef
        self.staging_dir = Path(staging_dir) if staging_dir is not None else None
        self.best_score = -np.inf
        self._last_seen_n = 0

    @property
    def command_evaluation(self) -> "CommandEvaluationPlan | None":
        """The live stage's evaluation plan its evaluation callback rolls (``None``: a command-free stage).

        Only a :class:`CommandEvaluationPlan` counts: SB3's own ``EvalCallback`` has no plan, and
        ``StageAwareEvalCallback`` holds ``None`` under command_mode "none".
        """
        plan = getattr(self.eval_callback, "command_evaluation", None)
        return plan if isinstance(plan, CommandEvaluationPlan) else None

    def score(self, returns: Any, families: "Sequence[str] | None" = None) -> float:
        """One evaluation's risk-adjusted score: pooled ``mean - risk_coef * std``, or family-aware on a live stage.

        On a live stage *families* (each episode's command family, aligned
        with *returns*) is required.
        """
        values = np.asarray(returns, dtype=float)
        plan = self.command_evaluation
        if plan is None:
            return float(values.mean() - self.risk_coef * values.std())
        if families is None:
            raise ValueError("a live-command stage's robust score needs each evaluation episode's command family")
        return family_robust_score(values, families, plan.family_weights(), risk_coef=self.risk_coef)

    def prior_score(self, returns: Any, families: "Sequence[str] | None" = None) -> "float | None":
        """The score of a prior session's evaluation (a same-stage resume): ``None`` when it cannot be scored.

        A prior evaluation of a live stage is scored on the families its
        record carries (*families*, ``evaluations.npz``'s
        ``command_families``), as the tracker scored it when it was rolled;
        a record without them (written before they were kept) is scored on
        this session's plan, which a resume that keeps the run's seed rolled
        at every evaluation.  One with another episode count than the plan's
        is not this panel's and is not scored.
        """
        plan = self.command_evaluation
        if plan is None:
            return self.score(returns)
        values = np.asarray(returns, dtype=float).reshape(-1)
        if values.size != len(plan.families):
            return None
        recorded = None if families is None or len(families) != values.size else list(families)
        return self.score(values, plan.families if recorded is None else recorded)

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None) or []
        n = len(results)
        if n <= self._last_seen_n:
            return True
        self._last_seen_n = n

        latest = np.asarray(results[-1], dtype=float)
        families = None
        if self.command_evaluation is not None:
            recorded = getattr(self.eval_callback, "evaluations_command_families", None) or []
            if len(recorded) != n or len(recorded[-1]) != latest.size:
                raise RuntimeError(
                    f"RobustBestModel: evaluation {n} of a live-command stage has {latest.size} returns but its "
                    f"evaluation callback recorded {len(recorded)} evaluation(s) of command families"
                    + (f", the last with {len(recorded[-1])}" if recorded else "")
                    + "; the family-aware score needs each episode's family (decision D-D62)"
                )
            families = list(recorded[-1])
        score = self.score(latest, families)
        if score <= self.best_score:
            return True
        self.best_score = score

        # getattr: tests build this callback without __init__.
        staging_dir = getattr(self, "staging_dir", None)
        path = (staging_dir if staging_dir is not None else self.model_dir) / "robust_best_model"
        self.model.save(str(path))
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            # The command slice at identity in every sidecar (decision D-D60).
            hold_vec_normalize_command_identity(vec_env)
            vec_env.save(str(path) + "_vecnorm.pkl")
        if staging_dir is not None:
            publish_staged_pair(path, self.model_dir / "robust_best_model", zip_last=True)
        if families is None:
            logger.info(
                "RobustBestModel: new best risk-adjusted score %.1f (mean %.1f - %.1f*std %.1f) at step %d",
                score,
                latest.mean(),
                self.risk_coef,
                latest.std(),
                self.num_timesteps,
            )
        else:
            logger.info(
                "RobustBestModel: new best family-aware risk-adjusted score %.1f at step %d (%s)",
                score,
                self.num_timesteps,
                "; ".join(
                    f"{family} {count} ep: mean {mean:.1f} - {self.risk_coef:.1f}*std {std:.1f}"
                    for family, (count, mean, std) in family_statistics(latest, families).items()
                ),
            )
        return True


def select_handoff_checkpoint(model_dir: Path) -> tuple[str, str, str] | None:
    """The checkpoint the curriculum actually promotes to the next stage.

    Preference order matches next-stage loading: the risk-adjusted
    ``robust_best_model``, then SB3's mean-reward ``best_model`` — each
    only when its matched VecNormalize stats exist, so obs normalization
    matches the policy weights. Returns ``(name, model_path_without_ext,
    vecnorm_path)`` or ``None`` when neither candidate is complete.

    A stage whose gate chooses its handoff by the gated statistic
    (``command_tracking/v1``, decision D-D61) records the choice in
    ``models/handoff_choice.json`` (:data:`HANDOFF_CHOICE_FILENAME`); a
    valid record (:func:`recorded_handoff_choice`: its schema, a pair in
    *model_dir* whose two files hash to the digests it records) names the
    pair returned, so every reader of the handoff -- the replays, the
    selected evaluation, the stage gate, the next stage's load, reuse and
    publication -- follows the one choice.  An invalid record is logged and
    ignored, and the order above applies; the command-tracking judge
    refuses a handoff that is not the recorded choice, or a choice whose
    ranked pairs are no longer the stage's
    (:func:`handoff_choice_candidates_problem`), so a stale record certifies
    nothing.
    """
    choice, problem = recorded_handoff_choice(model_dir)
    if choice is not None:
        name = str(choice["selected"])
        return name, str(model_dir / name), str(model_dir / f"{name}_vecnorm.pkl")
    if problem is not None and (Path(model_dir) / HANDOFF_CHOICE_FILENAME).exists():
        logger.warning("%s is ignored: %s", Path(model_dir) / HANDOFF_CHOICE_FILENAME, problem)
    for candidate in REWARD_HANDOFF_NAMES:
        cand_zip = model_dir / f"{candidate}.zip"
        cand_vecnorm = model_dir / f"{candidate}_vecnorm.pkl"
        if cand_zip.exists() and cand_vecnorm.exists():
            return candidate, str(model_dir / candidate), str(cand_vecnorm)
    return None


#: The pairs a stage hands on when no choice is recorded, in preference order: the risk-adjusted
#: ``robust_best_model``, then SB3's mean-reward ``best_model`` (:func:`select_handoff_checkpoint`).
REWARD_HANDOFF_NAMES: tuple[str, ...] = ("robust_best_model", "best_model")
#: The suffix of a stage's final pair (``models/<stage_label>_final``), the third pair a choice ranks.
FINAL_PAIR_SUFFIX = "_final"

#: The record of a handoff chosen by the gated statistic, in the stage's ``models/`` (decision D-D61).
HANDOFF_CHOICE_FILENAME = "handoff_choice.json"
HANDOFF_CHOICE_SCHEMA = "mesozoic.handoff-choice/v1"


def _file_sha256(path: Path) -> str:
    """``result_bundle.hashing.sha256_file``'s portable ``sha256:<hex>`` digest (that package imports this one)."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _is_stem(name: Any) -> TypeGuard[str]:
    """A bare checkpoint stem: a non-empty string with no path part."""
    return isinstance(name, str) and bool(name) and Path(name).name == name and name not in (".", "..")


def _pair_digests(model_dir: Path, name: str) -> "tuple[str, str] | None":
    """The digests of ``<name>.zip`` and ``<name>_vecnorm.pkl`` in *model_dir*; ``None`` when the pair is not complete."""
    zip_path, vecnorm_path = model_dir / f"{name}.zip", model_dir / f"{name}_vecnorm.pkl"
    if not (zip_path.is_file() and vecnorm_path.is_file()):
        return None
    return _file_sha256(zip_path), _file_sha256(vecnorm_path)


def _selected_problem(record: Any, model_dir: Path) -> str | None:
    """Why *record* does not name a pair in *model_dir* by its current digests, or ``None``."""
    if not isinstance(record, dict) or record.get("schema") != HANDOFF_CHOICE_SCHEMA:
        schema = record.get("schema") if isinstance(record, dict) else type(record).__name__
        return f"its schema is {schema!r}, not {HANDOFF_CHOICE_SCHEMA!r}"
    name = record.get("selected")
    if not _is_stem(name):
        return f"it selects {name!r}, not a checkpoint stem in {model_dir}"
    try:
        digests = _pair_digests(model_dir, name)
    except OSError as exc:
        return f"the selected pair cannot be read ({exc})"
    if digests is None:
        return f"the selected pair {name}.zip / {name}_vecnorm.pkl is not complete"
    if digests != (record.get("checkpoint_sha256"), record.get("normalization_sha256")):
        return (
            f"the selected pair {name}.zip / {name}_vecnorm.pkl no longer hashes to the recorded digests "
            "(it was rewritten after the choice)"
        )
    return None


def recorded_handoff_choice(model_dir: "str | Path") -> "tuple[dict[str, Any] | None, str | None]":
    """``(record, problem)``: the stage's handoff choice when it is valid, else ``None`` and why.

    Valid means: ``models/handoff_choice.json`` is a JSON object of schema
    :data:`HANDOFF_CHOICE_SCHEMA` whose ``selected`` is a bare checkpoint
    stem in *model_dir* (no path), whose ``<selected>.zip`` and
    ``<selected>_vecnorm.pkl`` both exist, and whose ``checkpoint_sha256``
    and ``normalization_sha256`` are those two files' digests -- a pair
    rewritten after the choice (a resumed stage's new ``robust_best_model``)
    no longer is the pair that was chosen.  ``(None, None)`` when there is
    no record at all.  Never raises.

    Only the selected pair is hashed here (every reader of the handoff
    calls this); whether the other pairs the choice ranked are still the
    stage's is :func:`handoff_choice_candidates_problem`, which the
    command-tracking judge and the code that makes the choice check.
    """
    from ..file_io import read_json_object

    path = Path(model_dir) / HANDOFF_CHOICE_FILENAME
    if not path.exists():
        return None, None
    record = read_json_object(path)
    if record is None:
        return None, "it cannot be read as a JSON object"
    problem = _selected_problem(record, Path(model_dir))
    if problem is not None:
        return None, problem
    return record, None


def handoff_choice_candidates_problem(record: "dict[str, Any]", model_dir: "str | Path") -> str | None:
    """Why *record* is no longer a choice among the pairs its stage holds, or ``None`` when it still is.

    A choice is the best of the pairs it ranked, so it stands only while
    those are still the stage's pairs.  Its ``candidates`` list names each
    ranked pair (``name``, a bare stem, each once) with the digests of its
    two files when it was ranked (``checkpoint_sha256``,
    ``normalization_sha256``); the selected pair is among them under the
    record's own digests; every ranked pair still hashes to its recorded
    digests; and every complete pair in *model_dir* a choice ranks -- the
    reward-chosen ones (:data:`REWARD_HANDOFF_NAMES`) and the final pair
    (``*_final``) -- is among them.  So a pair rewritten after the choice
    (training continued in the stage directory) or one that appeared after
    it, never compared, makes the choice stale.  Never raises.
    """
    directory = Path(model_dir)
    candidates = record.get("candidates") if isinstance(record, Mapping) else None
    if not isinstance(candidates, list) or not candidates:
        return "it records no candidates, so the pairs it was chosen among are unknown"
    ranked: dict[str, tuple[Any, Any]] = {}
    for entry in candidates:
        name = entry.get("name") if isinstance(entry, Mapping) else None
        if not _is_stem(name) or name in ranked:
            return f"it records a candidate {name!r} that is not a distinct checkpoint stem"
        ranked[name] = (entry.get("checkpoint_sha256"), entry.get("normalization_sha256"))
    selected = record.get("selected")
    if not _is_stem(selected) or ranked.get(selected) != (
        record.get("checkpoint_sha256"),
        record.get("normalization_sha256"),
    ):
        return f"its selected pair {selected!r} is not among its candidates under the digests it records"
    try:
        for name, digests in ranked.items():
            if _pair_digests(directory, name) != digests:
                return f"the candidate {name} no longer hashes to the digests it was ranked under"
        present = {
            stem
            for stem in {path.name[: -len(".zip")] for path in directory.glob("*.zip")}
            if (stem in REWARD_HANDOFF_NAMES or stem.endswith(FINAL_PAIR_SUFFIX))
            and (directory / f"{stem}_vecnorm.pkl").is_file()
        }
    except OSError as exc:
        return f"the stage's pairs cannot be read ({exc})"
    unranked = sorted(present - set(ranked))
    if unranked:
        return f"the stage holds pairs the choice never ranked ({', '.join(unranked)})"
    return None


def write_handoff_choice(model_dir: "str | Path", record: "dict[str, Any]") -> Path:
    """Write *record* as ``models/handoff_choice.json`` atomically; it must already be a valid choice.

    Refuses (``ValueError``) a record :func:`recorded_handoff_choice` would
    not honor once written, or one :func:`handoff_choice_candidates_problem`
    would call stale, so a written choice is never silently ignored.  The
    previous record is kept when the new one is refused.
    """
    from ..file_io import atomic_write_json

    directory = Path(model_dir)
    problem = _selected_problem(record, directory) or handoff_choice_candidates_problem(record, directory)
    if problem is not None:
        raise ValueError(f"the handoff choice is refused: {problem}")
    return atomic_write_json(directory / HANDOFF_CHOICE_FILENAME, record)


def publish_staged_pair(staged: "str | Path", destination: "str | Path", *, zip_last: bool) -> None:
    """Publish ``<staged>.zip`` and ``<staged>_vecnorm.pkl`` over ``<destination>.*``, never as a mixed pair.

    For the fixed-name pairs (``best_model``, ``robust_best_model``,
    ``<label>_final``) that ``train()`` saves to local scratch when the stage
    directory is on a Drive/GCS mount (CU-3). Each file goes through
    :func:`~environments.shared.file_io.atomic_copy`, so neither is ever seen
    truncated. Two files cannot be replaced in one step, though, and these
    names are reused: a reclaim between the copies would leave a new file
    beside the previous pair's other file, a mixed pair that every existence
    and integrity check accepts. So the destination file published last is
    removed first, and a reclaim part-way through leaves at worst one file
    without its partner (the new pair's, or the previous pair's before the
    first copy lands), which that pair's readers already treat as incomplete:

    * ``zip_last=True`` (the best and robust-best handoff pairs): at worst a
      sidecar without its zip, which :func:`select_handoff_checkpoint` skips;
    * ``zip_last=False`` (the final pair): at worst a zip without its
      sidecar, which :func:`checkpoint_pair_problem` reports, so the SB3
      notebook's RESUME cell and chain loop treat it as a final pair cut
      short (D-D16), as they treat a truncated one. Before it saves the pair,
      ``train()`` puts an empty placeholder zip at the destination and then
      removes the destination sidecar, so a reclaim before this publish
      starts also leaves a final zip that check rejects (the placeholder,
      alone or, for an instant, beside the previous sidecar).

    Without a staged sidecar (a model with no VecNormalize wrapper) the zip
    is published alone and the destination's sidecar removed, so no earlier
    sidecar is left beside it. The staged files are removed once published.
    An ``OSError`` propagates, as a direct save's would.
    """
    from ..file_io import atomic_copy

    staged_zip, staged_pkl = Path(f"{staged}.zip"), Path(f"{staged}_vecnorm.pkl")
    destination_zip, destination_pkl = Path(f"{destination}.zip"), Path(f"{destination}_vecnorm.pkl")
    if not staged_pkl.exists():
        order = [(staged_zip, destination_zip)]
        destination_pkl.unlink(missing_ok=True)
    elif zip_last:
        order = [(staged_pkl, destination_pkl), (staged_zip, destination_zip)]
    else:
        order = [(staged_zip, destination_zip), (staged_pkl, destination_pkl)]
    order[-1][1].unlink(missing_ok=True)
    for source, target in order:
        atomic_copy(source, target)
    for source, _ in order:
        source.unlink(missing_ok=True)


def checkpoint_pair_problem(zip_path: Path, vecnorm_path: Path) -> str | None:
    """``None`` when an SB3 checkpoint zip and its VecNormalize sidecar can be loaded, else why not.

    A runtime reclaimed mid-write leaves a truncated or orphaned pair, and a pair
    written straight to a Drive mount can be cut short in its sync window. This is
    the check :func:`newest_intact_periodic_pair` runs over every periodic candidate,
    the notebook's RESUME cell over the final pair, and the chain loop over the final
    pair before its JUDGE branch loads it. A zip whose sidecar is missing is never
    trusted: a loaded policy under fresh normalization statistics collapses silently.
    """
    zip_path, vecnorm_path = Path(zip_path), Path(vecnorm_path)
    if not vecnorm_path.exists():
        return f"missing matched VecNormalize sidecar {vecnorm_path.name}"
    try:
        with zipfile.ZipFile(zip_path) as archive:
            bad_member = archive.testzip()
            names = archive.namelist()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"corrupt archive member {bad_member!r}")
        # A truncated SB3 checkpoint can still open: zipfile locks onto a nested
        # torch archive's end-of-directory record, so testzip alone passes.
        # Require SB3's own members in the OUTER archive.
        if "data" not in names or not any(name.endswith("policy.pth") for name in names):
            raise zipfile.BadZipFile(
                f"outer archive lacks SB3 members (truncated checkpoint; found {sorted(names)[:5]})"
            )
    except Exception as exc:
        return f"bad/truncated checkpoint zip {zip_path.name} ({exc})"
    try:
        with open(vecnorm_path, "rb") as handle:
            pickle.load(handle)
    except Exception as exc:
        return f"VecNormalize sidecar {vecnorm_path.name} does not unpickle ({exc})"
    return None


def newest_intact_periodic_pair(
    model_dir: "str | Path", name_prefix: str
) -> "tuple[tuple[int, Path, Path] | None, list[str]]":
    """The newest periodic checkpoint pair in *model_dir* that :func:`checkpoint_pair_problem` passes, and the skips.

    SB3's ``CheckpointCallback`` saves ``{name_prefix}_{steps}_steps.zip`` and,
    with ``save_vecnormalize=True``, a matched
    ``{name_prefix}_vecnormalize_{steps}_steps.pkl``. A zip is a candidate when
    ``policy_loading._PERIODIC_CHECKPOINT_RE`` (the pattern the trainer and the
    command line read) matches its whole stem with *name_prefix* as the prefix:
    ``stage2_100_steps.zip`` is ``stage2``'s, ``stage22_100_steps.zip`` and
    ``stage2_x_100_steps.zip`` are not. The newest pair is exactly the one an
    ungraceful runtime reclaim may have left truncated or orphaned mid-write, so
    the candidates are walked newest step first, by the count parsed from the
    name (never the mtime, which on a Drive mount is upload order), and a
    candidate with a problem falls back to the next older step-point. The
    sidecar is named from the parsed count, as SB3 names it, and two names of
    one count are walked in name order.

    Returns ``(pair, skipped)``. *pair* is ``(steps, zip_path, vecnorm_path)``
    for the first intact candidate, or ``None``; *skipped* holds
    ``"<zip name>: <problem>"`` for each candidate passed over, newest first.
    ``(None, [])`` means the glob listed no periodic checkpoint of
    *name_prefix* in *model_dir*: it holds none, does not exist or is not a
    directory, or ``Path.glob`` could not list it. The glob drops some
    listing errors (a ``PermissionError`` on 3.11, any ``OSError`` on 3.12
    and 3.13), so an unreadable directory that holds pairs also gives
    ``(None, [])``, as the RESUME cell's own glob did before CU-6.
    ``(None, skipped)`` with *skipped* non-empty means none of them is
    intact. Nothing is printed or written here, and nothing is raised of its
    own: an ``OSError`` the glob does not drop (on 3.11 and 3.12 from its
    check that *model_dir* is a directory, and on 3.11 from the listing) or
    one from the pair check's ``Path.exists`` (an I/O or permission error on
    a mount) propagates, and the skips found before it are not returned. The
    SB3 notebook's RESUME cell, whose walk this was until cleanup CU-6,
    prints each skip as a warning once the walk returns and words its own
    refusals.

    *name_prefix* is a stage label (:func:`~environments.shared.stage_manifest.stage_label`):
    ``stage{N}`` or a stage id of the form ``[a-z][a-z0-9_]*``, so non-empty,
    on one line and free of glob metacharacters. The pattern's ``(.+)``
    admits no other prefix, so an empty one, or one holding a newline (which
    ``STAGE_ID_PATTERN``'s ``$`` lets through at the end of an id), finds no
    candidate.
    """
    from ..policy_loading import _PERIODIC_CHECKPOINT_RE

    directory = Path(model_dir)
    candidates: list[tuple[int, Path]] = []
    for zip_path in sorted(directory.glob(f"{name_prefix}_*_steps.zip")):
        match = _PERIODIC_CHECKPOINT_RE.fullmatch(zip_path.stem)
        if match is not None and match.group(1) == name_prefix:
            candidates.append((int(match.group(2)), zip_path))
    skipped: list[str] = []
    for steps, zip_path in sorted(candidates, key=lambda candidate: candidate[0], reverse=True):
        vecnorm_path = directory / f"{name_prefix}_vecnormalize_{steps}_steps.pkl"
        problem = checkpoint_pair_problem(zip_path, vecnorm_path)
        if problem is None:
            return (steps, zip_path, vecnorm_path), skipped
        skipped.append(f"{zip_path.name}: {problem}")
    return None, skipped


def seed_resume_eval_state(
    eval_callback: Any,
    callbacks: "list[Any]",
    prior_npz_path: "str | Path",
    *,
    max_timesteps: "int | None" = None,
) -> int:
    """Seed eval history and best-model trackers from a prior session's record.

    On a same-stage resume every best tracker restarts at ``-inf`` and SB3's
    ``EvalCallback`` starts an empty ``evaluations.npz`` — so the first
    post-resume evaluation unconditionally overwrote ``best_model.zip`` /
    ``robust_best_model.zip`` even when it scored far below the pre-crash
    best, and the published eval history was replaced with only the
    post-resume slice (review TC2).  This seeds, from the stage directory's
    published ``evaluations.npz``:

    * ``eval_callback.best_mean_reward`` and ``RobustBestModelCallback``'s
      ``best_score`` — computed over the FULL prior record, so a best model
      saved at any pre-crash point stays protected (on a live-command stage
      the robust score is the family-aware one on the families the record
      carries, or on this session's command panel for a record written
      before they were kept, :meth:`RobustBestModelCallback.prior_score`,
      decision D-D62);
    * ``eval_callback.evaluations_*`` — the in-memory history SB3 re-saves
      on every evaluation, so the npz keeps the whole run's record (on a
      live stage with each evaluation's command families,
      ``evaluations_command_families``: the record's, or the plan's for a
      record written before they were kept, so the republished record
      carries them for every row).  Entries
      newer than ``max_timesteps`` (the resumed checkpoint's step count) are
      dropped from the HISTORY so the republished timesteps stay monotonic
      when the crash — or a fallback to an older checkpoint — postdates the
      last evaluation; the best trackers still cover them;
    * the parallel per-eval capture series and eval cursors of the stage
      diagnostics callbacks (``StageAwareEvalCallback.seed_prior_history``,
      ``_last_seen_n_evals`` / ``_last_published_n`` counters), so consumers
      that index those series by the absolute evaluation index stay aligned
      and the seeded history is not reprocessed as new evaluations;
    * the plateau callback's ``gate_progress.npz`` rows, from the file beside
      *prior_npz_path* (``StageGatePlateauCallback.seed_prior_gate_progress``),
      so the resumed session appends to the prior rows instead of rewriting
      the file with its own, and the hop and posture watches' cursors over
      those rows;
    * every callback's own record in that directory (its
      ``seed_prior_record``: the eval-episode ledger ``eval_episodes.csv``,
      the training-episode ledger ``train_episodes.csv`` and the policy
      probe's ``policy_probe.npz``), each keeping its prior rows at or
      before ``max_timesteps`` (the probe holds them until it has built its
      state bank in ``_on_training_start`` and found it equal to the prior one);
    * the rollout diagnostics, from the ``diagnostics.npz`` beside
      *prior_npz_path* (``DiagnosticsCallback.seed_prior_diagnostics``),
      keeping its rows at or before ``max_timesteps`` on each of its clocks.
      Seeded first, whatever the evaluation record holds: a resume whose
      ``evaluations.npz`` is missing, unreadable or ragged still continues
      the rollout history.

    A prior record whose per-eval episode count differs from this session's
    ``n_eval_episodes`` seeds only the scalar best trackers: mixing row
    lengths would make SB3's next ``np.savez`` raise on the ragged array,
    wedging every resume retry.

    Returns the number of prior evaluations seeded into the HISTORY (0 when
    no usable record exists — a fresh stage directory resumes cleanly with
    empty state, with a prominent warning since the best-tracker protection
    is then absent).
    """
    path = Path(prior_npz_path)
    # diagnostics.npz is written beside evaluations.npz (both in the stage directory) on its own rollout clock, so
    # it is seeded from the file next to the record whether or not the evaluation history below can be.
    for callback in callbacks:
        seed_diagnostics = getattr(callback, "seed_prior_diagnostics", None)
        if callable(seed_diagnostics):
            seed_diagnostics(path.with_name("diagnostics.npz"), max_timesteps=max_timesteps)
    if not path.exists():
        logger.warning(
            "Resume: no prior evaluation record at %s — best-model trackers and eval history "
            "start fresh. If this resume continues an interrupted run, point the run at the "
            "interrupted stage directory so its evaluations.npz seeds this session.",
            path,
        )
        return 0
    try:
        with np.load(path) as data:
            timesteps = [int(value) for value in data["timesteps"]]
            results = [np.asarray(row) for row in data["results"]]
            ep_lengths = [np.asarray(row) for row in data["ep_lengths"]]
            successes = [np.asarray(row) for row in data["successes"]] if "successes" in data else None
            # Each evaluation's command families (a live stage's record, decision D-D62); None without them.
            families = recorded_command_families(data)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        logger.warning("Could not seed resume eval state from %s: %s", path, exc)
        return 0
    if not results:
        return 0

    # Best trackers cover the FULL record — a pre-crash best_model.zip must
    # stay protected even when its evaluation postdates the resumed
    # checkpoint and is therefore dropped from the history below.
    best_mean = max(float(np.asarray(row, dtype=float).mean()) for row in results)
    eval_callback.best_mean_reward = best_mean
    for callback in callbacks:
        if isinstance(callback, RobustBestModelCallback):
            # A live stage's prior evaluations are scored by family (RobustBestModelCallback.prior_score): on the
            # families the record carries, else on this session's plan; a row of another episode count is not
            # scored.
            scores = [
                score
                for score in (
                    callback.prior_score(row, None if families is None else families[index])
                    for index, row in enumerate(results)
                )
                if score is not None
            ]
            if len(scores) < len(results):
                logger.warning(
                    "Resume: %d of %d prior evaluations at %s do not hold this session's %d-episode command panel; "
                    "the robust tracker is seeded from the other %d",
                    len(results) - len(scores),
                    len(results),
                    path,
                    len(callback.command_evaluation.families) if callback.command_evaluation is not None else 0,
                    len(scores),
                )
            callback.best_score = max(scores) if scores else -np.inf

    if max_timesteps is not None:
        keep = sum(1 for value in timesteps if value <= max_timesteps)
        if keep < len(results):
            logger.info(
                "Resume: dropping %d prior evaluation(s) recorded after the resumed checkpoint "
                "(step %s) so the republished history stays monotonic; the best trackers still "
                "cover them.",
                len(results) - keep,
                f"{max_timesteps:,}",
            )
            timesteps = timesteps[:keep]
            results = results[:keep]
            ep_lengths = ep_lengths[:keep]
            if successes is not None:
                successes = successes[:keep]
            if families is not None:
                families = families[:keep]

    expected_episodes = getattr(eval_callback, "n_eval_episodes", None)
    if expected_episodes is not None and any(len(row) != expected_episodes for row in results):
        logger.warning(
            "Resume: prior evaluations at %s were recorded with a different episode count than "
            "this session's n_eval_episodes=%s — seeding only the best-model trackers; the "
            "republished evaluations.npz will hold this session's evaluations only.",
            path,
            expected_episodes,
        )
        for callback in callbacks:
            if isinstance(callback, RobustBestModelCallback):
                callback._last_seen_n = 0
        return 0

    seeded = len(results)
    eval_callback.evaluations_timesteps = timesteps
    eval_callback.evaluations_results = results
    eval_callback.evaluations_length = ep_lengths
    if successes is not None and len(successes) == seeded:
        # Inert since decision D-D59: only a success-gated (hunt) stage's
        # record carries the rows, and SB3 writes the successes array only
        # when the CURRENT session's success buffer is non-empty, which it
        # never is (StageAwareEvalCallback does not collect is_success), so
        # the seeded rows do not survive the next publish.
        eval_callback.evaluations_successes = successes

    # Stage diagnostics keep parallel per-eval series indexed by the absolute
    # evaluation index; pad them so post-resume evaluations stay aligned.
    seed_prior = getattr(eval_callback, "seed_prior_history", None)
    if callable(seed_prior):
        seed_prior(seeded)
    plan = getattr(eval_callback, "command_evaluation", None)
    if isinstance(plan, CommandEvaluationPlan):
        # A live stage's families, aligned with the seeded results, so the next save of evaluations.npz carries
        # them for every row: the record's, or the plan's (what the robust tracker scored them on) for a record
        # written before they were kept.
        eval_callback.evaluations_command_families = (
            [list(row) for row in families] if families is not None else [list(plan.families) for _ in results]
        )
    # gate_progress.npz is published beside evaluations.npz (both to the stage
    # directory), so the prior session's file is the one next to the record.
    gate_rows = 0
    for callback in callbacks:
        seed_gate_progress = getattr(callback, "seed_prior_gate_progress", None)
        if callable(seed_gate_progress):
            gate_rows = seed_gate_progress(path.with_name("gate_progress.npz"), max_timesteps=max_timesteps)
    # The records written beside it (the eval- and training-episode ledgers, the policy probe), each keeping its
    # prior rows at or before the resumed step, as gate_progress.npz does.
    for callback in callbacks:
        seed_record = getattr(callback, "seed_prior_record", None)
        if callable(seed_record):
            seed_record(path.parent, max_timesteps=max_timesteps)

    for callback in callbacks:
        if isinstance(callback, RobustBestModelCallback):
            # The seeded history must not read as a fresh evaluation of the
            # loaded model.
            callback._last_seen_n = seeded
        # Eval-count cursors (StageGatePlateauCallback, the collapse
        # backstop, the npz publisher): start past the seeded history so it
        # is not reprocessed as new evaluations. The collapse backstop then
        # re-arms from post-resume evaluations only — the same fresh-run
        # behavior it had before seeding existed.
        if hasattr(callback, "_last_seen_n_evals"):
            callback._last_seen_n_evals = seeded
        if hasattr(callback, "_last_published_n"):
            callback._last_published_n = seeded
        # The hop and posture watches' cursors count gate_progress rows, which need not be one per evaluation.
        if isinstance(callback, (HopRegimeWatchCallback, PostureDriftWatchCallback, PostureTrendWatchCallback)):
            callback._last_seen = gate_rows

    logger.info(
        "Resume: seeded %d prior evaluations from %s (best mean %.1f)",
        seeded,
        path,
        best_mean,
    )
    return seeded


class PublishEvalArtifactsCallback(BaseCallback):  # type: ignore[misc]
    """Atomically publish EvalCallback's ``evaluations.npz`` to the stage dir.

    The paired ``EvalCallback`` writes its npz to fast local scratch;
    this callback copies it to the (possibly Drive/GCS-FUSE mounted)
    stage directory after every new evaluation and again at training
    end, via copy-to-temp + rename, so the published file is never
    observed truncated. Place it *after* the ``EvalCallback`` in the
    callback list so it runs on the same step a new eval completes.

    Args:
        eval_callback: The ``EvalCallback`` whose npz to publish.
        publish_dir: Directory to publish ``evaluations.npz`` into
            (typically the stage directory).
        verbose: Verbosity level.
    """

    def __init__(self, eval_callback: Any, publish_dir: "str | Path", verbose: int = 0):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PublishEvalArtifactsCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.publish_dir = Path(publish_dir)
        self._last_published_n = 0

    def _source_npz(self) -> "Path | None":
        # SB3 stores log_path as the "<dir>/evaluations" file prefix.
        prefix = getattr(self.eval_callback, "log_path", None)
        return Path(str(prefix) + ".npz") if prefix else None

    def _publish(self) -> None:
        from ..file_io import atomic_copy

        src = self._source_npz()
        if src is not None and src.exists():
            atomic_copy(src, self.publish_dir / "evaluations.npz")

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None) or []
        if len(results) > self._last_published_n:
            self._last_published_n = len(results)
            self._publish()
        return True

    def _on_training_end(self) -> None:
        self._publish()


class PublishCheckpointPairCallback(BaseCallback):  # type: ignore[misc]
    """Atomically publish periodic checkpoints written to local scratch.

    SB3's ``CheckpointCallback`` streams the checkpoint zip and its
    VecNormalize sidecar directly onto the destination filesystem.  On a
    Drive/GCS FUSE mount an ungraceful runtime reclaim landing inside the
    write (or DriveFS's post-write sync window) can leave the newest pair —
    the sole artifact the resume path depends on — truncated or orphaned
    (review CO2). ``evaluations.npz`` got the local-scratch + atomic-publish
    treatment for exactly this hazard; this extends it to checkpoints.

    Pair with a ``CheckpointCallback`` whose ``save_path`` is fast local
    scratch, place this immediately after it in the callback list (same
    ``save_freq`` cadence), and point ``publish_dir`` at the real model
    directory.  Every ``{prefix}_*_steps.*`` file in the scratch dir is
    copied via copy-to-temp + rename and removed locally on success, so the
    published pair is never observed truncated and scratch stays small.

    Args:
        local_dir: The paired ``CheckpointCallback``'s ``save_path``.
        publish_dir: The stage's real ``models/`` directory.
        name_prefix: The paired callback's ``name_prefix``.
        save_freq: The paired callback's cadence, already divided by
            ``n_envs`` (see :class:`CheckpointRetentionCallback` for why the
            cadence gate matters).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        local_dir: "str | Path",
        publish_dir: "str | Path",
        name_prefix: str,
        save_freq: int,
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PublishCheckpointPairCallback.")
        if save_freq < 1:
            raise ValueError("save_freq must be at least 1")
        super().__init__(verbose)
        self.local_dir = Path(local_dir)
        self.publish_dir = Path(publish_dir)
        self.name_prefix = name_prefix
        self.save_freq = int(save_freq)

    def _publish(self) -> None:
        from ..file_io import atomic_copy

        # Sidecars first, model zip LAST: each copy is atomic but the pair is
        # not, and a reclaim between the two must only ever leave a harmless
        # sidecar-without-zip — never a zip whose missing sidecar trips the
        # resume paths' fail-closed guards.
        for infix, extension in reversed(_CHECKPOINT_KINDS):
            for path in sorted(self.local_dir.glob(f"{self.name_prefix}_{infix}*_steps.{extension}")):
                try:
                    atomic_copy(path, self.publish_dir / path.name)
                except OSError:
                    logger.warning("Could not publish checkpoint %s", path, exc_info=True)
                    continue
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Could not remove local staged checkpoint %s", path, exc_info=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._publish()
        return True

    def _on_training_end(self) -> None:
        self._publish()


class SaveVecNormalizeCallback(BaseCallback):  # type: ignore[misc]
    """Save VecNormalize stats whenever triggered.

    Intended for use as ``callback_on_new_best`` in SB3's ``EvalCallback``
    so that the VecNormalize wrapper is saved alongside ``best_model.zip``.
    This ensures the observation normalization statistics match the policy
    weights when the best model is loaded for evaluation or next-stage
    curriculum training.

    Example::

        save_vecnorm_cb = SaveVecNormalizeCallback(
            save_path=str(model_dir / "best_model_vecnorm.pkl"),
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=save_vecnorm_cb,
            ...
        )

    Args:
        save_path: Destination path for the VecNormalize ``.pkl`` file.
        verbose: Verbosity level.
        publish_dir: When set (``train()`` sets it on a Drive/GCS mount, with
            the ``EvalCallback``'s ``best_model_save_path`` and *save_path* in
            local scratch), the pair SB3 just saved there is published into
            *publish_dir* by :func:`publish_staged_pair` (CU-3). *save_path*
            must then end in ``_vecnorm.pkl``; the pair's name is what precedes it.
    """

    def __init__(self, save_path: str, verbose: int = 0, publish_dir: "str | Path | None" = None):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for SaveVecNormalizeCallback.")
        if publish_dir is not None and not str(save_path).endswith("_vecnorm.pkl"):
            raise ValueError(f"save_path must end in _vecnorm.pkl when publish_dir is set, got {save_path}")
        super().__init__(verbose)
        self.save_path = save_path
        self.publish_dir = Path(publish_dir) if publish_dir is not None else None

    def _on_step(self) -> bool:
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            # The command slice at identity in every sidecar (decision D-D60).
            hold_vec_normalize_command_identity(vec_env)
            vec_env.save(self.save_path)
            logger.info("VecNormalize saved to: %s", self.save_path)
        # getattr: tests build this callback without __init__.
        publish_dir = getattr(self, "publish_dir", None)
        if publish_dir is not None:
            staged = Path(str(self.save_path).removesuffix("_vecnorm.pkl"))
            publish_staged_pair(staged, publish_dir / staged.name, zip_last=True)
        return True


def load_vecnorm_stats(
    vecnorm_path: str,
    train_env,
    eval_env=None,
    *,
    current_plant: PlantIdentity | None = None,
    allow_legacy_plant: bool = False,
    unsafe_skip_plant_validation: bool = False,
    carry_ret_rms: bool = False,
    command_mode: str,
) -> bool:
    """Load VecNormalize running statistics from a previous stage into new envs.

    By default only observation normalization (``obs_rms``) is carried
    forward.  Return normalization (``ret_rms``) is deliberately **reset**
    across stage boundaries because the reward distribution changes between
    curriculum stages (new reward components, different weight magnitudes).
    Carrying stale ``ret_rms`` produces badly scaled normalized rewards that
    destabilise policy gradients during the critical first updates of a new
    stage.

    A **same-stage resume** is the opposite situation: the reward
    distribution is identical and the correct ``ret_rms`` sits in the loaded
    ``.pkl`` — resetting it re-clips the first post-resume rollouts against
    fresh unit variance (review TC6).  Pass ``carry_ret_rms=True`` there.

    Args:
        vecnorm_path: Path to a ``_vecnorm.pkl`` file saved by a previous stage.
        train_env: The new stage's training ``VecNormalize`` wrapper.
            Its existing ``training`` / ``norm_reward`` flags are preserved
            (set by ``create_vec_env`` based on the algorithm — SAC keeps
            ``norm_reward=False`` to avoid replay-buffer reward-scale drift).
        eval_env: Optional evaluation ``VecNormalize`` wrapper.
            ``training`` is set to ``False``; ``norm_reward`` is disabled.
        current_plant: Expected plant identity. Required unless the explicit
            unsafe inspection-only escape hatch is enabled.
        allow_legacy_plant: Explicitly allow a pre-contract normalization
            artifact with no identity.  Incompatible tagged artifacts are
            always rejected.
        unsafe_skip_plant_validation: Deliberately load without checking the
            plant. This is for low-level inspection/tests only and cannot be
            combined with either validation option.
        command_mode: The ``command_mode`` of the stage that will read
            these statistics (required: every caller states it).  The
            identity rule (decision D-D60): training holds the command slice
            of ``obs_rms`` at mean 0 / var 1, so statistics saved since then
            load as they are.  Statistics whose slice is not at identity --
            saved before the rule (a ``"none"`` stage's slice decayed toward
            variance 0 as ``1e-4 / count``), or edited -- are refused with
            :class:`~environments.shared.command_frame.CommandNormalizationError`
            for a live mode, before either destination changes, since a live
            command read through them would not mean what it means to every
            other stage's policy; for ``"none"`` identity is re-imposed on
            BOTH destinations (count kept) and a warning names the file, as
            a ``"none"`` stage's zero command normalises to zero under either.
            A same-stage resume follows the same rule: its own saves were
            held at identity.

    Returns:
        ``True`` if stats were loaded, ``False`` if the file was not found.
    """
    from pathlib import Path as _Path

    from environments.shared.command_frame import (
        COMMAND_MODE_NONE,
        COMMAND_MODES,
        command_identity_gap,
        hold_command_identity,
        require_command_identity,
    )
    from environments.shared.plant_contract import PlantCompatibilityError

    if command_mode not in COMMAND_MODES:
        raise ValueError(f"command_mode={command_mode!r} is not one of {COMMAND_MODES}")

    if unsafe_skip_plant_validation:
        if current_plant is not None or allow_legacy_plant:
            raise ValueError("unsafe_skip_plant_validation cannot be combined with current_plant or allow_legacy_plant")
        logger.warning(
            "UNSAFE: loading %s without plant compatibility validation; "
            "do not use these statistics for training or evaluation until verified",
            vecnorm_path,
        )
    elif current_plant is None:
        raise PlantCompatibilityError(
            f"refusing to load {vecnorm_path} without current_plant; pass the current PlantIdentity "
            "or use unsafe_skip_plant_validation=True only for deliberate low-level inspection"
        )

    if not sb3_compat._SB3_AVAILABLE:
        logger.warning("stable-baselines3 not available; skipping VecNormalize load.")
        return False

    from stable_baselines3.common.vec_env import VecNormalize

    path = _Path(vecnorm_path)
    if not path.exists():
        logger.debug("VecNormalize file not found: %s", vecnorm_path)
        return False

    logger.info("Loading VecNormalize stats from: %s", vecnorm_path)
    from environments.shared.policy_loading import load_vecnormalize

    # Statistics of another observation width cannot even bind to the venv;
    # load_vecnormalize then refuses them by their recorded plant identity
    # rather than by SB3's bare shape assertion.
    prev_norm = load_vecnormalize(
        path,
        train_env.venv,
        plant_identity=None if unsafe_skip_plant_validation else current_plant,
        allow_legacy_plant=allow_legacy_plant,
        vec_normalize_cls=VecNormalize,
    )
    if not unsafe_skip_plant_validation:
        from environments.shared.plant_contract import validate_model_plant

        assert current_plant is not None
        # Validate before copying obs_rms so an incompatible artifact cannot
        # partially mutate either destination environment.
        validate_model_plant(
            prev_norm,
            current_plant,
            artifact=str(path),
            allow_legacy=allow_legacy_plant,
        )

    # The identity rule (decision D-D60), judged before either destination
    # changes: a live stage refuses a slice off identity; a "none" stage
    # re-imposes it on the loaded object, which both destinations then carry.
    gap = command_identity_gap(prev_norm.obs_rms)
    if gap != 0.0:
        if command_mode != COMMAND_MODE_NONE:
            require_command_identity(prev_norm.obs_rms, artifact=str(path), command_mode=command_mode)
        hold_command_identity(prev_norm.obs_rms)
        logger.warning(
            "%s: the command slice of its observation statistics was off identity (largest deviation %.3g; saved "
            "before decision D-D60 held it at mean 0 / variance 1); identity re-imposed for this command-free "
            "stage, whose zero command normalises to zero under either",
            vecnorm_path,
            gap,
        )

    # Carry forward observation statistics — the observation WIDTH is
    # identical across stages (the plant identity check above guarantees
    # it), so the running mean/var remain valid, the command slice at
    # identity.  ret_rms is copied only for same-stage resumes
    # (carry_ret_rms=True): across stage boundaries the reward distribution
    # changes, so stale return statistics would produce incorrectly scaled
    # normalised rewards for PPO.  train_env.training / norm_reward are left
    # as configured by create_vec_env (algorithm-aware).
    train_env.obs_rms = prev_norm.obs_rms
    eval_obs_rms = prev_norm.obs_rms.copy() if eval_env is not None else None
    if carry_ret_rms:
        # Only the running statistics: the per-env `returns` accumulators were
        # re-zeroed by VecNormalize.set_venv and correctly start at zero (the
        # resumed envs begin from a reset).
        train_env.ret_rms = prev_norm.ret_rms
        logger.info("obs_rms and ret_rms carried forward (same-stage resume)")
    else:
        logger.info("obs_rms carried forward; ret_rms reset (reward distribution changed)")

    if eval_env is not None:
        eval_env.obs_rms = eval_obs_rms
        eval_env.training = False
        eval_env.norm_reward = False

    return True


#: Periodic checkpoints kept by default: five, the rollback depth the JAX
#: trainer kept (its ``max_checkpoints``) until D-D17 retired it, so stages
#: trained before and after the retirement keep the same history.
DEFAULT_MAX_CHECKPOINTS = 5

#: The ``[curriculum]`` key of the archive rule (:func:`prune_periodic_checkpoints`); absent or 0: no archive.
CHECKPOINT_ARCHIVE_KEY = "checkpoint_archive_every_timesteps"


def checkpoint_archive_every(curriculum: Any) -> int:
    """The stage's ``checkpoint_archive_every_timesteps`` (0 when absent): a whole, non-negative step count.

    Refuses a bool, a non-integer and a negative value with a ``ValueError``
    naming the key, so a typo cannot silently turn the archive off.
    """
    value = (curriculum or {}).get(CHECKPOINT_ARCHIVE_KEY, 0)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(
            f"[curriculum] {CHECKPOINT_ARCHIVE_KEY} must be a whole number of steps, 0 or more; got {value!r}"
        )
    return int(value)


def archived_steps(steps: "set[int] | list[int]", archive_every_timesteps: int) -> set[int]:
    """The steps the archive rule keeps: the largest step present in each interval ``(k A, (k+1) A]``.

    ``k = (s - 1) // A`` for a step ``s``; nothing with ``A <= 0``.  A function
    of the step counts alone, so a restart or a resume keeps the same set.
    """
    if archive_every_timesteps <= 0:
        return set()
    newest: dict[int, int] = {}
    for step in steps:
        interval = (int(step) - 1) // archive_every_timesteps
        newest[interval] = max(newest.get(interval, int(step)), int(step))
    return set(newest.values())


#: The three artifact families SB3's ``CheckpointCallback`` emits, as
#: ``(infix, extension)``.  Its filenames are
#: ``f"{name_prefix}_{infix}{num_timesteps}_steps.{ext}"``
#: (``stable_baselines3.common.callbacks.CheckpointCallback._checkpoint_path``).
_CHECKPOINT_KINDS: tuple[tuple[str, str], ...] = (
    ("", "zip"),
    ("vecnormalize_", "pkl"),
    ("replay_buffer_", "pkl"),
)


def prune_periodic_checkpoints(
    model_dir: "str | Path",
    name_prefix: str,
    max_checkpoints: int,
    *,
    archive_every_timesteps: int = 0,
) -> list[Path]:
    """Delete all but the newest *max_checkpoints* periodic checkpoints, and the archive's.

    With ``archive_every_timesteps = A > 0`` a step outside the newest
    *max_checkpoints* is also kept when it is the largest step present in its
    interval ``(k A, (k+1) A]`` (:func:`archived_steps`): one checkpoint per
    ``A`` steps survives the run, the record of when a posture set in, at a
    fraction of keeping every one.  ``A = 0`` (the default) is the rule
    without an archive.

    Matches only SB3's periodic ``{prefix}_{steps}_steps.*`` files and their
    ``vecnormalize_`` / ``replay_buffer_`` siblings.  ``best_model``,
    ``robust_best_model`` and ``stage<N>_final`` do not fit that pattern and
    are therefore unreachable from here by construction rather than by an
    exclusion list that could fall out of date.

    Ordering is by the **step count parsed from the filename**, not by mtime:
    on a Drive or GCS-FUSE mount the modification times reflect when the
    bytes finished uploading, which is not the order they were produced in.

    A checkpoint whose model ``.zip`` is pruned takes its matched
    ``vecnormalize_`` statistics with it — keeping orphaned statistics would
    leave a ``.pkl`` that no longer belongs to any policy in the directory.

    Returns the paths actually deleted, oldest step first.  A file that
    cannot be unlinked is logged and skipped rather than aborting the rest:
    these are archival copies, and one locked file must not leave the
    directory half-pruned.  ``max_checkpoints <= 0`` disables pruning
    entirely and returns ``[]``.
    """
    directory = Path(model_dir)
    if max_checkpoints <= 0 or not directory.is_dir():
        return []

    # Group every artifact by the step count in its name, so a step is only
    # ever dropped as a complete set.
    by_step: dict[int, list[Path]] = {}
    for infix, extension in _CHECKPOINT_KINDS:
        pattern = f"{name_prefix}_{infix}*_steps.{extension}"
        for path in directory.glob(pattern):
            remainder = path.name[len(f"{name_prefix}_{infix}") : -len(f"_steps.{extension}")]
            if not remainder.isdigit():
                # The remainder between the prefix and `_steps.<ext>` must be
                # the step count and nothing else. Anything unparsed is not a
                # file this function put here, so it is left alone rather than
                # deleted on a guess.
                continue
            by_step.setdefault(int(remainder), []).append(path)

    archived = archived_steps(set(by_step), archive_every_timesteps)
    doomed_steps = sorted(step for step in sorted(by_step, reverse=True)[max_checkpoints:] if step not in archived)
    removed: list[Path] = []
    for step in doomed_steps:
        for path in by_step[step]:
            try:
                path.unlink()
            except OSError:
                logger.warning("Could not prune checkpoint %s", path, exc_info=True)
                continue
            removed.append(path)
    return removed


class CheckpointRetentionCallback(BaseCallback):  # type: ignore[misc]
    """Keep only the newest *max_checkpoints* periodic checkpoints.

    SB3's ``CheckpointCallback`` never deletes anything.  At the shipped
    ``save_freq = 500_000`` a 6M-step stage 1 leaves **twelve** 4.02 MB
    policy zips plus their VecNormalize sidecars — measured on run
    ``20260801_021545``, where ``models/`` totals 60 MB of which 48 MB is
    periodic history, and a three-stage run is ~200 MB. Those files live on
    a Drive mount and nothing in this repository reads them: they exist for
    manual rollback, which needs recent history, not all of it.

    Place this **after** the ``CheckpointCallback`` in the callback list so
    it prunes on the same step a new checkpoint lands.  It never touches
    ``best_model``, ``robust_best_model`` or ``stage<N>_final`` — see
    :func:`prune_periodic_checkpoints` for why that is structural rather
    than an exclusion list.

    Args:
        model_dir: Directory the paired ``CheckpointCallback`` saves into.
        name_prefix: Its ``name_prefix`` (``f"stage{N}"``).
        save_freq: The paired callback's ``save_freq``, already divided by
            ``n_envs``. Pruning fires on the same ``n_calls`` cadence, which
            matters: ``_on_step`` runs on *every* training step, and globbing
            a Drive mount six million times a stage would cost far more than
            the storage it reclaims.
        max_checkpoints: How many step-points to keep. ``0`` disables
            pruning, which is how a stage opts out of retention entirely.
        archive_every_timesteps: Also keep the newest step-point of every
            interval of this many steps (``0``: no archive); see
            :func:`prune_periodic_checkpoints`.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        model_dir: "str | Path",
        name_prefix: str,
        save_freq: int,
        max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
        verbose: int = 0,
        archive_every_timesteps: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for CheckpointRetentionCallback.")
        if save_freq < 1:
            raise ValueError("save_freq must be at least 1")
        super().__init__(verbose)
        self.model_dir = Path(model_dir)
        self.name_prefix = name_prefix
        self.save_freq = int(save_freq)
        self.max_checkpoints = int(max_checkpoints)
        self.archive_every_timesteps = int(archive_every_timesteps)

    def _prune(self) -> None:
        archive = int(getattr(self, "archive_every_timesteps", 0))  # getattr: tests build this without __init__
        removed = prune_periodic_checkpoints(
            self.model_dir, self.name_prefix, self.max_checkpoints, archive_every_timesteps=archive
        )
        if removed:
            logger.info(
                "Pruned %d periodic checkpoint file(s), keeping the newest %d step-points%s: %s",
                len(removed),
                self.max_checkpoints,
                f" and the newest of every {archive:,} steps" if archive > 0 else "",
                ", ".join(sorted(path.name for path in removed)),
            )

    def _on_step(self) -> bool:
        # Same trigger CheckpointCallback uses, so pruning happens on exactly
        # the steps a new checkpoint was written and never in between.
        if self.n_calls % self.save_freq == 0:
            self._prune()
        return True

    def _on_training_end(self) -> None:
        # A stage can stop between save points (EvalCollapseEarlyStopCallback,
        # or a budget that is not a multiple of save_freq), leaving the last
        # prune stale. Cheap once, at the end.
        self._prune()
