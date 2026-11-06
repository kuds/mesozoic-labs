"""Runs that never count: not as certified, replicates, reusable ancestors, widening parents or committed results.

The maintainer's stance-landing decision of 2026-11-06 (cited by date until the
next records change assigns its decision id; its record is §30 of
``docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md``): the T. rex
stance recipe that the r10 phase-critic screen runs passed the six-heading
certificate with (the stance stagger penalty, the per-foot centre-of-pressure
band, the lateral lean, the settled leg reference and mirror-KL PPO, task
``1e5964b6``) landed on ``main`` for its certification by two NEW seeds.  The
nine r10 stance screen runs trained that recipe on branch commits, and their
task, gate, plant and recipe digests equal the landed recipe's byte for byte,
so without this module replicate discovery would count them as replicates of
a new seed and trunk selection could reuse one as the certified stance.  They
never count:

* ``stance6_r10LTSB_s42``, ``_s60`` and ``_s80``, trained at
  ``8737cd3d86431de1bdd0fcdb49a0d956a45e084d`` (the arm-LTSB screen branch);
* ``stance7_r10LTSBoff_s42``, ``_s60`` and ``_s80`` and
  ``stance7_r10LTSBpc_s42``, ``_s60`` and ``_s80``, trained at
  ``1dfe2327f773007fa7fd68460dc0c53740936ff9`` (the phase-critic screen
  branch).

A run is excluded when ANY of four records names one of them, so neither a
missing ``provenance.json`` (a stage directory copied into a run folder without
it, which ``ancestors.run_id_for`` names by its directory), nor a same-stage
resume on ``main`` (which rewrites ``stage_config.json`` with ``main``'s commit
while ``provenance.json`` keeps the original), nor a new run that loads a
screen's weights lets one count:

(a) the run's ``provenance.json`` records a ``repository_commit`` in
    :data:`EXCLUDED_COMMITS`;
(b) a stage directory's ``stage_config.json`` records a top-level
    ``git_commit`` in :data:`EXCLUDED_COMMITS`;
(c) a stage directory's ``gate_verdict.json`` hashes a handoff checkpoint in
    :data:`EXCLUDED_HANDOFF_SHA256` (``checkpoint_sha256``, in its
    ``sha256:`` form);
(d) a stage directory's ``stage_config.json`` ``run`` block records a load
    from one of them (the load-lineage keys ``config.save_stage_config``
    writes for a ``--load``, a same-stage resume or a widened root): a
    checkpoint digest (``parent_checkpoint_sha256``,
    ``resume_checkpoint_sha256``, ``widened_from_checkpoint_sha256``) in
    :data:`EXCLUDED_HANDOFF_SHA256` or :data:`EXCLUDED_CHECKPOINT_SHA256`, a
    path (``load_path``, ``resume_load_path``, ``widened_from_path``) with a
    component in :data:`EXCLUDED_RUN_IDS` (a load from a screen's run
    folder), or a parent run id (``parent_run_id``,
    ``widened_from_run_id``) in it.  A stage trained further from a screen's
    weights carries that screen's training, whatever commit it ran at: a
    ``--load`` of a screen's final or periodic checkpoint, a resume of a
    copied and re-judged stage directory, a widened copy.

The handoff digests are the screens' own records: the three phase-critic runs'
``gate_verdict.json``, the three stance6 runs' offline judge files committed under
``docs/investigations/stance_2026_10/stance6_results/complete/``, and the three
switch-off runs' ``provenance.json`` (their ``robust_best_model.zip``
``model_hash``, the file their verdicts judge).  The other checkpoint digests
are the six stance7 runs' fetched ``stage1_final.zip`` and periodic
``stage1_<steps>_steps.zip`` files; the stance6 runs' were never fetched, so a
load of one is caught by its path or its parent run id only.  Readers here
never raise on a missing or malformed file: an unreadable record decides
nothing, and the callers' own rules still apply to it.

The callers: :func:`environments.shared.replication.discover_replicates` and
``discover_replicates_for_run`` (an excluded sibling is skipped, an excluded
run counts no replicate), :mod:`environments.shared.ancestors` (reuse rule 9,
and the older-interface widening hint), ``scripts/widen_checkpoint.py`` (an
excluded parent is refused), the result-bundle audit (rule (a) on the
bundle's provenance, (b) and (d) on every stage config it hashes) and the
species catalog (an excluded run is never a promotable or committed result).
A later screen needs its own entry here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

#: The decision that keeps these runs out, cited by date (its record is §30 of
#: ``docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md``).
EXCLUSION_DECISION = "the maintainer's stance-landing decision of 2026-11-06"

#: Full commit -> the runs that trained at it.
EXCLUDED_COMMITS: Mapping[str, str] = {
    # The arm-LTSB screen branch (the module docstring names both branches).
    "8737cd3d86431de1bdd0fcdb49a0d956a45e084d": "the r10 stance screen runs stance6_r10LTSB_s42, _s60 and _s80",
    # The phase-critic screen branch.
    "1dfe2327f773007fa7fd68460dc0c53740936ff9": (
        "the r10 stance screen runs stance7_r10LTSBoff_s42, _s60 and _s80 and stance7_r10LTSBpc_s42, _s60 and _s80"
    ),
}

#: The nine screen runs by id: their ``provenance.json`` ``run_id`` and their run folders' names (rule (d) reads
#: them in a recorded load path and in a recorded parent run id).
EXCLUDED_RUN_IDS: frozenset[str] = frozenset(
    {
        "stance6_r10LTSB_s42",
        "stance6_r10LTSB_s60",
        "stance6_r10LTSB_s80",
        "stance7_r10LTSBoff_s42",
        "stance7_r10LTSBoff_s60",
        "stance7_r10LTSBoff_s80",
        "stance7_r10LTSBpc_s42",
        "stance7_r10LTSBpc_s60",
        "stance7_r10LTSBpc_s80",
    }
)

#: The screens' handoff checkpoints (``robust_best_model.zip``), ``sha256:``-prefixed -> the run that wrote it.
EXCLUDED_HANDOFF_SHA256: Mapping[str, str] = {
    "sha256:77959e30f4f20e1a3bb184d210b4430a14b6df1a046cdbc2e5b279487fbfc299": "stance7_r10LTSBpc_s42",
    "sha256:92ac93e518ccfd4bbfdc15a25817477517f58463e544065a12b0e12d1e9591ad": "stance7_r10LTSBpc_s60",
    "sha256:b03e12823f9003ee8ad9f73a7ef485ab8f9be4ee23946c395f7f0f333abe44f5": "stance7_r10LTSBpc_s80",
    "sha256:034a105f136dcef45381cd2948cbb1390133578914739d94ee780b1f9a426163": "stance7_r10LTSBoff_s42",
    "sha256:47ee71dcda386f5f833ff500054976d288a95ecc3e0e2845d09a4ce655e9bd1c": "stance7_r10LTSBoff_s60",
    "sha256:22a61b91daf0265938980bd4be550720a4de802d198bcf07e9c81f9efc547beb": "stance7_r10LTSBoff_s80",
    "sha256:ba5e46a9fb20f4edeaba05e8070f0bd868146a02a6eb0dd2866105bc09846866": "stance6_r10LTSB_s42",
    "sha256:23d9b74a5d3e39123cd8d32fa6d1e20c3b27203d33271ed5068b8e80b699f3ce": "stance6_r10LTSB_s60",
    "sha256:55ca16f498ccd7f4bc017459098c1e271fc4dfa9e3a65cc8be2ec7a721dbaba6": "stance6_r10LTSB_s80",
}

#: The six stance7 runs' other fetched checkpoints (``01_stance/models/``): run -> file name -> ``sha256:`` digest.
_SCREEN_CHECKPOINTS: Mapping[str, Mapping[str, str]] = {
    "stance7_r10LTSBpc_s42": {
        "stage1_1048576_steps.zip": "sha256:bec9648edbff799e769de3ae6fe8ef0427c1cb9ca526afa7cb5c6c39b5647597",
        "stage1_2097152_steps.zip": "sha256:901e664be945616c7468595d4e35cdbfcfc532abe3cde606589ea8707d958fd6",
        "stage1_3145728_steps.zip": "sha256:3e6309ca98dd903ab4c55ebc0b18d7c6ae08d931be70a27ff863130cca4e71bb",
        "stage1_4194304_steps.zip": "sha256:fe7c590e34fc3727143fb59d17501f4310e264a8acb1148999d723e947e40338",
        "stage1_5242880_steps.zip": "sha256:f829adb18a3e8175acb92d65383c108235194efcc728ffb2c1a07bda393123a9",
        "stage1_6291456_steps.zip": "sha256:22222afedda98bfb351c147f6c25512d3dc1e88a04753d03ee2e1e07bb0f1fd8",
        "stage1_7340032_steps.zip": "sha256:eaac816f606564f38a3378d6949b7faa3e542718f29901389cb7853285f79023",
        "stage1_8388608_steps.zip": "sha256:d10887578038b712b52d631250be76dddc619008e57113cfbcbef4930d4fbe37",
        "stage1_9437184_steps.zip": "sha256:a8ac5e909a4d48cc0323834ee5196882d3d955e975f9dfc14d71773bae792164",
        "stage1_10485760_steps.zip": "sha256:c2feed2db50fd51a5cfa19ba057863d4c485c1ed92cf4e9eebe300df919c1e6c",
        "stage1_11534336_steps.zip": "sha256:179692977e6d9bfbd6bec07740c361f68643fc3f7d6fd4b2ef6cb5da9b6b0cee",
        "stage1_final.zip": "sha256:1c9d87c3d0f817b048de429bb80cdc89fcd2c83d5fbafd7f444af0b5474b9ff6",
    },
    "stance7_r10LTSBpc_s60": {
        "stage1_1048576_steps.zip": "sha256:ba10c90a3e7465782f32cbc90aca2cabe4504459848be7add6b13c60bd978a6f",
        "stage1_2097152_steps.zip": "sha256:83a159274355d361739377759c639c06c8c432fe7779aa119295c09bce314abe",
        "stage1_3145728_steps.zip": "sha256:16a84e59271357922d774bca8a9c5bfb55b5d4d44319a721c3b0254e6aed0bc8",
        "stage1_4194304_steps.zip": "sha256:25949ff5cb033c076ecae9564917f04fadf3ed5e568035db13716c9b0dd3d745",
        "stage1_5242880_steps.zip": "sha256:cddccb0ddfc6a7720fa66cca39cd2de9ea043e4a2d6e2ae30175af8e36fafc7a",
        "stage1_6291456_steps.zip": "sha256:d686013f946d7958f274fb20420b309352767f42b0d3308cb71c6c96732a0c36",
        "stage1_7340032_steps.zip": "sha256:8477dbf982a936717556fb730de2114454ad4c94e1bb91177e16399f584983fa",
        "stage1_8388608_steps.zip": "sha256:7a87bdae53981aaaaaf6573adf2c21db5f0c451faeaf0f6bb3c8fbc57814f00f",
        "stage1_9437184_steps.zip": "sha256:c8cb977ce068e38ad803ab02135485a6cb6e10bcc7e8ebe2afab9801634aba85",
        "stage1_10485760_steps.zip": "sha256:b273e4dae302567f82915ddb293893d481b3fc63e7bcd3ce08c75eca6fe6062e",
        "stage1_11534336_steps.zip": "sha256:738c54eaf5a43332ffdbbe87652d55b7f1c28c2487be36126857ae3769b69f20",
        "stage1_final.zip": "sha256:05b1638f18b130aad16dc45607809de1992c4c3b07fea1c32897424dc9ef03fe",
    },
    "stance7_r10LTSBpc_s80": {
        "stage1_1048576_steps.zip": "sha256:54f3a3d2bc9aa4811cfcb0667e17cd3bed3782693298746c199dfdd047c43832",
        "stage1_2097152_steps.zip": "sha256:7d3b8522db4da72e036d4a79d1a653368a6c6a0dc3004250ab914bd6588c4772",
        "stage1_3145728_steps.zip": "sha256:98b9a7e687ebecf6fbb8da30195d7c1852fa233ff71b7fca29b7522963cfad9c",
        "stage1_4194304_steps.zip": "sha256:48e432fd469ca8a16b47e7fbd6b76a97b2eb38543e33a371d355e8f911887e77",
        "stage1_5242880_steps.zip": "sha256:70cff30a2ccbcc6c13a8d3996803ec0d1972da97318f482cda07ec6efd527fb5",
        "stage1_6291456_steps.zip": "sha256:d64b48b9c83369ee79c12e207ab79c527244619a123b14d16c543a3d83ba6d10",
        "stage1_7340032_steps.zip": "sha256:5de5a13b358b7ad4b50d4977fbcea63bae549badcaf04e82a092fa52940a499a",
        "stage1_8388608_steps.zip": "sha256:9cf39f4cd3a0ce3deb5784ca914f9ffc92e5229f1d689b194175cb564bbdf2c5",
        "stage1_9437184_steps.zip": "sha256:e5e4d39854c0cd552ecd25105fccfc7acee22c320a4f2c89f1554a46042b762a",
        "stage1_10485760_steps.zip": "sha256:f9063ae0a0943d6b56ea5831e392f9f5ddb40b210614f4aa3ba135497cb095a8",
        "stage1_11534336_steps.zip": "sha256:397d1f83741c8990b0bb744794ecef64186c7445fe3d971ede32651bf9aa44b2",
        "stage1_final.zip": "sha256:1a4fef1f99722417d089819094acebece54e2c020b7253d4c285d6bd16e21df4",
    },
    "stance7_r10LTSBoff_s42": {
        "stage1_1048576_steps.zip": "sha256:9d06988f9ce9c38da9d316a42ce994da6673bc77c8c75034dc157db78693c9cb",
        "stage1_2097152_steps.zip": "sha256:97b4c488a72792ad79cd86da6b9ed2e0bc08c61806af957e534234ec530b15fb",
        "stage1_3145728_steps.zip": "sha256:35364ca572caee8019254044713ce67c6fd2b33b1b7e0e64bfc45c1fb432fd24",
        "stage1_3670016_steps.zip": "sha256:3fe3c7520fe304a223af725f9f9827ae7e621d54b401e48d35979cd5831f95e6",
        "stage1_4194304_steps.zip": "sha256:e7ddffb0c60b24405708395c679708990cf37f621240431e90b8a364d0488068",
        "stage1_5242880_steps.zip": "sha256:b38383d9f7eb86a2830bc3a106d9097b4b1a6376d583b1ca1a9277e9fa78e3e7",
        "stage1_6291456_steps.zip": "sha256:5627f0656129e6fdfad3ba28582fb1c2ffa1db25ba4bd65a3b97fd0d917350cf",
        "stage1_7340032_steps.zip": "sha256:ebaf648fce3257c5a745ca2ad5853e19a04216904d4ddc6e35dd3d5e9e9fe682",
        "stage1_8388608_steps.zip": "sha256:3853505233d37840e4a0768d8ab87d24df8128e347bbc27191d8b0f66a54e32f",
        "stage1_9437184_steps.zip": "sha256:be3470789ad19f3c9b92d40027e46c85760df953bb25fed068d4a1573beafb96",
        "stage1_10485760_steps.zip": "sha256:347a43d4b4f3c1e4f2171f718f428b5359d5955135ad567332d0c2ebfc68d26f",
        "stage1_11534336_steps.zip": "sha256:d14f0430d83c3effa2f6fe5c271d50f6bd6dfa6fc651e54d0c35ce856b7b4bb4",
        "stage1_final.zip": "sha256:3712ecaf080488e4397b1639119e360cc07a09b089a2166ad6e9c0cfa3286e57",
    },
    "stance7_r10LTSBoff_s60": {
        "stage1_1048576_steps.zip": "sha256:51ca2643e28306dd1d86a45b57e1f16e268d2d0b48dd2b87ee0702aa0dd5c219",
        "stage1_2097152_steps.zip": "sha256:5527f8b3e53ee6c1cfbd03e9a5d7e50424e67b7f95527bdcf558c932f52a0ed1",
        "stage1_3145728_steps.zip": "sha256:92a7f196e1a5a59cd12a5fc03e8f65f02e778bd5ae46ef1119ebef8be7a2cdd3",
        "stage1_4194304_steps.zip": "sha256:bb2d11c8c0214a6e437d4577f7a9a5dc24c6efcf8c34c9220167cb8ab2b5db4b",
        "stage1_5242880_steps.zip": "sha256:4968f60bb4512c72a6a46ac32566326ac8db810d42cf34d44690d3671071b601",
        "stage1_6291456_steps.zip": "sha256:f838e2b3cc19dd2f4865c6f49609ec8ebe3c5684dc9066c8ed136f96a26015d2",
        "stage1_7340032_steps.zip": "sha256:7100cbe2bff6ba74017ef162971590efaf7f6656af020f39bb75c85a0de55354",
        "stage1_8388608_steps.zip": "sha256:13d2c83ceb1c40aba97ed816edd0f552f3ad36d6d261b103d1c9ffa7191772a8",
        "stage1_9437184_steps.zip": "sha256:5d4e55940553981188ac0a5c4d032700df28595efec051ec127090687f791179",
        "stage1_10485760_steps.zip": "sha256:21e782e5b9a5a10f952044cc7234412710961846f08593a851bf0adaf79327e1",
        "stage1_11534336_steps.zip": "sha256:f3bdea5e3e8ab4d42174b596b97891b1c98bc55e63eb1afb17319d41ed569ce8",
        "stage1_final.zip": "sha256:9e5431b3ad0196073033f5a0881f05a082de00d946b220a63674925e7d148333",
    },
    "stance7_r10LTSBoff_s80": {
        "stage1_1048576_steps.zip": "sha256:528e13b29aae9023c5fb7d2a1175a99505d74953523433279637ec890acbd124",
        "stage1_2097152_steps.zip": "sha256:5ff7e229533f6c1f7fb5524aeb7ef979f2a81617128711a3a70706ab6b41f329",
        "stage1_3145728_steps.zip": "sha256:b6bba4853296d7e3abd64d67255c676800bee3ef90528bb7ef0fc2da0fab6071",
        "stage1_4194304_steps.zip": "sha256:3d07cfcd8b43edaeee8e2fc336d4442df206911f0ff8a0a4818c1985cbb88151",
        "stage1_5242880_steps.zip": "sha256:101b7c428a2c50342c3fca13c41f4177054e11349b310ca846d25101535dd725",
        "stage1_6291456_steps.zip": "sha256:5220c2cc3dbf80842b02254fed9899ee47ed29ad666b916dbebb2328efc3eb6f",
        "stage1_7340032_steps.zip": "sha256:6891faf65c38816a61e869d170728361f8930aff0445c709e3703b23fd0e310e",
        "stage1_8388608_steps.zip": "sha256:9354ee01e999866e1f76a1ece00bcfcc896e17a88b19ac3fa0cbed16311079dc",
        "stage1_9437184_steps.zip": "sha256:5fd3f808061506400e8d1816a82cc5fa994293c75569054149ab9e7de3b4b298",
        "stage1_10485760_steps.zip": "sha256:e6395fc3060d0fa26ce2eb16c7d8aba26d937b33a39e9a7dca55c7a4007478f6",
        "stage1_11534336_steps.zip": "sha256:83d22b778948b719aa2b30f53f2eaa41022b88f517a06899798c72564035f16d",
        "stage1_final.zip": "sha256:f002647b4fd54b4a705e8bc2117d305957445402daccd308a9693c0fddb6dfe6",
    },
}

#: Those checkpoints flattened, ``sha256:``-prefixed -> ``<run>/<file>``.
EXCLUDED_CHECKPOINT_SHA256: Mapping[str, str] = {
    digest: f"{run}/{name}" for run, files in _SCREEN_CHECKPOINTS.items() for name, digest in files.items()
}

#: Rule (d)'s ``run``-block keys, a subset of ``config.LOAD_LINEAGE_KEYS``, ``RESUME_LINEAGE_KEYS`` and
#: ``WIDEN_LINEAGE_KEYS``: the digests of the file a stage loaded, the paths as given to the trainer, the run ids.
LINEAGE_DIGEST_KEYS = ("parent_checkpoint_sha256", "resume_checkpoint_sha256", "widened_from_checkpoint_sha256")
LINEAGE_PATH_KEYS = ("load_path", "resume_load_path", "widened_from_path")
LINEAGE_RUN_ID_KEYS = ("parent_run_id", "widened_from_run_id")

_SUFFIX = (
    f"; branch screen runs never count as certified, as replicates or as reusable ancestors ({EXCLUSION_DECISION})"
)


def excluded_commit_reason(commit: Any) -> str | None:
    """Why a run recorded at *commit* never counts, or None (any non-string, or a commit not listed)."""
    if not isinstance(commit, str):
        return None
    runs = EXCLUDED_COMMITS.get(commit.strip().lower())
    return None if runs is None else f"it was trained at commit {commit.strip()}, {runs}{_SUFFIX}"


def _normalized_sha256(digest: Any) -> str | None:
    if not isinstance(digest, str) or not digest.strip():
        return None
    value = digest.strip().lower()
    return value if value.startswith("sha256:") else f"sha256:{value}"


def excluded_handoff_reason(digest: Any) -> str | None:
    """Why a checkpoint hashing to *digest* (with or without the ``sha256:`` prefix) never counts, or None."""
    value = _normalized_sha256(digest)
    run = EXCLUDED_HANDOFF_SHA256.get(value) if value is not None else None
    return None if run is None else f"its handoff checkpoint {value} is the r10 stance screen run {run}'s{_SUFFIX}"


def excluded_checkpoint_reason(digest: Any) -> str | None:
    """Why a checkpoint hashing to *digest* never counts: a screen's handoff, or another checkpoint of a screen."""
    reason = excluded_handoff_reason(digest)
    if reason is not None:
        return reason
    value = _normalized_sha256(digest)
    name = EXCLUDED_CHECKPOINT_SHA256.get(value) if value is not None else None
    return None if name is None else f"the checkpoint {value} is the r10 stance screen run checkpoint {name}{_SUFFIX}"


def excluded_run_id_reason(run_id: Any) -> str | None:
    """Why a record naming *run_id* as its parent never counts: the id is a screen run's, or None."""
    if not isinstance(run_id, str) or run_id.strip() not in EXCLUDED_RUN_IDS:
        return None
    return f"it names the r10 stance screen run {run_id.strip()}{_SUFFIX}"


def excluded_path_reason(path: Any) -> str | None:
    """Why a checkpoint at *path* never counts: one of its components is a screen run's folder, or None."""
    if not isinstance(path, str):
        return None
    for part in re.split(r"[\\/]+", path):
        if part in EXCLUDED_RUN_IDS:
            return f"the path {path} is inside the r10 stance screen run {part}'s folder{_SUFFIX}"
    return None


def excluded_lineage_reason(run_block: Any) -> str | None:
    """Rule (d) on a ``stage_config.json`` ``run`` block: the stage loaded, resumed or widened a screen's checkpoint."""
    if not isinstance(run_block, Mapping):
        return None
    for keys, read in (
        (LINEAGE_DIGEST_KEYS, excluded_checkpoint_reason),
        (LINEAGE_PATH_KEYS, excluded_path_reason),
        (LINEAGE_RUN_ID_KEYS, excluded_run_id_reason),
    ):
        for key in keys:
            reason = read(run_block.get(key))
            if reason is not None:
                return f"run.{key}: {reason}"
    return None


def excluded_stage_config_reason(stage_config: Any) -> str | None:
    """Rules (b) and (d) on a parsed ``stage_config.json``: its recorded commit, or a load from a screen."""
    if not isinstance(stage_config, Mapping):
        return None
    reason = excluded_commit_reason(stage_config.get("git_commit"))
    if reason is not None:
        return reason
    return excluded_lineage_reason(stage_config.get("run"))


def _read_object(path: Path) -> Mapping[str, Any] | None:
    from .file_io import read_json_object

    return read_json_object(path)  # None for a missing, unreadable or non-object file


def excluded_stage_reason(stage_dir: "str | Path") -> str | None:
    """Rules (b), (c) and (d) on one stage directory: its recorded commit, its verdict's handoff, its load lineage."""
    path = Path(stage_dir)
    stage_config = _read_object(path / "stage_config.json")
    reason = excluded_commit_reason(stage_config.get("git_commit")) if stage_config is not None else None
    if reason is not None:
        return f"{path.name}/stage_config.json: {reason}"
    verdict = _read_object(path / "gate_verdict.json")
    if verdict is not None:
        reason = excluded_handoff_reason(verdict.get("checkpoint_sha256"))
        if reason is not None:
            return f"{path.name}/gate_verdict.json: {reason}"
    reason = excluded_lineage_reason(stage_config.get("run")) if stage_config is not None else None
    if reason is not None:
        return f"{path.name}/stage_config.json {reason}"
    return None


def excluded_run_reason(run_dir: "str | Path", stage_dir: "str | Path | None" = None) -> str | None:
    """Why the run at *run_dir* never counts, or None.

    Rule (a) on its ``provenance.json``; rules (b), (c) and (d) on
    *stage_dir* when given, else on every immediate subdirectory of
    *run_dir* (every stage-directory layout generation is one level down).
    """
    from .result_bundle import DEFAULT_PROVENANCE_NAME

    path = Path(run_dir)
    provenance = _read_object(path / DEFAULT_PROVENANCE_NAME)
    if provenance is not None:
        reason = excluded_commit_reason(provenance.get("repository_commit"))
        if reason is not None:
            return f"{DEFAULT_PROVENANCE_NAME}: {reason}"
    if stage_dir is not None:
        return excluded_stage_reason(stage_dir)
    try:
        children = sorted(child for child in path.iterdir() if child.is_dir())
    except OSError:
        return None
    for child in children:
        reason = excluded_stage_reason(child)
        if reason is not None:
            return reason
    return None
