# Provenance of this folder (round-zero posture study)

This folder preserves the body-shape and posture study supplied by the project owner on
2026-11-07 as `T_rex_body_study.zip`. It had not been pushed to any branch: PR #664 was titled as
this study, but its head commit (`b5c18bce…`) is the same as PR #663's and contains none of these
files. The archive is the "newer local or unpushed posture study" that the reconciliation
looked for.

| Item | Value |
|---|---|
| Archive | `T_rex_body_study.zip`, 26,776,412 bytes |
| Archive SHA-256 | `671b6b6884ab8b79a2e950a2106a6d142938297e8956db2a420963c9e7b4fd95` |
| Internal checksums | All 102 entries of `SHA256SUMS` verified (`shasum -a 256 -c SHA256SUMS`) on the extracted archive |
| Pinned inputs | `source/pr662/model.xml` and `components.json` are byte-identical to PR #662 head `e103f7ff…` (Git blobs `43152d1c…` and `5da76a30…`) |
| PR #663 copies | `source/pr663/FINDINGS.md`, `Partner-brief.pdf` and `visual/redesign-at-60cm.png` are byte-identical to PR #663 head `b5c18bce…` |
| Author | Not recorded in the archive. Its README links the original R3 handoff to a private ChatGPT library file that cannot be opened from this repository |

## What is here, and what is not

Every script, input, document, numeric result (JSON/CSV) and review plate (PNG) from the archive
is copied unchanged, at its original relative path. Three regenerable bulk outputs are left out:

* `study/model/generated/*.xml`: nine 1.6 MB MJCF models. They are rebuilt in ~8 s by
  `python reproduce.py --only-models`. Their per-candidate ledgers (`*.json`, with the original XML
  SHA-256 in `xml_sha256`) are kept.
* `study/visuals/Inspect-3D.html` (8.6 MB offline viewer): rebuilt by `study/visuals/viewer.py`.
* `T_rex_body_study.pdf` (7.5 MB review book): rebuilt by `python reproduce.py`.

`study/model/prior_geometry_v1.zip` (superseded models) is also not copied. The `*.zip` rule in
`.gitignore` excludes it, and the pipeline does not need it. It remains in the original archive.

## Reproduction check performed in this repository (2026-11-07)

Environment: Linux x86-64, Python 3.13.18, MuJoCo 3.10.0, numpy 2.5.4, scipy 1.18.1, pillow 12.3,
trimesh 5.2.0, matplotlib 3.11.2, reportlab 5.1.0. The checks ran on a fresh extraction, with
commands run from the extracted root.

* `python reproduce.py --only-models` regenerated all nine models in 7.7 s. Five XMLs (B00, B10,
  B20, C00, C10) are byte-identical to the archive. A00, A10, A20 and C20 differ: one or two cover
  panels are clipped differently at a keep-out boundary. The worst case is A20, at
  0.34 g of shell mass. All ledgers agree to within 1e-6 kg except those panels. This is a numerical
  platform difference. It does not change the design.
* `python reproduce.py` (full pipeline including screens, plates, viewer and PDF) completed in
  99 s with exit status 0. Every pass/fail outcome in `study/mechanics/SCREENING.md` is
  reproduced. Tabulated values differ by at most one unit in the last printed digit (for example A10
  peak hip roll 9.84 vs 9.89 N·m, B00 transfer tilt 12.9° vs 13.0°). The regenerated PDF is
  not byte-identical (fonts and embedded rasters).

The archive's own `VERIFICATION.md` reports byte-identical regeneration on its authors' platform.
This repository's check confirms numerical reproduction but not byte identity.

## How this study is used by the round-one study

The round-one study in `../designlab` does not import this code. It is an independent generator
covering four height classes. It reuses three things from here: the cited anatomical references
(`study/reference/references.md`), the hardware facts in `study/hardware/component_basis.md`, and the
single-support finding (≈ 9.7 N·m hip roll on the PR #662 geometry). That finding was re-derived
independently with `designlab.statics` on the same PR #662 model (see `../BASELINE.md`).
