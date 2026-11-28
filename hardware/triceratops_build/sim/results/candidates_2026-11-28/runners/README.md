# Runners of the candidate experiments E1-E5 (28 Nov 2026)

These are the scripts that ran, summarised and reviewed experiments E1-E5 of
`design/PREREG_CANDIDATES_2026_11_28.md`. They were copied here byte for byte from the session's scratch folder
after the runs, so that the record outlives that folder. Each result file's `provenance` block names its runner by
the scratch path it ran from and gives its sha256; those hashes match these copies (checked 28 Nov 2026). The
scripts use the build's frozen shared code, `sim/candidate_checks.py`.

| Folder | Scripts |
|---|---|
| `E1/` | `run_e1.py` (episodes and clearance), `summarise_e1.py`, `md_tables_e1.py`, `beyond_cap_e1.py` (supporting, after a result) |
| `E2/` | `e2_run.py` (E2(a) and E2(b)), `e2_checks.py`, `check_repro.py`, `e2_tables.py`, `e2_review.py`, `e2a_x6_beside.py` and `e2b_vs_R_supporting.py` (supporting, after a result) |
| `E3/` | `run_e3.py` (demo courses through the build's thermal model) |
| `E4/` | `run_e4.py` (the forced-air source test), `run_e4_review.py` |
| `E5/` | `run_e5.py` (tuning and held-out runs), `harness_e5.py`, `record_e5.py`, `md_e5.py`, `check_push_applied.py` |
| `E5_pitch0/` | the pitch-0 branch, changed after a result: `make_runner.py`, `run_e5_pitch0.py`, `record_p0.py`, `md_p0.py`, `review_checks.py` |

They read and write absolute scratch paths (`TRIKE_SCRATCH` sets the shared code's), and the policy runs need the
checkpoints there, so a rerun needs those paths or an edit. They are records, not maintained tools: nothing else in
the build imports them, and no test or CI job runs them.

**Line ends.** Python's `csv` module wrote the CSV files here with CRLF line ends. The repository's `.gitattributes`
(`* text=auto eol=lf`) stores and checks them out with LF, so the sha256 that a review addendum records for a CSV
(`E2/e2_review_addendum.json`, `E4/e4_review_addendum.json`) is of the CRLF bytes. Turning a checkout's LF back into
CRLF reproduces it (checked for all four recorded CSVs, 28 Nov 2026). No CSV has a bare LF.
