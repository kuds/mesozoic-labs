# Runners of the candidate experiments E1-E5 (28 Nov 2026)

These are the scripts that ran, summarised and reviewed experiments E1-E5 of
`design/PREREG_CANDIDATES_2026_11_28.md`. They were copied here byte for byte from the session's scratch folder
after the runs, so that the record outlives that folder (each copy byte-identical to its original, checked
28 Nov 2026 by `../lead_check/`). Each result file's `provenance` block names its runner by the scratch path it ran from and gives its
sha256. For 19 of the 24 copies a result file records a sha256, and each matches (`../lead_check/`). The other five
have no recorded hash; their sha256, from these copies:

| Script | sha256 |
|---|---|
| `E1/md_tables_e1.py` | `df773067d6f62477b822d72bd94d7e9fe6fef97183ea4bfe7c4d6dbbbf889d47` |
| `E5/check_push_applied.py` | `3d3eca7d5a5a7007ef453ca93f5883c5bb1d9a3743afeb2d0715006c12327c0b` |
| `E5/md_e5.py` | `0391383f08c9b2fb8662d6de6e5cac4c20a5f7371cc09c9b35f5d0949fe1015c` |
| `E5_pitch0/make_runner.py` | `1b29f705bd7b505767b48d61633d470d4834752e98ef6fb3da28bf7ebe67dd1d` |
| `E5_pitch0/md_p0.py` | `62fc68d33f63aa6f1fa2d2948c31a8e42d99ade70dc5f9a74397fa676e1e9b3c` |

The scripts use the build's frozen shared code, `sim/candidate_checks.py`.

| Folder | Scripts |
|---|---|
| `E1/` | `run_e1.py` (episodes and clearance), `summarise_e1.py`, `md_tables_e1.py`, `beyond_cap_e1.py` (supporting, after a result) |
| `E2/` | `e2_run.py` (E2(a) and E2(b)), `e2_checks.py`, `check_repro.py`, `e2_tables.py`, `e2_review.py`, `e2a_x6_beside.py` and `e2b_vs_R_supporting.py` (supporting, after a result) |
| `E3/` | `run_e3.py` (demo courses through the build's thermal model) |
| `E4/` | `run_e4.py` (the forced-air source test), `run_e4_review.py` |
| `E5/` | `run_e5.py` (tuning and held-out runs), `harness_e5.py`, `record_e5.py`, `md_e5.py`, `check_push_applied.py` (the push check, after a result) |
| `E5_pitch0/` | the pitch-0 branch, changed after a result: `make_runner.py`, `run_e5_pitch0.py`, `record_p0.py`, `md_p0.py`, `review_checks.py` |

They read and write absolute scratch paths (`TRIKE_SCRATCH` sets the shared code's), and the policy runs need the
checkpoints there, so a rerun needs those paths or an edit. They are records, not maintained tools: nothing else in
the build imports them, and no test or CI job runs them.

**Line ends.** Python's `csv` module wrote the CSV files here with CRLF line ends. The repository's `.gitattributes`
(`* text=auto eol=lf`) stores and checks them out with LF, so the sha256 that a review addendum records for a CSV
(`E2/e2_review_addendum.json`, `E4/e4_review_addendum.json`) is of the CRLF bytes. Turning a checkout's LF back into
CRLF reproduces it (checked for all four recorded CSVs, 28 Nov 2026). No CSV has a bare LF.
