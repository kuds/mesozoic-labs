# How `STAGE2_75_REPORT.md` and its owner page are put together

The report's prose is `template.md`; its tables are filled by `assemble.py` from this run's digest
(`../digest/DIGEST.md`, `../digest/digest.json`) and recount (`../rereads/joint_range_aside.json`), so that no table
cell is typed by hand. `turn_sections.py` writes the turning-room texts (`TURN_ANSWER.md`, `TURN_SECTION.md`) from
`../views/turnpath/<set>/turnpath.json` and `../views/clips.json`. `owner_page.py` builds the owner page from the
report's own sections and inlines the clips (media under `logs/barviews/`, not in git; made with
`python -m designlab.stage2.barviews clips ...`).

From `hardware/trex_form_study`:

    python stage2/runs/2026-11-15-75cm/digest/digest.py
    python stage2/runs/2026-11-15-75cm/rereads/term_aside.py
    python stage2/runs/2026-11-15-75cm/report_src/turn_sections.py
    python stage2/runs/2026-11-15-75cm/report_src/assemble.py            # writes STAGE2_75_REPORT.md
    python stage2/runs/2026-11-15-75cm/report_src/owner_page.py OUT.html  # needs pandoc and ffmpeg, and the clips' media

Nothing here is in a record's key. The page was published as a private artifact on 16 Nov 2026.
