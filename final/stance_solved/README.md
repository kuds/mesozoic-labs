# Is the T. rex stance solved? (checked review, 2026-11-02)

The maintainer asked, after the phase-critic round: "would you consider the stance behavior solved? ... we now have a
quality model that can stand, that meets our criteria across multiple seeds. Does it make sense now moving on to
additional behaviors?"

A read-only review answered it on 2026-11-02, 20:41-21:18 UTC: four evidence agents, one adversarial verifier
per evidence note, and one synthesis. Nothing was trained, replayed or fetched; no Drive, network or git write.

**Read `synth/ANSWER.md` first.** It is the checked answer: it uses each claim only as the verifiers left it.

| Folder | What it is |
|---|---|
| `synth/ANSWER.md` | The checked answer (`synth/NOTE.md` is an identical copy). |
| `criteria/` | Every written stance criterion, marked met / not met / not measured for the three PHASE finals. |
| `skeptic/` | The strongest case that the stance is not yet a quality stance or a walking parent, weighed. |
| `readiness/` | What moving on to walking requires: plan rules, warm-start code, recipes, a hidden-clock audit. |
| `strength/` | How strong the evidence across seeds is. |
| `verify/<slice>/` | Each verifier's `VERIFY.md` and its own check scripts and outputs. |
| `VERDICTS.json` | The structured verdicts: every claim the verifiers judged WRONG, OVERCLAIM or UNDERCLAIM, with the correction, and the points each note missed. |
| `RUNLOG_TAIL.txt` | Run-log lines 973 onward: the events after this branch's `run_log.txt` copy, which ends at line 972. |

**Caution.** The evidence notes (`criteria/`, `skeptic/`, `readiness/`, `strength/` `NOTE.md`) are as written, before
verification, and contain claims the verifiers corrected. Two examples: the skeptic note says the maintainer has
not ruled on head height, but D-D46 made head posture report-only; the strength note gives "roughly 3 in 4", which
is false precision. Check `VERDICTS.json` before citing a note. (`verify/<slice>/NOTE.md` is the verifier's working
copy of the note it checked.)

Left out of this copy: copies of repository files that the agents took for reading (main's and the run branch's
configs, code and plan docs; read them from the branches themselves), the curriculum extracts `*.tmp`, and empty
stderr files.

One process slip: the criteria verifier's own self-check grep pattern contained the null-device path string (a
search pattern, not a redirect). No agent wrote outside its own folder; the phase-critic worktree is unchanged
at 1dfe2327.
