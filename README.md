# End-of-day safety copy, 2026-10-30 (never merged)

The T. rex tail-habit research's work that existed only in a session container when development stopped for the
day, pushed at the maintainer's word so that a container reset cannot lose it. This orphan branch holds no repository
code and is never merged or opened as a pull request.

- why_probe/: the no-training probe of why the critic values a raised tail, with each lens's pre-registration
  (PREREG.txt, hashed before data), scripts, outputs and notes, and the independent verifier's adjudication: the phase
  proxy (A) is favoured strongly; a genuine tail-floor termination risk (B) is excluded. The rollout arrays (npz) are
  left out; the scripts that made them are kept. SKIPPED.txt lists any text file left out for size (none).
- run_log.txt: the session's run log as it stood when this copy was made (the state to resume is its last entry).
- long_blind/: the paused long critic-blind study (from s42's 2,097,152 to about 12M): the harness with crash resume,
  its tests and validations' logs, the end-of-run measurement tool built on the r10 round's pinned tools, the draft
  criteria, and the partial build check. Not reviewed or approved; nothing was trained.
- records_pr_patches/: the 7 unreviewed commits of the records PR (branch claude/records-critic-blind on main
  358651fe, not pushed), as git format-patch files; apply with git am on 358651fe. They have not been through the
  checkers or the fixer.

Built 2026-10-30 18:35 UTC. FILELIST.txt and SHA256SUMS cover every file (sha256sum -c SHA256SUMS).
