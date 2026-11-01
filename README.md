# Phase-critic Colab round (PC) pre-registration: durable copy (never merged)

The pre-registration of the T. rex r10 LTSB stance round with a critic-only episode phase input ([ppo] critic_phase,
notebook CRITIC_PHASE) on branch claude/screen-trex-r10-phasecritic (pushed; never merged): three PHASE runs and three
concurrent switch-OFF runs (seeds 60, 42, 80; 12M steps each on Colab L4s), the one test PHASE vs OFF (tail drive,
exact 3-vs-3 permutation, LOWER only at 1/20), STANCE6 report only. Approved by the maintainer on 2026-11-01 (run_log.txt).
This orphan branch holds the text so the results stay tied to their pinned criteria. It holds no repository code.

- round/: criteria.txt (the approved text, byte for byte) and APPROVED.
- criteria.txt, COLAB_STEPS.txt: the installed texts; design/: the design notes and analyses.
- tools/: the measurement port, scorer, pins and their tests; build/, check_code/, adapt/: the code build, the
  independent code check and the measurement adapter; review/: the reviews, fixes and re-review; amend/: the
  six-run amendment, its re-review, fixes and second re-review; pin/: the pin-step validations on the approved HEAD.
- run_log.txt: the session's run log as it stood when this copy was made. SKIPPED: 0 files over 5 MB.

Built 2026-11-01 03:07 UTC. FILELIST.txt and SHA256SUMS cover every file (sha256sum -c SHA256SUMS).

Redaction: in 7 staged copies (review/fixer/commit_msg_m1.txt, build/NOTES.txt, build/scratch/msg1-5.txt) an AI model product name in draft commit trailers was replaced by "[name removed]"; the originals stay in the container. The pushed branch commits carry "Co-Authored-By: Claude <noreply@anthropic.com>" (reworded before the push; trees unchanged).
