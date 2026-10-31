"""Fixer: criteria.txt edits, batch 3 (the fixer's re-validation and the disk measured at the end)."""
from pathlib import Path

p = Path("/home/user/work/phase_screen/criteria.txt")
s = p.read_text()
R = []
R.append(("""  V2's records hold no terminal observation (fixer, design D4).""",
"""  V2's records hold no terminal observation (fixer, design D4).
  After the fixer's run_bs.py change (P-B1, P-i3) V1 (both arms, 9401), V2 (control and critic_phase at 9401 and 9411,
  no update, to 2,359,296) and V3 were re-run into <PS>/validate/fixer by <PS>/review/fixer/bin/fixer_validate.sh
  [measured, <PS>/review/fixer/logs/val/ and logs/fixer_compare_2.out]: the builder's v1_independent.py, v2_check.py
  and v3_check.py all PASS on them; against the builder's runs, V1's saved pairs are equal (the zip's tensors and
  data; the pkl byte for byte), and every V2 run's per-rollout evidence (31 rollouts), digests at load, start and
  end, ledger, probe file, CHECK and loop markers at 2,097,152 and 2,359,296 are equal, the saved 2,359,296 zip
  differing only in data.tensorboard_log; each run's events file is under <PS>/tmp/no_update_<run>/tb/PPO_0 and none
  was written under /tmp/tb_buffer or anywhere outside <PS> by these runs; critic_phase's phase_state_end holds (the
  actor's phase columns unchanged, their Adam moments 0). V5 was not re-run: bs_hooks.py and the arm are unchanged."""))
R.append(("""- Disk: free space on / was 2.0 GB at 12:57 UTC today [measured, df -h /, WORK/CMDLOG.txt] and 1.8 GB at 13:57 UTC
  [measured, df -h /, WORK/ADAPT_NOTES.txt]: BELOW the launch rule.""",
"""- Disk: free space on / was 2.0 GB at 12:57 UTC today [measured, df -h /, WORK/CMDLOG.txt] and 1.8 GB at 13:57 UTC
  [measured, df -h /, WORK/ADAPT_NOTES.txt]: BELOW the launch rule; 1,597,640,704 B at 15:30 UTC, below even the
  per-start floor of 1.5 GiB [measured, df -B1 /, <PS>/review/fixer/notes.txt] (the fixer's re-validation under
  <PS>/validate/fixer holds 112.6 MB, removable after the re-review; other lanes' folders grew meanwhile): about
  1.2 GB must be freed for the 2.6 GiB rule (process review P-m1)."""))
for old, new in R:
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)
p.write_text(s)
print("applied", len(R))
