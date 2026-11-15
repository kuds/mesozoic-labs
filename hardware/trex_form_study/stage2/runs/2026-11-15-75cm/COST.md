# Cost of the bar's episodes

Set `F25F9cl`, body `75-M-H-10`, block `seeds3042`, 20 episodes per record; made 2026-11-15T19:43:38Z by `designlab.stage2.bar dry-run` (it writes nothing; a smoke estimate, not a result).

Rates (wall seconds per simulated second, one process), measured on Stage 1's own committed condition `walk/brisk` of `75-M-H-10` (`stage1/runs/2026-11-08`) under each row type's row at 0 ms, after a warm-up episode that is not counted, on episodes run their full length, seeds [3042, 3043, 3044]; never on a reference of a Stage 2 set.  The machine's load at the time is in the block below: a rate measured beside other processes is an upper estimate, so a unit's estimate from it is no bound.

| Row type | Row measured | Wall s per simulated s |
|---|---|---|
| R2 | `R2/42@0ms` | 0.1574 |
| R0 | `R0@0ms` | 0.0954 |

```json
{
 "schema": "bar-cost-1",
 "rates": {
  "R2": {
   "wall_per_sim_s": 0.15740039675984782,
   "row": "R2/42@0ms",
   "wall_s": 9.725455714997224,
   "simulated_s": 61.78799999999839,
   "planned_s": 20.59599371365785,
   "episodes": [
    {
     "seed": 3042,
     "wall_s": 3.322732239998004,
     "simulated_s": 20.595999999999464,
     "fell": false
    },
    {
     "seed": 3043,
     "wall_s": 3.6487555479980074,
     "simulated_s": 20.595999999999464,
     "fell": false
    },
    {
     "seed": 3044,
     "wall_s": 2.753967927001213,
     "simulated_s": 20.595999999999464,
     "fell": false
    }
   ],
   "full_length": true,
   "warm_up": {
    "seed": 3042,
    "wall_s": 3.5156777980009792,
    "simulated_s": 20.595999999999464,
    "counted": false
   }
  },
  "R0": {
   "wall_per_sim_s": 0.09541879767919491,
   "row": "R0@0ms",
   "wall_s": 5.895736671001941,
   "simulated_s": 61.78799999999839,
   "planned_s": 20.59599371365785,
   "episodes": [
    {
     "seed": 3042,
     "wall_s": 1.9656845089994022,
     "simulated_s": 20.595999999999464,
     "fell": false
    },
    {
     "seed": 3043,
     "wall_s": 2.2773108430010325,
     "simulated_s": 20.595999999999464,
     "fell": false
    },
    {
     "seed": 3044,
     "wall_s": 1.6527413190015068,
     "simulated_s": 20.595999999999464,
     "fell": false
    }
   ],
   "full_length": true,
   "warm_up": {
    "seed": 3042,
    "wall_s": 1.5806745259978925,
    "simulated_s": 20.595999999999464,
    "counted": false
   }
  }
 },
 "measured_on": {
  "set": "S1",
  "stage1_run": "stage1/runs/2026-11-08",
  "body": "75-M-H-10",
  "condition": "walk/brisk",
  "cut_s": null,
  "seeds": [
   3042,
   3043,
   3044
  ],
  "delay_ms": 0,
  "warm_up": "one episode per row type before the counted ones; not counted",
  "note": "Stage 1's own committed condition, never a reference of a Stage 2 set"
 },
 "load": {
  "before": {
   "load_average": [
    0.341796875,
    0.8984375,
    1.48486328125
   ],
   "at": "2026-11-15T19:43:16Z",
   "runnable": 3,
   "processes": 114
  },
  "after": {
   "load_average": [
    1.09716796875,
    1.03564453125,
    1.51806640625
   ],
   "at": "2026-11-15T19:43:38Z",
   "runnable": 3,
   "processes": 116
  },
  "cpu_count": 4,
  "note": "the machine's load while the rates were measured; a rate measured beside other processes is an upper estimate"
 },
 "at": "2026-11-15T19:43:38Z"
}
```

| Row | Condition | Episodes | Planned s each | Core-hours | Status |
|---|---|---|---|---|---|
| `R2/42@10ms` | `walk/v0.4` | 20 | 24.38 | 0.0213 | to run |
| `R2/43@10ms` | `walk/v0.4` | 20 | 24.38 | 0.0213 | to run |
| `R0@10ms` | `walk/v0.4` | 20 | 24.38 | 0.0129 | to run |
| `R2/42@10ms` | `walk/v0.4_mu0.54` | 20 | 24.38 | 0.0213 | to run |
| `R2/43@10ms` | `walk/v0.4_mu0.54` | 20 | 24.38 | 0.0213 | to run |
| `R0@10ms` | `walk/v0.4_mu0.54` | 20 | 24.38 | 0.0129 | to run |
| `R2/42@10ms` | `turn/q90_left` | 20 | 23.90 | 0.0209 | to run |
| `R2/43@10ms` | `turn/q90_left` | 20 | 23.90 | 0.0209 | to run |
| `R0@10ms` | `turn/q90_left` | 20 | 23.90 | 0.0127 | to run |
| `R2/42@10ms` | `turn/q90_left_mu0.54` | 20 | 23.90 | 0.0209 | to run |
| `R2/43@10ms` | `turn/q90_left_mu0.54` | 20 | 23.90 | 0.0209 | to run |
| `R0@10ms` | `turn/q90_left_mu0.54` | 20 | 23.90 | 0.0127 | to run |
| `R2/42@10ms` | `turn/q90_right` | 20 | 24.39 | 0.0213 | to run |
| `R2/43@10ms` | `turn/q90_right` | 20 | 24.39 | 0.0213 | to run |
| `R0@10ms` | `turn/q90_right` | 20 | 24.39 | 0.0129 | to run |
| `R2/42@10ms` | `turn/q90_right_mu0.54` | 20 | 24.39 | 0.0213 | to run |
| `R2/43@10ms` | `turn/q90_right_mu0.54` | 20 | 24.39 | 0.0213 | to run |
| `R0@10ms` | `turn/q90_right_mu0.54` | 20 | 24.39 | 0.0129 | to run |
| `R2/42@10ms` | `nudge/forward` | 20 | 24.79 | 0.0217 | to run |
| `R2/43@10ms` | `nudge/forward` | 20 | 24.79 | 0.0217 | to run |
| `R0@10ms` | `nudge/forward` | 20 | 24.79 | 0.0131 | to run |
| `R2/42@10ms` | `nudge/forward_mu0.54` | 20 | 24.79 | 0.0217 | to run |
| `R2/43@10ms` | `nudge/forward_mu0.54` | 20 | 24.79 | 0.0217 | to run |
| `R0@10ms` | `nudge/forward_mu0.54` | 20 | 24.79 | 0.0131 | to run |
| `R2/42@10ms` | `nudge/backward` | 20 | 24.79 | 0.0217 | to run |
| `R2/43@10ms` | `nudge/backward` | 20 | 24.79 | 0.0217 | to run |
| `R0@10ms` | `nudge/backward` | 20 | 24.79 | 0.0131 | to run |
| `R2/42@10ms` | `nudge/backward_mu0.54` | 20 | 24.79 | 0.0217 | to run |
| `R2/43@10ms` | `nudge/backward_mu0.54` | 20 | 24.79 | 0.0217 | to run |
| `R0@10ms` | `nudge/backward_mu0.54` | 20 | 24.79 | 0.0131 | to run |
| `R2/42@10ms` | `nudge/left` | 20 | 24.79 | 0.0217 | to run |
| `R2/43@10ms` | `nudge/left` | 20 | 24.79 | 0.0217 | to run |
| `R0@10ms` | `nudge/left` | 20 | 24.79 | 0.0131 | to run |
| `R2/42@10ms` | `nudge/left_mu0.54` | 20 | 24.79 | 0.0217 | to run |
| `R2/43@10ms` | `nudge/left_mu0.54` | 20 | 24.79 | 0.0217 | to run |
| `R0@10ms` | `nudge/left_mu0.54` | 20 | 24.79 | 0.0131 | to run |
| `R2/42@10ms` | `nudge/right` | 20 | 24.79 | 0.0217 | to run |
| `R2/43@10ms` | `nudge/right` | 20 | 24.79 | 0.0217 | to run |
| `R0@10ms` | `nudge/right` | 20 | 24.79 | 0.0131 | to run |
| `R2/42@10ms` | `nudge/right_mu0.54` | 20 | 24.79 | 0.0217 | to run |
| `R2/43@10ms` | `nudge/right_mu0.54` | 20 | 24.79 | 0.0217 | to run |
| `R0@10ms` | `nudge/right_mu0.54` | 20 | 24.79 | 0.0131 | to run |
| `R2/42@10ms` | `stop/two_step` | 20 | 24.88 | 0.0218 | to run |
| `R2/43@10ms` | `stop/two_step` | 20 | 24.88 | 0.0218 | to run |
| `R0@10ms` | `stop/two_step` | 20 | 24.88 | 0.0132 | to run |
| `R2/42@10ms` | `stop/two_step_mu0.54` | 20 | 24.88 | 0.0218 | to run |
| `R2/43@10ms` | `stop/two_step_mu0.54` | 20 | 24.88 | 0.0218 | to run |
| `R0@10ms` | `stop/two_step_mu0.54` | 20 | 24.88 | 0.0132 | to run |

Total: 960 episodes, 6.557 simulated hours, 0.897 core-hours (0.897 to run).

