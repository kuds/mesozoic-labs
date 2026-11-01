# ------------------------------------------------------------------------------------------------- PC (phase critic)
#: PC: the critic phase as the branch defines it (environments/shared/critic_phase.py; PS criteria.txt lines 134-143):
#: phi(k) = (min(k, S)/S, 1 if k >= S else 0, gamma ** max(0, S - k)), S 200 (stance_width_settle_steps), gamma 0.98.
PHASE_SETTLE = 200
PHASE_GAMMA = 0.98
ACTOR_OBS_DIM = 63
PHASE_WIDTH = 3
#: PC launch-check fact (8) and C4 (p): the switch as stage_config.json writes it (pin critic_phase_switch, a
#: {"path": ..., "value": ...} object; the builder's train_base records it as hyperparameters.critic_phase = true).
SWITCH_PATH_DEFAULT = "hyperparameters.critic_phase"
#: PC C8: THE MANIPULATION CHECK's record (manip_check.py) at <panels>/<RUN_ID>/<label>/c8/manip_check.json.
MANIP_KIND = "c8"
MANIP_RECORD = "manip_check.json"
MANIP_SCHEMA = "pc-manip-check/1"
MANIP_SEEDS = (3042, 3043)
MANIP_BANK_LIMIT = 500
#: PC: launch-check faults that allow REPLACEMENT (ii): ROUND's (1)-(5) and (7), plus (8) (PC RUN HANDLING).
STOP_FAULTS = (1, 2, 3, 4, 5, 7, 8)
