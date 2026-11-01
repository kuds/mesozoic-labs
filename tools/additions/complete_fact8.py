    # (8) PC: the critic-phase switch ON in 01_stance/stage_config.json (a FAULT like (1)-(5) and (7))
    if sc is None:
        F[8] = fact("(8) critic-phase switch ON in stage_config.json", None, None,
                    f"not computed: stage_config.json {sh}")
    else:
        F[8] = critic_phase_fact("(8) critic-phase switch ON in stage_config.json", sc, pins)
