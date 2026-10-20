"""Floor contacts per foot, read from the physics: the code the gait recorder and the T. rex env both import.

Scratch (the stance4 copband design, never merged).  The T. rex's opt-in
per-foot CoP band prices each step's whole-foot centre of pressure, the
judge's K2b reading, so it must reduce the step's contacts with the
recorder's own code.  That code lives here rather than in
:mod:`environments.shared.gait`, because no species env module imports the
gait package (``test_no_species_env_module_imports_the_gait_package``, with
no exception).

Modules, bottom-up:

* :mod:`~environments.shared.floor_contact.reduction` -- the contact
  threshold, the normal-force decode, the floor-contact classification, the
  per-substep force sums, the sole and whole-foot centres of pressure and
  their normalization; pure (the standard library and numpy), because the
  gait package's pure modules import from it;
* :mod:`~environments.shared.floor_contact.morphology` -- feet, legs, support
  geoms, soles and body weight read off a live env, with each embodiment
  class's ``FOOT_SUPPORT`` entry; imports mujoco (``gait/morphology.py``
  re-exports it);
* :mod:`~environments.shared.floor_contact.step_cop` -- ``StepFootCop``, the
  band's per-step reader: it copies each substep's contacts and reduces one
  control step in one fused pass, with :mod:`.reduction`'s decode,
  classification, thresholds and normalization and the recorder's
  arithmetic.

Layering: this package imports nothing from ``gait``, ``curriculum`` or
``reporting``, and this file imports nothing, so a pure reader of
:mod:`.reduction` loads no mujoco.
"""
