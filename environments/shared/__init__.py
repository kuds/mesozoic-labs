"""Shared utilities for Mesozoic Labs dinosaur environments.

Import each name from its submodule, e.g. ``from environments.shared.config
import load_stage_config``.  The package re-exports nothing on purpose:
importing any submodule runs this file first, and every species environment
does, so whatever it imported would be paid for by ``import environments``.
"""
