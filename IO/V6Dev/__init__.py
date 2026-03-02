"""
V6 Advanced Decision & Auto-Tuning System
==========================================

Modular components that integrate with the existing V5 light controller
without modifying any V5 files.

Modules
-------
engagement_score     Composite engagement quality scoring
predictive_context   Learned time-of-day profiles from report history
strategy_bandit      Thompson Sampling for almost-engaged strategies
feedback_learning_v6 Bidirectional feedback with decay
falloff_strategies   Per-mode anisotropic falloff shaping + V6 gestures
smart_autotuner      Gradient-informed auto-tuning
mode_intelligence    Predictive mode transitions + CROWD sub-modes
modifier_resolver    Intent-based modifier chain resolution
v6_integration       Bridge layer wiring everything into V5

Quick start::

    from V6Dev.v6_integration import V6Integration

    v6 = V6Integration(meta, sliders, db, behavior, light,
                        reports_dir='reports/daily')

    # In the V5 main loop (replaces auto_tuner.update()):
    behavior_params = v6.tick(behavior_status, behavior_params,
                              tracked_people, dt, now)
"""

from .v6_integration import V6Integration

__all__ = ['V6Integration']
