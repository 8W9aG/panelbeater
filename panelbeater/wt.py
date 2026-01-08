"""Handles wavetrainer interaction."""

import datetime

import wavetrainer


def create_wt() -> wavetrainer.trainer.Trainer:  # pyright: ignore
    """Creates a wavetrainer instance."""
    return wavetrainer.create(
        "panelbeater-train",
        walkforward_timedelta=datetime.timedelta(days=30),
        validation_size=datetime.timedelta(days=365),
        test_size=datetime.timedelta(days=365),
        allowed_models={"catboost"},
        max_false_positive_reduction_steps=0,
        use_power_transformer=True,
    )
