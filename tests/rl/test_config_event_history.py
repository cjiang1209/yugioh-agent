from yugioh_rl.config import TrainingConfig


def test_default_disabled():
    assert TrainingConfig().event_history_dim == 0


def test_settable():
    assert TrainingConfig(event_history_dim=32).event_history_dim == 32
