from yugioh_rl.config import TrainingConfig


def test_deck_selection_defaults():
    c = TrainingConfig()
    assert c.deck_allocation == "random"
    assert c.mirror_decks is False


def test_deck_selection_overridable():
    c = TrainingConfig(deck_allocation="balanced", mirror_decks=True)
    assert c.deck_allocation == "balanced"
    assert c.mirror_decks is True
