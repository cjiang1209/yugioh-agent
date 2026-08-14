from yugioh_core.constants import (
    TYPE_MONSTER,
)
from yugioh_env.server.board_state import render_board


class _FakeCardDB:
    def get_card_name(self, code):
        return f"Card{code}"

    def get_card(self, code):
        return {"type": TYPE_MONSTER, "attack": 1000, "defense": 1000, "level": 4}


def test_render_board_uses_live_stats_not_db_defaults():
    """Deterministic codes-only-bug guard: live atk/def differ from DB defaults."""
    raw = {
        "agent": {
            "hand": [],
            "monsters": [
                {
                    "code": 999,
                    "position": 4,
                    "sequence": 0,
                    "type": TYPE_MONSTER,
                    "attack": 9999,
                    "defense": 8888,
                    "level": 8,
                }
            ],
            "spells_traps": [],
            "grave": [],
            "banished": [],
            "extra": [],
            "lp": 8000,
            "deck_count": 30,
            "hand_count": 0,
            "extra_count": 0,
        },
        "opponent": {
            "hand": [],
            "monsters": [],
            "spells_traps": [],
            "grave": [],
            "banished": [],
            "extra": [],
            "lp": 8000,
            "deck_count": 30,
            "hand_count": 0,
            "extra_count": 0,
        },
        "agent_player": 0,
    }
    mon = render_board(raw, _FakeCardDB(), open_cards=True)["player"]["monsters"][0]
    assert mon["attack"] == 9999 and mon["defense"] == 8888 and mon["level"] == 8
