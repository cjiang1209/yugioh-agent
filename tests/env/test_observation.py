"""Test observation encoding.

Encode→decode roundtrip tests live in test_feature_roundtrip.py.
This file covers observation-building logic that doesn't go through
the decoder (e.g. query_fn integration, visibility rules).
"""


def test_action_meta_length_matches_actions(lib, db_path, script_dirs):
    """action_meta length must equal action_mask length (32 for active obs).
    This is the §6 length-parity invariant from the spec."""
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment
    env = YuGiOhEnvironment({})
    obs = env.reset(seed=42)
    assert len(obs.action_meta) == len(obs.actions) == len(obs.action_mask)
    # Inactive slots are None; only the legal-action prefix may carry meta
    legal_count = sum(obs.action_mask)
    for i in range(legal_count, len(obs.action_mask)):
        assert obs.action_meta[i] is None


def test_terminal_observation_lists_empty(lib, db_path, script_dirs):
    """On done=True, actions, action_mask, and action_meta are all empty.
    This is an intentional drift from the previous action_mask=[0]*32 behavior
    (§3 of spec) — kept consistent so the three lists never differ in length."""
    import random
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment
    from yugioh_env.models import YuGiOhAction
    env = YuGiOhEnvironment({})
    obs = env.reset(seed=42)
    rng = random.Random(0)
    while not obs.done:
        legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
        if not legal:
            break
        obs = env.step(YuGiOhAction(action_index=rng.choice(legal)))
    assert obs.done
    assert obs.actions == []
    assert obs.action_mask == []
    assert obs.action_meta == []


# ─── Board controller relativization invariants (Tests B1, B2) ────────────────

def _build_obs_with_card_on_engine_player_1(agent_player: int):
    """Synthesize an observation with one face-up monster on engine player 1's
    field, using a fake query_fn. Returns the cards array."""
    from yugioh_env.observation import build_observation
    from yugioh_env.game_state import GameState

    gs = GameState()  # default LP/zones/phase

    def fake_query(player: int, loc: int):
        # One face-up Atk monster on engine player 1's MZONE; nothing elsewhere.
        if player == 1 and loc == 0x04:  # LOCATION_MZONE
            return [{
                "code": 46986414, "position": 0x01,  # POS_FACEUP_ATTACK
                "is_public": 1, "is_hidden": 0,
            }]
        return []

    obs = build_observation(gs, current_msg=None, agent_player=agent_player,
                            query_fn=fake_query)
    return obs["cards"]


def test_board_controller_relativizes_when_agent_player_is_0():
    """B1: with agent_player=0, a card on engine player 1's MZONE shows
    controller=1 (opponent) in the board encoding."""
    from yugioh_core.encoding import decode_u32
    cards = _build_obs_with_card_on_engine_player_1(agent_player=0)
    found = None
    for i in range(cards.shape[0]):
        if decode_u32(cards[i], 0) == 46986414:
            found = i
            break
    assert found is not None, "test card not found in board encoding"
    # encode_card layout: byte 7 = controller (after code[0-3], location[4],
    # sequence[5], position[6])
    assert int(cards[found, 7]) == 1, "engine player 1's card → relative=1 when agent=0"


def test_board_controller_relativizes_when_agent_player_is_1():
    """B2: with agent_player=1, the same engine-player-1 card now shows
    controller=0 (agent's own) in the board encoding."""
    from yugioh_core.encoding import decode_u32
    cards = _build_obs_with_card_on_engine_player_1(agent_player=1)
    found = None
    for i in range(cards.shape[0]):
        if decode_u32(cards[i], 0) == 46986414:
            found = i
            break
    assert found is not None, "test card not found in board encoding"
    assert int(cards[found, 7]) == 0, "engine player 1's card → relative=0 when agent=1"


# Cross-source consistency between board and action encodings in a real
# episode is currently not implementable. The intent is to verify, over many
# turns of actual play, that any card referenced by both the board encoding
# and a legal action carries consistent attributes across the two paths
# (controller, location, sequence, etc. — anything the model could read from
# both sources). It catches drift between the board-build and action-extract
# pipelines that single-source unit tests cannot, including bugs introduced
# by future encoding changes that touch only one side. The check would be
# restricted to non-hidden cards (own cards + opponent's face-up cards),
# since hidden-zone entries are intentionally redacted on the board side.
#
# Why it can't run today: `_parse_query_buffer` in
# `yugioh_env/observation.py` parses the engine query wire format
# incorrectly (it expects a `uint32 total_size` per card, but the engine
# writes `uint16 entry_size + uint32 flag + value` per field), so the
# parser breaks on the first field of every card and returns empty dicts.
# Every entry in `obs.cards` therefore has `code == 0`, and no cross-source
# lookup ever matches a real card. The web UI takes a different path
# through the correct parser at `yugioh_env/server/board_state.py` (see
# lines 48-56 for the documented wire format), which is why UI rendering
# works while the RL observation pipeline silently feeds the model all-zero
# board cards. This is a pre-existing bug, tracked separately; once the
# parser is fixed, this test should be added.
