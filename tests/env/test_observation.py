"""Test observation encoding.

Encode→decode roundtrip tests live in test_feature_roundtrip.py.
This file covers observation-building logic that doesn't go through
the decoder (e.g. query_fn integration, visibility rules).
"""


def test_action_descriptors_cover_exactly_the_legal_actions(lib, db_path, script_dirs):
    """One descriptor per legal action and nothing else, on a live prompt."""
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment
    from yugioh_rl.obs_encoder import encode_observation

    env = YuGiOhEnvironment({})
    obs = env.reset(seed=42)
    assert obs.action_descriptors
    assert obs.num_actions == len(obs.action_descriptors)
    assert int(encode_observation(obs)["action_mask"].sum()) == obs.num_actions


def test_terminal_observation_actions_zeroed(lib, db_path, script_dirs):
    """On done=True there is no active prompt: action_descriptors is empty and
    the arrays the network reads encode to shaped zeros."""
    import random

    from yugioh_core.encoding import ACTION_FEATURES, MAX_ACTIONS
    from yugioh_env.models import YuGiOhAction
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment
    from yugioh_rl.obs_encoder import encode_observation

    env = YuGiOhEnvironment({})
    obs = env.reset(seed=42)
    rng = random.Random(0)
    while not obs.done:
        if obs.num_actions == 0:
            break
        obs = env.step(YuGiOhAction(action_index=rng.randrange(obs.num_actions)))
    assert obs.done
    assert obs.action_descriptors == []
    encoded = encode_observation(obs)
    assert encoded["actions"].shape == (MAX_ACTIONS, ACTION_FEATURES)
    assert not encoded["actions"].any()
    assert encoded["action_mask"].shape == (MAX_ACTIONS,) and not encoded["action_mask"].any()


def test_terminal_obs_keeps_real_board_but_zeroes_actions(
    lib, db_path, script_dirs, deck_path
) -> None:
    """Always picking slot 0 against a random opponent reaches a finished duel
    well inside the engine's step cap.
    """
    import numpy as np

    from yugioh_core.encoding import ACTION_FEATURES, MAX_ACTIONS
    from yugioh_env.deck_parser import parse_ydk
    from yugioh_env.models import YuGiOhAction
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment
    from yugioh_rl.obs_encoder import encode_observation

    deck = parse_ydk(deck_path)
    env = YuGiOhEnvironment(
        config={
            "db_path": str(db_path),
            "script_dirs": [str(d) for d in script_dirs],
            "opponent": "random",
        }
    )
    obs = env.reset(seed=1, deck0=deck, deck1=deck, agent_player=0)
    while not obs.done:
        obs = env.step(YuGiOhAction(action_index=0))

    assert obs.action_descriptors == []
    encoded = encode_observation(obs)
    assert encoded["actions"].shape == (MAX_ACTIONS, ACTION_FEATURES)
    assert not encoded["actions"].any()
    assert encoded["action_mask"].shape == (MAX_ACTIONS,)
    assert encoded["action_mask"].dtype == np.int8
    assert obs.cards, "final board must NOT be empty"
    assert encoded["cards"].any(), "final board must NOT be zeroed"


# ─── Board controller relativization invariants (Tests B1, B2) ────────────────


def _build_obs_with_card_on_engine_player_1(agent_player: int):
    """Synthesize an observation with one face-up monster on engine player 1's
    field, using a fake query_fn. Returns the structured card list."""
    from yugioh_env.game_state import GameState
    from yugioh_env.observation import build_observation

    gs = GameState()  # default LP/zones/phase

    def fake_query(player: int, loc: int):
        # One face-up Atk monster on engine player 1's MZONE; nothing elsewhere.
        if player == 1 and loc == LOCATION_MZONE:
            return [
                {
                    "code": 46986414,
                    "position": POS_FACEUP_ATTACK,  # POS_FACEUP_ATTACK
                    "is_public": 1,
                    "is_hidden": 0,
                }
            ]
        return []

    obs = build_observation(gs, current_msg=None, agent_player=agent_player, query_fn=fake_query)
    return obs["cards"]


def _find_test_card(cards):
    found = [c for c in cards if c.code == 46986414]
    assert found, "test card not found on the board"
    return found[0]


def test_board_controller_relativizes_when_agent_player_is_0():
    """B1: with agent_player=0, a card on engine player 1's MZONE shows
    controller=1 (opponent) on the board."""
    card = _find_test_card(_build_obs_with_card_on_engine_player_1(agent_player=0))
    assert card.controller == 1, "engine player 1's card → relative=1 when agent=0"


def test_board_controller_relativizes_when_agent_player_is_1():
    """B2: with agent_player=1, the same engine-player-1 card now shows
    controller=0 (agent's own) on the board."""
    card = _find_test_card(_build_obs_with_card_on_engine_player_1(agent_player=1))
    assert card.controller == 0, "engine player 1's card → relative=0 when agent=1"


def test_board_and_action_controller_agree_on_real_episode(lib, db_path, script_dirs):
    """Cross-source consistency: every legal card-bearing action over a real
    episode whose card is present in the board encoding must carry the same
    controller byte on both sides.

    Restricted to non-hidden cards: opponent's face-down cards are
    intentionally redacted in the board encoding (`is_public == 0`,
    `code == 0`), so they cannot be cross-checked. The test focuses on
    cards where both sources have a real value.

    Asserts a minimum of 5 cross-checks were performed so the test is
    meaningful — without the guard, an episode that only ever presents
    card-less prompts (TO_BP, TO_EP, etc.) would silently no-op.
    """
    import random

    from yugioh_env.models import YuGiOhAction
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment({})
    obs = env.reset(seed=1234)
    rng = random.Random(0)

    steps_run = 0
    card_actions_checked = 0
    max_steps = 80  # cap so the test stays fast; enough to cross several turns

    while not obs.done and steps_run < max_steps:
        if obs.num_actions == 0:
            break

        # Build a (code, location, sequence) → controller lookup from the
        # board. Skip entries without a real code (face-down opponent cards).
        board_by_id: dict[tuple[int, int, int], int] = {}
        for c in obs.cards:
            if c.code == 0:
                continue
            board_by_id.setdefault((c.code, c.location, c.sequence), c.controller)

        # Walk the card-bearing actions and cross-check the controller.
        for ai, d in enumerate(obs.action_descriptors):
            card = getattr(d, "card", None)
            if card is None or card.code == 0 or card.location == 0:
                continue
            board_ctrl = board_by_id.get((card.code, card.location, card.sequence))
            if board_ctrl is None:
                continue  # action references a card that isn't on the board
            assert card.controller == board_ctrl, (
                f"controller drift: action[{ai}] (code={card.code}, "
                f"loc=0x{card.location:02x}, seq={card.sequence}) "
                f"ctrl={card.controller} but board_ctrl={board_ctrl}"
            )
            card_actions_checked += 1

        obs = env.step(YuGiOhAction(action_index=rng.randrange(obs.num_actions)))
        steps_run += 1

    assert card_actions_checked >= 5, (
        f"Test D no-op: {steps_run} steps but only {card_actions_checked} "
        f"card-bearing actions were cross-checked. The cross-source invariant "
        f"was barely (or never) exercised. Increase max_steps or check that "
        f"the parser fix is producing real codes in the board encoding."
    )


def test_prompt_meta_populated_on_select_msg(lib, db_path, script_dirs):
    """Reset and step into a SELECT_CARD prompt; assert obs.prompt_meta has
    the expected keys."""
    import random

    from yugioh_env.models import YuGiOhAction
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment({})
    obs = env.reset(seed=1234)
    rng = random.Random(0)

    # Walk a few prompts to look for any non-None prompt_meta with structure.
    for _ in range(20):
        if obs.done:
            break
        if obs.prompt_meta is not None:
            assert isinstance(obs.prompt_meta, dict)
            # The dict may be empty (for prompts like idle_cmd that have no
            # extra fields) or carry per-prompt-type fields.
            # `msg_type` is a documented wire-contract field that openenv
            # HTTP clients receive via Pydantic model_dump().
            assert "msg_type" in obs.prompt_meta, (
                f"prompt_meta missing required 'msg_type' wire field: {obs.prompt_meta!r}"
            )
            assert isinstance(obs.prompt_meta["msg_type"], int)
            return  # found a populated prompt_meta — test passes
        if obs.num_actions == 0:
            break
        obs = env.step(YuGiOhAction(action_index=rng.randrange(obs.num_actions)))

    # If we never observed a non-None prompt_meta, the wiring is broken.
    assert obs.prompt_meta is not None or obs.done, (
        "prompt_meta was None across 20 prompts; wiring may be broken"
    )


def test_prompt_meta_none_on_terminal(lib, db_path, script_dirs):
    """After the duel ends, obs.prompt_meta is None."""
    import random

    from yugioh_env.models import YuGiOhAction
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment({})
    obs = env.reset(seed=1234)
    rng = random.Random(0)

    # Play to completion.
    while not obs.done:
        if obs.num_actions == 0:
            break
        obs = env.step(YuGiOhAction(action_index=rng.randrange(obs.num_actions)))

    assert obs.done
    assert obs.prompt_meta is None


# ─── Chain encoder gap tests ────────────────────────────────────────────────


from yugioh_core.constants import (
    LOCATION_MZONE,
    POS_FACEUP_ATTACK,
)
from yugioh_core.encoding import MAX_PENDING_CHAIN, decode_u32
from yugioh_env.game_state import ChainLink, GameState
from yugioh_env.observation import build_observation


def _gs_with_links(n, controller):
    gs = GameState()
    for i in range(n):
        gs.pending_chain.append(
            ChainLink(
                code=1000 + i,
                desc=0,
                controller=controller,
                location=LOCATION_MZONE,
                sequence=i,
                chain_link=i + 1,
            )
        )
    gs.chain_count = n
    return gs


def test_encoder_relativizes_controller_for_agent_player_1():
    # Link controlled by engine player 1. With agent_player=1, that is "agent" (0).
    gs = _gs_with_links(1, controller=1)
    obs = build_observation(gs, None, agent_player=1, query_fn=lambda p, loc: [])
    assert obs["pending_chain"][0, 12] == 0  # relativized to agent
    # With agent_player=0, the same raw controller 1 relativizes to opponent (1).
    obs0 = build_observation(gs, None, agent_player=0, query_fn=lambda p, loc: [])
    assert obs0["pending_chain"][0, 12] == 1


def test_encoder_truncates_beyond_max_pending_chain():
    gs = _gs_with_links(MAX_PENDING_CHAIN + 2, controller=0)
    obs = build_observation(gs, None, agent_player=0, query_fn=lambda p, loc: [])
    pc = obs["pending_chain"]
    assert pc.shape == (MAX_PENDING_CHAIN, pc.shape[1])
    for i in range(MAX_PENDING_CHAIN):
        assert decode_u32(pc[i], 0) == 1000 + i
