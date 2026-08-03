"""Test observation encoding.

Encode→decode roundtrip tests live in test_feature_roundtrip.py.
This file covers observation-building logic that doesn't go through
the decoder (e.g. query_fn integration, visibility rules).
"""


def test_action_descriptors_length_matches_actions(lib, db_path, script_dirs):
    """action_descriptors length must equal action_mask length (32 for active
    obs). This is the §6 length-parity invariant from the spec."""
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment({})
    obs = env.reset(seed=42)
    assert len(obs.action_descriptors) == len(obs.actions) == len(obs.action_mask)
    # Inactive slots are None; only the legal-action prefix may carry a descriptor
    legal_count = sum(obs.action_mask)
    for i in range(legal_count, len(obs.action_mask)):
        assert obs.action_descriptors[i] is None


def test_terminal_observation_actions_zeroed(lib, db_path, script_dirs):
    """On done=True there is no active prompt: actions/action_mask are shaped
    all-zero arrays and action_descriptors is empty."""
    import random

    from yugioh_core.encoding import ACTION_FEATURES, MAX_ACTIONS
    from yugioh_env.models import YuGiOhAction
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment({})
    obs = env.reset(seed=42)
    rng = random.Random(0)
    while not obs.done:
        legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
        if not legal:
            break
        obs = env.step(YuGiOhAction(action_index=rng.choice(legal)))
    assert obs.done
    assert obs.actions.shape == (MAX_ACTIONS, ACTION_FEATURES) and not obs.actions.any()
    assert obs.action_mask.shape == (MAX_ACTIONS,) and not obs.action_mask.any()
    assert obs.action_descriptors == []


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

    assert obs.actions.shape == (MAX_ACTIONS, ACTION_FEATURES)
    assert not obs.actions.any()
    assert obs.action_mask.shape == (MAX_ACTIONS,) and obs.action_mask.dtype == np.int8
    assert obs.action_descriptors == []
    assert obs.cards.any(), "final board must NOT be zeroed"


# ─── Board controller relativization invariants (Tests B1, B2) ────────────────


def _build_obs_with_card_on_engine_player_1(agent_player: int):
    """Synthesize an observation with one face-up monster on engine player 1's
    field, using a fake query_fn. Returns the cards array."""
    from yugioh_env.game_state import GameState
    from yugioh_env.observation import build_observation

    gs = GameState()  # default LP/zones/phase

    def fake_query(player: int, loc: int):
        # One face-up Atk monster on engine player 1's MZONE; nothing elsewhere.
        if player == 1 and loc == 0x04:  # LOCATION_MZONE
            return [
                {
                    "code": 46986414,
                    "position": 0x01,  # POS_FACEUP_ATTACK
                    "is_public": 1,
                    "is_hidden": 0,
                }
            ]
        return []

    obs = build_observation(gs, current_msg=None, agent_player=agent_player, query_fn=fake_query)
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

    from yugioh_core.encoding import decode_u32
    from yugioh_env.models import YuGiOhAction
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    env = YuGiOhEnvironment({})
    obs = env.reset(seed=1234)
    rng = random.Random(0)

    steps_run = 0
    card_actions_checked = 0
    max_steps = 80  # cap so the test stays fast; enough to cross several turns

    while not obs.done and steps_run < max_steps:
        legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
        if not legal:
            break

        # Build a (code, location, sequence) → controller lookup from the
        # board encoding. Skip slots without a real code (empty slots,
        # face-down opponent cards).
        board_by_id: dict[tuple[int, int, int], int] = {}
        for c in obs.cards:
            code = decode_u32(c, 0)
            if code == 0:
                continue
            location = c[4]
            sequence = c[5]
            controller = c[7]
            board_by_id.setdefault((code, location, sequence), controller)

        # Walk legal card-bearing actions and cross-check the controller.
        for ai, mask_v in enumerate(obs.action_mask):
            if mask_v != 1:
                continue
            af = obs.actions[ai]
            a_code = decode_u32(af, 2)
            a_loc = af[7]
            if a_code == 0 or a_loc == 0:
                continue
            a_ctrl = af[6]
            a_seq = af[8] | (af[9] << 8)
            board_ctrl = board_by_id.get((a_code, a_loc, a_seq))
            if board_ctrl is None:
                continue  # action references a card that isn't in obs.cards
            assert a_ctrl == board_ctrl, (
                f"controller drift: action[{ai}] (code={a_code}, "
                f"loc=0x{a_loc:02x}, seq={a_seq}) ctrl={a_ctrl} but "
                f"board_ctrl={board_ctrl}"
            )
            card_actions_checked += 1

        obs = env.step(YuGiOhAction(action_index=rng.choice(legal)))
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
        legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
        if not legal:
            break
        obs = env.step(YuGiOhAction(action_index=rng.choice(legal)))

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
        legal = [i for i, m in enumerate(obs.action_mask) if m == 1]
        if not legal:
            break
        obs = env.step(YuGiOhAction(action_index=rng.choice(legal)))

    assert obs.done
    assert obs.prompt_meta is None


# ─── Chain encoder gap tests ────────────────────────────────────────────────


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
                location=0x04,
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
