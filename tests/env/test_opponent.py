"""Tests for opponent policies and seed determinism."""

import random
import tempfile

import numpy as np
import pytest

from tests.env.conftest import MINIMAL_MSGS, obs_from_msg
from yugioh_core.constants import MSG_SELECT_BATTLECMD, MSG_SELECT_IDLECMD, MSG_SELECT_YESNO
from yugioh_core.encoding import (
    MAX_ACTIONS,
)
from yugioh_env.deck_parser import parse_ydk
from yugioh_env.models import CardCommand, YuGiOhAction, YuGiOhObservation
from yugioh_env.opponent import GreedyOpponent, Opponent, RandomOpponent
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

# ---------------------------------------------------------------------------
# RandomOpponent
# ---------------------------------------------------------------------------


def _yesno_obs() -> YuGiOhObservation:
    """A genuine observation for a 2-action (yes/no) prompt."""
    return obs_from_msg({"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 0})


def test_random_opponent_deterministic_with_seed():
    """Same seed should produce identical action sequences."""
    obs = _yesno_obs()
    results = []
    for _ in range(2):
        opp = RandomOpponent(seed=42)
        actions = [opp.select_action(obs) for _ in range(20)]
        results.append(actions)
    assert results[0] == results[1]


def test_random_opponent_reseed_restores_determinism():
    """Calling reseed() should reset the RNG to produce the same sequence."""
    obs = _yesno_obs()
    opp = RandomOpponent(seed=99)

    run1 = [opp.select_action(obs) for _ in range(20)]

    opp.reseed(99)
    run2 = [opp.select_action(obs) for _ in range(20)]

    assert run1 == run2


def test_random_opponent_different_seeds_differ():
    """Different seeds should (almost certainly) produce different sequences."""
    obs = _yesno_obs()

    opp1 = RandomOpponent(seed=1)
    opp2 = RandomOpponent(seed=2)
    run1 = [opp1.select_action(obs) for _ in range(50)]
    run2 = [opp2.select_action(obs) for _ in range(50)]

    assert run1 != run2


def test_random_opponent_zero_actions_returns_zero():
    """An observation with no legal actions (all-zero mask) returns 0."""
    obs = YuGiOhObservation(action_mask=np.zeros(MAX_ACTIONS, dtype=np.int8))
    opp = RandomOpponent(seed=0)
    assert opp.select_action(obs) == 0


def test_pick_action_random_seeded():
    """Client-side pick_action_random is deterministic when random module is seeded."""
    # Import here to avoid polluting module-level random state
    from cli.play_client import pick_action_random

    mask = [1, 1, 1, 1, 0, 0, 0, 0] + [0] * 24  # 4 legal actions
    obs = YuGiOhObservation(
        action_mask=mask,
        done=False,
        reward=0.0,
    )

    results = []
    for _ in range(2):
        random.seed(123)
        actions = [pick_action_random(obs) for _ in range(20)]
        results.append(actions)

    assert results[0] == results[1]


# ---------------------------------------------------------------------------
# GreedyOpponent
#
# Each test mutates MINIMAL_MSGS[MSG_SELECT_IDLECMD]/[MSG_SELECT_BATTLECMD]
# (which is branch-complete: every category populated) by emptying out the
# higher-priority categories one at a time. This proves the priority order
# itself (not just "some action is picked") -- swapping any two categories'
# priority, or matching by absolute position instead of "first slot of this
# category", would flip at least one of these assertions.
# ---------------------------------------------------------------------------


def test_greedy_opponent_reseed_is_noop():
    """GreedyOpponent.reseed() should not raise."""
    opp = GreedyOpponent()
    opp.reseed(42)  # should be a no-op


def _idle_msg(**overrides) -> dict:
    return {
        **MINIMAL_MSGS[MSG_SELECT_IDLECMD],
        "msg_type": MSG_SELECT_IDLECMD,
        **overrides,
    }


def _battle_msg(**overrides) -> dict:
    return {
        **MINIMAL_MSGS[MSG_SELECT_BATTLECMD],
        "msg_type": MSG_SELECT_BATTLECMD,
        **overrides,
    }


def test_greedy_opponent_idle_prefers_summon_over_everything():
    """summonable is non-empty -> first summon action (index 0) wins,
    even though sp_summonable/sset/to_bp are all also available."""
    opp = GreedyOpponent()
    obs = obs_from_msg(_idle_msg())
    assert opp.select_action(obs) == 0
    assert isinstance(obs.action_descriptors[0], CardCommand)


def test_greedy_opponent_idle_prefers_sp_summon_when_no_summon():
    """With summonable empty, sp_summon (now the first category) wins."""
    opp = GreedyOpponent()
    obs = obs_from_msg(_idle_msg(summonable=[]))
    assert opp.select_action(obs) == 0


def test_greedy_opponent_idle_prefers_sset_over_reposition_and_mset():
    """With summon/sp_summon empty, sset wins even though reposition/mset
    (lower priority but earlier in extraction order) are offered first."""
    opp = GreedyOpponent()
    obs = obs_from_msg(_idle_msg(summonable=[], sp_summonable=[]))
    # order: repositionable(0), mset(1), sset(2), activatable(3), to_bp(4), to_ep(5)
    assert opp.select_action(obs) == 2


def test_greedy_opponent_idle_moves_to_battle_phase_as_last_resort():
    """With summon/sp_summon/sset all empty, falls through to the to_bp action."""
    opp = GreedyOpponent()
    obs = obs_from_msg(_idle_msg(summonable=[], sp_summonable=[], sset=[]))
    # order: repositionable(0), mset(1), activatable(2), to_bp(3), to_ep(4)
    assert opp.select_action(obs) == 3


def test_greedy_opponent_idle_falls_back_to_last_action():
    """With no summon/sp_summon/sset/to_bp, falls back to the final action."""
    opp = GreedyOpponent()
    obs = obs_from_msg(_idle_msg(summonable=[], sp_summonable=[], sset=[], to_bp=0))
    num_actions = int(obs.action_mask.sum())
    assert opp.select_action(obs) == num_actions - 1


def test_greedy_opponent_battle_prefers_attack():
    """attackable is non-empty -> first attack action wins over ending."""
    opp = GreedyOpponent()
    obs = obs_from_msg(_battle_msg())
    # order: activatable(0), attackable(1), to_m2(2), to_ep(3)
    assert opp.select_action(obs) == 1


def test_greedy_opponent_battle_falls_back_to_last_action():
    """With no attackable actions, falls back to the final action."""
    opp = GreedyOpponent()
    obs = obs_from_msg(_battle_msg(attackable=[]))
    num_actions = int(obs.action_mask.sum())
    assert opp.select_action(obs) == num_actions - 1


def test_greedy_opponent_other_prompts_pick_first_option():
    """For non-idle/non-battle prompts, GreedyOpponent picks index 0."""
    opp = GreedyOpponent()
    obs = _yesno_obs()
    assert opp.select_action(obs) == 0


def test_greedy_opponent_single_legal_action_short_circuits():
    """num_actions <= 1 always returns 0 without inspecting descriptors."""
    opp = GreedyOpponent()
    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    mask[0] = 1
    obs = YuGiOhObservation(action_mask=mask, prompt_meta={"msg_type": MSG_SELECT_IDLECMD})
    assert opp.select_action(obs) == 0


def test_greedy_opponent_zero_legal_actions_returns_zero():
    opp = GreedyOpponent()
    obs = YuGiOhObservation(action_mask=np.zeros(MAX_ACTIONS, dtype=np.int8))
    assert opp.select_action(obs) == 0


# ---------------------------------------------------------------------------
# Opponent receives the canonical YuGiOhObservation
# ---------------------------------------------------------------------------


def test_opponent_receives_full_observation(lib, db_path, script_dirs, deck_path) -> None:
    """A single agent step does NOT reliably reach a multi-action opponent
    prompt, so step until the spy fires (bounded), then assert."""
    seen = {}

    class Spy(Opponent):
        needs_board_state = True  # the assertions below read global_state

        def select_action(self, obs):
            if "obs" not in seen:
                seen["obs"] = obs
                # Snapshot both players' hand counts at the moment the
                # opponent is asked to decide, so we can check the
                # observation was built from the OPPONENT's perspective,
                # not the agent's.
                seen["hand"] = list(env._duel.game_state.hand_count)
                # Capture the most recent _build_seat_observation call
                # synchronously, right here, before any further engine
                # processing can append unrelated later calls (e.g. the
                # agent's own observation being built afterwards).
                seen["call"] = calls[-1] if calls else None
            return 0

        def reseed(self, seed):
            pass

    deck = parse_ydk(deck_path)  # parsed dict, not a path
    env = YuGiOhEnvironment(
        config={
            "db_path": str(db_path),
            "script_dirs": [str(d) for d in script_dirs],
        }
    )
    try:
        env.set_opponent(Spy())

        # Independently record every (msg_type, seat) actually passed into
        # _build_seat_observation, by wrapping the bound method on this
        # instance. This lets us verify the opponent's obs was built from
        # THIS call (not some stale mapper/msg), which the Opponent boundary
        # cannot show us: it only ever sees `obs`.
        real_build = env._build_seat_observation
        calls: list[tuple[int | None, int]] = []

        def spy_build(mapper, **kwargs):
            result = real_build(mapper, **kwargs)
            calls.append((mapper.msg_type, mapper.agent_player))
            return result

        env._build_seat_observation = spy_build

        obs = env.reset(seed=1, deck0=deck, deck1=deck, agent_player=0)
        for _ in range(200):
            if "obs" in seen or obs.done:
                break
            obs = env.step(YuGiOhAction(action_index=0))
        assert "obs" in seen, "opponent never got a prompt; raise the bound"

        opp_obs = seen["obs"]
        assert isinstance(opp_obs, YuGiOhObservation)
        assert opp_obs.prompt_meta is not None
        assert any(d is not None for d in opp_obs.action_descriptors)

        # The opponent's observation must be built from the OPPONENT's
        # perspective, so "my hand" (global_state[11]) must equal the actual
        # OPPONENT's hand count. Handing the builder the agent's mapper would
        # silently read the agent's own hand count instead.
        assert opp_obs.global_state[11] == seen["hand"][1 - env._agent_player]

        # prompt_meta must describe the prompt the OPPONENT is answering.
        # Each mapper carries the seat it was updated for, so recording it
        # catches the agent's mapper being passed in the opponent's place --
        # prompt_meta would then describe the agent's last prompt.
        assert seen.get("call"), (
            "opponent-seat observation must be built via _build_seat_observation"
        )
        msg_type, mapper_seat = seen["call"]
        assert opp_obs.prompt_meta["msg_type"] == msg_type
        assert mapper_seat == 1 - env._agent_player
    finally:
        env.close()


# ---------------------------------------------------------------------------
# ModelOpponent tests (require torch + yugioh_rl)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _make_synthetic_checkpoint(path: str) -> None:
    """Create a minimal valid checkpoint file with default config."""
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.network import YuGiOhNet

    config = TrainingConfig()
    net = YuGiOhNet.from_config(config)
    torch.save(
        {
            "update": 1,
            "global_step": 100,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": {},
            "config": config,
        },
        path,
    )


def _dummy_obs() -> YuGiOhObservation:
    """Create a dummy observation with valid shapes; first 3 actions legal."""
    mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
    mask[:3] = 1
    return YuGiOhObservation(action_mask=mask)


def test_model_opponent_construction():
    """ModelOpponent loads a checkpoint and enters eval mode."""
    from yugioh_env.opponent import ModelOpponent

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_synthetic_checkpoint(f.name)
        opp = ModelOpponent(f.name, device="cpu")
        assert not opp._impl._network.training


def test_model_opponent_select_action():
    """ModelOpponent returns a valid action index within bounds."""
    from yugioh_env.opponent import ModelOpponent

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_synthetic_checkpoint(f.name)
        opp = ModelOpponent(f.name, device="cpu")

        obs = _dummy_obs()
        action = opp.select_action(obs)
        assert 0 <= action < int(obs.action_mask.sum())


def test_model_opponent_deterministic():
    """Same checkpoint and observation should produce the same action."""
    from yugioh_env.opponent import ModelOpponent

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_synthetic_checkpoint(f.name)
        opp = ModelOpponent(f.name, device="cpu")

        obs = _dummy_obs()
        a1 = opp.select_action(obs)
        a2 = opp.select_action(obs)
        assert a1 == a2


def test_model_opponent_reseed_noop():
    """ModelOpponent.reseed() should not raise."""
    from yugioh_env.opponent import ModelOpponent

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_synthetic_checkpoint(f.name)
        opp = ModelOpponent(f.name, device="cpu")
        opp.reseed(42)  # should be a no-op


def test_model_opponent_semantic_checkpoint():
    """ModelOpponent works with a semantic-mode checkpoint (no embeddings file on disk)."""
    from yugioh_env.opponent import ModelOpponent
    from yugioh_rl.config import TrainingConfig
    from yugioh_rl.network import TextEmbeddingLookup, YuGiOhNet

    # Build a semantic-mode network from a synthetic embeddings file
    codes = list(range(1, 21))
    embeddings = torch.randn(len(codes), 384)
    codes_tensor = torch.tensor(codes, dtype=torch.int64)
    sorted_indices = codes_tensor.argsort()
    sorted_codes = codes_tensor[sorted_indices]
    sorted_embeddings = embeddings[sorted_indices]
    padded = torch.cat([torch.zeros(1, 384), sorted_embeddings], dim=0)

    text_lookup = TextEmbeddingLookup(sorted_codes, padded, text_embed_dim=32)
    config = TrainingConfig(text_embed_dim=32, learned_embed_dim=8)
    net = YuGiOhNet(config, text_lookup)

    # Save checkpoint (no embeddings file path in config)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(
            {
                "update": 1,
                "global_step": 100,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": {},
                "config": config,
            },
            f.name,
        )
        ckpt_path = f.name

    # Load ModelOpponent — should NOT attempt to read an embeddings file
    opp = ModelOpponent(ckpt_path, device="cpu")
    assert opp._impl._network.text_lookup is not None

    # Verify select_action works
    obs = _dummy_obs()
    action = opp.select_action(obs)
    assert 0 <= action < int(obs.action_mask.sum())

    import os

    os.unlink(ckpt_path)


def test_model_opponent_env_config_missing_checkpoint():
    """opponent_type='model' without checkpoint should raise ValueError."""
    from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

    with pytest.raises(ValueError, match="checkpoint path"):
        YuGiOhEnvironment(config={"opponent": "model:"})
