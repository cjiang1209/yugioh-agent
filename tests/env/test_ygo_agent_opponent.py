"""Tests for yugioh_env.ygo_agent.opponent.YGOAgentOpponent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from yugioh_core.constants import MSG_SELECT_YESNO
from yugioh_core.encoding import (
    ACTION_FEATURES,
    CARD_FEATURES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    MAX_CARDS,
    encode_u16,
)
from yugioh_env.ygo_agent.opponent import DEFAULT_URL, YGOAgentOpponent


class TestYGOAgentOpponent:
    def test_default_url(self):
        opp = YGOAgentOpponent()
        assert opp._base_url == DEFAULT_URL

    def test_custom_url(self):
        opp = YGOAgentOpponent("http://myhost:5000")
        assert opp._base_url == "http://myhost:5000"

    def test_needs_observation_true(self):
        opp = YGOAgentOpponent()
        assert opp.needs_observation is True

    def test_reseed_creates_duel(self):
        opp = YGOAgentOpponent()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"duelId": "abc-123", "index": 0}
        mock_resp.raise_for_status = MagicMock()
        with patch("yugioh_env.ygo_agent.opponent.requests") as mock_req:
            mock_req.post.return_value = mock_resp
            mock_req.delete.return_value = MagicMock()
            opp.reseed(42)
        assert opp._duel_id == "abc-123"
        assert opp._index == 0

    def test_reseed_deletes_old_duel(self):
        opp = YGOAgentOpponent()
        opp._duel_id = "old-id"
        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"duelId": "new-id", "index": 0}
        mock_post_resp.raise_for_status = MagicMock()
        with patch("yugioh_env.ygo_agent.opponent.requests") as mock_req:
            mock_req.post.return_value = mock_post_resp
            mock_req.delete.return_value = MagicMock()
            opp.reseed(0)
        # Should have called delete with old duel id
        mock_req.delete.assert_called_once()
        assert "old-id" in str(mock_req.delete.call_args)
        assert opp._duel_id == "new-id"

    def test_select_action_calls_predict(self):
        opp = YGOAgentOpponent()
        opp._duel_id = "test-duel"
        opp._index = 0
        obs = {
            "cards": np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8),
            "global_state": np.zeros(GLOBAL_FEATURES, dtype=np.uint8),
            "actions": np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8),
            "action_mask": np.zeros(MAX_ACTIONS, dtype=np.int8),
        }
        obs["global_state"][4] = 1  # turn
        obs["global_state"][5], obs["global_state"][6] = encode_u16(0x04)  # phase: main1
        obs["global_state"][7] = 1  # is_my_turn
        opp.set_observation(obs)

        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "predict_results": {
                "action_preds": [
                    {"prob": 0.8, "response": 1, "can_finish": False},
                    {"prob": 0.2, "response": 0, "can_finish": False},
                ],
                "win_rate": 0.6,
            },
            "index": 1,
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("yugioh_env.ygo_agent.opponent.requests") as mock_req:
            mock_req.post.return_value = mock_resp
            # yes/no: actions with category 0 (yes) and 1 (no)
            action = opp.select_action(msg, 2)
        # Best action has response=1 (yes) → maps to category 0 → action index 0
        assert action == 0
        assert opp._index == 1

    def test_select_action_no_duel_returns_zero(self):
        opp = YGOAgentOpponent()
        opp._duel_id = None
        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        assert opp.select_action(msg, 2) == 0

    def test_select_action_short_circuits_announce_card(self):
        """announce_card is not supported by the ygo-agent server; the opponent
        returns 0 without making an HTTP call."""
        from yugioh_core.constants import MSG_ANNOUNCE_CARD

        opp = YGOAgentOpponent()
        opp._duel_id = "test-duel"
        opp.set_observation(
            {
                "cards": np.zeros((MAX_CARDS, CARD_FEATURES), dtype=np.uint8),
                "global_state": np.zeros(GLOBAL_FEATURES, dtype=np.uint8),
            }
        )
        msg = {"msg_type": MSG_ANNOUNCE_CARD, "player": 0, "opcodes": []}
        with patch("yugioh_env.ygo_agent.opponent.requests") as mock_req:
            result = opp.select_action(msg, num_actions=3)
        assert result == 0
        mock_req.post.assert_not_called()


class TestYGOAgentOpponentFactory:
    def test_parse_ygo_agent_no_url(self):
        from yugioh_env.opponent import parse_opponent_spec

        assert parse_opponent_spec("ygo-agent") == ("ygo-agent", "")

    def test_parse_ygo_agent_with_url(self):
        from yugioh_env.opponent import parse_opponent_spec

        assert parse_opponent_spec("ygo-agent:http://host:3000") == (
            "ygo-agent",
            "http://host:3000",
        )

    def test_make_ygo_agent_default(self):
        from yugioh_env.opponent import make_opponent

        opp = make_opponent("ygo-agent")
        assert isinstance(opp, YGOAgentOpponent)
        assert opp._base_url == DEFAULT_URL

    def test_make_ygo_agent_custom_url(self):
        from yugioh_env.opponent import make_opponent

        opp = make_opponent("ygo-agent:http://myhost:5000")
        assert isinstance(opp, YGOAgentOpponent)
        assert opp._base_url == "http://myhost:5000"
