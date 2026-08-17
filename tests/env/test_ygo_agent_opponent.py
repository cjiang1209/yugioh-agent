"""Tests for yugioh_env.ygo_agent.opponent.YGOAgentOpponent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.env.conftest import MINIMAL_MSGS, obs_from_msg
from yugioh_core.constants import MSG_SELECT_YESNO, SELECT_MSGS
from yugioh_env.ygo_agent.bridge import _ACTION_MSG_TRANSLATORS
from yugioh_env.ygo_agent.opponent import (
    _SERVER_UNSUPPORTED_MSGS,
    DEFAULT_URL,
    YGOAgentOpponent,
)


class TestYGOAgentOpponent:
    def test_default_url(self):
        opp = YGOAgentOpponent()
        assert opp._base_url == DEFAULT_URL

    def test_custom_url(self):
        opp = YGOAgentOpponent("http://myhost:5000")
        assert opp._base_url == "http://myhost:5000"

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

        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        # select_yesno yields exactly two Confirm descriptors: yes=True at
        # slot 0, yes=False at slot 1.
        obs = obs_from_msg(msg)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "predict_results": {
                "action_preds": [
                    # Best (highest prob) has response=0 → "no" → Confirm(yes=False).
                    {"prob": 0.8, "response": 0, "can_finish": False},
                    {"prob": 0.2, "response": 1, "can_finish": False},
                ],
                "win_rate": 0.6,
            },
            "index": 1,
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("yugioh_env.ygo_agent.opponent.requests") as mock_req:
            mock_req.post.return_value = mock_resp
            action, _ = opp.select_action(obs)
        # response=0 ("no") matches the Confirm(yes=False) descriptor, which is
        # slot 1 (slot 0 is always yes=True) — deliberately non-zero so this
        # assertion can't be satisfied by match_response's slot-0 fallback.
        assert action == 1
        assert opp._index == 1

    def test_select_action_no_duel_returns_zero(self):
        opp = YGOAgentOpponent()
        opp._duel_id = None
        msg = {"msg_type": MSG_SELECT_YESNO, "player": 0, "desc": 30}
        obs = obs_from_msg(msg)
        action, _ = opp.select_action(obs)
        assert action == 0

    @pytest.mark.parametrize("msg_type", sorted(_SERVER_UNSUPPORTED_MSGS))
    def test_select_action_short_circuits_unsupported(self, msg_type):
        """The ygo-agent server has no schema for these prompts, so answer 0
        without an HTTP call.

        Parametrized over the table itself: a newly declared unsupported type
        cannot slip in untested, and a type that belongs in neither this table
        nor the translator table is caught by
        test_every_agent_facing_prompt_is_translated_or_declared_unsupported.
        """
        opp = YGOAgentOpponent()
        opp._duel_id = "test-duel"
        obs = obs_from_msg({**MINIMAL_MSGS[msg_type], "msg_type": msg_type})
        with patch("yugioh_env.ygo_agent.opponent.requests") as mock_req:
            result, _ = opp.select_action(obs)
        assert result == 0
        mock_req.post.assert_not_called()

    def test_every_agent_facing_prompt_is_translated_or_declared_unsupported(self):
        """A prompt in neither table reaches translate_action_msg and raises
        ValueError mid-duel, which select_action does not catch."""
        uncovered = set(SELECT_MSGS) - set(_ACTION_MSG_TRANSLATORS) - set(_SERVER_UNSUPPORTED_MSGS)
        assert not uncovered, (
            f"agent-facing prompts with no ygo-agent translator and no explicit "
            f"unsupported declaration: {sorted(uncovered)}"
        )


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
