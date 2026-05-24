"""Unit tests for IdleCmdHandler and BattleCmdHandler.

Uses FakeConnection (same pattern as test_mud_protocol.py).
"""

from __future__ import annotations

import asyncio
from collections import deque

import pytest

from yugioh_core.constants import LOCATION_BANISHED
from yugioh_mud.agent import PassiveAgent, RandomAgent
from yugioh_mud.cmd_handler import (
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    BattleCmdHandler,
    IdleCmdHandler,
    parse_cardspec,
)
from yugioh_mud.game_state import CardEntry, MUDGameState
from yugioh_mud.text_parser import MUDTextParser, ParsedPrompt, PromptType


class FakeConnection:
    """Test double: feeds canned recv lines and records sent lines."""

    def __init__(self, lines: list[str]) -> None:
        self._inbox: deque[str] = deque(lines)
        self.sent: list[str] = []

    async def send_line(self, text: str) -> None:
        self.sent.append(text)

    async def recv_line(self) -> str:
        if not self._inbox:
            raise StopIteration("No more canned lines")
        return self._inbox.popleft()

    def add_lines(self, lines: list[str]) -> None:
        self._inbox.extend(lines)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# parse_cardspec
# ---------------------------------------------------------------------------


class TestParseCardspec:
    def test_hand(self):
        assert parse_cardspec("h1") == (LOCATION_HAND, 0)
        assert parse_cardspec("h5") == (LOCATION_HAND, 4)

    def test_monster(self):
        assert parse_cardspec("m1") == (LOCATION_MZONE, 0)
        assert parse_cardspec("m3") == (LOCATION_MZONE, 2)

    def test_spell(self):
        assert parse_cardspec("s2") == (LOCATION_SZONE, 1)

    def test_grave(self):
        assert parse_cardspec("g1") == (LOCATION_GRAVE, 0)

    def test_extra(self):
        assert parse_cardspec("x1") == (LOCATION_EXTRA, 0)

    def test_removed(self):
        assert parse_cardspec("r1") == (LOCATION_BANISHED, 0)
        assert parse_cardspec("r3") == (LOCATION_BANISHED, 2)

    def test_unknown(self):
        assert parse_cardspec("??") == (0, 0)
        assert parse_cardspec("") == (0, 0)


# ---------------------------------------------------------------------------
# IdleCmdHandler
# ---------------------------------------------------------------------------


def _make_idle_prompt(options: list[str] | None = None) -> ParsedPrompt:
    """Create a minimal IDLE_CMD prompt."""
    return ParsedPrompt(
        prompt_type=PromptType.IDLE_CMD,
        options=options or ["b", "e"],
    )


class TestIdleCmdHandlerPassive:
    """PassiveAgent always sends 'e' (END_PHASE)."""

    def test_passive_ends_phase_no_usable_cards(self):
        """With no usable cards, passive agent sends 'e'."""
        conn = FakeConnection(
            [
                # Response to "?" — no usable cards, just re-prompt
                "Select a card:",
            ]
        )
        prompt = _make_idle_prompt(["b", "e"])
        agent = PassiveAgent()
        parser = MUDTextParser()

        ended = _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        assert not ended
        assert conn.sent[0] == "?"
        assert conn.sent[1] == "e"
        # structured_actions should have phase transitions only
        assert len(prompt.structured_actions) == 2
        assert prompt.structured_actions[0].category == 6  # to_bp
        assert prompt.structured_actions[1].category == 7  # to_ep

    def test_passive_ends_phase_with_usable_cards(self):
        """Even with usable cards, passive agent still sends 'e'."""
        conn = FakeConnection(
            [
                # Response to "?"
                "Summonable in attack position: h1, h3",
                "Activatable: h2",
                "Select a card:",
            ]
        )
        prompt = _make_idle_prompt(["b", "e"])
        agent = PassiveAgent()
        parser = MUDTextParser()

        ended = _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        assert not ended
        assert conn.sent == ["?", "e"]
        # Should have 3 card actions + 2 phase transitions
        assert len(prompt.structured_actions) == 5
        cats = [a.category for a in prompt.structured_actions]
        assert 0 in cats  # summon
        assert 5 in cats  # activate
        assert 6 in cats  # to_bp
        assert 7 in cats  # to_ep


class TestIdleCmdHandlerRandom:
    """RandomAgent picks from structured_actions."""

    def test_random_picks_card_action(self):
        """RandomAgent can pick a card action and execute submenu."""
        # We need a seed that picks a card action (not END_PHASE)
        for seed in range(50):
            conn = FakeConnection(
                [
                    # Response to "?"
                    "Summonable in attack position: h1",
                    "Select a card:",
                    # Submenu after selecting h1
                    "s: Summon in attack position.",
                    "t: Set in defense position.",
                    "z: back.",
                    "Select action for Blue-Eyes White Dragon",
                ]
            )
            prompt = _make_idle_prompt(["b", "e"])
            agent = RandomAgent(seed=seed)
            parser = MUDTextParser()

            ended = _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

            assert not ended
            # Check if agent picked the card action (h1)
            if "h1" in conn.sent:
                # After "?", then "h1", then a submenu letter
                assert conn.sent[0] == "?"
                assert conn.sent[1] == "h1"
                assert conn.sent[2] in ("s", "t", "z")
                return
        pytest.fail("No seed out of 50 picked a card action")

    def test_random_picks_phase_transition(self):
        """RandomAgent can pick a phase transition."""
        for seed in range(50):
            conn = FakeConnection(
                [
                    "Summonable in attack position: h1",
                    "Select a card:",
                ]
            )
            prompt = _make_idle_prompt(["b", "e"])
            agent = RandomAgent(seed=seed)
            parser = MUDTextParser()

            ended = _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

            if not ended and conn.sent[-1] in ("b", "e"):
                assert conn.sent[0] == "?"
                return
        pytest.fail("No seed out of 50 picked a phase transition")

    def test_multi_category_same_card(self):
        """A card in multiple categories creates multiple actions."""
        conn = FakeConnection(
            [
                "Summonable in attack position: h1",
                "Settable: h1",
                "Select a card:",
            ]
        )
        prompt = _make_idle_prompt(["e"])
        agent = PassiveAgent()
        parser = MUDTextParser()

        _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        # h1 appears in both summon (cat 0) and settable (cat 4)
        card_actions = [a for a in prompt.structured_actions if a.cardspec == "h1"]
        assert len(card_actions) == 2
        cats = {a.category for a in card_actions}
        assert cats == {0, 4}


class TestIdleCmdHandlerDuelEnd:
    """Duel-end detection during idle handler."""

    def test_duel_end_during_probe(self):
        conn = FakeConnection(
            [
                "You won (ran out of cards to draw).",
            ]
        )
        prompt = _make_idle_prompt(["e"])
        agent = PassiveAgent()
        parser = MUDTextParser()

        ended = _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        assert ended

    def test_duel_end_during_submenu(self):
        """Duel ends while waiting for submenu after card selection."""
        for seed in range(50):
            conn = FakeConnection(
                [
                    "Summonable in attack position: h1",
                    "Select a card:",
                    # After sending h1, duel ends
                    "You lost (LP became 0).",
                ]
            )
            prompt = _make_idle_prompt(["e"])
            agent = RandomAgent(seed=seed)
            parser = MUDTextParser()

            ended = _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

            if "h1" in conn.sent:
                assert ended
                return
        pytest.fail("No seed picked h1")


class TestIdleCmdHandlerMultiEffect:
    """Multi-effect card handling (v → va/vb)."""

    def test_single_effect_sends_v(self):
        """Single-effect card: submenu has 'v', handler sends 'v'."""
        for seed in range(100):
            conn = FakeConnection(
                [
                    "Activatable: s1",
                    "Select a card:",
                    # Submenu for s1
                    "v: Activate Mirror Force.",
                    "z: back.",
                    "Select action for Mirror Force",
                ]
            )
            prompt = _make_idle_prompt(["e"])
            agent = RandomAgent(seed=seed)
            parser = MUDTextParser()

            _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

            if "s1" in conn.sent:
                assert conn.sent[-1] == "v"
                return
        pytest.fail("No seed picked s1")

    def test_multi_effect_sends_va(self):
        """Multi-effect card: submenu has 'va'/'vb', handler sends 'va'."""
        for seed in range(100):
            conn = FakeConnection(
                [
                    "Activatable: s1",
                    "Select a card:",
                    # Submenu with multi-effect
                    "va: First effect.",
                    "vb: Second effect.",
                    "z: back.",
                    "Select action for Mystic Card",
                ]
            )
            prompt = _make_idle_prompt(["e"])
            agent = RandomAgent(seed=seed)
            parser = MUDTextParser()

            _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

            if "s1" in conn.sent:
                # Should send "va" (first effect) since "v" is not in menu
                assert conn.sent[-1] == "va"
                return
        pytest.fail("No seed picked s1")


class TestIdleCmdHandlerGameState:
    """Card code resolution from game state."""

    def test_resolves_card_code_from_hand(self):
        conn = FakeConnection(
            [
                "Summonable in attack position: h1",
                "Select a card:",
            ]
        )
        prompt = _make_idle_prompt(["e"])
        agent = PassiveAgent()
        parser = MUDTextParser()
        gs = MUDGameState()
        gs.my_hand = [
            CardEntry(name="Blue-Eyes", code=89631139, spec="h1"),
        ]

        _run(IdleCmdHandler.handle(conn, prompt, agent, gs, parser, verbose=False))

        h1_action = [a for a in prompt.structured_actions if a.cardspec == "h1"][0]
        assert h1_action.card_code == 89631139
        assert h1_action.location == LOCATION_HAND
        assert h1_action.sequence == 0

    def test_resolves_card_code_from_graveyard(self):
        conn = FakeConnection(
            [
                "Activatable: g1",
                "Select a card:",
            ]
        )
        prompt = _make_idle_prompt(["e"])
        agent = PassiveAgent()
        parser = MUDTextParser()
        gs = MUDGameState()
        gs.my_graveyard = [
            CardEntry(name="Monster Reborn", code=83764718, spec="g1"),
        ]

        _run(IdleCmdHandler.handle(conn, prompt, agent, gs, parser, verbose=False))

        g1_action = [a for a in prompt.structured_actions if a.cardspec == "g1"][0]
        assert g1_action.card_code == 83764718
        assert g1_action.location == LOCATION_GRAVE
        assert g1_action.sequence == 0

    def test_resolves_card_code_from_banished(self):
        conn = FakeConnection(
            [
                "Activatable: r1",
                "Select a card:",
            ]
        )
        prompt = _make_idle_prompt(["e"])
        agent = PassiveAgent()
        parser = MUDTextParser()
        gs = MUDGameState()
        gs.my_banished = [
            CardEntry(name="Necroface", code=70426919, spec="r1"),
        ]

        _run(IdleCmdHandler.handle(conn, prompt, agent, gs, parser, verbose=False))

        r1_action = [a for a in prompt.structured_actions if a.cardspec == "r1"][0]
        assert r1_action.card_code == 70426919
        assert r1_action.location == LOCATION_BANISHED
        assert r1_action.sequence == 0


class TestIdleCmdHandlerReprompt:
    """Handler handles re-prompt (ParsedPrompt) as terminator."""

    def test_reprompt_as_terminator(self):
        """When '?' response includes a re-sent prompt, use it as terminator."""
        conn = FakeConnection(
            [
                "Summonable in attack position: h1",
                # Full re-sent prompt (parser accumulates this)
                "Select a card on which to perform an action.",
                "b: Enter the battle phase.",
                "e: End phase.",
                "Select a card:",
            ]
        )
        prompt = _make_idle_prompt(["b", "e"])
        agent = PassiveAgent()
        parser = MUDTextParser()

        ended = _run(IdleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        assert not ended
        assert len(prompt.structured_actions) >= 1


# ---------------------------------------------------------------------------
# BattleCmdHandler
# ---------------------------------------------------------------------------


def _make_battle_prompt(options: list[str] | None = None) -> ParsedPrompt:
    return ParsedPrompt(
        prompt_type=PromptType.BATTLE_MENU,
        options=options or ["a", "e"],
    )


class TestBattleCmdHandlerPassive:
    def test_passive_ends_phase(self):
        """PassiveAgent sends 'e' without probing."""
        # No attack/activate probing needed since agent will pick END_PHASE
        # But the handler probes first, then decides.
        conn = FakeConnection(
            [
                # Probe attack: send "a" → card list → "Select a card:"
                "m1: Blue-Eyes White Dragon",
                "z: back.",
                "Select a card:",
                # Back: re-sent battle menu (discarded)
                "Battle menu:",
                "a: Attack.",
                "e: End phase.",
                "Select an option:",
            ]
        )
        prompt = _make_battle_prompt(["a", "e"])
        agent = PassiveAgent()
        parser = MUDTextParser()

        ended = _run(BattleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        assert not ended
        # Should probe "a", then "z" back, then send "e"
        assert "a" in conn.sent
        assert "z" in conn.sent
        assert conn.sent[-1] == "e"

    def test_passive_no_attack_option(self):
        """Battle menu with only 'e' (no attack available)."""
        conn = FakeConnection([])
        prompt = _make_battle_prompt(["e"])
        agent = PassiveAgent()
        parser = MUDTextParser()

        ended = _run(BattleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        assert not ended
        assert conn.sent == ["e"]


class TestBattleCmdHandlerRandom:
    def test_random_picks_attack(self):
        """RandomAgent can pick an attack action."""
        for seed in range(100):
            conn = FakeConnection(
                [
                    # Probe attack
                    "m1: Blue-Eyes White Dragon",
                    "z: back.",
                    "Select a card:",
                    # Discard re-sent menu
                    "Battle menu:",
                    "a: Attack.",
                    "e: End phase.",
                    "Select an option:",
                    # Execution: send "a" again, then card list, then send cardspec
                    "m1: Blue-Eyes White Dragon",
                    "z: back.",
                    "Select a card:",
                ]
            )
            prompt = _make_battle_prompt(["a", "e"])
            agent = RandomAgent(seed=seed)
            parser = MUDTextParser()

            ended = _run(BattleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

            # Check if agent picked the attack
            if conn.sent.count("a") == 2:  # probe + execute
                assert not ended
                assert conn.sent[-1] == "m1"
                return
        pytest.fail("No seed out of 100 picked attack")

    def test_random_picks_to_m2(self):
        """RandomAgent can pick 'to main phase 2' transition."""
        for seed in range(100):
            conn = FakeConnection(
                [
                    # Probe attack
                    "m1: Blue-Eyes White Dragon",
                    "z: back.",
                    "Select a card:",
                    # Discard re-sent menu
                    "Battle menu:",
                    "a: Attack.",
                    "m: Main phase 2.",
                    "e: End phase.",
                    "Select an option:",
                ]
            )
            prompt = _make_battle_prompt(["a", "m", "e"])
            agent = RandomAgent(seed=seed)
            parser = MUDTextParser()

            ended = _run(BattleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

            if not ended and conn.sent[-1] == "m":
                return
        pytest.fail("No seed out of 100 picked m")


class TestBattleCmdHandlerActivate:
    def test_probe_and_execute_activate(self):
        """Probes activate cards and can execute activation."""
        for seed in range(100):
            conn = FakeConnection(
                [
                    # Probe activate ("c")
                    "s1: Mirror Force",
                    "z: back.",
                    "Enter a line of text.",
                    # Discard re-sent menu
                    "Battle menu:",
                    "c: Activate.",
                    "e: End phase.",
                    "Select an option:",
                    # Execution: send "c", then card list, then cardspec
                    "s1: Mirror Force",
                    "z: back.",
                    "Enter a line of text.",
                ]
            )
            prompt = _make_battle_prompt(["c", "e"])
            agent = RandomAgent(seed=seed)
            parser = MUDTextParser()

            ended = _run(BattleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

            if conn.sent.count("c") == 2:  # probe + execute
                assert not ended
                assert conn.sent[-1] == "s1"
                return
        pytest.fail("No seed out of 100 picked activate")


class TestBattleCmdHandlerDuelEnd:
    def test_duel_end_during_probe(self):
        conn = FakeConnection(
            [
                # Probe attack, then duel ends
                "You won (opponent LP became 0).",
            ]
        )
        prompt = _make_battle_prompt(["a", "e"])
        agent = PassiveAgent()
        parser = MUDTextParser()

        ended = _run(BattleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        assert ended

    def test_duel_end_during_discard(self):
        """Duel ends while discarding re-sent battle menu after probe."""
        conn = FakeConnection(
            [
                # Probe attack
                "m1: Blue-Eyes White Dragon",
                "z: back.",
                "Select a card:",
                # Duel ends during discard
                "You lost (LP became 0).",
            ]
        )
        prompt = _make_battle_prompt(["a", "e"])
        agent = PassiveAgent()
        parser = MUDTextParser()

        ended = _run(BattleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        assert ended


class TestBattleCmdHandlerBothProbes:
    def test_attack_and_activate_probed(self):
        """Both attack and activate options are probed."""
        conn = FakeConnection(
            [
                # Probe attack
                "m1: Blue-Eyes White Dragon",
                "m2: Dark Magician",
                "z: back.",
                "Select a card:",
                # Discard re-sent menu
                "Battle menu:",
                "a: Attack.",
                "c: Activate.",
                "e: End phase.",
                "Select an option:",
                # Probe activate
                "s1: Mirror Force",
                "z: back.",
                "Enter a line of text.",
                # Discard re-sent menu
                "Battle menu:",
                "a: Attack.",
                "c: Activate.",
                "e: End phase.",
                "Select an option:",
            ]
        )
        prompt = _make_battle_prompt(["a", "c", "e"])
        agent = PassiveAgent()  # will pick END_PHASE
        parser = MUDTextParser()

        ended = _run(BattleCmdHandler.handle(conn, prompt, agent, None, parser, verbose=False))

        assert not ended
        # Should have probed both: "a", "z", "c", "z", then "e"
        assert conn.sent[0] == "a"
        assert conn.sent[1] == "z"
        assert conn.sent[2] == "c"
        assert conn.sent[3] == "z"
        assert conn.sent[4] == "e"
        # structured_actions: 1 activate + 2 attacks + 1 end_phase
        assert len(prompt.structured_actions) == 4
        cats = [a.category for a in prompt.structured_actions]
        assert cats.count(0) == 1  # activate
        assert cats.count(1) == 2  # attacks
        assert cats.count(3) == 1  # to_ep


class TestBattleCmdHandlerGameState:
    def test_resolves_card_code_from_mzone(self):
        conn = FakeConnection(
            [
                # Probe attack
                "m1: Blue-Eyes White Dragon",
                "z: back.",
                "Select a card:",
                # Discard menu
                "Battle menu:",
                "a: Attack.",
                "e: End phase.",
                "Select an option:",
            ]
        )
        prompt = _make_battle_prompt(["a", "e"])
        agent = PassiveAgent()
        parser = MUDTextParser()
        gs = MUDGameState()
        gs.my_mzone = [
            CardEntry(name="Blue-Eyes", code=89631139, spec="m1"),
        ]

        _run(BattleCmdHandler.handle(conn, prompt, agent, gs, parser, verbose=False))

        m1_action = [a for a in prompt.structured_actions if a.cardspec == "m1"][0]
        assert m1_action.card_code == 89631139
        assert m1_action.location == LOCATION_MZONE
