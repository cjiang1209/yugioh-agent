"""MUD duel game state tracker.

Maintains zone contents, LP, turn number, and current phase by consuming
``ParsedEvent`` objects from the text parser.  Supports periodic resync
from ground-truth ``hand``/``tab``/``score`` command responses.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from yugioh_mud.card_lookup import CardNameLookup
from yugioh_mud.text_parser import EventType, ParsedEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Card entry — one card tracked in a zone
# ---------------------------------------------------------------------------

@dataclass
class CardEntry:
    """A single card tracked in a zone."""
    name: str = ""
    code: int = 0       # passcode from CardNameLookup (0 = unknown)
    position: str = ""  # "face-up attack", "face-down defense", etc.
    spec: str = ""      # zone spec e.g. "m1", "s2"


# ---------------------------------------------------------------------------
# Resync response parsers
# ---------------------------------------------------------------------------

# Score response: "Your LP: {n} Opponent LP: {n}"
_SCORE_LP_RE = re.compile(
    r"^Your LP: (\d+) Opponent LP: (\d+)$")
_SCORE_COUNTS_RE = re.compile(
    r"^(Hand|Deck|Grave|Removed): You: (\d+) Opponent: (\d+)$")
_SCORE_TURN_RE = re.compile(r"^It's your turn\.$")
_SCORE_OPP_TURN_RE = re.compile(r"^It's (.+?)'s turn\.$")

# Hand response: "{spec} {card_name}" (e.g. "h1 Dark Magician")
_HAND_CARD_RE = re.compile(r"^(h\d+) (.+)$")

# Tab response: monster zone "m{n}: {name} ({atk}/{def}) ..."
_TAB_MONSTER_RE = re.compile(
    r"^(m\d+): (.+?) \((\d+)/(\d+)\)")
# Tab response: spell/trap zone "s{n}: {name} {position}"
_TAB_SPELL_RE = re.compile(r"^(s\d+): (.+?) (face-.+)$")
# Tab response: face-down (no name) "m{n}: face-down defense" etc.
_TAB_FACEDOWN_RE = re.compile(r"^([ms]\d+): (face-.+)$")


# ---------------------------------------------------------------------------
# MUDGameState
# ---------------------------------------------------------------------------

class MUDGameState:
    """Tracks duel state from parsed events.

    Zones are lists of ``CardEntry``.  Opponent hand and face-down S/T
    are tracked as count only (``code=0``, ``name=""``).
    """

    def __init__(self, card_lookup: CardNameLookup | None = None) -> None:
        self._lookup = card_lookup

        # LP
        self.my_lp: int = 8000
        self.opp_lp: int = 8000

        # Turn / phase
        self.turn: int = 0
        self.phase: str = ""
        self.is_my_turn: bool = False

        # Zones — own
        self.my_hand: list[CardEntry] = []
        self.my_mzone: list[CardEntry] = []
        self.my_szone: list[CardEntry] = []
        self.my_graveyard: list[CardEntry] = []
        self.my_banished: list[CardEntry] = []

        # Zones — opponent
        self.opp_hand_count: int = 0
        self.opp_mzone: list[CardEntry] = []
        self.opp_szone: list[CardEntry] = []
        self.opp_graveyard: list[CardEntry] = []
        self.opp_banished: list[CardEntry] = []

    def _resolve_code(self, name: str) -> int:
        """Look up passcode for *name*, returning 0 if unknown."""
        if not name or self._lookup is None:
            return 0
        return self._lookup.name_to_code(name) or 0

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def update(self, event: ParsedEvent) -> None:
        """Update state from a single parsed event."""
        handler = _EVENT_HANDLERS.get(event.event_type)
        if handler is not None:
            handler(self, event)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_new_turn(self, ev: ParsedEvent) -> None:
        self.turn += 1
        self.is_my_turn = not ev.is_opponent

    def _on_new_phase(self, ev: ParsedEvent) -> None:
        self.phase = ev.phase

    def _on_damage(self, ev: ParsedEvent) -> None:
        if ev.is_opponent:
            self.opp_lp = ev.new_lp
        else:
            self.my_lp = ev.new_lp

    def _on_recover(self, ev: ParsedEvent) -> None:
        if ev.is_opponent:
            self.opp_lp = ev.new_lp
        else:
            self.my_lp = ev.new_lp

    def _on_pay_lp(self, ev: ParsedEvent) -> None:
        if ev.is_opponent:
            self.opp_lp = ev.new_lp
        else:
            self.my_lp = ev.new_lp

    def _on_draw(self, ev: ParsedEvent) -> None:
        if ev.is_opponent:
            self.opp_hand_count += ev.amount
        else:
            # We don't know card names from "Drew N cards:" alone.
            # Individual card lines following the draw header are not
            # parsed as separate events (they're sub-lines of the draw
            # message). For now, add unknown entries.
            for _ in range(ev.amount):
                self.my_hand.append(CardEntry())

    def _on_summon(self, ev: ParsedEvent) -> None:
        entry = CardEntry(
            name=ev.card_name,
            code=self._resolve_code(ev.card_name),
            position=ev.position,
            spec=ev.card_spec,
        )
        if ev.is_opponent:
            self.opp_mzone.append(entry)
        else:
            self.my_mzone.append(entry)

    _on_sp_summon = _on_summon
    _on_flip_summon = _on_summon

    def _on_set(self, ev: ParsedEvent) -> None:
        entry = CardEntry(
            name=ev.card_name,
            code=self._resolve_code(ev.card_name),
            position=ev.position,
            spec=ev.card_spec,
        )
        # Determine zone from spec: "m*" = monster, "s*" = spell/trap
        if ev.card_spec.startswith("s"):
            if ev.is_opponent:
                self.opp_szone.append(entry)
            else:
                self.my_szone.append(entry)
        else:
            if ev.is_opponent:
                self.opp_mzone.append(entry)
            else:
                self.my_mzone.append(entry)

    def _on_pos_change(self, ev: ParsedEvent) -> None:
        spec = ev.card_spec
        # Search own zones first, then opponent
        for zone in (self.my_mzone, self.opp_mzone):
            for card in zone:
                if card.spec == spec:
                    card.position = ev.position
                    if ev.card_name:
                        card.name = ev.card_name
                        card.code = self._resolve_code(ev.card_name)
                    return

    def _on_attack(self, ev: ParsedEvent) -> None:
        pass  # Attack declaration doesn't change zone state

    def _on_chaining(self, ev: ParsedEvent) -> None:
        pass  # Chain activation doesn't move cards

    def _on_destroy(self, ev: ParsedEvent) -> None:
        # Destroy removes from field; the subsequent TO_GRAVEYARD/BANISHED
        # event handles adding to the destination zone.
        self._remove_from_field(ev.card_spec, ev.card_name)

    def _on_to_graveyard(self, ev: ParsedEvent) -> None:
        removed = self._remove_from_field(ev.card_spec, ev.card_name)
        entry = removed or CardEntry(
            name=ev.card_name,
            code=self._resolve_code(ev.card_name),
            spec=ev.card_spec,
        )
        if ev.is_opponent:
            self.opp_graveyard.append(entry)
        else:
            self.my_graveyard.append(entry)

    def _on_banished(self, ev: ParsedEvent) -> None:
        removed = self._remove_from_field(ev.card_spec, ev.card_name)
        entry = removed or CardEntry(
            name=ev.card_name,
            code=self._resolve_code(ev.card_name),
            spec=ev.card_spec,
        )
        if ev.is_opponent:
            self.opp_banished.append(entry)
        else:
            self.my_banished.append(entry)

    def _on_to_hand(self, ev: ParsedEvent) -> None:
        self._remove_from_field(ev.card_spec, ev.card_name)
        if ev.is_opponent:
            self.opp_hand_count += 1
        else:
            self.my_hand.append(CardEntry(
                name=ev.card_name,
                code=self._resolve_code(ev.card_name),
                spec=ev.card_spec,
            ))

    def _on_to_deck(self, ev: ParsedEvent) -> None:
        self._remove_from_field(ev.card_spec, ev.card_name)
        # Deck contents not tracked

    def _on_to_extra_deck(self, ev: ParsedEvent) -> None:
        self._remove_from_field(ev.card_spec, ev.card_name)
        # Extra deck contents not tracked

    def _on_tribute(self, ev: ParsedEvent) -> None:
        self._remove_from_field(ev.card_spec, ev.card_name)
        # Tributed card goes to GY — tracked by subsequent TO_GRAVEYARD event

    def _on_discard(self, ev: ParsedEvent) -> None:
        if ev.is_opponent:
            self.opp_hand_count = max(0, self.opp_hand_count - 1)
        else:
            self._remove_from_hand(ev.card_name)
        # Discarded card goes to GY — tracked by subsequent TO_GRAVEYARD event

    def _on_equip(self, ev: ParsedEvent) -> None:
        pass  # Equip doesn't change zone membership

    def _on_shuffle(self, ev: ParsedEvent) -> None:
        pass  # Shuffle doesn't change tracked state

    def _on_win(self, ev: ParsedEvent) -> None:
        pass  # Terminal — no state update needed

    def _on_lose(self, ev: ParsedEvent) -> None:
        pass  # Terminal — no state update needed

    # ------------------------------------------------------------------
    # Zone helpers
    # ------------------------------------------------------------------

    def _remove_from_field(
        self, spec: str, name: str = "",
    ) -> CardEntry | None:
        """Remove a card from field zones by spec, returning it if found."""
        for zone in (self.my_mzone, self.my_szone,
                     self.opp_mzone, self.opp_szone):
            for i, card in enumerate(zone):
                if card.spec == spec:
                    return zone.pop(i)
                # Also match by name if spec doesn't match (spec may
                # have changed after zone switches)
                if name and card.name == name and not card.spec:
                    return zone.pop(i)
        return None

    def _remove_from_hand(self, name: str) -> CardEntry | None:
        """Remove a card from own hand by name."""
        for i, card in enumerate(self.my_hand):
            if card.name == name:
                return self.my_hand.pop(i)
        # If not found by name, remove last unknown entry
        for i in range(len(self.my_hand) - 1, -1, -1):
            if not self.my_hand[i].name:
                return self.my_hand.pop(i)
        return None

    # ------------------------------------------------------------------
    # Resync from server commands
    # ------------------------------------------------------------------

    def resync_score(self, lines: list[str]) -> None:
        """Parse ``score`` command response and overwrite LP/counts."""
        for line in lines:
            m = _SCORE_LP_RE.match(line)
            if m:
                old_my, old_opp = self.my_lp, self.opp_lp
                self.my_lp = int(m.group(1))
                self.opp_lp = int(m.group(2))
                if old_my != self.my_lp or old_opp != self.opp_lp:
                    logger.warning(
                        "Resync LP drift: my %d→%d, opp %d→%d",
                        old_my, self.my_lp, old_opp, self.opp_lp)
                continue

            m = _SCORE_COUNTS_RE.match(line)
            if m:
                label = m.group(1)
                my_count, opp_count = int(m.group(2)), int(m.group(3))
                if label == "Hand":
                    if len(self.my_hand) != my_count:
                        logger.warning(
                            "Resync hand drift: tracked %d, actual %d",
                            len(self.my_hand), my_count)
                    if self.opp_hand_count != opp_count:
                        logger.warning(
                            "Resync opp hand drift: tracked %d, actual %d",
                            self.opp_hand_count, opp_count)
                        self.opp_hand_count = opp_count
                elif label == "Grave":
                    if len(self.my_graveyard) != my_count:
                        logger.warning(
                            "Resync GY drift: tracked %d, actual %d",
                            len(self.my_graveyard), my_count)
                continue

            if _SCORE_TURN_RE.match(line):
                self.is_my_turn = True
            elif _SCORE_OPP_TURN_RE.match(line):
                self.is_my_turn = False

    def resync_hand(self, lines: list[str]) -> None:
        """Parse ``hand`` command response and overwrite own hand."""
        new_hand: list[CardEntry] = []
        for line in lines:
            if line == "No cards.":
                break
            m = _HAND_CARD_RE.match(line)
            if m:
                name = m.group(2)
                new_hand.append(CardEntry(
                    name=name,
                    code=self._resolve_code(name),
                    spec=m.group(1),
                ))
        if new_hand or any(l == "No cards." for l in lines):
            old_count = len(self.my_hand)
            self.my_hand = new_hand
            if old_count != len(self.my_hand):
                logger.warning(
                    "Resync hand: %d → %d cards",
                    old_count, len(self.my_hand))

    def resync_tab(self, lines: list[str], opponent: bool = False) -> None:
        """Parse ``tab``/``tab2`` response and overwrite field zones."""
        new_mzone: list[CardEntry] = []
        new_szone: list[CardEntry] = []
        for line in lines:
            if line in ("Your table:", "Opponent's table:",
                        "Table is empty."):
                continue
            # Try face-down first (subset of monster/spell patterns)
            m = _TAB_FACEDOWN_RE.match(line)
            if m:
                spec, pos = m.group(1), m.group(2)
                entry = CardEntry(position=pos, spec=spec)
                if spec.startswith("m"):
                    new_mzone.append(entry)
                else:
                    new_szone.append(entry)
                continue
            m = _TAB_MONSTER_RE.match(line)
            if m:
                spec, name = m.group(1), m.group(2)
                new_mzone.append(CardEntry(
                    name=name,
                    code=self._resolve_code(name),
                    spec=spec,
                ))
                continue
            m = _TAB_SPELL_RE.match(line)
            if m:
                spec, name, pos = m.group(1), m.group(2), m.group(3)
                new_szone.append(CardEntry(
                    name=name,
                    code=self._resolve_code(name),
                    position=pos,
                    spec=spec,
                ))
                continue
        if opponent:
            self.opp_mzone = new_mzone
            self.opp_szone = new_szone
        else:
            self.my_mzone = new_mzone
            self.my_szone = new_szone


# Handler dispatch table (avoids long if/elif chain in update())
_EVENT_HANDLERS: dict[EventType, Callable[[MUDGameState, ParsedEvent], None]] = {
    EventType.NEW_TURN: MUDGameState._on_new_turn,
    EventType.NEW_PHASE: MUDGameState._on_new_phase,
    EventType.DAMAGE: MUDGameState._on_damage,
    EventType.RECOVER: MUDGameState._on_recover,
    EventType.PAY_LP: MUDGameState._on_pay_lp,
    EventType.DRAW: MUDGameState._on_draw,
    EventType.SUMMON: MUDGameState._on_summon,
    EventType.SP_SUMMON: MUDGameState._on_sp_summon,
    EventType.FLIP_SUMMON: MUDGameState._on_flip_summon,
    EventType.SET: MUDGameState._on_set,
    EventType.POS_CHANGE: MUDGameState._on_pos_change,
    EventType.ATTACK: MUDGameState._on_attack,
    EventType.CHAINING: MUDGameState._on_chaining,
    EventType.DESTROY: MUDGameState._on_destroy,
    EventType.TO_GRAVEYARD: MUDGameState._on_to_graveyard,
    EventType.BANISHED: MUDGameState._on_banished,
    EventType.TO_HAND: MUDGameState._on_to_hand,
    EventType.TO_DECK: MUDGameState._on_to_deck,
    EventType.TO_EXTRA_DECK: MUDGameState._on_to_extra_deck,
    EventType.TRIBUTE: MUDGameState._on_tribute,
    EventType.DISCARD: MUDGameState._on_discard,
    EventType.EQUIP: MUDGameState._on_equip,
    EventType.SHUFFLE: MUDGameState._on_shuffle,
    EventType.WIN: MUDGameState._on_win,
    EventType.LOSE: MUDGameState._on_lose,
}
