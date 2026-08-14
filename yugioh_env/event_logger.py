"""Human-readable event log formatter for duel messages.

Converts parsed informational messages (summons, attacks, damage, etc.) into
descriptive strings using card names from the database. Events are accumulated
between choice points and delivered as part of observations.
"""

from __future__ import annotations

import logging

from yugioh_core.constants import (
    LOCATION_BANISHED,
    LOCATION_DECK,
    LOCATION_EXTRA,
    LOCATION_GRAVE,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_OVERLAY,
    LOCATION_SZONE,
    MSG_ATTACK,
    MSG_CHAIN_DISABLED,
    MSG_CHAIN_NEGATED,
    MSG_CHAINING,
    MSG_DAMAGE,
    MSG_DRAW,
    MSG_EQUIP,
    MSG_FLIPSUMMONING,
    MSG_MOVE,
    MSG_NEW_PHASE,
    MSG_NEW_TURN,
    MSG_PAY_LPCOST,
    MSG_POS_CHANGE,
    MSG_RECOVER,
    MSG_SET,
    MSG_SPSUMMONING,
    MSG_SUMMONING,
    MSG_TOSS_COIN,
    MSG_TOSS_DICE,
    PHASE_NAMES,
    POS_FACEDOWN,
    POS_FACEDOWN_ATTACK,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEUP_ATTACK,
    POS_FACEUP_DEFENSE,
)
from yugioh_core.string_resolver import CardTextResolver

logger = logging.getLogger(__name__)

_POS_NAMES = {
    POS_FACEUP_ATTACK: "FU-Atk",
    POS_FACEDOWN_ATTACK: "FD-Atk",
    POS_FACEUP_DEFENSE: "FU-Def",
    POS_FACEDOWN_DEFENSE: "FD-Def",
    POS_FACEUP: "FU",
    POS_FACEDOWN: "FD",
}


def _location_name(location: int, sequence: int) -> str:
    """Return human-readable location string."""
    if location == LOCATION_HAND:
        return "Hand"
    if location == LOCATION_MZONE:
        return f"MZone-{sequence}"
    if location == LOCATION_SZONE:
        return f"SZone-{sequence}"
    if location == LOCATION_GRAVE:
        return "GY"
    if location == LOCATION_BANISHED:
        return "Banished"
    if location == LOCATION_DECK:
        return "Deck"
    if location == LOCATION_EXTRA:
        return "Extra Deck"
    if location == LOCATION_OVERLAY:
        return "Overlay"
    return f"loc=0x{location:02x}"


def _position_suffix(location: int, position: int) -> str:
    """Return ``[position]`` string, or empty if not applicable."""
    # Position is only meaningful for field cards (monster/spell-trap zones)
    if location not in (LOCATION_MZONE, LOCATION_SZONE):
        return ""
    name = _POS_NAMES.get(position)
    if name:
        return f"[{name}]"
    # For spell/trap zone, face-up/face-down without atk/def
    if location == LOCATION_SZONE:
        if position & POS_FACEUP:
            return "[FU]"
        if position & POS_FACEDOWN:
            return "[FD]"
    return ""


def format_card(
    code: int,
    name: str,
    location: int,
    sequence: int = 0,
    position: int = 0,
) -> str:
    """Format card info as ``[code: name][location][position]``."""
    loc_str = _location_name(location, sequence)
    pos_str = _position_suffix(location, position)
    return f"[{code}: {name}][{loc_str}]{pos_str}"


class CardInfo:
    """Complete snapshot of a card's identity and field placement."""

    __slots__ = ("code", "controller", "location", "sequence", "position")

    def __init__(
        self,
        code: int = 0,
        controller: int = 0,
        location: int = 0,
        sequence: int = 0,
        position: int = 0,
    ) -> None:
        self.code = code
        self.controller = controller
        self.location = location
        self.sequence = sequence
        self.position = position


_EMPTY_CARD = CardInfo()


class FieldTracker:
    """Tracks full card info on the field from informational messages.

    MSG_ATTACK, MSG_BATTLE, and MSG_EQUIP only provide location info (no card
    codes or positions). This tracker resolves them by recording placements
    from summon/move/set/position-change messages.
    """

    def __init__(self) -> None:
        self._field: dict[tuple[int, int, int], CardInfo] = {}

    def update(self, msg: dict) -> None:
        """Update tracker from an informational message."""
        msg_type = msg.get("msg_type")

        if msg_type in (MSG_SUMMONING, MSG_SPSUMMONING, MSG_FLIPSUMMONING, MSG_SET):
            code = msg.get("code", 0)
            if code:
                con = msg.get("controller", 0)
                loc = msg.get("location", 0)
                seq = msg.get("sequence", 0)
                pos = msg.get("position", 0)
                self._field[(con, loc, seq)] = CardInfo(code, con, loc, seq, pos)

        elif msg_type == MSG_MOVE:
            code = msg.get("code", 0)
            if code:
                cur_con = msg.get("cur_controller", 0)
                cur_loc = msg.get("cur_location", 0)
                cur_seq = msg.get("cur_sequence", 0)
                cur_pos = msg.get("cur_position", 0)
                self._field[(cur_con, cur_loc, cur_seq)] = CardInfo(
                    code,
                    cur_con,
                    cur_loc,
                    cur_seq,
                    cur_pos,
                )
                # Remove from previous location
                prev_con = msg.get("prev_controller", 0)
                prev_loc = msg.get("prev_location", 0)
                prev_seq = msg.get("prev_sequence", 0)
                self._field.pop((prev_con, prev_loc, prev_seq), None)

        elif msg_type == MSG_POS_CHANGE:
            con = msg.get("controller", 0)
            loc = msg.get("location", 0)
            seq = msg.get("sequence", 0)
            cur_pos = msg.get("cur_position", 0)
            code = msg.get("code", 0)
            key = (con, loc, seq)
            prev = self._field.get(key)
            if prev is not None:
                prev.position = cur_pos
            elif code:
                self._field[key] = CardInfo(code, con, loc, seq, cur_pos)

    def get(self, controller: int, location: int, sequence: int) -> CardInfo:
        """Look up full card info by field position.

        Returns a ``CardInfo`` with all-zero fields if the position is unknown.
        """
        return self._field.get((controller, location, sequence), _EMPTY_CARD)

    def reset(self) -> None:
        self._field.clear()


def enrich_messages(messages: list[dict], field_tracker: FieldTracker) -> list[dict]:
    """Return copies of ``messages`` with structural facts stamped in.

    Location-only messages (MSG_ATTACK, MSG_EQUIP) name their cards only by
    field position; this fills in the card ``code`` and ``position`` from the
    tracker so downstream consumers can format without a tracker. The tracker
    is advanced over every message (in order) so its placement state stays
    consistent. Input dicts are never mutated.

    Enrichment scope is limited to the location-only messages the describer
    renders; all other messages pass through as shallow copies.

    Drift detection: a message that references a real (non-zero) field location
    the tracker has no card for means the tracker's reconstruction has diverged
    from the engine's actual field (a missed placement). That is unexpected and
    is logged as a warning rather than silently rendered as an unknown card.
    """

    def stamp(out: dict, msg: dict, prefix: str, msg_type) -> None:
        """Look up ``<prefix>_{controller,location,sequence}`` in the tracker and
        stamp ``<prefix>_code``/``<prefix>_position`` onto ``out``. Warns on
        drift (a non-zero location the tracker has no card for)."""
        controller = msg.get(f"{prefix}_controller", 0)
        location = msg.get(f"{prefix}_location", 0)
        sequence = msg.get(f"{prefix}_sequence", 0)
        info = field_tracker.get(controller, location, sequence)
        if location != 0 and info.code == 0:
            logger.warning(
                "State reconstruction drift: %s of msg_type=%s at "
                "controller=%d location=0x%02x sequence=%d has no tracked card",
                prefix,
                msg_type,
                controller,
                location,
                sequence,
            )
        out[f"{prefix}_code"] = info.code
        out[f"{prefix}_position"] = info.position

    enriched: list[dict] = []
    for msg in messages:
        field_tracker.update(msg)
        msg_type = msg.get("msg_type")
        out = dict(msg)

        if msg_type == MSG_ATTACK:
            stamp(out, msg, "attacker", msg_type)
            if msg.get("target_location", 0) != 0:
                stamp(out, msg, "target", msg_type)

        elif msg_type == MSG_EQUIP:
            stamp(out, msg, "equip", msg_type)
            stamp(out, msg, "target", msg_type)

        enriched.append(out)
    return enriched


class EventDescriber:
    """Materializes engine messages into human-readable event-log lines.

    Counterpart to ActionDescriber. Owns the stable materialization deps
    (card_db, sys_strings). Messages must be pre-enriched via enrich_messages()
    before being passed to describe(), which is a pure message→text formatter.

    Env-independent by design: takes no env handle and reads no env-private
    state.
    """

    def __init__(self, card_db, sys_strings: dict[int, str] | None = None) -> None:
        self._text = CardTextResolver(card_db, sys_strings=sys_strings)

    def describe(self, messages: list[dict], agent_player: int) -> list[str]:
        events: list[str] = []

        def tag(p: int) -> str:
            return "You:" if p == agent_player else "Opponent:"

        def card_str(code: int, location: int, sequence: int = 0, position: int = 0) -> str:
            name = self._text.card_name(code) or "?"
            return format_card(code, name, location, sequence, position)

        for msg in messages:
            msg_type = msg.get("msg_type")

            if msg_type == MSG_NEW_TURN:
                p = msg.get("player", 0)
                events.append(f"{tag(p)} Turn start")

            elif msg_type == MSG_NEW_PHASE:
                phase = msg.get("phase", 0)
                phase_name = PHASE_NAMES.get(phase, f"0x{phase:02x}")
                events.append(f"Phase: {phase_name}")

            elif msg_type == MSG_DRAW:
                p = msg.get("player", 0)
                count = len(msg.get("cards", []))
                s = "s" if count != 1 else ""
                events.append(f"{tag(p)} Draw {count} card{s}")

            elif msg_type == MSG_SUMMONING:
                p = msg.get("controller", 0)
                c = card_str(
                    msg.get("code", 0),
                    msg.get("location", 0),
                    msg.get("sequence", 0),
                    msg.get("position", 0),
                )
                events.append(f"{tag(p)} Normal Summon {c}")

            elif msg_type == MSG_SPSUMMONING:
                p = msg.get("controller", 0)
                c = card_str(
                    msg.get("code", 0),
                    msg.get("location", 0),
                    msg.get("sequence", 0),
                    msg.get("position", 0),
                )
                events.append(f"{tag(p)} Special Summon {c}")

            elif msg_type == MSG_FLIPSUMMONING:
                p = msg.get("controller", 0)
                c = card_str(
                    msg.get("code", 0),
                    msg.get("location", 0),
                    msg.get("sequence", 0),
                    msg.get("position", 0),
                )
                events.append(f"{tag(p)} Flip Summon {c}")

            elif msg_type == MSG_CHAINING:
                p = msg.get("controller", 0)
                c = card_str(
                    msg.get("code", 0),
                    msg.get("location", 0),
                    msg.get("sequence", 0),
                    msg.get("position", 0),
                )
                effect = self._text.effect_text(msg.get("desc", 0))
                chain_link = msg.get("chain_link")
                prefix = f"[Chain {chain_link}] " if chain_link else ""
                suffix = f": {effect}" if effect else ""
                events.append(f"{tag(p)} {prefix}Activate {c}{suffix}")

            elif msg_type == MSG_CHAIN_NEGATED:
                chain_link = msg.get("chain_link")
                events.append(f"[Chain {chain_link}] negated" if chain_link else "Chain is negated")

            elif msg_type == MSG_CHAIN_DISABLED:
                chain_link = msg.get("chain_link")
                events.append(
                    f"[Chain {chain_link}] disabled" if chain_link else "Chain is disabled"
                )

            elif msg_type == MSG_ATTACK:
                a_con = msg.get("attacker_controller", 0)
                attacker_str = card_str(
                    msg.get("attacker_code", 0),
                    msg.get("attacker_location", 0),
                    msg.get("attacker_sequence", 0),
                    msg.get("attacker_position", 0),
                )
                if msg.get("target_location", 0) == 0:
                    events.append(f"{tag(a_con)} {attacker_str} attacks directly")
                else:
                    t_con = msg.get("target_controller", 0)
                    target_str = card_str(
                        msg.get("target_code", 0),
                        msg.get("target_location", 0),
                        msg.get("target_sequence", 0),
                        msg.get("target_position", 0),
                    )
                    events.append(f"{tag(a_con)} {attacker_str} attacks {tag(t_con)} {target_str}")

            elif msg_type == MSG_DAMAGE:
                p = msg.get("player", 0)
                events.append(f"{tag(p)} Take {msg.get('amount', 0)} damage")

            elif msg_type == MSG_RECOVER:
                p = msg.get("player", 0)
                events.append(f"{tag(p)} Recover {msg.get('amount', 0)} LP")

            elif msg_type == MSG_PAY_LPCOST:
                p = msg.get("player", 0)
                events.append(f"{tag(p)} Pay {msg.get('amount', 0)} LP")

            elif msg_type == MSG_MOVE:
                code = msg.get("code", 0)
                prev_con = msg.get("prev_controller", 0)
                prev_loc = msg.get("prev_location", 0)
                prev_seq = msg.get("prev_sequence", 0)
                prev_pos = msg.get("prev_position", 0)
                cur_loc = msg.get("cur_location", 0)
                c = card_str(code, prev_loc, prev_seq, prev_pos)

                cur_con = msg.get("cur_controller", 0)

                if cur_loc == LOCATION_GRAVE:
                    events.append(f"{tag(prev_con)} {c} is sent to Graveyard")
                elif cur_loc == LOCATION_BANISHED:
                    events.append(f"{tag(prev_con)} {c} is banished")
                elif cur_loc == LOCATION_DECK:
                    events.append(f"{tag(cur_con)} {c} is returned to Deck")
                elif cur_loc == LOCATION_HAND and prev_loc == LOCATION_DECK:
                    events.append(f"{tag(cur_con)} {c} is added to Hand")
                elif cur_loc == LOCATION_HAND:
                    events.append(f"{tag(prev_con)} {c} is returned to Hand")

            elif msg_type == MSG_POS_CHANGE:
                p = msg.get("controller", 0)
                prev_pos = msg.get("prev_position", 0)
                cur_pos = msg.get("cur_position", 0)
                c = card_str(
                    msg.get("code", 0), msg.get("location", 0), msg.get("sequence", 0), prev_pos
                )
                pos_name = _POS_NAMES.get(cur_pos, f"0x{cur_pos:x}")
                events.append(f"{tag(p)} {c} changes position to {pos_name}")

            elif msg_type == MSG_SET:
                p = msg.get("controller", 0)
                c = card_str(
                    msg.get("code", 0),
                    msg.get("location", 0),
                    msg.get("sequence", 0),
                    msg.get("position", 0),
                )
                events.append(f"{tag(p)} Set {c}")

            elif msg_type == MSG_EQUIP:
                equip_str = card_str(
                    msg.get("equip_code", 0),
                    msg.get("equip_location", 0),
                    msg.get("equip_sequence", 0),
                    msg.get("equip_position", 0),
                )
                target_str = card_str(
                    msg.get("target_code", 0),
                    msg.get("target_location", 0),
                    msg.get("target_sequence", 0),
                    msg.get("target_position", 0),
                )
                events.append(f"{equip_str} is equipped to {target_str}")

            elif msg_type == MSG_TOSS_COIN:
                results = msg.get("results", [])
                names = ["Heads" if r else "Tails" for r in results]
                events.append(f"Coin toss: {', '.join(names)}")

            elif msg_type == MSG_TOSS_DICE:
                results = msg.get("results", [])
                events.append(f"Dice roll: {', '.join(str(r) for r in results)}")

        return events
