"""Human-readable event log formatter for duel messages.

Converts parsed informational messages (summons, attacks, damage, etc.) into
descriptive strings using card names from the database. Events are accumulated
between choice points and delivered as part of observations.
"""

from __future__ import annotations

from typing import Callable

from yugioh_env.constants import (
    LOCATION_DECK,
    LOCATION_HAND,
    LOCATION_MZONE,
    LOCATION_SZONE,
    LOCATION_GRAVE,
    LOCATION_BANISHED,
    LOCATION_EXTRA,
    LOCATION_OVERLAY,
    POS_FACEUP_ATTACK,
    POS_FACEDOWN_ATTACK,
    POS_FACEUP_DEFENSE,
    POS_FACEDOWN_DEFENSE,
    POS_FACEUP,
    POS_FACEDOWN,
    MSG_NEW_TURN,
    MSG_NEW_PHASE,
    MSG_DRAW,
    MSG_SUMMONING,
    MSG_SPSUMMONING,
    MSG_FLIPSUMMONING,
    MSG_CHAINING,
    MSG_CHAIN_NEGATED,
    MSG_ATTACK,
    MSG_DAMAGE,
    MSG_RECOVER,
    MSG_PAY_LPCOST,
    MSG_MOVE,
    MSG_POS_CHANGE,
    MSG_SET,
    MSG_EQUIP,
    MSG_TOSS_COIN,
    MSG_TOSS_DICE,
    PHASE_DRAW,
    PHASE_STANDBY,
    PHASE_MAIN1,
    PHASE_BATTLE_START,
    PHASE_BATTLE_STEP,
    PHASE_DAMAGE,
    PHASE_DAMAGE_CAL,
    PHASE_BATTLE,
    PHASE_MAIN2,
    PHASE_END,
)

_PHASE_NAMES = {
    PHASE_DRAW: "Draw",
    PHASE_STANDBY: "Standby",
    PHASE_MAIN1: "Main 1",
    PHASE_BATTLE_START: "Battle Start",
    PHASE_BATTLE_STEP: "Battle Step",
    PHASE_DAMAGE: "Damage",
    PHASE_DAMAGE_CAL: "Damage Calc",
    PHASE_BATTLE: "Battle",
    PHASE_MAIN2: "Main 2",
    PHASE_END: "End",
}

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
        if position & 0x5:  # POS_FACEUP
            return "[FU]"
        if position & 0xA:  # POS_FACEDOWN
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
                    code, cur_con, cur_loc, cur_seq, cur_pos,
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


def format_events(
    messages: list[dict],
    agent_player: int,
    get_name_fn: Callable[[int], str],
    field_tracker: FieldTracker,
) -> list[str]:
    """Convert parsed message dicts into human-readable event strings.

    Args:
        messages: List of parsed message dicts from the engine.
        agent_player: The player index of the agent (0 or 1).
        get_name_fn: Callable that maps card code -> card name string.
        field_tracker: Persistent tracker that maps field positions to card
            codes. Updated in-place so it retains knowledge across calls
            (i.e. across steps within an episode).

    Returns:
        List of human-readable event descriptions.
    """
    events: list[str] = []
    tracker = field_tracker

    def tag(p: int) -> str:
        return "You:" if p == agent_player else "Opponent:"

    def card_str(code: int, location: int, sequence: int = 0, position: int = 0) -> str:
        name = get_name_fn(code) if code else "?"
        return format_card(code, name, location, sequence, position)

    def card_str_from_info(info: CardInfo) -> str:
        name = get_name_fn(info.code) if info.code else "?"
        return format_card(info.code, name, info.location, info.sequence, info.position)

    def card_str_from_loc(controller: int, location: int, sequence: int) -> str:
        return card_str_from_info(tracker.get(controller, location, sequence))

    for msg in messages:
        tracker.update(msg)
        msg_type = msg.get("msg_type")

        if msg_type == MSG_NEW_TURN:
            p = msg.get("player", 0)
            events.append(f"{tag(p)} Turn start")

        elif msg_type == MSG_NEW_PHASE:
            phase = msg.get("phase", 0)
            phase_name = _PHASE_NAMES.get(phase, f"0x{phase:02x}")
            events.append(f"Phase: {phase_name}")

        elif msg_type == MSG_DRAW:
            p = msg.get("player", 0)
            count = len(msg.get("cards", []))
            s = "s" if count != 1 else ""
            events.append(f"{tag(p)} Draw {count} card{s}")

        elif msg_type == MSG_SUMMONING:
            p = msg.get("controller", 0)
            c = card_str(msg.get("code", 0), msg.get("location", 0), msg.get("sequence", 0), msg.get("position", 0))
            events.append(f"{tag(p)} Normal Summon {c}")

        elif msg_type == MSG_SPSUMMONING:
            p = msg.get("controller", 0)
            c = card_str(msg.get("code", 0), msg.get("location", 0), msg.get("sequence", 0), msg.get("position", 0))
            events.append(f"{tag(p)} Special Summon {c}")

        elif msg_type == MSG_FLIPSUMMONING:
            p = msg.get("controller", 0)
            c = card_str(msg.get("code", 0), msg.get("location", 0), msg.get("sequence", 0), msg.get("position", 0))
            events.append(f"{tag(p)} Flip Summon {c}")

        elif msg_type == MSG_CHAINING:
            p = msg.get("controller", 0)
            c = card_str(msg.get("code", 0), msg.get("location", 0), msg.get("sequence", 0), msg.get("position", 0))
            events.append(f"{tag(p)} Activate {c}")

        elif msg_type == MSG_CHAIN_NEGATED:
            events.append("Chain is negated")

        elif msg_type == MSG_ATTACK:
            a_con = msg.get("attacker_controller", 0)
            a_loc = msg.get("attacker_location", 0)
            a_seq = msg.get("attacker_sequence", 0)
            t_loc = msg.get("target_location", 0)
            attacker_str = card_str_from_loc(a_con, a_loc, a_seq)
            if t_loc == 0:
                events.append(f"{tag(a_con)} {attacker_str} attacks directly")
            else:
                t_con = msg.get("target_controller", 0)
                t_seq = msg.get("target_sequence", 0)
                target_str = card_str_from_loc(t_con, t_loc, t_seq)
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
            c = card_str(msg.get("code", 0), msg.get("location", 0), msg.get("sequence", 0), prev_pos)
            pos_name = _POS_NAMES.get(cur_pos, f"0x{cur_pos:x}")
            events.append(f"{tag(p)} {c} changes position to {pos_name}")

        elif msg_type == MSG_SET:
            p = msg.get("controller", 0)
            c = card_str(msg.get("code", 0), msg.get("location", 0), msg.get("sequence", 0), msg.get("position", 0))
            events.append(f"{tag(p)} Set {c}")

        elif msg_type == MSG_EQUIP:
            equip_str = card_str_from_loc(msg.get("equip_controller", 0), msg.get("equip_location", 0), msg.get("equip_sequence", 0))
            target_str = card_str_from_loc(msg.get("target_controller", 0), msg.get("target_location", 0), msg.get("target_sequence", 0))
            events.append(f"{equip_str} is equipped to {target_str}")

        elif msg_type == MSG_TOSS_COIN:
            results = msg.get("results", [])
            names = ["Heads" if r else "Tails" for r in results]
            events.append(f"Coin toss: {', '.join(names)}")

        elif msg_type == MSG_TOSS_DICE:
            results = msg.get("results", [])
            events.append(f"Dice roll: {', '.join(str(r) for r in results)}")

    return events
