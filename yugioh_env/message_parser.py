"""Binary message buffer parsing for ygopro-core (edo9300 fork).

The buffer from OCG_DuelGetMessage contains concatenated messages.
Each message is prefixed with a 4-byte length (uint32 LE, edo9300 fork),
followed by a 1-byte message type, then message-specific data.
"""

from __future__ import annotations

import struct
import logging
from typing import Any

from yugioh_env.constants import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


class BinaryReader:
    """Sequential reader for little-endian binary data."""

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes | memoryview):
        self._data = bytes(data) if isinstance(data, memoryview) else data
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos

    def u8(self) -> int:
        val = self._data[self._pos]
        self._pos += 1
        return val

    def u16(self) -> int:
        val = struct.unpack_from("<H", self._data, self._pos)[0]
        self._pos += 2
        return val

    def u32(self) -> int:
        val = struct.unpack_from("<I", self._data, self._pos)[0]
        self._pos += 4
        return val

    def i32(self) -> int:
        val = struct.unpack_from("<i", self._data, self._pos)[0]
        self._pos += 4
        return val

    def u64(self) -> int:
        val = struct.unpack_from("<Q", self._data, self._pos)[0]
        self._pos += 8
        return val

    def read_bytes(self, n: int) -> bytes:
        val = self._data[self._pos : self._pos + n]
        self._pos += n
        return val

    def skip(self, n: int) -> None:
        self._pos += n

    def read_card_loc(self) -> dict:
        """Read a card location: controller(u8), location(u8), sequence(u32), position(u32)."""
        con = self.u8()
        loc = self.u8()
        seq = self.u32()
        pos = self.u32()
        return {"controller": con, "location": loc, "sequence": seq, "position": pos}

    def read_card_loc_short(self) -> dict:
        """Read a short card location: controller(u8), location(u8), sequence(u8)."""
        con = self.u8()
        loc = self.u8()
        seq = self.u8()
        return {"controller": con, "location": loc, "sequence": seq}


def _read_card_info(r: BinaryReader) -> dict:
    """Read standard card info: code(u32) + location info."""
    code = r.u32()
    loc = r.read_card_loc()
    loc["code"] = code
    return loc


# ─── Message parsers ─────────────────────────────────────────────────────────


def _parse_select_battlecmd(r: BinaryReader) -> dict:
    """MSG_SELECT_BATTLECMD: player can choose battle actions."""
    player = r.u8()
    # Activatable cards
    act_count = r.u32()
    activatable = []
    for _ in range(act_count):
        card = {
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
            "desc": r.u64(),
        }
        activatable.append(card)
    # Attackable cards
    atk_count = r.u32()
    attackable = []
    for _ in range(atk_count):
        card = {
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u8(),
            "direct_attackable": r.u8(),
        }
        attackable.append(card)
    # Can go to Main Phase 2?
    to_m2 = r.u8()
    # Can go to End Phase?
    to_ep = r.u8()
    return {
        "msg_type": MSG_SELECT_BATTLECMD,
        "player": player,
        "activatable": activatable,
        "attackable": attackable,
        "to_m2": to_m2,
        "to_ep": to_ep,
    }


def _parse_select_idlecmd(r: BinaryReader) -> dict:
    """MSG_SELECT_IDLECMD: player chooses main phase actions."""
    player = r.u8()
    # Summonable
    sum_count = r.u32()
    summonable = []
    for _ in range(sum_count):
        summonable.append({"code": r.u32(), "controller": r.u8(), "location": r.u8(), "sequence": r.u32()})
    # Special summonable
    sps_count = r.u32()
    sp_summonable = []
    for _ in range(sps_count):
        sp_summonable.append({"code": r.u32(), "controller": r.u8(), "location": r.u8(), "sequence": r.u32()})
    # Repositionable
    repos_count = r.u32()
    repositionable = []
    for _ in range(repos_count):
        repositionable.append({"code": r.u32(), "controller": r.u8(), "location": r.u8(), "sequence": r.u32()})
    # Set-able monsters
    mset_count = r.u32()
    mset = []
    for _ in range(mset_count):
        mset.append({"code": r.u32(), "controller": r.u8(), "location": r.u8(), "sequence": r.u32()})
    # Set-able spells/traps
    sset_count = r.u32()
    sset = []
    for _ in range(sset_count):
        sset.append({"code": r.u32(), "controller": r.u8(), "location": r.u8(), "sequence": r.u32()})
    # Activatable
    act_count = r.u32()
    activatable = []
    for _ in range(act_count):
        activatable.append({
            "code": r.u32(), "controller": r.u8(), "location": r.u8(),
            "sequence": r.u32(), "desc": r.u64(),
        })
    # Can enter battle phase?
    to_bp = r.u8()
    # Can enter end phase?
    to_ep = r.u8()
    # Can shuffle hand?
    shuffle_hand = r.u8()
    return {
        "msg_type": MSG_SELECT_IDLECMD,
        "player": player,
        "summonable": summonable,
        "sp_summonable": sp_summonable,
        "repositionable": repositionable,
        "mset": mset,
        "sset": sset,
        "activatable": activatable,
        "to_bp": to_bp,
        "to_ep": to_ep,
        "shuffle_hand": shuffle_hand,
    }


def _parse_select_effectyn(r: BinaryReader) -> dict:
    """MSG_SELECT_EFFECTYN: yes/no for activating an effect."""
    player = r.u8()
    code = r.u32()
    loc = r.read_card_loc()
    desc = r.u64()
    return {
        "msg_type": MSG_SELECT_EFFECTYN,
        "player": player,
        "code": code,
        **loc,
        "desc": desc,
    }


def _parse_select_yesno(r: BinaryReader) -> dict:
    """MSG_SELECT_YESNO: generic yes/no question."""
    player = r.u8()
    desc = r.u64()
    return {"msg_type": MSG_SELECT_YESNO, "player": player, "desc": desc}


def _parse_select_option(r: BinaryReader) -> dict:
    """MSG_SELECT_OPTION: choose one from N options."""
    player = r.u8()
    count = r.u8()
    options = [r.u64() for _ in range(count)]
    return {"msg_type": MSG_SELECT_OPTION, "player": player, "options": options}


def _parse_select_card(r: BinaryReader) -> dict:
    """MSG_SELECT_CARD: select cards from a list."""
    player = r.u8()
    cancelable = r.u8()
    min_select = r.u32()
    max_select = r.u32()
    count = r.u32()
    cards = []
    for _ in range(count):
        cards.append({
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
            "subsequence": r.u32(),
        })
    return {
        "msg_type": MSG_SELECT_CARD,
        "player": player,
        "cancelable": cancelable,
        "min": min_select,
        "max": max_select,
        "cards": cards,
    }


def _parse_select_chain(r: BinaryReader) -> dict:
    """MSG_SELECT_CHAIN: select a chain to activate (or pass)."""
    player = r.u8()
    spe_count = r.u8()  # special count / forced
    forced = r.u8()
    hint_timing = r.u32()
    other_timing = r.u32()
    count = r.u32()
    chains = []
    for _ in range(count):
        chains.append({
            "flag": r.u8(),
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
            "subsequence": r.u32(),
            "desc": r.u64(),
        })
    return {
        "msg_type": MSG_SELECT_CHAIN,
        "player": player,
        "spe_count": spe_count,
        "forced": forced,
        "hint_timing": hint_timing,
        "other_timing": other_timing,
        "chains": chains,
    }


def _parse_select_place(r: BinaryReader) -> dict:
    """MSG_SELECT_PLACE / MSG_SELECT_DISFIELD: select a field zone."""
    player = r.u8()
    count = r.u8()
    field_mask = r.u32()
    return {
        "msg_type": MSG_SELECT_PLACE,
        "player": player,
        "count": count,
        "field_mask": field_mask,
    }


def _parse_select_position(r: BinaryReader) -> dict:
    """MSG_SELECT_POSITION: select card position (ATK/DEF/face-up/face-down)."""
    player = r.u8()
    code = r.u32()
    positions = r.u8()
    return {
        "msg_type": MSG_SELECT_POSITION,
        "player": player,
        "code": code,
        "positions": positions,
    }


def _parse_select_tribute(r: BinaryReader) -> dict:
    """MSG_SELECT_TRIBUTE: select cards to tribute."""
    player = r.u8()
    cancelable = r.u8()
    min_select = r.u32()
    max_select = r.u32()
    count = r.u32()
    cards = []
    for _ in range(count):
        cards.append({
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
            "release_param": r.u8(),
        })
    return {
        "msg_type": MSG_SELECT_TRIBUTE,
        "player": player,
        "cancelable": cancelable,
        "min": min_select,
        "max": max_select,
        "cards": cards,
    }


def _parse_select_sum(r: BinaryReader) -> dict:
    """MSG_SELECT_SUM: select cards whose stats sum to a value."""
    player = r.u8()
    select_type = r.u8()  # 0 = exact, 1 = at least
    target_sum = r.u32()
    min_select = r.u32()
    max_select = r.u32()
    # Must-select cards
    must_count = r.u32()
    must_cards = []
    for _ in range(must_count):
        must_cards.append({
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
            "param": r.u32(),
        })
    # Optional cards
    opt_count = r.u32()
    optional_cards = []
    for _ in range(opt_count):
        optional_cards.append({
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
            "param": r.u32(),
        })
    return {
        "msg_type": MSG_SELECT_SUM,
        "player": player,
        "select_type": select_type,
        "target_sum": target_sum,
        "min": min_select,
        "max": max_select,
        "must_cards": must_cards,
        "optional_cards": optional_cards,
    }


def _parse_select_unselect_card(r: BinaryReader) -> dict:
    """MSG_SELECT_UNSELECT_CARD: select/unselect from two lists."""
    player = r.u8()
    finishable = r.u8()
    cancelable = r.u8()
    min_select = r.u32()
    max_select = r.u32()
    select_count = r.u32()
    selectable = []
    for _ in range(select_count):
        selectable.append({
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
            "subsequence": r.u32(),
        })
    unselect_count = r.u32()
    unselectable = []
    for _ in range(unselect_count):
        unselectable.append({
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
            "subsequence": r.u32(),
        })
    return {
        "msg_type": MSG_SELECT_UNSELECT_CARD,
        "player": player,
        "finishable": finishable,
        "cancelable": cancelable,
        "min": min_select,
        "max": max_select,
        "selectable": selectable,
        "unselectable": unselectable,
    }


def _parse_sort_card(r: BinaryReader) -> dict:
    """MSG_SORT_CARD: arrange cards in order."""
    player = r.u8()
    count = r.u32()
    cards = []
    for _ in range(count):
        cards.append({
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u32(),
            "sequence": r.u32(),
        })
    return {"msg_type": MSG_SORT_CARD, "player": player, "cards": cards}


def _parse_announce_race(r: BinaryReader) -> dict:
    """MSG_ANNOUNCE_RACE: announce monster race/type."""
    player = r.u8()
    count = r.u8()
    available = r.u64()
    return {
        "msg_type": MSG_ANNOUNCE_RACE,
        "player": player,
        "count": count,
        "available": available,
    }


def _parse_announce_attrib(r: BinaryReader) -> dict:
    """MSG_ANNOUNCE_ATTRIB: announce attribute."""
    player = r.u8()
    count = r.u8()
    available = r.u32()
    return {
        "msg_type": MSG_ANNOUNCE_ATTRIB,
        "player": player,
        "count": count,
        "available": available,
    }


def _parse_announce_card(r: BinaryReader) -> dict:
    """MSG_ANNOUNCE_CARD: announce a card (filter by opcodes)."""
    player = r.u8()
    count = r.u8()
    opcodes = [r.u64() for _ in range(count)]
    return {"msg_type": MSG_ANNOUNCE_CARD, "player": player, "opcodes": opcodes}


def _parse_announce_number(r: BinaryReader) -> dict:
    """MSG_ANNOUNCE_NUMBER: announce a number."""
    player = r.u8()
    count = r.u8()
    numbers = [r.u64() for _ in range(count)]
    return {"msg_type": MSG_ANNOUNCE_NUMBER, "player": player, "numbers": numbers}


def _parse_rock_paper_scissors(r: BinaryReader) -> dict:
    """MSG_ROCK_PAPER_SCISSORS: choose rock/paper/scissors."""
    player = r.u8()
    return {"msg_type": MSG_ROCK_PAPER_SCISSORS, "player": player}


def _parse_select_counter(r: BinaryReader) -> dict:
    """MSG_SELECT_COUNTER: select counters to remove."""
    player = r.u8()
    counter_type = r.u16()
    count = r.u16()
    card_count = r.u32()
    cards = []
    for _ in range(card_count):
        cards.append({
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u8(),
            "counter_count": r.u16(),
        })
    return {
        "msg_type": MSG_SELECT_COUNTER,
        "player": player,
        "counter_type": counter_type,
        "count": count,
        "cards": cards,
    }


def _parse_sort_chain(r: BinaryReader) -> dict:
    """MSG_SORT_CHAIN: arrange chain order."""
    player = r.u8()
    count = r.u32()
    cards = []
    for _ in range(count):
        cards.append({
            "code": r.u32(),
            "controller": r.u8(),
            "location": r.u8(),
            "sequence": r.u32(),
        })
    return {"msg_type": MSG_SORT_CHAIN, "player": player, "cards": cards}


# ─── Info message parsers (non-response) ─────────────────────────────────────

def _parse_new_turn(r: BinaryReader) -> dict:
    player = r.u8()
    return {"msg_type": MSG_NEW_TURN, "player": player}


def _parse_new_phase(r: BinaryReader) -> dict:
    phase = r.u16()
    return {"msg_type": MSG_NEW_PHASE, "phase": phase}


def _parse_win(r: BinaryReader) -> dict:
    player = r.u8()
    reason = r.u8()
    return {"msg_type": MSG_WIN, "player": player, "reason": reason}


def _parse_draw(r: BinaryReader) -> dict:
    player = r.u8()
    count = r.u32()
    cards = [r.u32() for _ in range(count)]
    return {"msg_type": MSG_DRAW, "player": player, "cards": cards}


def _parse_damage(r: BinaryReader) -> dict:
    player = r.u8()
    amount = r.u32()
    return {"msg_type": MSG_DAMAGE, "player": player, "amount": amount}


def _parse_recover(r: BinaryReader) -> dict:
    player = r.u8()
    amount = r.u32()
    return {"msg_type": MSG_RECOVER, "player": player, "amount": amount}


def _parse_lpupdate(r: BinaryReader) -> dict:
    player = r.u8()
    lp = r.u32()
    return {"msg_type": MSG_LPUPDATE, "player": player, "lp": lp}


def _parse_pay_lpcost(r: BinaryReader) -> dict:
    player = r.u8()
    amount = r.u32()
    return {"msg_type": MSG_PAY_LPCOST, "player": player, "amount": amount}


def _parse_move(r: BinaryReader) -> dict:
    code = r.u32()
    prev_con = r.u8()
    prev_loc = r.u8()
    prev_seq = r.u32()
    prev_pos = r.u32()
    cur_con = r.u8()
    cur_loc = r.u8()
    cur_seq = r.u32()
    cur_pos = r.u32()
    reason = r.u32()
    return {
        "msg_type": MSG_MOVE,
        "code": code,
        "prev_controller": prev_con,
        "prev_location": prev_loc,
        "prev_sequence": prev_seq,
        "prev_position": prev_pos,
        "cur_controller": cur_con,
        "cur_location": cur_loc,
        "cur_sequence": cur_seq,
        "cur_position": cur_pos,
        "reason": reason,
    }


def _parse_pos_change(r: BinaryReader) -> dict:
    code = r.u32()
    cc = r.u8()
    cl = r.u8()
    cs = r.u8()
    pp = r.u8()
    cp = r.u8()
    return {
        "msg_type": MSG_POS_CHANGE,
        "code": code,
        "controller": cc,
        "location": cl,
        "sequence": cs,
        "prev_position": pp,
        "cur_position": cp,
    }


def _parse_set(r: BinaryReader) -> dict:
    code = r.u32()
    loc = r.read_card_loc()
    return {"msg_type": MSG_SET, "code": code, **loc}


def _parse_swap(r: BinaryReader) -> dict:
    code1 = r.u32()
    loc1 = r.read_card_loc()
    code2 = r.u32()
    loc2 = r.read_card_loc()
    return {
        "msg_type": MSG_SWAP,
        "code1": code1, **{f"card1_{k}": v for k, v in loc1.items()},
        "code2": code2, **{f"card2_{k}": v for k, v in loc2.items()},
    }


def _parse_summoning(r: BinaryReader) -> dict:
    code = r.u32()
    loc = r.read_card_loc()
    return {"msg_type": MSG_SUMMONING, "code": code, **loc}


def _parse_spsummoning(r: BinaryReader) -> dict:
    code = r.u32()
    loc = r.read_card_loc()
    return {"msg_type": MSG_SPSUMMONING, "code": code, **loc}


def _parse_flipsummoning(r: BinaryReader) -> dict:
    code = r.u32()
    loc = r.read_card_loc()
    return {"msg_type": MSG_FLIPSUMMONING, "code": code, **loc}


def _parse_chaining(r: BinaryReader) -> dict:
    code = r.u32()
    loc = r.read_card_loc()
    triggering_con = r.u8()
    triggering_loc = r.u8()
    triggering_seq = r.u32()
    desc = r.u64()
    chain_count = r.u32()
    return {
        "msg_type": MSG_CHAINING,
        "code": code,
        **loc,
        "triggering_controller": triggering_con,
        "triggering_location": triggering_loc,
        "triggering_sequence": triggering_seq,
        "desc": desc,
        "chain_count": chain_count,
    }


def _parse_attack(r: BinaryReader) -> dict:
    attacker_con = r.u8()
    attacker_loc = r.u8()
    attacker_seq = r.u8()
    r.u8()  # padding
    target_con = r.u8()
    target_loc = r.u8()
    target_seq = r.u8()
    r.u8()  # padding
    return {
        "msg_type": MSG_ATTACK,
        "attacker_controller": attacker_con,
        "attacker_location": attacker_loc,
        "attacker_sequence": attacker_seq,
        "target_controller": target_con,
        "target_location": target_loc,
        "target_sequence": target_seq,
    }


def _parse_battle(r: BinaryReader) -> dict:
    attacker_con = r.u8()
    attacker_loc = r.u8()
    attacker_seq = r.u8()
    attacker_atk = r.u32()
    attacker_def = r.u32()
    r.u8()  # destroyed flag
    target_con = r.u8()
    target_loc = r.u8()
    target_seq = r.u8()
    target_atk = r.u32()
    target_def = r.u32()
    r.u8()  # destroyed flag
    return {
        "msg_type": MSG_BATTLE,
        "attacker_controller": attacker_con,
        "attacker_location": attacker_loc,
        "attacker_sequence": attacker_seq,
        "attacker_atk": attacker_atk,
        "attacker_def": attacker_def,
        "target_controller": target_con,
        "target_location": target_loc,
        "target_sequence": target_seq,
        "target_atk": target_atk,
        "target_def": target_def,
    }


def _parse_hint(r: BinaryReader) -> dict:
    hint_type = r.u8()
    player = r.u8()
    data = r.u64()
    return {"msg_type": MSG_HINT, "hint_type": hint_type, "player": player, "data": data}


def _parse_start(r: BinaryReader) -> dict:
    # MSG_START format in edo9300: player_type(1) + lp1(4) + lp2(4) + draw1(2) + draw2(2) + extra1(2) + extra2(2)
    r.u8()  # player type
    lp0 = r.u32()
    lp1 = r.u32()
    deck0 = r.u16()
    deck1 = r.u16()
    extra0 = r.u16()
    extra1 = r.u16()
    return {
        "msg_type": MSG_START,
        "lp": [lp0, lp1],
        "deck_count": [deck0, deck1],
        "extra_count": [extra0, extra1],
    }


def _parse_card_selected(r: BinaryReader) -> dict:
    count = r.u32()
    cards = []
    for _ in range(count):
        cards.append({"controller": r.u8(), "location": r.u8(), "sequence": r.u32(), "subsequence": r.u32()})
    return {"msg_type": MSG_CARD_SELECTED, "cards": cards}


def _parse_become_target(r: BinaryReader) -> dict:
    count = r.u32()
    cards = []
    for _ in range(count):
        cards.append({"controller": r.u8(), "location": r.u8(), "sequence": r.u32()})
    return {"msg_type": MSG_BECOME_TARGET, "cards": cards}


def _parse_toss_coin(r: BinaryReader) -> dict:
    player = r.u8()
    count = r.u8()
    results = [r.u8() for _ in range(count)]
    return {"msg_type": MSG_TOSS_COIN, "player": player, "results": results}


def _parse_toss_dice(r: BinaryReader) -> dict:
    player = r.u8()
    count = r.u8()
    results = [r.u8() for _ in range(count)]
    return {"msg_type": MSG_TOSS_DICE, "player": player, "results": results}


def _parse_hand_res(r: BinaryReader) -> dict:
    res0 = r.u8()
    res1 = r.u8()
    return {"msg_type": MSG_HAND_RES, "results": [res0, res1]}


def _parse_equip(r: BinaryReader) -> dict:
    ec = r.u8()
    el = r.u8()
    es = r.u8()
    r.u8()  # padding
    tc = r.u8()
    tl = r.u8()
    ts = r.u8()
    r.u8()  # padding
    return {
        "msg_type": MSG_EQUIP,
        "equip_controller": ec,
        "equip_location": el,
        "equip_sequence": es,
        "target_controller": tc,
        "target_location": tl,
        "target_sequence": ts,
    }


def _parse_field_disabled(r: BinaryReader) -> dict:
    field = r.u32()
    return {"msg_type": MSG_FIELD_DISABLED, "field_mask": field}


def _parse_card_hint(r: BinaryReader) -> dict:
    r.u8()  # controller
    r.u8()  # location
    r.u8()  # sequence
    r.u8()  # padding
    hint_type = r.u8()
    value = r.u64()
    return {"msg_type": MSG_CARD_HINT, "hint_type": hint_type, "value": value}


def _parse_shuffle_deck(r: BinaryReader) -> dict:
    player = r.u8()
    return {"msg_type": MSG_SHUFFLE_DECK, "player": player}


def _parse_shuffle_hand(r: BinaryReader) -> dict:
    player = r.u8()
    count = r.u32()
    cards = [r.u32() for _ in range(count)]
    return {"msg_type": MSG_SHUFFLE_HAND, "player": player, "cards": cards}


def _parse_shuffle_extra(r: BinaryReader) -> dict:
    player = r.u8()
    count = r.u32()
    cards = [r.u32() for _ in range(count)]
    return {"msg_type": MSG_SHUFFLE_EXTRA, "player": player, "cards": cards}


def _parse_shuffle_set_card(r: BinaryReader) -> dict:
    loc = r.u8()
    count = r.u8()
    # old positions then new positions
    old_locs = []
    for _ in range(count):
        old_locs.append(r.read_card_loc())
    new_locs = []
    for _ in range(count):
        new_locs.append(r.read_card_loc())
    return {"msg_type": MSG_SHUFFLE_SET_CARD, "location": loc, "old": old_locs, "new": new_locs}


def _parse_confirm_decktop(r: BinaryReader) -> dict:
    player = r.u8()
    count = r.u32()
    cards = []
    for _ in range(count):
        cards.append({"code": r.u32(), "controller": r.u8(), "location": r.u8(), "sequence": r.u32()})
    return {"msg_type": MSG_CONFIRM_DECKTOP, "player": player, "cards": cards}


def _parse_confirm_cards(r: BinaryReader) -> dict:
    player = r.u8()
    count = r.u32()
    cards = []
    for _ in range(count):
        cards.append({"code": r.u32(), "controller": r.u8(), "location": r.u8(), "sequence": r.u32()})
    return {"msg_type": MSG_CONFIRM_CARDS, "player": player, "cards": cards}


def _parse_add_counter(r: BinaryReader) -> dict:
    counter_type = r.u16()
    con = r.u8()
    loc = r.u8()
    seq = r.u8()
    count = r.u16()
    return {
        "msg_type": MSG_ADD_COUNTER,
        "counter_type": counter_type,
        "controller": con,
        "location": loc,
        "sequence": seq,
        "count": count,
    }


def _parse_remove_counter(r: BinaryReader) -> dict:
    counter_type = r.u16()
    con = r.u8()
    loc = r.u8()
    seq = r.u8()
    count = r.u16()
    return {
        "msg_type": MSG_REMOVE_COUNTER,
        "counter_type": counter_type,
        "controller": con,
        "location": loc,
        "sequence": seq,
        "count": count,
    }


def _parse_card_target(r: BinaryReader) -> dict:
    ec = r.u8()
    el = r.u8()
    es = r.u8()
    r.u8()
    tc = r.u8()
    tl = r.u8()
    ts = r.u8()
    r.u8()
    return {
        "msg_type": MSG_CARD_TARGET,
        "equip_controller": ec, "equip_location": el, "equip_sequence": es,
        "target_controller": tc, "target_location": tl, "target_sequence": ts,
    }


def _parse_cancel_target(r: BinaryReader) -> dict:
    ec = r.u8()
    el = r.u8()
    es = r.u8()
    r.u8()
    tc = r.u8()
    tl = r.u8()
    ts = r.u8()
    r.u8()
    return {
        "msg_type": MSG_CANCEL_TARGET,
        "equip_controller": ec, "equip_location": el, "equip_sequence": es,
        "target_controller": tc, "target_location": tl, "target_sequence": ts,
    }


def _parse_deck_top(r: BinaryReader) -> dict:
    player = r.u8()
    count = r.u32()
    code = r.u32()
    return {"msg_type": MSG_DECK_TOP, "player": player, "count": count, "code": code}


def _parse_noop(r: BinaryReader) -> dict:
    """Parser for messages with no additional data."""
    return {}


# ─── Dispatch table ──────────────────────────────────────────────────────────

MSG_PARSERS: dict[int, Any] = {
    # Player-choice messages (require response)
    MSG_SELECT_BATTLECMD: _parse_select_battlecmd,
    MSG_SELECT_IDLECMD: _parse_select_idlecmd,
    MSG_SELECT_EFFECTYN: _parse_select_effectyn,
    MSG_SELECT_YESNO: _parse_select_yesno,
    MSG_SELECT_OPTION: _parse_select_option,
    MSG_SELECT_CARD: _parse_select_card,
    MSG_SELECT_CHAIN: _parse_select_chain,
    MSG_SELECT_PLACE: _parse_select_place,
    MSG_SELECT_DISFIELD: _parse_select_place,  # same format
    MSG_SELECT_POSITION: _parse_select_position,
    MSG_SELECT_TRIBUTE: _parse_select_tribute,
    MSG_SELECT_SUM: _parse_select_sum,
    MSG_SELECT_UNSELECT_CARD: _parse_select_unselect_card,
    MSG_SORT_CARD: _parse_sort_card,
    MSG_SORT_CHAIN: _parse_sort_chain,
    MSG_ANNOUNCE_RACE: _parse_announce_race,
    MSG_ANNOUNCE_ATTRIB: _parse_announce_attrib,
    MSG_ANNOUNCE_CARD: _parse_announce_card,
    MSG_ANNOUNCE_NUMBER: _parse_announce_number,
    MSG_ROCK_PAPER_SCISSORS: _parse_rock_paper_scissors,
    MSG_SELECT_COUNTER: _parse_select_counter,
    # Info messages
    MSG_HINT: _parse_hint,
    MSG_START: _parse_start,
    MSG_WIN: _parse_win,
    MSG_NEW_TURN: _parse_new_turn,
    MSG_NEW_PHASE: _parse_new_phase,
    MSG_DRAW: _parse_draw,
    MSG_DAMAGE: _parse_damage,
    MSG_RECOVER: _parse_recover,
    MSG_LPUPDATE: _parse_lpupdate,
    MSG_PAY_LPCOST: _parse_pay_lpcost,
    MSG_MOVE: _parse_move,
    MSG_POS_CHANGE: _parse_pos_change,
    MSG_SET: _parse_set,
    MSG_SWAP: _parse_swap,
    MSG_SUMMONING: _parse_summoning,
    MSG_SUMMONED: _parse_noop,
    MSG_SPSUMMONING: _parse_spsummoning,
    MSG_SPSUMMONED: _parse_noop,
    MSG_FLIPSUMMONING: _parse_flipsummoning,
    MSG_FLIPSUMMONED: _parse_noop,
    MSG_CHAINING: _parse_chaining,
    MSG_CHAINED: _parse_noop,
    MSG_CHAIN_SOLVING: _parse_noop,
    MSG_CHAIN_SOLVED: _parse_noop,
    MSG_CHAIN_END: _parse_noop,
    MSG_CHAIN_NEGATED: _parse_noop,
    MSG_CHAIN_DISABLED: _parse_noop,
    MSG_ATTACK: _parse_attack,
    MSG_BATTLE: _parse_battle,
    MSG_ATTACK_DISABLED: _parse_noop,
    MSG_DAMAGE_STEP_START: _parse_noop,
    MSG_DAMAGE_STEP_END: _parse_noop,
    MSG_CARD_SELECTED: _parse_card_selected,
    MSG_BECOME_TARGET: _parse_become_target,
    MSG_TOSS_COIN: _parse_toss_coin,
    MSG_TOSS_DICE: _parse_toss_dice,
    MSG_HAND_RES: _parse_hand_res,
    MSG_EQUIP: _parse_equip,
    MSG_UNEQUIP: _parse_noop,
    MSG_FIELD_DISABLED: _parse_field_disabled,
    MSG_CARD_HINT: _parse_card_hint,
    MSG_SHUFFLE_DECK: _parse_shuffle_deck,
    MSG_SHUFFLE_HAND: _parse_shuffle_hand,
    MSG_SHUFFLE_EXTRA: _parse_shuffle_extra,
    MSG_SHUFFLE_SET_CARD: _parse_shuffle_set_card,
    MSG_CONFIRM_DECKTOP: _parse_confirm_decktop,
    MSG_CONFIRM_CARDS: _parse_confirm_cards,
    MSG_CONFIRM_EXTRATOP: _parse_confirm_decktop,  # same format
    MSG_DECK_TOP: _parse_deck_top,
    MSG_ADD_COUNTER: _parse_add_counter,
    MSG_REMOVE_COUNTER: _parse_remove_counter,
    MSG_CARD_TARGET: _parse_card_target,
    MSG_CANCEL_TARGET: _parse_cancel_target,
    MSG_MISSED_EFFECT: _parse_noop,
    MSG_WAITING: _parse_noop,
    MSG_RETRY: _parse_noop,
    MSG_UPDATE_DATA: _parse_noop,
    MSG_UPDATE_CARD: _parse_noop,
    MSG_REFRESH_DECK: _parse_noop,
    MSG_SWAP_GRAVE_DECK: _parse_noop,
    MSG_REVERSE_DECK: _parse_noop,
    MSG_RANDOM_SELECTED: _parse_become_target,  # same format
    MSG_BE_CHAIN_TARGET: _parse_noop,
    MSG_CREATE_RELATION: _parse_noop,
    MSG_RELEASE_RELATION: _parse_noop,
    MSG_MATCH_KILL: _parse_noop,
    MSG_CUSTOM_MSG: _parse_noop,
    MSG_REMOVE_CARDS: _parse_noop,
    MSG_TAG_SWAP: _parse_noop,
    MSG_RELOAD_FIELD: _parse_noop,
    MSG_AI_NAME: _parse_noop,
    MSG_SHOW_HINT: _parse_noop,
    MSG_PLAYER_HINT: _parse_noop,
}


def parse_messages(buffer: bytes) -> list[dict]:
    """Parse a complete message buffer from OCG_DuelGetMessage.

    In the edo9300 fork, each message is prefixed with a 4-byte length (uint32 LE).

    Returns:
        List of parsed message dicts, each with at least a 'msg_type' key.
    """
    messages = []
    r = BinaryReader(buffer)

    while r.remaining >= 4:
        msg_len = r.u32()
        if msg_len == 0 or r.remaining < 1:
            break

        start_pos = r.pos
        msg_type = r.u8()

        parser = MSG_PARSERS.get(msg_type)
        if parser is not None:
            try:
                msg = parser(r)
                if "msg_type" not in msg:
                    msg["msg_type"] = msg_type
                messages.append(msg)
            except Exception as e:
                logger.warning("Failed to parse msg_type=%d: %s", msg_type, e)
                # Skip the rest of this message
                r._pos = start_pos + msg_len
        else:
            logger.debug("Unknown msg_type=%d, skipping %d bytes", msg_type, msg_len)

        # Ensure we advance to the next message boundary
        expected_end = start_pos + msg_len
        if r.pos < expected_end:
            r._pos = expected_end
        elif r.pos > expected_end:
            logger.warning(
                "Parser overread for msg_type=%d: read %d bytes but length was %d",
                msg_type, r.pos - start_pos, msg_len,
            )

    return messages
