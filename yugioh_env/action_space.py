"""Action masking and action-to-response mapping."""

from __future__ import annotations

import logging
from itertools import combinations
import numpy as np

from yugioh_env.constants import *  # noqa: F401,F403
from yugioh_env.observation import MAX_ACTIONS, ACTION_FEATURES
from yugioh_env import response_builder as rb

logger = logging.getLogger(__name__)


class ActionMapper:
    """Maps parsed SELECT messages to a fixed-size action space.

    Extracts available actions from a SELECT message, provides action masks
    and feature vectors, and converts action indices to binary responses.

    Stateless: all multi-step orchestration (e.g. accumulating card picks)
    is handled by the caller.  ``action_to_response`` returns ``None`` for
    intermediate picks whose ``build_response`` is ``None``; the caller
    re-presents updated choices by calling ``update`` with an augmented
    message.
    """

    def __init__(self):
        self._actions: list[dict] = []
        self._msg_type: int = 0
        self._msg: dict = {}

    def update(self, msg: dict) -> None:
        """Update with a new SELECT message. Extracts all legal actions."""
        self._msg = msg
        self._msg_type = msg.get("msg_type", 0)
        self._actions = []

        handler = _ACTION_EXTRACTORS.get(self._msg_type)
        if handler:
            self._actions = handler(msg)

        if len(self._actions) > MAX_ACTIONS:
            logger.warning(
                "Action count %d exceeds MAX_ACTIONS=%d for msg_type=%d, truncating",
                len(self._actions), MAX_ACTIONS, self._msg_type,
            )
            self._actions = self._actions[:MAX_ACTIONS]

    @property
    def num_actions(self) -> int:
        return len(self._actions)

    def get_action_mask(self) -> np.ndarray:
        """Return binary mask: 1 for legal actions, 0 otherwise."""
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        mask[: len(self._actions)] = 1
        return mask

    def get_action_features(self) -> np.ndarray:
        """Return feature vectors for each action slot."""
        features = np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.uint8)
        for i, action in enumerate(self._actions):
            features[i] = _encode_action(action, self._msg_type)
        return features

    def get_action_index(self, idx: int) -> int:
        """Return the card/item index for action *idx*."""
        if idx < 0 or idx >= len(self._actions):
            raise ValueError(f"Action index {idx} out of range [0, {len(self._actions)})")
        return self._actions[idx].get("index", 0)

    def action_to_response(self, idx: int) -> bytes | None:
        """Convert an action index to the binary response buffer.

        Returns ``None`` when the action's ``build_response`` is ``None``
        (intermediate multi-step pick).  The caller should accumulate the
        pick, call ``update`` with an augmented message, and re-present.
        """
        if idx < 0 or idx >= len(self._actions):
            raise ValueError(f"Action index {idx} out of range [0, {len(self._actions)})")
        action = self._actions[idx]
        br = action.get("build_response")
        if br is None:
            return None
        return br()


# ─── Action extraction per message type ──────────────────────────────────────

def _extract_idle_actions(msg: dict) -> list[dict]:
    """Extract actions from MSG_SELECT_IDLECMD."""
    actions = []

    for i, card in enumerate(msg.get("summonable", [])):
        actions.append({
            "category": 0, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "build_response": lambda cat=0, idx=i: rb.build_select_idlecmd_response(cat, idx),
        })

    for i, card in enumerate(msg.get("sp_summonable", [])):
        actions.append({
            "category": 1, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "build_response": lambda cat=1, idx=i: rb.build_select_idlecmd_response(cat, idx),
        })

    for i, card in enumerate(msg.get("repositionable", [])):
        actions.append({
            "category": 2, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "build_response": lambda cat=2, idx=i: rb.build_select_idlecmd_response(cat, idx),
        })

    for i, card in enumerate(msg.get("mset", [])):
        actions.append({
            "category": 3, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "build_response": lambda cat=3, idx=i: rb.build_select_idlecmd_response(cat, idx),
        })

    for i, card in enumerate(msg.get("sset", [])):
        actions.append({
            "category": 4, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "build_response": lambda cat=4, idx=i: rb.build_select_idlecmd_response(cat, idx),
        })

    for i, card in enumerate(msg.get("activatable", [])):
        actions.append({
            "category": 5, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "build_response": lambda cat=5, idx=i: rb.build_select_idlecmd_response(cat, idx),
        })

    if msg.get("to_bp"):
        actions.append({
            "category": 6, "index": 0, "code": 0, "location": 0, "sequence": 0,
            "build_response": lambda: rb.build_select_idlecmd_response(6, 0),
        })

    if msg.get("to_ep"):
        actions.append({
            "category": 7, "index": 0, "code": 0, "location": 0, "sequence": 0,
            "build_response": lambda: rb.build_select_idlecmd_response(7, 0),
        })

    return actions


def _extract_battle_actions(msg: dict) -> list[dict]:
    """Extract actions from MSG_SELECT_BATTLECMD."""
    actions = []

    for i, card in enumerate(msg.get("activatable", [])):
        actions.append({
            "category": 0, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "build_response": lambda cat=0, idx=i: rb.build_select_battlecmd_response(cat, idx),
        })

    for i, card in enumerate(msg.get("attackable", [])):
        actions.append({
            "category": 1, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "build_response": lambda cat=1, idx=i: rb.build_select_battlecmd_response(cat, idx),
        })

    if msg.get("to_m2"):
        actions.append({
            "category": 2, "index": 0, "code": 0, "location": 0, "sequence": 0,
            "build_response": lambda: rb.build_select_battlecmd_response(2, 0),
        })

    if msg.get("to_ep"):
        actions.append({
            "category": 3, "index": 0, "code": 0, "location": 0, "sequence": 0,
            "build_response": lambda: rb.build_select_battlecmd_response(3, 0),
        })

    return actions


def _extract_effectyn_actions(msg: dict) -> list[dict]:
    return [
        {"category": 0, "index": 0, "code": msg.get("code", 0), "location": 0, "sequence": 0,
         "build_response": lambda: rb.build_select_yesno_response(True)},
        {"category": 1, "index": 0, "code": msg.get("code", 0), "location": 0, "sequence": 0,
         "build_response": lambda: rb.build_select_yesno_response(False)},
    ]


def _extract_yesno_actions(msg: dict) -> list[dict]:
    return [
        {"category": 0, "index": 0, "code": 0, "location": 0, "sequence": 0,
         "build_response": lambda: rb.build_select_yesno_response(True)},
        {"category": 1, "index": 0, "code": 0, "location": 0, "sequence": 0,
         "build_response": lambda: rb.build_select_yesno_response(False)},
    ]


def _extract_option_actions(msg: dict) -> list[dict]:
    options = msg.get("options", [])
    return [
        {"category": 0, "index": i, "code": 0, "location": 0, "sequence": 0,
         "build_response": lambda idx=i: rb.build_select_option_response(idx)}
        for i, _ in enumerate(options)
    ]


def _extract_card_actions(msg: dict) -> list[dict]:
    """For MSG_SELECT_CARD: present individual card picks per step.

    The caller drives multi-step selection by injecting ``_selected``
    (list of already-chosen card indices) into the message before each
    ``update`` call.  On each step this extractor presents the remaining
    cards; picks that complete the selection get a real ``build_response``,
    intermediate picks get ``None``.  A "finish" action (category=1)
    appears when ``min < max`` and enough cards are already selected.
    """
    cards = msg.get("cards", [])
    selected: list[int] = msg.get("_selected", [])
    min_sel = msg.get("min", 1)
    max_sel = msg.get("max", min_sel)
    selected_set = set(selected)
    if not cards:
        return []

    actions: list[dict] = []
    for i, card in enumerate(cards):
        if i in selected_set:
            continue
        new_selected = selected + [i]
        completes = len(new_selected) >= max_sel
        actions.append({
            "category": 0, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "num_selected": len(new_selected),
            "build_response": (
                (lambda idxs=new_selected: rb.build_select_card_response(idxs))
                if completes else None
            ),
        })
        if len(actions) >= MAX_ACTIONS:
            return actions

    # "Finish" when min < max and enough cards are already selected
    if min_sel < max_sel and len(selected) >= min_sel:
        actions.append({
            "category": 1, "index": 0, "code": 0, "location": 0, "sequence": 0,
            "num_selected": len(selected),
            "build_response": lambda idxs=list(selected): rb.build_select_card_response(idxs),
        })

    return actions


def _extract_chain_actions(msg: dict) -> list[dict]:
    chains = msg.get("chains", [])
    forced = msg.get("forced", 0)
    actions = []
    for i, chain in enumerate(chains):
        actions.append({
            "category": 0, "index": i, "code": chain.get("code", 0),
            "location": chain.get("location", 0), "sequence": chain.get("sequence", 0),
            "build_response": lambda idx=i: rb.build_select_chain_response(idx),
        })
    if not forced:
        actions.append({
            "category": 1, "index": 0, "code": 0, "location": 0, "sequence": 0,
            "build_response": lambda: rb.build_select_chain_response(-1),
        })
    return actions


def _extract_place_actions(msg: dict) -> list[dict]:
    """Extract place selections from field mask.

    The field_mask is from the selecting player's perspective:
    bits 0-15 = selector's own zones, bits 16-31 = opponent's zones.
    The response must use absolute player numbers.
    """
    field_mask = msg.get("field_mask", 0)
    selecting_player = msg.get("player", 0)
    # Map relative position (0=self, 1=opponent) to absolute player number
    abs_player = [selecting_player, 1 - selecting_player]
    actions = []
    for rel_player in range(2):
        base_m = rel_player * 16
        base_s = rel_player * 16 + 8
        for seq in range(7):
            bit = base_m + seq
            if bit < 32 and not (field_mask & (1 << bit)):
                actions.append({
                    "category": 0, "index": len(actions),
                    "code": 0, "location": LOCATION_MZONE, "sequence": seq,
                    "build_response": lambda p=abs_player[rel_player], s=seq: rb.build_select_place_response(
                        p, LOCATION_MZONE, s
                    ),
                })
        for seq in range(6):
            bit = base_s + seq
            if bit < 32 and not (field_mask & (1 << bit)):
                actions.append({
                    "category": 1, "index": len(actions),
                    "code": 0, "location": LOCATION_SZONE, "sequence": seq,
                    "build_response": lambda p=abs_player[rel_player], s=seq: rb.build_select_place_response(
                        p, LOCATION_SZONE, s
                    ),
                })
    return actions


def _extract_position_actions(msg: dict) -> list[dict]:
    positions = msg.get("positions", 0)
    actions = []
    for pos_val in [POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE]:
        if positions & pos_val:
            actions.append({
                "category": 0, "index": pos_val, "code": msg.get("code", 0),
                "location": 0, "sequence": 0,
                "build_response": lambda pv=pos_val: rb.build_select_position_response(pv),
            })
    return actions


def _extract_tribute_actions(msg: dict) -> list[dict]:
    """For MSG_SELECT_TRIBUTE: generate valid tribute combinations.

    The engine requires sum(release_param) >= min for selected cards.
    """
    cards = msg.get("cards", [])
    min_sel = msg.get("min", 1)
    max_sel = msg.get("max", min_sel)
    if not cards:
        return []
    actions = []
    for size in range(1, min(max_sel, len(cards)) + 1):
        for combo in combinations(range(len(cards)), size):
            total_release = sum(cards[i].get("release_param", 1) for i in combo)
            if total_release >= min_sel:
                indices = list(combo)
                card = cards[indices[0]]
                actions.append({
                    "category": 0, "index": indices[0], "code": card.get("code", 0),
                    "location": card.get("location", 0), "sequence": card.get("sequence", 0),
                    "build_response": lambda idxs=indices: rb.build_select_card_response(idxs),
                })
                if len(actions) >= MAX_ACTIONS:
                    return actions
    return actions


def _extract_sum_actions(msg: dict) -> list[dict]:
    optional = msg.get("optional_cards", [])
    return [
        {"category": 0, "index": i, "code": card.get("code", 0),
         "location": card.get("location", 0), "sequence": card.get("sequence", 0),
         "build_response": lambda idx=i: rb.build_select_sum_response([idx])}
        for i, card in enumerate(optional)
    ]


def _extract_unselect_actions(msg: dict) -> list[dict]:
    actions = []
    for i, card in enumerate(msg.get("selectable", [])):
        actions.append({
            "category": 0, "index": i, "code": card.get("code", 0),
            "location": card.get("location", 0), "sequence": card.get("sequence", 0),
            "build_response": lambda idx=i: rb.build_select_unselect_card_response(idx),
        })
    if msg.get("finishable"):
        actions.append({
            "category": 1, "index": 0, "code": 0, "location": 0, "sequence": 0,
            "build_response": lambda: rb.build_select_unselect_card_response(-1),
        })
    return actions


def _extract_sort_actions(msg: dict) -> list[dict]:
    """For sort: each position choice is an action. Simplified: return identity order."""
    cards = msg.get("cards", [])
    count = len(cards)
    # Each action = place card i first in the ordering
    return [
        {"category": 0, "index": i, "code": card.get("code", 0),
         "location": 0, "sequence": 0,
         "build_response": lambda idx=i, n=count: rb.build_sort_card_response(
             [idx] + [j for j in range(n) if j != idx]
         )}
        for i, card in enumerate(cards)
    ]


def _extract_announce_race_actions(msg: dict) -> list[dict]:
    available = msg.get("available", 0)
    actions = []
    for bit in range(64):
        race = 1 << bit
        if available & race:
            actions.append({
                "category": 0, "index": bit, "code": 0, "location": 0, "sequence": 0,
                "build_response": lambda r=race: rb.build_announce_race_response(r),
            })
            if len(actions) >= MAX_ACTIONS:
                break
    return actions


def _extract_announce_attrib_actions(msg: dict) -> list[dict]:
    available = msg.get("available", 0)
    actions = []
    for bit in range(8):
        attrib = 1 << bit
        if available & attrib:
            actions.append({
                "category": 0, "index": bit, "code": 0, "location": 0, "sequence": 0,
                "build_response": lambda a=attrib: rb.build_announce_attrib_response(a),
            })
    return actions


def _extract_announce_number_actions(msg: dict) -> list[dict]:
    numbers = msg.get("numbers", [])
    return [
        {"category": 0, "index": i, "code": 0, "location": 0, "sequence": 0,
         "build_response": lambda n=num: rb.build_announce_number_response(n)}
        for i, num in enumerate(numbers)
    ]


def _extract_rps_actions(msg: dict) -> list[dict]:
    return [
        {"category": 0, "index": c, "code": 0, "location": 0, "sequence": 0,
         "build_response": lambda choice=c: rb.build_rock_paper_scissors_response(choice)}
        for c in [1, 2, 3]  # rock, paper, scissors
    ]


def _extract_counter_actions(msg: dict) -> list[dict]:
    """Simplified: select first card's counters."""
    cards = msg.get("cards", [])
    count = msg.get("count", 0)
    if not cards:
        return []
    # Simple: distribute all counters from first available card
    actions = []
    for i, card in enumerate(cards):
        cc = card.get("counter_count", 0)
        if cc > 0:
            counters = [0] * len(cards)
            counters[i] = min(cc, count)
            actions.append({
                "category": 0, "index": i, "code": card.get("code", 0),
                "location": 0, "sequence": 0,
                "build_response": lambda c=counters: rb.build_select_counter_response(c),
            })
    return actions


_ACTION_EXTRACTORS = {
    MSG_SELECT_IDLECMD: _extract_idle_actions,
    MSG_SELECT_BATTLECMD: _extract_battle_actions,
    MSG_SELECT_EFFECTYN: _extract_effectyn_actions,
    MSG_SELECT_YESNO: _extract_yesno_actions,
    MSG_SELECT_OPTION: _extract_option_actions,
    MSG_SELECT_CARD: _extract_card_actions,
    MSG_SELECT_CHAIN: _extract_chain_actions,
    MSG_SELECT_PLACE: _extract_place_actions,
    MSG_SELECT_DISFIELD: _extract_place_actions,
    MSG_SELECT_POSITION: _extract_position_actions,
    MSG_SELECT_TRIBUTE: _extract_tribute_actions,
    MSG_SELECT_SUM: _extract_sum_actions,
    MSG_SELECT_UNSELECT_CARD: _extract_unselect_actions,
    MSG_SORT_CARD: _extract_sort_actions,
    MSG_SORT_CHAIN: _extract_sort_actions,
    MSG_ANNOUNCE_RACE: _extract_announce_race_actions,
    MSG_ANNOUNCE_ATTRIB: _extract_announce_attrib_actions,
    MSG_ANNOUNCE_NUMBER: _extract_announce_number_actions,
    MSG_ROCK_PAPER_SCISSORS: _extract_rps_actions,
    MSG_SELECT_COUNTER: _extract_counter_actions,
}


def _encode_action(action: dict, msg_type: int) -> np.ndarray:
    """Encode a single action as a feature vector.

    Layout (12 bytes):
        [0]    msg_type      (uint8)
        [1]    category      (uint8)
        [2:6]  code          (uint32 LE - card passcode)
        [6]    location      (uint8)
        [7]    sequence      (uint8)
        [8]    index         (uint8)
        [9]    num_selected  (uint8 - number of cards in combo, default 1)
        [10]   extra_idx_0   (uint8 - index of 2nd selected card)
        [11]   extra_idx_1   (uint8 - index of 3rd selected card)
    """
    feat = np.zeros(ACTION_FEATURES, dtype=np.uint8)
    feat[0] = msg_type & 0xFF
    feat[1] = action.get("category", 0)
    code = action.get("code", 0)
    feat[2] = code & 0xFF
    feat[3] = (code >> 8) & 0xFF
    feat[4] = (code >> 16) & 0xFF
    feat[5] = (code >> 24) & 0xFF
    feat[6] = action.get("location", 0) & 0xFF
    feat[7] = min(action.get("sequence", 0), 255)
    feat[8] = action.get("index", 0) & 0xFF
    feat[9] = action.get("num_selected", 1)
    extra = action.get("extra_indices", [])
    if len(extra) >= 1:
        feat[10] = extra[0] & 0xFF
    if len(extra) >= 2:
        feat[11] = extra[1] & 0xFF
    return feat
