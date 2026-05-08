"""Resolve engine string IDs to display text.

The OCG engine emits 64-bit string IDs (``desc``) for prompts like
``MSG_SELECT_OPTION`` and per-link descriptions in ``MSG_SELECT_CHAIN``.
A string ID is encoded as ``(passcode << 20) | (n & 0xfffff)``:

  - ``passcode == 0``: a system string (engine-internal hint), looked up
    by ``n`` in ``strings.conf``.  Not yet supported — sysstrings always
    return ``None`` until a sysstring table is wired in.
  - ``passcode != 0``: a per-card option string, looked up as
    ``texts.str{n+1}`` for ``id = passcode`` in ``cards.cdb``.

Failed lookups return ``None``; callers fall back to a placeholder.
"""

from __future__ import annotations

from yugioh_core.card_database import CardDatabase


class StringResolver:
    """Resolve engine string IDs to display text via a card database.

    Sysstring resolution is reserved for a future wire-in: pass a
    ``sys_strings`` mapping (parsed from ``strings.conf``) once available.
    """

    def __init__(
        self,
        card_db: CardDatabase,
        sys_strings: dict[int, str] | None = None,
    ):
        self._card_db = card_db
        self._sys = sys_strings or {}

    def resolve(self, desc_u64: int) -> str | None:
        passcode = desc_u64 >> 20
        n = desc_u64 & 0xfffff
        if passcode == 0:
            return self._sys.get(n)
        return self._card_db.get_card_string(passcode, n)
