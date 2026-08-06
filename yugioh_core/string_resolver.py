"""Resolve engine string IDs to display text.

The OCG engine emits 64-bit string IDs (``desc``) for prompts like
``MSG_SELECT_OPTION`` and per-link descriptions in ``MSG_SELECT_CHAIN``.
A string ID is encoded as ``(passcode << 20) | (n & 0xfffff)``:

  - ``passcode == 0``: a system string (engine-internal hint), looked up
    by ``n`` in a ``sys_strings`` mapping parsed from ``strings.conf``.
    Returns ``None`` when the mapping is empty or doesn't contain ``n``.
  - ``passcode != 0``: a per-card option string, looked up as
    ``texts.str{n+1}`` for ``id = passcode`` in ``cards.cdb``.

Failed lookups return ``None``; callers fall back to a placeholder.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from pathlib import Path

from yugioh_core.card_database import CardDatabase

_SYS_STRING_RE = re.compile(r"^!system\s+(\d+)\s+(.*)$")

logger = logging.getLogger(__name__)


def parse_sys_strings(path: str | Path) -> dict[int, str]:
    """Parse a `strings.conf` file's `!system` entries into {id: text}.

    Other sections (`!counter`, `!setname`, `!victory`) are ignored — only
    sysstrings feed the current resolver. Raises FileNotFoundError if the
    path doesn't exist; callers should pre-check (env-side decision so the
    resolver stays pure).
    """
    table: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _SYS_STRING_RE.match(line.rstrip("\r\n"))
            if m:
                table[int(m.group(1))] = m.group(2)
    return table


class StringResolver:
    """Resolve engine string IDs to display text via a card database.

    Pass ``sys_strings`` (a mapping parsed from ``strings.conf`` via
    ``parse_sys_strings``) to enable sysstring resolution; otherwise
    sysstring lookups return ``None`` and callers fall back to placeholders.
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
        n = desc_u64 & 0xFFFFF
        if passcode == 0:
            return self._sys.get(n)
        return self._card_db.get_card_string(passcode, n)


class CardTextResolver:
    """Resolves card display text: names (via card_db) and effect descriptors
    (via StringResolver). Single source for the card-text lookups that action
    and event materialization both need.
    """

    def __init__(self, card_db, sys_strings: dict[int, str] | None = None) -> None:
        self._card_db = card_db
        self._resolver: StringResolver | None = (
            StringResolver(card_db, sys_strings=sys_strings) if sys_strings is not None else None
        )

    def card_name(self, code: int) -> str:
        return self._card_db.get_card_name(code) if code else ""

    def effect_text(self, desc: int) -> str | None:
        return self._resolver.resolve(desc) if (self._resolver and desc) else None


def load_sys_strings(strings_path: str | Path | None = None) -> dict[int, str] | None:
    """Load the sysstring table from strings.conf, or None if the file is absent.

    Resolves the path from ``strings_path``, then ``YUGIOH_STRINGS_PATH``, then
    ``<repo_root>/assets/strings.conf``. Returns None (with a warning) when the
    file does not exist, so callers can fall back to placeholder labels. The
    single source of truth for locating and parsing strings.conf. Parsing is
    memoized per resolved path so repeated callers (e.g. the action/event/chain
    resolvers built at app startup) don't re-read the file.
    """
    if strings_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        strings_path = os.environ.get(
            "YUGIOH_STRINGS_PATH", str(repo_root / "assets" / "strings.conf")
        )
    return _load_sys_strings_cached(str(Path(strings_path)))


@functools.cache
def _load_sys_strings_cached(strings_path: str) -> dict[int, str] | None:
    path = Path(strings_path)
    if not path.is_file():
        logger.warning(
            "strings.conf not found at %s; sysstring labels will use placeholders. "
            "Run `make assets` to download.",
            path,
        )
        return None
    return parse_sys_strings(path)
