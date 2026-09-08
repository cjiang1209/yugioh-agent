"""Core-side observer seam for serving-only per-chunk frame capture.

No ``server/`` imports — the env can call ``on_chunk`` without depending on the
serving layer, and the env itself satisfies ``DuelView`` via its read-only accessors.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DuelView(Protocol):
    @property
    def agent_player(self) -> int: ...
    @property
    def game_state(self): ...  # GameState | None (None when no duel)
    def query_location(self, player: int, location: int) -> list[dict]: ...  # needs a live duel


@runtime_checkable
class FrameObserver(Protocol):
    def on_chunk(
        self, events: list[dict], view: DuelView
    ) -> None: ...  # the ONLY core-facing method
