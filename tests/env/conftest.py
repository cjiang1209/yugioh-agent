"""Env-specific test fixtures (lib, script_dirs, duel)."""

from __future__ import annotations

from pathlib import Path

import pytest


def obs_from_msg(msg: dict, *, _selected: list[int] | None = None):
    """Build a YuGiOhObservation from a single SELECT message.

    Mirrors what the server's _make_observation produces for that message,
    using _build_action_meta_list and _build_prompt_meta to populate the
    parallel meta fields. The optional _selected list seeds the mapper's
    multi-step selection state for tests that need it (e.g., tribute
    finish actions).
    """
    from yugioh_env.action_space import ActionMapper
    from yugioh_env.models import YuGiOhObservation
    from yugioh_env.server.yugioh_environment import (
        _build_action_meta_list,
        _build_prompt_meta,
    )

    mapper = ActionMapper()
    mapper.update(msg)
    if _selected is not None:
        mapper.update({**msg, "_selected": _selected})
    return YuGiOhObservation(
        cards=[],
        global_state=[],
        actions=mapper.get_action_features().tolist(),
        action_mask=mapper.get_action_mask().tolist(),
        action_meta=_build_action_meta_list(mapper.actions),
        prompt_meta=_build_prompt_meta(mapper),
        events=[],
        done=False,
        reward=0.0,
    )


@pytest.fixture
def script_dirs(project_root) -> list[Path]:
    dirs = [
        project_root / "third_party" / "CardScripts" / "official",
        project_root / "third_party" / "CardScripts" / "pre-release",
        project_root / "third_party" / "CardScripts",
    ]
    existing = [d for d in dirs if d.exists()]
    if not existing:
        pytest.skip("CardScripts not found. Set up git submodules.")
    return existing


@pytest.fixture
def lib():
    """Load the OCG core library."""
    try:
        from yugioh_env.lib_loader import load_library

        return load_library()
    except FileNotFoundError:
        pytest.skip("libocgcore not found. Run: make build")


@pytest.fixture
def duel(lib, card_db, script_dirs, deck_path):
    """Create a Duel instance ready to use."""
    from yugioh_env.duel import Duel

    d = Duel(lib, card_db, script_dirs)
    yield d
    d.destroy()
