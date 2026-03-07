"""Parse .ydk deck files."""

from __future__ import annotations

from pathlib import Path


def parse_ydk(path: str | Path) -> dict[str, list[int]]:
    """Parse a .ydk file into main/extra/side deck lists.

    Format:
        Lines starting with # or ! are section headers.
        #main -> main deck
        #extra -> extra deck
        !side -> side deck
        Blank lines and comment lines (starting with #created) are ignored.
        All other lines are card codes (integers).

    Returns:
        {"main": [int, ...], "extra": [int, ...], "side": [int, ...]}
    """
    deck: dict[str, list[int]] = {"main": [], "extra": [], "side": []}
    current_section = "main"

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#created"):
                continue
            if line == "#main":
                current_section = "main"
                continue
            if line == "#extra":
                current_section = "extra"
                continue
            if line == "!side":
                current_section = "side"
                continue
            if line.startswith("#") or line.startswith("!"):
                continue
            try:
                code = int(line)
                if code > 0:
                    deck[current_section].append(code)
            except ValueError:
                continue

    return deck
