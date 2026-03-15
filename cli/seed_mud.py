"""Seed MUD server with bot accounts and decks from .ydk files.

Run inside the MUD server's venv with cwd=third_party/yugioh-game/:
    python <repo_root>/cli/seed_mud.py [--deck path.ydk ...]
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_ydk(path: Path) -> dict:
    """Parse a .ydk file into MUD deck format {"cards": [...], "side": [...]}."""
    main = []
    extra = []
    side = []
    current = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line == "#main":
            current = main
        elif line == "#extra":
            current = extra
        elif line == "!side":
            current = side
        elif line and not line.startswith("#") and current is not None:
            try:
                current.append(int(line))
            except ValueError:
                continue
    return {"cards": main + extra, "side": side}


def main():
    parser = argparse.ArgumentParser(description="Seed MUD server accounts and decks")
    parser.add_argument(
        "--deck",
        action="append",
        dest="decks",
        metavar="PATH",
        help="Path to .ydk file (repeatable; default: all assets/decks/*.ydk)",
    )
    args = parser.parse_args()

    # Determine repo root (seed_mud.py lives at <repo>/cli/seed_mud.py)
    repo_root = Path(__file__).resolve().parent.parent

    # Collect deck files (resolve to absolute before chdir)
    if args.decks:
        deck_paths = [Path(d).resolve() for d in args.decks]
        for p in deck_paths:
            if not p.exists():
                print(f"Error: deck file not found: {p}", file=sys.stderr)
                sys.exit(1)
    else:
        deck_dir = repo_root / "assets" / "decks"
        deck_paths = sorted(deck_dir.glob("*.ydk"))
        if not deck_paths:
            print(f"Warning: no .ydk files found in {deck_dir}")

    # Import MUD server models (add MUD server dir to sys.path)
    mud_dir = repo_root / "third_party" / "yugioh-game"
    sys.path.insert(0, str(mud_dir))
    from ygo import models
    from ygo.models import Account, Deck

    # models.setup() uses relative paths (game.db, alembic.ini)
    os.chdir(mud_dir)
    Session = models.setup()
    session = Session()

    # ── Create accounts ──────────────────────────────────────────────────────
    # Names must be capitalized: the MUD server's login handler calls
    # .capitalize() on nickname input before querying the DB.
    accounts_info = [
        ("Player1", "player1pass"),
        ("Player2", "player2pass"),
    ]
    accounts = {}
    for name, password in accounts_info:
        existing = session.query(Account).filter_by(name=name).first()
        if existing:
            print(f"Account '{name}' already exists, skipping.")
            accounts[name] = existing
        else:
            account = Account()
            account.name = name
            account.set_password(password)
            session.add(account)
            session.commit()
            print(f"Created account '{name}'.")
            accounts[name] = account

    # ── Insert decks ─────────────────────────────────────────────────────────
    for deck_path in deck_paths:
        deck_name = deck_path.stem
        content = parse_ydk(deck_path)
        content_json = json.dumps(content)
        card_count = len(content["cards"])
        side_count = len(content["side"])
        for acct_name, acct in accounts.items():
            existing = (
                session.query(Deck)
                .filter_by(account_id=acct.id, name=deck_name)
                .first()
            )
            if existing:
                existing.content = content_json
                existing.public = True
                session.commit()
                print(f"Updated deck '{deck_name}' ({card_count} cards, {side_count} side) for {acct_name}.")
            else:
                deck = Deck()
                deck.account_id = acct.id
                deck.name = deck_name
                deck.content = content_json
                deck.public = True
                session.add(deck)
                session.commit()
                print(f"Created deck '{deck_name}' ({card_count} cards, {side_count} side) for {acct_name}.")

    session.close()
    print("Done.")


if __name__ == "__main__":
    main()
