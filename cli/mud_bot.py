"""MUD bot entry point — connects to the MUD server and plays duels."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os.path
import sys

from copy import copy

from yugioh_mud.action_translator import ActionTranslator
from yugioh_mud.agent import PassiveAgent, RandomAgent
from yugioh_mud.config import GUEST_CONFIG, HOST_CONFIG, MUDBotConfig
from yugioh_mud.connection import MUDConnection
from yugioh_mud.protocol import MUDProtocol
from yugioh_mud.text_parser import MUDTextParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yu-Gi-Oh! MUD bot client")

    server = parser.add_argument_group("server")
    server.add_argument("--host", type=str, default="localhost",
                        help="MUD server hostname (default: localhost)")
    server.add_argument("--port", type=int, default=8080,
                        help="MUD server WebSocket port (default: 8080)")
    server.add_argument("--nickname", type=str, default=None,
                        help="Login nickname (default: from profile)")
    server.add_argument("--password", type=str, default=None,
                        help="Login password (default: from profile)")

    room = parser.add_argument_group("room")
    room.add_argument("--profile", type=str, default="host",
                      choices=["host", "guest"],
                      help="Bot role: 'host' creates room, 'guest' joins (default: host)")
    room.add_argument("--join", type=str, default=None,
                      help="Nickname of host to join (guest profile only; default: Player1)")
    room.add_argument("--deck", type=str, default=None,
                      help="Deck name loaded in MUD DB (default: starter)")

    play = parser.add_argument_group("play")
    play.add_argument("--mode", type=str, default=None,
                      choices=["passive", "random"],
                      help="Play mode (default: random)")
    play.add_argument("--seed", type=int, default=None,
                      help="RNG seed (default: host=42, guest=137)")

    debug = parser.add_argument_group("debug")
    debug.add_argument("--verbose", action="store_true",
                       help="Log all sent/received lines")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MUDBotConfig:
    base = copy(HOST_CONFIG if args.profile == "host" else GUEST_CONFIG)

    # Override only explicitly provided args
    overrides = {}
    if args.host != "localhost":
        overrides["server_host"] = args.host
    if args.port != 8080:
        overrides["server_port"] = args.port
    if args.nickname is not None:
        overrides["nickname"] = args.nickname
    if args.password is not None:
        overrides["password"] = args.password
    if args.join is not None:
        overrides["join"] = args.join
    if args.deck is not None:
        overrides["deck"] = args.deck
    if args.mode is not None:
        overrides["mode"] = args.mode
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.verbose:
        overrides["verbose"] = True

    for key, value in overrides.items():
        setattr(base, key, value)

    return base


async def run(config: MUDBotConfig) -> None:
    conn = MUDConnection(config.server_host, config.server_port)
    try:
        await conn.connect()
        logging.info("Connected to ws://%s:%d", config.server_host, config.server_port)
        parser = MUDTextParser()
        if config.mode == "random":
            agent = RandomAgent(seed=config.seed)
        else:
            agent = PassiveAgent()
        translator = ActionTranslator()

        # Wire game state if cards.cdb is available
        game_state = None
        if os.path.exists(config.db_path):
            from yugioh_mud.card_lookup import CardNameLookup
            from yugioh_mud.game_state import MUDGameState
            lookup = CardNameLookup(config.db_path)
            game_state = MUDGameState(card_lookup=lookup)
            logging.info("Game state tracking enabled (db: %s)", config.db_path)
        else:
            logging.warning(
                "cards.cdb not found at %s — game state tracking disabled",
                config.db_path)

        proto = MUDProtocol(
            conn, config,
            text_parser=parser, agent=agent, action_translator=translator,
            game_state=game_state)
        await proto.run()
        logging.info("Reached state: %s", proto.state.name)
    finally:
        await conn.close()


def main() -> None:
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

    config = build_config(args)
    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        logging.info("Interrupted")
        sys.exit(0)
    except Exception:
        logging.exception("Bot failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
