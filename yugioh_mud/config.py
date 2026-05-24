"""MUD bot configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MUDBotConfig:
    """Configuration for the MUD WebSocket bot."""

    # Server
    server_host: str = "localhost"
    server_port: int = 8080
    nickname: str = ""
    password: str = ""

    # Room
    profile: str = ""  # "host" or "guest"
    join: str = ""  # host nickname to join (guest only)
    deck: str = "blue_eyes"

    # Play
    mode: str = "random"
    seed: int = 42

    # Database
    db_path: str = "assets/cards.cdb"

    # Model
    checkpoint: str = ""  # path to .pt checkpoint
    device: str = "cpu"  # torch device

    # Debug
    verbose: bool = False


HOST_CONFIG = MUDBotConfig(
    nickname="Player1",
    password="player1pass",
    profile="host",
    seed=42,
)

GUEST_CONFIG = MUDBotConfig(
    nickname="Player2",
    password="player2pass",
    profile="guest",
    seed=137,
)
