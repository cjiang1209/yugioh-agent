"""Async WebSocket connection to the MUD server."""

from __future__ import annotations

import websockets


class MUDConnection:
    """Thin wrapper around a websockets client connection."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._closed = False

    async def connect(self) -> None:
        """Open the WebSocket connection."""
        uri = f"ws://{self._host}:{self._port}"
        self._ws = await websockets.connect(uri)
        self._closed = False

    async def send_line(self, text: str) -> None:
        """Send a single text frame."""
        assert self._ws is not None and not self._closed
        await self._ws.send(text)

    async def recv_line(self) -> str:
        """Receive a single text frame, stripping trailing whitespace."""
        assert self._ws is not None and not self._closed
        msg = await self._ws.recv()
        return msg.rstrip()

    async def close(self) -> None:
        """Close the connection."""
        if self._ws is not None and not self._closed:
            self._closed = True
            await self._ws.close()
