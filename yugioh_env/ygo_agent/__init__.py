"""YGO-Agent cross-play integration.

Provides an HTTP bridge to ygo-agent's inference server, allowing
ygo-agent trained models to play as opponents in this repo's
eval and leaderboard systems.
"""

from __future__ import annotations

from yugioh_env.ygo_agent.opponent import YGOAgentOpponent

__all__ = ["YGOAgentOpponent"]
