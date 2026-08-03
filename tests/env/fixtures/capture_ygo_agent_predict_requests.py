"""Capture the golden ygo-agent predict requests.

Run from the project root:

    python tests/env/fixtures/capture_ygo_agent_predict_requests.py

Produces ``tests/env/fixtures/ygo_agent_predict_requests.json``, read by
``tests/env/test_ygo_agent_bridge.py``'s byte-equality tests.

Regenerate **only** when the outbound wire contract changes deliberately and
you want to lock the new shape in. The file records what the ygo-agent server
is sent; a change to it is a compatibility event for that server, not an
implementation detail. This script captures from the *current* bridge, so
running it always makes the tests pass -- it will just as happily bless a bug.
After regenerating:

    python -m pytest tests/env/test_ygo_agent_bridge.py -v

must pass on the same commit, and the fixture diff must be reviewed and
committed alongside the code change that motivated it.

The cases come from the same tables the tests use -- ``MINIMAL_MSGS`` for one
prompt per translated msg type, ``MULTI_STEP_CASES`` for the mid-selection
prompts -- so the requests written here and the requests asserted there
cannot describe different prompts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# tests/env/fixtures/<this_file> → project root is three parents up.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.env.conftest import MINIMAL_MSGS, MULTI_STEP_CASES, obs_from_msg

from yugioh_env.ygo_agent.bridge import _ACTION_MSG_TRANSLATORS, build_predict_input

OUT = Path(__file__).with_name("ygo_agent_predict_requests.json")


def main() -> None:
    requests: dict[str, dict] = {}

    for msg_type in sorted(_ACTION_MSG_TRANSLATORS):
        obs = obs_from_msg({**MINIMAL_MSGS[msg_type], "msg_type": msg_type})
        requests[str(msg_type)] = build_predict_input(obs, prev_action_idx=0)

    for key, (msg, selected) in MULTI_STEP_CASES.items():
        obs = obs_from_msg(msg, _selected=selected)
        requests[key] = build_predict_input(obs, prev_action_idx=0)

    OUT.write_text(json.dumps(requests, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(requests)} requests to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
