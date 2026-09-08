"""Named constants for idle/battle action categories.

These replace magic numbers duplicated between action_space.py and cmd_handler.py.
"""

# Idle (MSG_SELECT_IDLECMD)
IDLE_SUMMON = 0
IDLE_SP_SUMMON = 1
IDLE_REPOSITION = 2
IDLE_MSET = 3
IDLE_SSET = 4
IDLE_ACTIVATE = 5
IDLE_TO_BP = 6
IDLE_TO_EP = 7

# Battle (MSG_SELECT_BATTLECMD)
BATTLE_ACTIVATE = 0
BATTLE_ATTACK = 1
BATTLE_TO_M2 = 2
BATTLE_TO_EP = 3
