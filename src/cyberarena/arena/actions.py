"""Discrete action sets for the simulated arena.

Every action is an abstract, simulated move that flips node attributes by probability. Nothing here touches
real hosts. ``action_id`` strings are stable: later phases key MITRE ATT&CK / D3FEND tags on them.
"""

from __future__ import annotations

from enum import IntEnum


class RedAction(IntEnum):
    RECON = 0
    PHISH = 1
    EXPLOIT = 2
    ESCALATE = 3
    LATERAL_MOVE = 4
    EXFILTRATE = 5
    WAIT = 6

    @property
    def action_id(self) -> str:
        return self.name.lower()


class BlueAction(IntEnum):
    MONITOR = 0
    ISOLATE = 1
    PATCH = 2
    RESTORE = 3
    RESET_CREDENTIALS = 4
    WAIT = 5

    @property
    def action_id(self) -> str:
        return self.name.lower()


SIDES = ("red", "blue")
ACTIONS: dict[str, type[IntEnum]] = {"red": RedAction, "blue": BlueAction}
RED_ACTION_IDS: tuple[str, ...] = tuple(a.action_id for a in RedAction)
BLUE_ACTION_IDS: tuple[str, ...] = tuple(a.action_id for a in BlueAction)
ACTION_IDS: dict[str, tuple[str, ...]] = {"red": RED_ACTION_IDS, "blue": BLUE_ACTION_IDS}


def action_from_id(side: str, action_id: str) -> IntEnum:
    return ACTIONS[side][action_id.upper()]


def n_actions(side: str) -> int:
    return len(ACTIONS[side])
