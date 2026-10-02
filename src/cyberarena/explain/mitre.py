"""Static, versioned MITRE tags for arena actions.

Red ``action_id`` -> MITRE ATT&CK (Enterprise) technique; blue ``action_id`` -> MITRE D3FEND technique where the
simulated move maps cleanly, else ``None``. The arena's moves are abstract probability flips, so a tag says
"this simulated move plays the role of technique X" and nothing more.

Bump the version constants (and ``MAPPING_VERSION``) whenever a row changes.
"""

from __future__ import annotations

from cyberarena.arena.actions import BLUE_ACTION_IDS, RED_ACTION_IDS

ATTACK_VERSION = "v16"  # MITRE ATT&CK Enterprise release the IDs/names below were taken from
D3FEND_VERSION = "1.0.0"  # MITRE D3FEND release the IDs/names below were taken from
MAPPING_VERSION = 1

# action_id -> (technique_id, technique_name, tactic). ``wait`` is an idle turn: no technique applies.
RED_ATTACK: dict[str, tuple[str, str, str] | None] = {
    "recon": ("T1046", "Network Service Discovery", "Discovery"),
    "phish": ("T1566", "Phishing", "Initial Access"),
    "exploit": ("T1190", "Exploit Public-Facing Application", "Initial Access"),
    "escalate": ("T1068", "Exploitation for Privilege Escalation", "Privilege Escalation"),
    "lateral_move": ("T1021", "Remote Services", "Lateral Movement"),
    "exfiltrate": ("T1041", "Exfiltration Over C2 Channel", "Exfiltration"),
    "wait": None,
}

# ``exploit`` against a host that is not internet-facing is exploitation of an internal service.
EXPLOIT_INTERNAL = ("T1210", "Exploitation of Remote Services", "Lateral Movement")
PUBLIC_FACING_ROLES = frozenset({"dmz"})

BLUE_D3FEND: dict[str, tuple[str, str, str] | None] = {
    "monitor": ("D3-NTA", "Network Traffic Analysis", "Detect"),
    "isolate": ("D3-NI", "Network Isolation", "Isolate"),
    "patch": ("D3-SU", "Software Update", "Harden"),
    "reset_credentials": ("D3-CRO", "Credential Rotation", "Harden"),
    "restore": None,  # simulated restore has no single clean D3FEND counterpart
    "wait": None,
}


def tag(actor: str, action_id: str, target_role: str | None = None) -> dict | None:
    """MITRE tag for one turn, in the docs/contracts.md ``mitre`` shape, or None."""
    if actor == "red":
        if action_id not in RED_ATTACK:
            raise KeyError(f"no ATT&CK mapping for red action {action_id!r}")
        row = RED_ATTACK[action_id]
        if action_id == "exploit" and target_role is not None and target_role not in PUBLIC_FACING_ROLES:
            row = EXPLOIT_INTERNAL
        framework, version = "ATT&CK", ATTACK_VERSION
    elif actor == "blue":
        if action_id not in BLUE_D3FEND:
            raise KeyError(f"no D3FEND mapping entry for blue action {action_id!r}")
        row = BLUE_D3FEND[action_id]
        framework, version = "D3FEND", D3FEND_VERSION
    else:
        raise ValueError(f"unknown actor {actor!r}")
    if row is None:
        return None
    tid, name, tactic = row
    return {"framework": framework, "version": version, "technique_id": tid,
            "technique_name": name, "tactic": tactic}  # fmt: skip


def coverage() -> dict[str, list[str]]:
    """Action ids with no table entry (should be empty for both sides)."""
    return {
        "red": [a for a in RED_ACTION_IDS if a not in RED_ATTACK],
        "blue": [a for a in BLUE_ACTION_IDS if a not in BLUE_D3FEND],
    }


def table() -> list[dict]:
    """Flat mapping table, for docs and the dashboard."""
    rows = []
    for side, mapping, fw, ver in (("red", RED_ATTACK, "ATT&CK", ATTACK_VERSION),
                                   ("blue", BLUE_D3FEND, "D3FEND", D3FEND_VERSION)):  # fmt: skip
        for action_id, row in mapping.items():
            rows.append({"actor": side, "action_id": action_id, "framework": fw, "version": ver,
                         "technique_id": row[0] if row else None,
                         "technique_name": row[1] if row else None,
                         "tactic": row[2] if row else None})  # fmt: skip
    rows.insert(3, {"actor": "red", "action_id": "exploit (non-dmz target)", "framework": "ATT&CK",
                    "version": ATTACK_VERSION, "technique_id": EXPLOIT_INTERNAL[0],
                    "technique_name": EXPLOIT_INTERNAL[1], "tactic": EXPLOIT_INTERNAL[2]})  # fmt: skip
    return rows
