"""Declarative parameter table for the Simulation Lab (contract: docs/contracts.md, "Simulation Lab").

``PARAMS`` is the single source of truth for three things:

- the ``train`` CLI flags of every ``target="cli"`` param (``add_cli_args``),
- validation of ``--env-json`` overrides and CLI values (``parse_env_json`` / ``validate_cli``),
- the JSON printed by ``--describe-params`` (``spec``).

Env-param defaults are read from ``ArenaConfig`` so the table never disagrees with the env. Importing this
module does not load TensorFlow or any classifier.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from typing import Any

from cyberarena.arena.env import ArenaConfig

SPEC_VERSION = 1


class ParamError(ValueError):
    """Bad parameter value or unknown key; the CLI turns it into exit code 2 with a one-line message."""


@dataclass(frozen=True)
class Param:
    key: str
    group: str
    label: str
    type: str  # int | float | bool | choice
    help: str
    target: str = "env"  # env -> --env-json key; cli -> its own flag
    default: Any = None  # cli params only; env params read ArenaConfig
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    nullable: bool = False
    advanced: bool = False
    # Hard bounds used for CLI validation when they differ from the UI range (e.g. tiny smoke-test runs).
    # Not part of the spec.
    hard_min: float | None = None

    @property
    def flag(self) -> str | None:
        return "--" + self.key.replace("_", "-") if self.target == "cli" else None

    @property
    def dest(self) -> str:
        return self.key


GROUPS: tuple[tuple[str, str], ...] = (
    ("network", "Network"),
    ("red", "Red team"),
    ("blue", "Blue team"),
    ("rewards", "Rewards"),
    ("training", "Training"),
    ("opponents", "Opponents"),
    ("memory", "Memory"),
)


def _f(key, group, label, help, lo, hi, step=0.01, **kw) -> Param:
    return Param(key, group, label, "float", help, min=lo, max=hi, step=step, **kw)


def _i(key, group, label, help, lo, hi, step=1, **kw) -> Param:
    return Param(key, group, label, "int", help, min=lo, max=hi, step=step, **kw)


_NOISE_HELP = {
    "recon": "Recon", "phish": "Phishing", "exploit": "Exploit", "escalate": "Escalation",
    "lateral_move": "Lateral move", "exfiltrate": "Exfiltration",
}  # fmt: skip
_LEAK_HELP = {"malware": "Malware", "phishing": "Phishing", "network": "Network"}

PARAMS: tuple[Param, ...] = (
    # ---- network
    _i("seed", "network", "Seed", "Seeds the graph, sensors and agents; same seed and settings = same run.",
       0, 999_999, target="cli", default=7),
    _i("n_nodes", "network", "Hosts", "null = seeded 12-16. More hosts give red more paths and blue more to watch.",
       10, 20, target="cli", default=None, nullable=True),
    _i("max_rounds", "network", "Round limit",
       "Rounds before blue wins by holding out. Raising it gives red more time to reach the crown jewel.", 10, 100),
    _i("pool_size", "network", "Sensor pool size",
       "Pre-scored classifier rows per sensor and class. Raising it gives more varied readings, slower start.",
       32, 2048, step=32, advanced=True),
    # ---- red success probabilities
    _f("p_phish", "red", "Phish success",
       "Chance a phishing email lands a foothold. Raising it makes initial access and re-entry easier.", 0.05, 0.9),
    _f("p_exploit", "red", "Exploit success",
       "Chance an exploit on an unpatched host succeeds. Raising it speeds red's spread.", 0.05, 0.95),
    _f("p_exploit_patched", "red", "Exploit success (patched)",
       "Chance an exploit still works on a patched host. Raising it weakens blue's patching.", 0.0, 0.5),
    _f("p_exploit_recon_bonus", "red", "Recon bonus",
       "Extra exploit chance on a host red has scanned. Raising it rewards careful reconnaissance.", 0.0, 0.5),
    _f("p_escalate", "red", "Escalation success",
       "Chance of gaining admin on a foothold. Raising it lets red reach exfiltration sooner.", 0.05, 0.95),
    _f("p_escalate_patched", "red", "Escalation success (patched)",
       "Chance escalation works on a patched host. Raising it weakens blue's patching.", 0.0, 0.6),
    _f("p_lateral", "red", "Lateral move success",
       "Chance an admin foothold pivots to a neighbour. Raising it speeds red through the network.", 0.1, 1.0),
    _f("p_exfiltrate", "red", "Exfiltration success",
       "Chance stealing from the crown jewel works once red holds it. Raising it shortens red's endgame.", 0.1, 1.0),
    _i("phish_max_footholds", "red", "Phish while footholds <=",
       "Phishing is offered only while red holds at most this many hosts. Raising it makes phishing an "
       "expansion tool, not just re-entry.", 0, 5),
    # ---- blue
    _f("detect_threshold", "blue", "Detection threshold",
       "Classifier score needed for monitoring to flag a host. Raising it means fewer false alarms but more "
       "missed intrusions.", 0.1, 0.95),
    *(_f(f"p_implant_leak.{m}", "blue", f"{lbl} implant leak",
         f"Chance a compromised host shows a malicious {m} reading when monitored. Raising it makes "
         "footholds easier to spot.", 0.0, 0.6)
      for m, lbl in _LEAK_HELP.items()),
    _f("p_reset_evicts", "blue", "Credential reset evicts",
       "Chance resetting credentials kicks a non-admin foothold out. Raising it makes resets a stronger "
       "response.", 0.0, 1.0),
    *(_f(f"noise.{a}", "blue", f"{lbl} noise",
         f"Chance {'an' if lbl[0] in 'AEIOU' else 'a'} {lbl.lower()} attempt makes its host emit a malicious sensor reading. Raising it makes that "
         "red action louder and easier for blue to detect.", 0.0, 1.0, advanced=True)
      for a, lbl in _NOISE_HELP.items()),
    # ---- rewards (red's view; blue gets the negation of red's progress terms)
    _f("r_win", "rewards", "Win reward", "Reward for winning. Raising it makes the final outcome dominate shaping.",
       0.1, 5.0, step=0.1, advanced=True),
    _f("r_new_foothold", "rewards", "New foothold",
       "Red's reward for each new host. Raising it pushes red to spread wide.", 0.0, 0.5, advanced=True),
    _f("r_server_foothold", "rewards", "Server foothold",
       "Extra red reward for owning a server. Raising it draws red toward servers.", 0.0, 0.5, advanced=True),
    _f("r_crown_foothold", "rewards", "Crown-jewel foothold",
       "Extra red reward for owning the crown jewel. Raising it makes red beeline for it.", 0.0, 1.0,
       advanced=True),
    _f("r_privilege", "rewards", "Privilege gain",
       "Red's reward for each escalation. Raising it encourages escalating before moving.", 0.0, 0.5,
       advanced=True),
    _f("r_foothold_lost", "rewards", "Foothold lost",
       "Red's penalty (blue's gain) when a foothold is removed. Raising it makes evictions matter more.",
       0.0, 0.5, advanced=True),
    _f("r_turn", "rewards", "Time pressure", "Red pays this per turn and blue earns it. Raising it hurries red.",
       0.0, 0.05, step=0.001, advanced=True),
    _f("r_isolate_true", "rewards", "Correct isolation",
       "Blue's reward for isolating a truly compromised host. Raising it makes blue quicker to isolate.",
       0.0, 0.5, advanced=True),
    _f("r_isolate_false", "rewards", "Wrong isolation",
       "Blue's penalty for isolating a clean host. Raising it makes blue more cautious.", 0.0, 0.5, advanced=True),
    _f("r_detect_true", "rewards", "True detection",
       "Blue's reward when monitoring flags a compromised host. Raising it encourages monitoring.", 0.0, 0.5,
       advanced=True),
    _f("r_detect_false", "rewards", "False positive",
       "Blue's penalty when monitoring flags a clean host. Raising it discourages noisy monitoring.", 0.0, 0.5,
       advanced=True),
    _f("r_isolated_upkeep", "rewards", "Isolation upkeep",
       "Blue's cost per isolated host per turn (business disruption). Raising it makes blue restore sooner.",
       0.0, 0.1, step=0.001, advanced=True),
    # ---- training
    _i("episodes", "training", "Episodes", "Training games. Raising it gives the learners more practice (slower).",
       100, 20_000, step=100, target="cli", default=2000, hard_min=1),
    _i("eval_every", "training", "Eval every",
       "Episodes between win-rate checkpoints. Lowering it gives a finer chart but slower runs.",
       50, 5000, step=50, target="cli", default=250, hard_min=1),
    _i("eval_n", "training", "Eval games",
       "Games per matchup at each checkpoint. Raising it gives a less noisy win-rate chart (slower).",
       20, 1000, step=10, target="cli", default=200, hard_min=1),
    _i("eval_log_n", "training", "Logged eval games",
       "Eval games per matchup per checkpoint saved for replay. Raising it gives more to replay, bigger logs.",
       0, 200, target="cli", default=10),
    _f("alpha", "training", "Learning rate",
       "How far each update moves a Q-value at the start. Raising it learns faster but less stably.",
       0.001, 1.0, step=0.001, target="cli", default=0.1),
    _f("alpha_end", "training", "Final learning rate",
       "Learning rate at the end of training (linear decay). Raising it keeps learning plastic late on.",
       0.0, 1.0, step=0.001, target="cli", default=0.01),
    _f("gamma", "training", "Discount",
       "How much future reward counts. Raising it makes agents plan further ahead.", 0.5, 0.999, step=0.001,
       target="cli", default=0.9),
    _f("eps_start", "training", "Exploration start",
       "Chance of a random move at the start. Use about 0.3 when continuing from a saved run.",
       0.0, 1.0, target="cli", default=1.0),
    _f("eps_end", "training", "Exploration end",
       "Chance of a random move after decay. Raising it keeps agents experimenting.", 0.0, 0.5, target="cli",
       default=0.02),
    _f("eps_decay_frac", "training", "Exploration decay",
       "Fraction of training over which exploration decays. Raising it explores for longer.", 0.05, 1.0,
       target="cli", default=0.6),
    # ---- opponents
    Param("baseline", "opponents", "Baseline type", "choice",
          "Fixed opponent the learners train against and are scored against. Random is much weaker.",
          target="cli", default="heuristic", choices=("heuristic", "random")),
    _f("red_baseline_noise", "opponents", "Baseline red noise",
       "Chance the baseline attacker plays a random move. Raising it makes it weaker and less predictable.",
       0.0, 1.0, target="cli", default=0.3),
    _f("blue_baseline_noise", "opponents", "Baseline blue noise",
       "Chance the baseline defender plays a random move. Raising it makes it weaker and less predictable.",
       0.0, 1.0, target="cli", default=0.4),
    # ---- memory
    Param("init_side", "memory", "Warm-start side", "choice",
          "With --init-from, which learners start from the saved run's Q-tables; the other starts fresh.",
          target="cli", default="both", choices=("both", "red", "blue")),
)

BY_KEY: dict[str, Param] = {p.key: p for p in PARAMS}
ENV_PARAMS: dict[str, Param] = {p.key: p for p in PARAMS if p.target == "env"}
CLI_PARAMS: tuple[Param, ...] = tuple(p for p in PARAMS if p.target == "cli")


def env_default(key: str) -> Any:
    d = ArenaConfig().to_dict()
    if "." in key:
        field, sub = key.split(".", 1)
        return d[field][sub]
    return d[key]


def default(p: Param) -> Any:
    return env_default(p.key) if p.target == "env" else p.default


def spec() -> dict[str, Any]:
    groups = []
    for gid, glabel in GROUPS:
        out = []
        for p in PARAMS:
            if p.group != gid:
                continue
            d: dict[str, Any] = {"key": p.key, "label": p.label, "type": p.type, "default": default(p)}
            if p.type in ("int", "float"):
                d.update(min=p.min, max=p.max, step=p.step)
            if p.choices is not None:
                d["choices"] = list(p.choices)
            if p.nullable:
                d["nullable"] = True
            d["help"] = p.help
            d["target"] = p.target
            if p.flag:
                d["flag"] = p.flag
            if p.advanced:
                d["advanced"] = True
            out.append(d)
        groups.append({"id": gid, "label": glabel, "params": out})
    return {"version": SPEC_VERSION, "groups": groups}


# --------------------------------------------------------------------------------------------- validation


def check_value(p: Param, value: Any, *, hard: bool = False) -> Any:
    """Type- and range-check one value; returns it coerced (int -> float for float params)."""
    if value is None:
        if p.nullable:
            return None
        raise ParamError(f"{p.key}: null not allowed")
    if p.type == "choice":
        if value not in (p.choices or ()):
            raise ParamError(f"{p.key}: {value!r} not one of {list(p.choices or ())}")
        return value
    if p.type == "bool":
        if not isinstance(value, bool):
            raise ParamError(f"{p.key}: expected true/false, got {value!r}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParamError(f"{p.key}: expected a number, got {value!r}")
    if p.type == "int":
        if isinstance(value, float) and not value.is_integer():
            raise ParamError(f"{p.key}: expected an integer, got {value!r}")
        value = int(value)
    else:
        value = float(value)
    lo = p.hard_min if hard and p.hard_min is not None else p.min
    if (lo is not None and value < lo) or (p.max is not None and value > p.max):
        raise ParamError(f"{p.key}: {value} out of range [{lo}, {p.max}]")
    return value


def parse_env_json(text: str | None) -> ArenaConfig:
    """Apply an ``--env-json`` object (flat or dotted keys, or a dict for a dict field) to ArenaConfig."""
    cfg = ArenaConfig().to_dict()
    if not text:
        return ArenaConfig(**cfg)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParamError(f"--env-json is not valid JSON: {e.msg} (char {e.pos})") from None
    if not isinstance(raw, dict):
        raise ParamError("--env-json must be a JSON object")
    flat: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            flat.update({f"{k}.{sk}": sv for sk, sv in v.items()})
        else:
            flat[k] = v
    for k, v in flat.items():
        p = ENV_PARAMS.get(k)
        if p is None:
            raise ParamError(f"--env-json: unknown key {k!r}")
        v = check_value(p, v)
        if "." in k:
            field, sub = k.split(".", 1)
            cfg[field] = {**cfg[field], sub: v}
        else:
            cfg[k] = v
    return ArenaConfig(**cfg)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    for p in CLI_PARAMS:
        kw: dict[str, Any] = {"dest": p.dest, "default": p.default, "help": p.help}
        if p.type == "choice":
            kw["choices"] = list(p.choices or ())
        elif p.type == "bool":
            kw["action"] = "store_true"
        else:
            kw["type"] = int if p.type == "int" else float
        parser.add_argument(p.flag, **kw)


def validate_cli(args: argparse.Namespace) -> None:
    for p in CLI_PARAMS:
        setattr(args, p.dest, check_value(p, getattr(args, p.dest), hard=True))


def env_config_fields() -> set[str]:
    """Flattened ArenaConfig field names (dict fields as dotted keys) -- what the spec's env params must cover."""
    out: set[str] = set()
    d = ArenaConfig().to_dict()
    for f in fields(ArenaConfig):
        if isinstance(d[f.name], dict):
            out.update(f"{f.name}.{k}" for k in d[f.name])
        else:
            out.add(f.name)
    return out
