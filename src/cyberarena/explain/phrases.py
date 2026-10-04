"""Plain-English phrases for the v4 agents' candidate-move features (``arena/features.py::FEATURE_NAMES``).

``describe(name, value, ctx)`` turns one attributed feature of the chosen move into a short clause for the
"why this host" rationale, e.g. ``t_dist = 0.25`` -> "host 13 is one hop from the crown jewel".

Placeholders in the table: ``{T}`` / ``{Ts}`` the target host ("host 13" / "host 13's", or "it" / "its" once the
host has been named), ``{S}`` / ``{Ss}`` the source host, ``{v}`` the value, ``{n}`` a count recovered from a
scaled value. Feature values are the agent's scaled inputs (roughly 0-1); counts are recovered only where the
scaling is a known cap. ``tests/explain/test_explain_phrases.py`` checks that every ``FEATURE_NAMES`` entry of
both sides has a phrase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

ACTION_NOUNS = {
    "recon": "a scan", "phish": "a phishing attempt", "exploit": "an exploit", "escalate": "a privilege escalation",
    "lateral_move": "a lateral move", "exfiltrate": "an exfiltration attempt", "wait": "a wait",
    "monitor": "a monitoring sweep", "isolate": "an isolation", "patch": "a patch", "restore": "a restore",
    "reset_credentials": "a credential reset",
}  # fmt: skip

# plural, for "<moves> rate lower here" (the chosen move is NOT of this kind, and that raised its Q)
ACTION_PLURALS = {
    "recon": "scans", "phish": "phishing attempts", "exploit": "exploits", "escalate": "privilege escalations",
    "lateral_move": "lateral moves", "exfiltrate": "exfiltration attempts", "wait": "waits",
    "monitor": "monitoring sweeps", "isolate": "isolations", "patch": "patches", "restore": "restores",
    "reset_credentials": "credential resets",
}  # fmt: skip

# gerund phrases for the runner-up move ("rather than {phrase}"); {t} is the host
ACTION_GERUNDS = {
    "recon": "scanning {t}", "phish": "phishing {t}", "exploit": "exploiting {t}",
    "escalate": "escalating privilege on {t}", "lateral_move": "moving laterally to {t}",
    "exfiltrate": "exfiltrating from {t}", "wait": "waiting", "monitor": "monitoring {t}",
    "isolate": "isolating {t}", "patch": "patching {t}", "restore": "restoring {t}",
    "reset_credentials": "resetting credentials on {t}",
}  # fmt: skip


@dataclass(frozen=True)
class Phrase:
    """``kind``: "flag" (on/off text), "score" (detector score), "level" (scaled value with text), "hops", "action"."""

    kind: str
    on: str = ""
    off: str = ""
    scale: float = 1.0  # {n} = round(value * scale)


def _flag(on: str, off: str) -> Phrase:
    return Phrase("flag", on, off)


def _level(text: str, scale: float = 1.0, zero: str = "") -> Phrase:
    return Phrase("level", text, zero, scale)


_SCORE = Phrase("score")

PHRASES: dict[str, Phrase] = {
    # ---- global situation (identical for every candidate of a turn, so near-zero attribution vs the mean move)
    "g_turn_frac": _level("the game is {pct} through its turn limit"),
    "g_footholds": _level("red holds {n} foothold(s)", 5, "red has no foothold"),
    "g_admin_footholds": _level("red has admin rights on {n} host(s)", 3, "red has no admin foothold"),
    "g_has_server": _flag("red already holds a server", "red holds no server yet"),
    "g_has_crown": _flag("red holds the crown jewel", "red does not hold the crown jewel yet"),
    "g_crown_admin": _flag("red is admin on the crown jewel", "red is not admin on the crown jewel"),
    "g_front_dist": _level("red's closest foothold is {hops} from the crown jewel"),
    "g_hot": _level("red's most-watched foothold reads {v:.2f} on the detectors"),
    "g_evasion_malware": _level("red's malware evasion is {v:.2f}", zero="red's malware activity is unblended"),
    "g_evasion_phishing": _level("red's phishing evasion is {v:.2f}", zero="red's phishing is unblended"),
    "g_evasion_network": _level("red's network evasion is {v:.2f}", zero="red's network traffic is unblended"),
    "g_detected": _level("blue has {n} host(s) flagged", 5, "blue has no host flagged"),
    "g_isolated": _level("blue has {n} host(s) isolated", 5, "blue has no host isolated"),
    "g_confirmed": _level("blue has confirmed {n} compromise(s)", 5, "blue has confirmed no compromise"),
    "g_patched_frac": _level("{pct} of the network is patched", zero="nothing is patched yet"),
    "g_top_undetected": _level("the most suspicious unflagged host reads {v:.2f}"),
    "g_crown_detected": _flag("the crown jewel is flagged", "the crown jewel is not flagged"),
    "g_crown_isolated": _flag("the crown jewel is isolated", "the crown jewel is not isolated"),
    "g_crown_score": _level("the crown jewel's top detector score is {v:.2f}"),
    # ---- target host
    "t_none": _flag("the move has no target host", "the move targets a host"),
    "t_dmz": _flag("{T} is a DMZ host", "{T} is not a DMZ host"),
    "t_workstation": _flag("{T} is a workstation", "{T} is not a workstation"),
    "t_server": _flag("{T} is a server", "{T} is not a server"),
    "t_crown": _flag("{T} is the crown jewel", "{T} is not the crown jewel"),
    "t_compromised": _flag("{T} is already compromised", "{T} is not compromised yet"),
    "t_privilege": _level("red has privilege {n} on {T}", 2, "red has no privilege on {T}"),
    "t_isolated": _flag("{T} is isolated", "{T} is not isolated"),
    "t_patched": _flag("{T} is patched", "{T} is unpatched"),
    "t_recon": _flag("{T} has been scanned", "{T} has not been scanned yet"),
    "t_score_malware": _SCORE,
    "t_score_phishing": _SCORE,
    "t_score_network": _SCORE,
    "t_score_max": _SCORE,
    "t_dist": Phrase("hops", "{T} is {hops} from the crown jewel"),
    "t_degree": _level("{T} is {conn} ({v:.2f} of the best-connected host's links)"),
    "t_closer": _flag("{T} is closer to the crown jewel than any foothold", "{T} is no closer to the crown jewel"),
    "t_open_nbrs": _level("{T} borders {n} scanned, uncompromised host(s)", 4, "{T} borders no open hosts"),
    "t_detected": _flag("{T} is flagged", "{T} is not flagged"),
    "t_confirmed": _flag("{T} is a confirmed compromise", "{T} is not a confirmed compromise"),
    "t_nbr_detected": _level("{n} of {Ts} neighbours are flagged", 4, "none of {Ts} neighbours are flagged"),
    # ---- source host (red)
    "s_has": _flag("red attacks from its foothold {S}", "the move needs no foothold"),
    "s_privilege": _level("red has privilege {n} on source {S}", 2, "red has no privilege on source {S}"),
    "s_recon": _flag("source {S} has been scanned", "source {S} has not been scanned"),
    "s_score_max": _level("source {S}'s top detector score is {v:.2f}"),
    "s_dist": Phrase("hops", "source {S} is {hops} from the crown jewel"),
    # ---- derived
    "p_success": _level("its success odds are {v:.2f}"),
}

# one-hot of the action (a_<action>) for both sides
for _a, _noun in ACTION_NOUNS.items():
    PHRASES[f"a_{_a}"] = Phrase("action", f"the move is {_noun}", f"{ACTION_PLURALS[_a]} rate lower here")


ROLE_FLAGS = {"t_dmz": "DMZ host", "t_workstation": "workstation", "t_server": "server", "t_crown": "crown jewel"}
ROLE_NAMES = {"dmz": "DMZ host", "workstation": "workstation", "server": "server"}


def _a(noun: str) -> str:
    if noun == "crown jewel":
        return "the crown jewel"
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def hops_text(n: int) -> str:
    if n <= 0:
        return "zero hops"
    word = NUMBER_WORDS.get(n, str(n))
    return f"{word} hop" if n == 1 else f"{word} hops"


def score_level(v: float) -> str:
    return "low" if v < 0.3 else ("moderate" if v < 0.5 else "high")


@dataclass
class PhraseContext:
    """Per-move context. ``t_hops`` / ``s_hops`` are graph distances to the crown jewel (exact, from graph.json)."""

    target: int | None = None
    source: int | None = None
    t_hops: int | None = None
    s_hops: int | None = None
    max_dist: int | None = None
    role: str | None = None  # the target's role ("server", ..., or "crown jewel")
    named: set = field(default_factory=set)  # subjects already named in this sentence (target only)

    def subject(self, key: str, possessive: bool = False) -> str:
        node = self.target if key == "T" else self.source
        if key == "T" and key in self.named:  # the source is always named, to keep the two hosts apart
            return "its" if possessive else "it"
        self.named.add(key)
        name = f"host {node}" if node is not None else "the target"
        return f"{name}'s" if possessive else name


def describe(name: str, value: float, ctx: PhraseContext | None = None) -> str:
    """Plain-English clause for feature ``name`` at ``value``. Unknown names fall back to "name = value"."""
    ctx = ctx or PhraseContext()
    p = PHRASES.get(name)
    if p is None:
        return f"{name} is {value:.2f}"
    v = float(value)
    if p.kind == "score":
        model = name.removeprefix("t_score_")
        model = "top detector" if model == "max" else model
        return f"{ctx.subject('T', True)} {model} score is {score_level(v)} ({v:.2f})"
    if p.kind in ("flag", "action"):
        text = p.on if v >= 0.5 else p.off
        if name in ROLE_FLAGS and v < 0.5 and ctx.role and ctx.role != ROLE_FLAGS[name]:
            text = "{T} is " + f"{_a(ctx.role)} rather than {_a(ROLE_FLAGS[name])}"
    elif p.kind == "hops":
        node_hops = ctx.t_hops if name.startswith("t_") else ctx.s_hops
        if node_hops is None and ctx.max_dist:
            node_hops = round(v * ctx.max_dist)
        if node_hops == 0:
            text = "{T} is the crown jewel itself" if name.startswith("t_") else "source {S} is the crown jewel"
        else:
            text = p.on.replace("{hops}", hops_text(node_hops) if node_hops is not None else f"{v:.2f} scaled hops")
    else:  # level
        n = round(v * p.scale)
        zero = n == 0 if p.scale != 1.0 else v == 0
        text = p.off if (p.off and zero) else p.on
        if "{hops}" in text:
            text = text.replace("{hops}", hops_text(round(v * ctx.max_dist)) if ctx.max_dist else f"{v:.2f}")
        conn = "well connected" if v >= 0.5 else ("moderately connected" if v >= 0.25 else "sparsely connected")
        text = text.replace("{n}", str(n)).replace("{pct}", f"{v:.0%}").replace("{conn}", conn)
        text = text.replace("{v:.2f}", f"{v:.2f}")
    for key in ("T", "S"):
        if "{" + key + "s}" in text:
            text = text.replace("{" + key + "s}", ctx.subject(key, True))
        if "{" + key + "}" in text:
            text = text.replace("{" + key + "}", ctx.subject(key))
    return text


def join_clauses(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]
