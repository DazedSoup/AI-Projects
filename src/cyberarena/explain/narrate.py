"""Plain-language move rationale, one per turn, for both red and blue.

Online: one Anthropic Messages API call per turn (default model ``claude-haiku-4-5``, overridable with
``--model`` or ``CYBERARENA_NARRATION_MODEL``), concurrency- and rate-limited, disk-cached by prompt hash.
Offline: a deterministic template built from the same structured facts, so tests and the dashboard work with no
key and no network.

Prompts describe abstract simulation state only (action ids, action values, node flags, detector scores, SHAP
features). They never ask for real-world exploit steps, commands or payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-haiku-4-5"
MODEL_ENV = "CYBERARENA_NARRATION_MODEL"
KEY_ENV = "ANTHROPIC_API_KEY"
PROMPT_VERSION = 1

# USD per 1M tokens (input, output), first-party API list prices. Used only for estimates.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

SYSTEM_PROMPT = (
    "You narrate moves in a turn-based red-team vs blue-team reinforcement-learning simulation. Hosts, attacks "
    "and defenses are abstract game mechanics that flip node attributes with fixed probabilities; nothing is "
    "real. Given one turn's structured state, write 1-2 plain-English sentences (at most 50 words) explaining "
    "the strategic reason the actor chose this move, citing the most relevant numbers (action values, "
    "detector scores, SHAP evidence). If the move was an exploration or noise pick rather than the top-ranked "
    "option, say so. Stay at the level of game strategy: never describe real-world exploit steps, commands, "
    "tools or payloads. Output only the sentences."
)

MATCHUPS = {
    "learned_vs_learned": "learned red vs learned blue",
    "red_learned_vs_blue_baseline": "learned red vs heuristic-baseline blue",
    "blue_learned_vs_red_baseline": "heuristic-baseline red vs learned blue",
}

VERBS = {
    "red": {
        "recon": "scouts {t}",
        "phish": "phishes {t}",
        "exploit": "exploits {t}",
        "escalate": "escalates privilege on {t}",
        "lateral_move": "moves laterally from {s} to {t}",
        "exfiltrate": "attempts exfiltration from {t}",
        "wait": "waits",
    },
    "blue": {
        "monitor": "monitors {t}",
        "isolate": "isolates {t}",
        "patch": "patches {t}",
        "restore": "restores {t}",
        "reset_credentials": "resets credentials on {t}",
        "wait": "waits",
    },
}


class NarrationError(RuntimeError):
    pass


class MissingCredentialsError(NarrationError):
    pass


# --------------------------------------------------------------------------------------------- turn facts


@dataclass
class TurnFacts:
    """Everything the prompt and the template need, extracted from one turn record."""

    turn: dict
    roles: dict[int, str]
    crown_jewel: int | None
    shap: list[dict] | None
    mitre: dict | None
    detect_threshold: float = 0.5
    max_rounds: int | None = None

    @property
    def actor(self) -> str:
        return self.turn["actor"]

    @property
    def agent(self) -> str:
        return self.turn.get("agent") or "learned"

    def host(self, node: int | None) -> str:
        if node is None:
            return "no host"
        bits = [self.roles.get(node, "host")]
        if node == self.crown_jewel:
            bits.append("crown jewel")
        return f"host {node} ({', '.join(bits)})"

    def node_state(self, node: int | None) -> dict | None:
        if node is None:
            return None
        for ns in self.turn.get("node_states") or []:
            if ns["id"] == node:
                return ns
        return None

    def ranked_values(self) -> list[tuple[str, float]]:
        dv = self.turn.get("decision_values") or {}
        return sorted(dv.items(), key=lambda kv: (-kv[1], kv[0]))

    def value_label(self) -> str:
        return "Q-values" if self.agent == "learned" else "heuristic priorities"


def _state_phrase(ns: dict) -> str:
    flags = [
        "compromised" if ns["compromised"] else "clean",
        "detected" if ns["detected"] else "undetected",
        "isolated" if ns["isolated"] else "connected",
        "patched" if ns["patched"] else "unpatched",
        f"privilege {ns['privilege']}",
    ]
    return "; ".join(flags)


def _scores_phrase(ns: dict) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in sorted(ns.get("scores", {}).items()))


def build_prompt(f: TurnFacts, top_values: int = 3, top_features: int = 3) -> str:
    """Compact structured user message for one turn."""
    t = f.turn
    lines = ["Simulated cyber-range turn (abstract game; no real systems)."]
    rnd = t["turn"] // 2 + 1
    of = f" of {f.max_rounds}" if f.max_rounds else ""
    game = MATCHUPS.get(t.get("matchup", ""), t.get("matchup", "unknown matchup"))
    lines.append(
        f"Episode {t['episode']}, turn {t['turn']} (round {rnd}{of}), {t.get('phase', 'train')} game: {game}."
    )
    policy = "learned Q-learning policy" if f.agent == "learned" else "scripted heuristic baseline"
    lines.append(f"Actor: {f.actor.upper()}, {policy}.")

    verb = (
        VERBS[f.actor]
        .get(t["action_id"], t["action_id"])
        .format(s=f.host(t.get("source")), t=f.host(t.get("target")))
    )
    outcome = "success" if t.get("success") else "failed"
    lines.append(f"Move: {t['action_id']} - {f.actor} {verb}: {outcome}. Reward {t.get('reward', 0):+.3f}.")
    if f.mitre:
        m = f.mitre
        lines.append(
            f"MITRE tag: {m['framework']} {m['technique_id']} {m['technique_name']} ({m['tactic']})."
        )

    ranked = f.ranked_values()
    if ranked:
        shown = ranked[:top_values]
        if t["action_id"] not in [a for a, _ in shown] and t["action_id"] in dict(ranked):
            shown.append((t["action_id"], dict(ranked)[t["action_id"]]))
        vals = ", ".join(f"{a} {v:.3g}" + (" (chosen)" if a == t["action_id"] else "") for a, v in shown)
        lines.append(f"Action values ({f.value_label()}; higher = preferred): {vals}.")
        if all(v == 0 for _, v in ranked):
            lines.append(
                "All action values are zero (state never visited in training), so the choice is a tie-break."
            )
    if t.get("explored"):
        lines.append("This move was an exploration/noise pick, not the top-ranked action.")

    ns = f.node_state(t.get("target"))
    if ns is not None:
        lines.append(f"Target {f.host(t['target'])} after the move: {_state_phrase(ns)}. "
                     f"Detector scores: {_scores_phrase(ns)} (blue flags a host at >= {f.detect_threshold:.2f}).")  # fmt: skip

    states = t.get("node_states") or []
    if states:
        comp = [n["id"] for n in states if n["compromised"]]
        det = [n["id"] for n in states if n["detected"] and n["compromised"]]
        iso = [n["id"] for n in states if n["isolated"]]
        cj = f.node_state(f.crown_jewel)
        cj_txt = ""
        if cj is not None:
            cj_txt = f"; crown jewel host {f.crown_jewel} is {'compromised' if cj['compromised'] else 'not compromised'}"
        lines.append(f"Network: {len(comp)}/{len(states)} hosts compromised (ids {comp or 'none'}; detected {det or 'none'}), "
                     f"{len(iso)} isolated{cj_txt}.")  # fmt: skip

    if f.shap:
        lines.append("Detector evidence this turn (SHAP, contribution to P(malicious)):")
        for s in f.shap[:3]:
            feats = ", ".join(f"{x['name']} {x['shap']:+.3f}" for x in s["top_features"][:top_features])
            lines.append(f"- {s['model']} model on host {s['node']}: P(malicious) {s['output']:.2f} vs baseline "
                         f"{s['base_value']:.2f}; top features: {feats or 'none'}.")  # fmt: skip
    if t.get("done"):
        lines.append(f"This move ends the game; winner: {t.get('winner')}.")
    lines.append(f"In 1-2 sentences, explain why {f.actor} made this move.")
    return "\n".join(lines)


def template_rationale(f: TurnFacts) -> str:
    """Deterministic offline rationale from the same facts."""
    t = f.turn
    a = t["action_id"]
    verb = VERBS[f.actor].get(a, a).format(s=f"host {t.get('source')}", t=f.host(t.get("target")))
    if a == "wait":
        outcome = ""
    elif a == "monitor":
        outcome = " and flags it" if t.get("success") else " and finds nothing new"
    else:
        outcome = " and succeeds" if t.get("success") else " but fails"
    first = f"{f.actor.capitalize()} {verb}{outcome}."

    ranked = f.ranked_values()
    if not ranked:
        why = "No action values were logged for this move"
    elif t.get("explored"):
        best, bv = ranked[0]
        why = f"This was an exploration/noise pick rather than its top-ranked option ({best}, {bv:.3g})"
    elif all(v == 0 for _, v in ranked):
        why = f"Its {f.value_label()} are all zero in this state, so the choice is a tie-break"
    else:
        alts = [(k, v) for k, v in ranked if k != a]
        chosen = dict(ranked).get(a)
        cv = f"{chosen:.3g}" if chosen is not None else "n/a"
        why = f"It ranks {a} highest among its {f.value_label()} ({cv}"
        why += f" vs {alts[0][0]} {alts[0][1]:.3g})" if alts else ")"

    ev = ""
    if f.shap:
        s = max(f.shap, key=lambda e: e["output"])  # the most suspicious detector reading drives detection
        ev = f"; the {s['model']} detector reads {s['output']:.2f} on host {s['node']} (baseline {s['base_value']:.2f})"
        if s["top_features"]:
            top = s["top_features"][0]
            ev += f", driven mostly by {top['name']} ({top['shap']:+.2f})"
    end = f" The move ends the game: {t.get('winner')} wins." if t.get("done") else ""
    return f"{first} {why}{ev}.{end}"


# ------------------------------------------------------------------------------------------------ parsing


def clean_text(text: str, max_sentences: int = 2, max_chars: int = 400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip().strip('"').strip()
    text = re.sub(r"^(rationale|answer|explanation)\s*:\s*", "", text, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sentences[:max_sentences]).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return text


def parse_response(message) -> tuple[str | None, dict]:
    """Extract the rationale from a Messages API response. Returns (text or None, meta)."""
    stop = getattr(message, "stop_reason", None)
    usage = getattr(message, "usage", None)
    meta = {
        "stop_reason": stop,
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }
    if stop == "refusal":
        details = getattr(message, "stop_details", None)
        meta["refusal_category"] = getattr(details, "category", None)
        return None, meta
    parts = [b.text for b in (getattr(message, "content", None) or []) if getattr(b, "type", None) == "text"]
    text = clean_text(" ".join(parts))
    return (text or None), meta


# --------------------------------------------------------------------------------------------- the client


def resolve_model(cli_value: str | None = None) -> str:
    return cli_value or os.environ.get(MODEL_ENV) or DEFAULT_MODEL


def require_credentials() -> None:
    """Load .env and fail loudly if no API key is available. Never prints the key."""
    try:
        from dotenv import load_dotenv

        from cyberarena.config import ROOT

        load_dotenv(ROOT / ".env")
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        pass
    if not (os.environ.get(KEY_ENV) or "").strip():
        raise MissingCredentialsError(
            f"{KEY_ENV} is not set, so online narration cannot run. Set it in your environment or in .env at the "
            f"repo root (copy .env.example), or rerun with --offline for deterministic template rationales."
        )


def request_params(model: str, max_tokens: int | None = None) -> dict:
    """Per-model request knobs. Haiku runs without thinking; newer models keep thinking on at low effort."""
    if model.startswith("claude-haiku"):
        return {"max_tokens": max_tokens or 200}
    return {"max_tokens": max_tokens or 2048, "output_config": {"effort": "low"}}


class RateLimiter:
    """Simple minimum-interval limiter shared by worker threads (requests per minute)."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)


@dataclass
class NarrationStats:
    prompts: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    template: int = 0
    refusals: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_chars: list[int] = field(default_factory=list)


class Narrator:
    def __init__(self, offline: bool = False, model: str | None = None, cache_path: Path | None = None,
                 client=None, concurrency: int = 4, rpm: float = 50.0, max_tokens: int | None = None):  # fmt: skip
        self.offline = offline
        self.model = resolve_model(model)
        self.cache_path = Path(cache_path) if cache_path else None
        self.concurrency = max(1, int(concurrency))
        self.limiter = RateLimiter(rpm)
        self.params = request_params(self.model, max_tokens)
        self.stats = NarrationStats()
        self._client = client
        self._cache: dict[str, dict] = {}
        self._cache_lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["value"]
        if not offline and client is None:
            require_credentials()
            import anthropic

            self._client = anthropic.Anthropic(max_retries=4)

    def cache_key(self, prompt: str) -> str:
        blob = json.dumps([PROMPT_VERSION, self.model, SYSTEM_PROMPT, self.params, prompt], sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_put(self, key: str, value: dict) -> None:
        with self._cache_lock:
            self._cache[key] = value
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": key, "value": value}, separators=(",", ":")) + "\n")

    def _call(self, prompt: str) -> tuple[str | None, dict]:
        import anthropic

        self.limiter.wait()
        try:
            msg = self._client.messages.create(
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                **self.params,
            )
        except anthropic.AuthenticationError as e:
            raise NarrationError(f"Anthropic API rejected the credentials (401). Check {KEY_ENV}.") from e
        except anthropic.NotFoundError as e:
            raise NarrationError(f"Model {self.model!r} not found or not available to this key.") from e
        except anthropic.APIStatusError as e:
            raise NarrationError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise NarrationError("Could not reach the Anthropic API (network error).") from e
        return parse_response(msg)

    def narrate_one(self, facts: TurnFacts) -> tuple[str, dict]:
        """Returns (rationale, meta). meta["source"] is "claude", "cache", "template" or "template:refusal"."""
        if self.offline:
            self.stats.template += 1
            return template_rationale(facts), {"source": "template"}
        prompt = build_prompt(facts)
        key = self.cache_key(prompt)
        self.stats.prompts += 1
        self.stats.prompt_chars.append(len(prompt))
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            self.stats.cache_hits += 1
            return hit["text"], {"source": "cache", "model": self.model}
        text, meta = self._call(prompt)
        self.stats.api_calls += 1
        self.stats.input_tokens += meta["input_tokens"]
        self.stats.output_tokens += meta["output_tokens"]
        if text is None:  # refusal or empty: keep going, but say so in the record
            self.stats.refusals += 1
            return template_rationale(facts), {"source": "template:refusal", "model": self.model, **meta}
        self._cache_put(key, {"text": text, **meta})
        return text, {"source": "claude", "model": self.model, **meta}

    def narrate_many(self, facts: Sequence[TurnFacts],
                     progress: Callable[[int, int], None] | None = None) -> list[tuple[str, dict]]:  # fmt: skip
        if self.offline or self.concurrency == 1:
            out = []
            for i, f in enumerate(facts):
                out.append(self.narrate_one(f))
                if progress:
                    progress(i + 1, len(facts))
            return out
        results: list[tuple[str, dict] | None] = [None] * len(facts)
        done = 0
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self.narrate_one, f): i for i, f in enumerate(facts)}
            for fut, i in futures.items():
                results[i] = fut.result()
                done += 1
                if progress:
                    progress(done, len(facts))
        return results  # type: ignore[return-value]


# ------------------------------------------------------------------------------------------- cost estimate


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~3.5 chars/token for this structured English). Online runs report real usage."""
    return math.ceil(len(text) / 3.5)


def estimate_cost(prompts: Sequence[str], model: str, output_tokens_per_call: int = 70) -> dict:
    sys_tokens = estimate_tokens(SYSTEM_PROMPT)
    in_tok = sum(estimate_tokens(p) + sys_tokens + 8 for p in prompts)
    out_tok = output_tokens_per_call * len(prompts)
    pin, pout = PRICES.get(model, PRICES[DEFAULT_MODEL])
    return {
        "model": model,
        "calls": len(prompts),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "usd": round(in_tok / 1e6 * pin + out_tok / 1e6 * pout, 6),
        "price_per_mtok": {"input": pin, "output": pout},
        "method": "chars/3.5 estimate, ~70 output tokens per call",
    }
