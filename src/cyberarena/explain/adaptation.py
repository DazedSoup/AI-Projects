"""Adaptive-learning facts per turn (v2 runs): red's evasion levels, detector versions, and detector retrains.

Everything here is computed from logged data only:
- ``classifier_inputs[].evasion`` / ``.version`` and the turn's ``detector_versions``;
- ``runs/<id>/learning.jsonl`` (streamed): ``detector_update`` rows (when each version was trained) and
  ``red_evasion`` rows (red's level history);
- red's previous logged move on the same host (this episode, or the latest earlier logged episode), to say whether
  a detector was retrained since then.

v1 runs (no version / evasion / detector_versions fields) get ``adaptation = None`` and no extra text.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path

from cyberarena.explain.shap_values import entry_version

# Abstract phrasing per sensor: "evasion" is interpolation toward benign rows in feature space, nothing more.
BLEND = {
    "network": ("network activity", "normal traffic"),
    "malware": ("malware footprint", "benign files"),
    "phishing": ("phishing lure", "legitimate mail"),
}


def is_adaptive_turn(turn: dict) -> bool:
    if "detector_versions" in turn:
        return True
    return any("version" in ci or "evasion" in ci for ci in turn.get("classifier_inputs") or [])


def turn_versions(turn: dict) -> dict[str, int]:
    """Detector versions in force on this turn: ``detector_versions``, else the versions on the logged rows."""
    dv = turn.get("detector_versions")
    if isinstance(dv, dict) and dv:
        return {m: int(v or 0) for m, v in dv.items()}
    out: dict[str, int] = {}
    for ci in turn.get("classifier_inputs") or []:
        if "version" in ci:
            out[ci["model"]] = max(out.get(ci["model"], 0), entry_version(ci))
    return out


def sort_key(turn: dict) -> tuple[int, int, int]:
    """Training-time order: eval games at checkpoint A come right after training episode A.

    **GUESS:** eval games run with the agents/detectors as of ``after_episode`` (eval ids are numbered after training).
    """
    if turn.get("phase") == "eval" and turn.get("after_episode") is not None:
        return (int(turn["after_episode"]), 1, int(turn["episode"]))
    return (int(turn["episode"]), 0, int(turn["episode"]))


# ------------------------------------------------------------------------------------------- learning.jsonl


@dataclass
class LearningLog:
    updates: dict[tuple[str, int], dict] = field(
        default_factory=dict
    )  # (model, version) -> detector_update row
    evasion: list[tuple[int, dict[str, float]]] = field(default_factory=list)  # (episode, levels), sorted
    caught: list[tuple[int, dict[str, float | None]]] = field(default_factory=list)  # (episode, caught_rate)

    @classmethod
    def load(cls, path: Path | None) -> LearningLog:
        """Stream ``learning.jsonl``; keeps only detector_update (minus eval dicts) and red_evasion rows."""
        log = cls()
        if path is None or not Path(path).exists():
            return log
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # a line being written right now
                    continue
                kind = rec.get("kind")
                if kind == "detector_update" and "model" in rec and "version" in rec:
                    log.updates[(rec["model"], int(rec["version"]))] = {
                        k: rec.get(k) for k in ("episode", "n_new", "loss_before", "loss_after")
                    }
                elif kind == "red_evasion" and isinstance(rec.get("levels"), dict):
                    log.evasion.append((int(rec.get("episode", 0)), dict(rec["levels"])))
                    if isinstance(rec.get("caught_rate"), dict):
                        log.caught.append((int(rec.get("episode", 0)), dict(rec["caught_rate"])))
        log.evasion.sort(key=lambda r: r[0])
        log.caught.sort(key=lambda r: r[0])
        return log

    def retrained_at(self, model: str, version: int) -> int | None:
        row = self.updates.get((model, version))
        return None if row is None else row.get("episode")

    def caught_rate(self, model: str, upto: int) -> float | None:
        """Latest logged red_evasion ``caught_rate`` for ``model`` at or before ``upto`` (rolling window, arena-defined)."""
        i = bisect.bisect_right([e for e, _ in self.caught], upto)
        for _, rates in reversed(self.caught[:i]):
            if rates.get(model) is not None:
                return round(float(rates[model]), 3)
        return None

    def evasion_change(self, model: str, level: float, upto: int) -> dict | None:
        """Most recent logged red_evasion level for ``model`` (at or before ``upto``) that differs from ``level``."""
        i = bisect.bisect_right([e for e, _ in self.evasion], upto)
        for ep, levels in reversed(self.evasion[:i]):
            prev = levels.get(model)
            if prev is not None and abs(float(prev) - level) > 1e-9:
                return {"level": round(float(prev), 3), "episode": ep}
        return None


# ---------------------------------------------------------------------------------- red's previous host moves


class RedMoveHistory:
    """Per host, the detector versions at red's last logged move on it in each logged episode.

    Filled during the indexing pass (``observe``), so it costs no extra read of ``episodes.jsonl``.
    """

    def __init__(self) -> None:
        self._by_host: dict[int, dict[int, tuple[tuple, dict[str, int]]]] = {}
        self._sorted: dict[int, list[tuple[tuple, int, dict[str, int]]]] | None = None

    def observe(self, turn: dict) -> None:
        if turn.get("actor") != "red" or turn.get("target") is None or not is_adaptive_turn(turn):
            return
        self._by_host.setdefault(int(turn["target"]), {})[int(turn["episode"])] = (
            sort_key(turn),
            turn_versions(turn),
        )
        self._sorted = None

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_host.values())

    def before(self, host: int, turn: dict) -> dict[str, int] | None:
        """Versions at red's last move on ``host`` in an earlier logged episode (by training-time order)."""
        if self._sorted is None:
            self._sorted = {
                h: sorted((k, ep, v) for ep, (k, v) in eps.items()) for h, eps in self._by_host.items()
            }
        rows = self._sorted.get(host) or []
        key = sort_key(turn)
        i = bisect.bisect_left([r[0] for r in rows], key)
        while i > 0:
            i -= 1
            if rows[i][1] != turn["episode"]:
                return rows[i][2]
        return None


# ------------------------------------------------------------------------------------------- per-turn facts


class AdaptationTracker:
    """Computes the ``adaptation`` field turn by turn (turns of one episode must arrive in order)."""

    def __init__(self, learning: LearningLog | None = None, history: RedMoveHistory | None = None):
        self.learning = learning or LearningLog()
        self.history = history or RedMoveHistory()
        self._episode: int | None = None
        self._in_episode: dict[int, dict[str, int]] = {}  # host -> versions at red's last move this episode

    def facts(self, turn: dict) -> dict | None:
        if turn["episode"] != self._episode:
            self._episode, self._in_episode = turn["episode"], {}
        if not is_adaptive_turn(turn):
            return None
        versions = turn_versions(turn)
        evasion: dict[str, float] = {}
        for ci in turn.get("classifier_inputs") or []:
            if ci.get("evasion") is not None:
                evasion[ci["model"]] = round(max(evasion.get(ci["model"], 0.0), float(ci["evasion"])), 3)

        host = turn.get("target")
        updated: list[str] = []
        previous: dict[str, int] = {}
        if host is not None:
            prev = self._in_episode.get(host)
            if prev is None:
                prev = self.history.before(int(host), turn)
            if prev:
                for m, v in sorted(versions.items()):
                    if m in prev and v > prev[m]:
                        updated.append(m)
                        previous[m] = prev[m]
            if turn.get("actor") == "red":
                self._in_episode[host] = versions

        out: dict = {"evasion": evasion, "detector_versions": versions, "updated_since_last_move": updated}
        retrained = {
            m: ep
            for m, v in versions.items()
            if v > 0 and (ep := self.learning.retrained_at(m, v)) is not None
        }
        if retrained:
            out["retrained_at"] = retrained
        if previous:
            out["previous_versions"] = previous
        upto = sort_key(turn)[0]
        changes = {
            m: c for m, s in evasion.items() if s > 0 and (c := self.learning.evasion_change(m, s, upto))
        }
        if changes:
            out["evasion_prev"] = changes
        caught = {
            m: c
            for m, s in evasion.items()
            if s > 0 and (c := self.learning.caught_rate(m, upto)) is not None
        }
        if caught:
            out["caught_rate"] = caught
        return out


# ------------------------------------------------------------------------------------------------- phrasing


def _primary_input(turn: dict, adaptation: dict) -> dict | None:
    """The row the sentence is about: on the target host if possible, most evasive first, then highest score."""
    cis = [ci for ci in turn.get("classifier_inputs") or [] if "model" in ci]
    if not cis:
        return None
    on_target = [ci for ci in cis if ci.get("node") == turn.get("target")] or cis
    return max(on_target, key=lambda ci: (float(ci.get("evasion") or 0.0), float(ci.get("score") or 0.0)))


def is_relevant(turn: dict, adaptation: dict | None) -> bool:
    """Mention adaptation only if the row was blended (evasion > 0) or its detector was retrained since red's last
    move on the host. A fine-tuned version alone is not news (every late game has one); it's named when mentioned."""
    if not adaptation:
        return False
    if _mentioned_updates(turn, adaptation):
        return True
    ci = _primary_input(turn, adaptation)
    return ci is not None and float(ci.get("evasion") or 0.0) > 0


def _mentioned_updates(turn: dict, adaptation: dict) -> list[str]:
    """Retrained models worth saying: the one that scored this turn's row (all of them if no row was logged)."""
    ups = list(adaptation.get("updated_since_last_move") or [])
    ci = _primary_input(turn, adaptation)
    return [m for m in ups if m == ci["model"]] if ci is not None else ups


def _detector_phrase(model: str, version: int, adaptation: dict) -> str:
    if version <= 0:
        return f"the pretrained {model} detector"
    ep = (adaptation.get("retrained_at") or {}).get(model)
    return f"{model} detector v{version}" + (f", retrained at game {ep}," if ep is not None else "")


def _updated_phrase(turn: dict, adaptation: dict) -> str:
    ups = _mentioned_updates(turn, adaptation)
    if not ups:
        return ""
    prev = adaptation.get("previous_versions") or {}
    cur = adaptation.get("detector_versions") or {}
    parts = [f"v{prev[m]}->v{cur[m]}" if len(ups) == 1 else f"{m} v{prev[m]}->v{cur[m]}" for m in ups
             if m in prev and m in cur]  # fmt: skip
    word = "detector was" if len(ups) == 1 else "detectors were"
    return f"the {' and '.join(ups)} {word} retrained since red last acted on host {turn.get('target')}" + (
        f" ({', '.join(parts)})" if parts else ""
    )


def adaptation_sentence(turn: dict, adaptation: dict | None, threshold: float = 0.5) -> str:
    """One factual sentence (or "") about evasion / detector version / retraining for this turn."""
    if not is_relevant(turn, adaptation):
        return ""
    ci = _primary_input(turn, adaptation)
    upd = _updated_phrase(turn, adaptation)
    if ci is None:
        return upd[0].upper() + upd[1:] + "." if upd else ""
    model, s, v = ci["model"], float(ci.get("evasion") or 0.0), entry_version(ci)
    score = float(ci.get("score") or 0.0)
    what, toward = BLEND.get(model, (f"{model} activity", "benign rows"))
    det = _detector_phrase(model, v, adaptation)
    flagged = score >= threshold
    if turn.get("actor") == "red":
        if s > 0:
            prev = (adaptation.get("evasion_prev") or {}).get(model)
            trend = ""
            if prev:
                trend = f", {'up' if s > prev['level'] else 'down'} from {prev['level']:.2f} at game {prev['episode']}"
            caught = (adaptation.get("caught_rate") or {}).get(model)
            if caught is not None:
                trend += f"; recent caught rate {caught:.2f}"
            head = f"Red blends its {what} toward {toward} (evasion {s:.2f}{trend})"
            tail = (f"{det} still flags it at {score:.2f}" if flagged
                    else f"{det} scores it only {score:.2f}, under the {threshold:.2f} alert line")  # fmt: skip
        else:
            head = f"Red's {what} on host {ci.get('node')} is unblended (evasion 0.00)"
            tail = f"{det} scores it {score:.2f}"
        text = f"{head}; {tail}"
    else:
        verdict = "above" if flagged else "below"
        text = f"{det[0].upper() + det[1:]} reads {score:.2f} on host {ci.get('node')} ({verdict} the {threshold:.2f} alert line)"
        if s > 0:
            text += f", on red {what} blended toward {toward} at evasion {s:.2f}"
    if upd:
        text += f"; {upd}"
    return text.rstrip(",") + "."


def adaptation_prompt_lines(turn: dict, adaptation: dict | None, threshold: float = 0.5) -> list[str]:
    """Structured facts for the online prompt (empty when not relevant, so v1 prompts are unchanged)."""
    if not is_relevant(turn, adaptation):
        return []
    header = (
        "Adaptive learning (detectors are fine-tuned during training; red blends its activity toward benign "
        "rows in feature space, 'evasion' 0-0.7, which also lowers its action success):"
    )
    lines = [header]
    for ci in turn.get("classifier_inputs") or []:
        m, v = ci["model"], entry_version(ci)
        ep = (adaptation.get("retrained_at") or {}).get(m)
        when = f", retrained at game {ep}" if ep is not None else ""
        s = float(ci.get("evasion") or 0.0)
        prev = (adaptation.get("evasion_prev") or {}).get(m)
        trend = f" (was {prev['level']:.2f} at game {prev['episode']})" if prev else ""
        caught = (adaptation.get("caught_rate") or {}).get(m)
        if caught is not None and s > 0:
            trend += f" (red's recent caught rate on {m} rows: {caught:.2f})"
        lines.append(f"- host {ci.get('node')}: {m} row at red evasion {s:.2f}{trend}, scored "
                     f"{float(ci.get('score') or 0):.2f} by {m} detector v{v}{when}.")  # fmt: skip
    upd = _updated_phrase(turn, adaptation)
    if upd:
        lines.append(f"- {upd[0].upper() + upd[1:]}.")
    return lines
