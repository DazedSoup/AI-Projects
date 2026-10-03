"""Learning telemetry (docs/contracts.md, "Adaptive detectors & learning telemetry (v2)"). Pure functions.

Everything here is derived from ``runs/<id>/learning.jsonl`` alone; nothing is simulated, smoothed into
existence or interpolated beyond what the rows say. Runs that predate adaptive learning have no such file:
:func:`load_learning` returns ``None`` and the pages show an empty state.

Row kinds: ``detector_update`` (a detector fine-tune with before/after evaluations), ``red_evasion`` (red's
evasion level per sensor), ``agent_stats`` (Q-table size, TD error, exploration, action mix over a rolling
window) and ``probe`` (Q-values for a fixed, hand-picked situation at each eval checkpoint).

Takeaway helpers return one plain-English sentence computed from the rows (HTML-safe: only numbers and
fixed words are interpolated besides probe descriptions, which are escaped).
"""

from __future__ import annotations

import html
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LEARNING_FILE = "learning.jsonl"
MODEL_ORDER = ("network", "malware", "phishing")
MODEL_LABEL = {"network": "network", "malware": "malware", "phishing": "phishing"}


class LearningFormatError(ValueError):
    """``learning.jsonl`` exists but a line is not valid JSON."""


@dataclass
class Learning:
    updates: pd.DataFrame  # one row per detector_update
    evasion: pd.DataFrame  # long: episode, model, level
    stats: pd.DataFrame  # episode, side, n_states, mean_abs_q, td_error, epsilon
    mix: pd.DataFrame  # long: episode, side, action, share
    probes: pd.DataFrame  # after_episode, side, probe_id, description, chosen, q (dict)

    @property
    def models(self) -> list[str]:
        seen = set(self.updates["model"]) | set(self.evasion["model"])
        return [m for m in MODEL_ORDER if m in seen] + sorted(seen - set(MODEL_ORDER))

    @property
    def dqn(self) -> bool:
        """Agent stats come from Q-networks (v4): ``loss``/``replay_size`` logged, no Q-table size."""
        s = self.stats
        return not s.empty and (s["loss"].notna().any() or s["replay_size"].notna().any()) and s["n_states"].isna().all()

    @property
    def empty(self) -> bool:
        return all(df.empty for df in (self.updates, self.evasion, self.stats, self.probes))


STATS_COLS = ["episode", "side", "n_states", "mean_abs_q", "td_error", "epsilon", "loss", "q_mean", "replay_size",
              "grad_steps", "win_rate"]  # fmt: skip
_UPD_COLS = ["episode", "model", "version", "n_new", "n_replay", "loss_before", "loss_after", "seconds",
             "clean_before", "clean_after", "before", "after", "red_level", "n_malicious"]  # fmt: skip


def parse_learning(lines) -> Learning:
    upd, eva, stats, mix, probes = [], [], [], [], []
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise LearningFormatError(f"{LEARNING_FILE}:{lineno}: invalid JSON ({e.msg})") from e
        kind = r.get("kind")
        if kind == "detector_update":
            b, a = r.get("before") or {}, r.get("after") or {}
            upd.append({"episode": int(r.get("episode", 0)), "model": r.get("model"),
                        "version": int(r.get("version", 0)), "n_new": r.get("n_new"), "n_replay": r.get("n_replay"),
                        "loss_before": r.get("loss_before"), "loss_after": r.get("loss_after"),
                        "seconds": r.get("seconds"), "clean_before": b.get("clean_auc"),
                        "clean_after": a.get("clean_auc"), "before": b, "after": a,
                        "red_level": r.get("red_level"), "n_malicious": r.get("n_malicious")})  # fmt: skip
        elif kind == "red_evasion":
            rec, ver = r.get("recall_at_level") or {}, r.get("detector_versions") or {}
            caught = r.get("caught_rate") or {}
            for m, s in (r.get("levels") or {}).items():
                eva.append({"episode": int(r.get("episode", 0)), "model": m, "level": float(s),
                            "recall": rec.get(m), "version": ver.get(m), "caught_rate": caught.get(m)})  # fmt: skip
        elif kind == "agent_stats":
            ep, side = int(r.get("episode", 0)), r.get("side")
            stats.append({"episode": ep, "side": side, "n_states": r.get("n_states"),
                          "mean_abs_q": r.get("mean_abs_q"), "td_error": r.get("td_error"),
                          "epsilon": r.get("epsilon"), "loss": r.get("loss"), "q_mean": r.get("q_mean"),
                          "replay_size": r.get("replay_size"), "grad_steps": r.get("grad_steps"),
                          "win_rate": r.get("win_rate")})  # fmt: skip
            for a, share in (r.get("action_mix") or {}).items():
                mix.append({"episode": ep, "side": side, "action": a, "share": float(share)})
        elif kind == "probe":
            probes.append({"after_episode": int(r.get("after_episode", 0)), "side": r.get("side"),
                           "probe_id": str(r.get("probe_id")), "description": r.get("description") or "",
                           "chosen": r.get("chosen"), "q": r.get("q") or {}})  # fmt: skip
    return Learning(
        updates=pd.DataFrame(upd, columns=_UPD_COLS).sort_values(["model", "version"]).reset_index(drop=True),
        evasion=pd.DataFrame(eva, columns=["episode", "model", "level", "recall", "version", "caught_rate"]).sort_values(["model", "episode"])
        .reset_index(drop=True),
        stats=pd.DataFrame(stats, columns=STATS_COLS)
        .sort_values(["side", "episode"]).reset_index(drop=True),
        mix=pd.DataFrame(mix, columns=["episode", "side", "action", "share"]),
        probes=pd.DataFrame(probes, columns=["after_episode", "side", "probe_id", "description", "chosen", "q"])
        .sort_values(["side", "probe_id", "after_episode"]).reset_index(drop=True),
    )  # fmt: skip


def load_learning(run_dir: Path) -> Learning | None:
    """Parsed ``learning.jsonl``, or ``None`` when the run predates adaptive learning (no file)."""
    path = Path(run_dir) / LEARNING_FILE
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return parse_learning(f)


# ============================================================================================== detector state


def level_key(d: dict, s: float) -> str | None:
    """The key of ``d`` (``{"0.3": …}``) whose level is nearest to ``s``."""
    if not d:
        return None
    return min(d, key=lambda k: abs(float(k) - s))


def at_level(evaluation: dict, s: float, metric: str = "recall") -> float | None:
    """``recall_by_level`` (falling back to ``auc_by_level``) at the level nearest ``s``."""
    d = evaluation.get(f"{metric}_by_level") or {}
    k = level_key(d, s)
    return float(d[k]) if k is not None else None


def detector_metric(lr: Learning) -> str:
    """``"recall"`` if the updates log ``recall_by_level``, else ``"auc"``."""
    for b in lr.updates["after"]:
        if (b or {}).get("recall_by_level"):
            return "recall"
    return "auc"


def evasion_at(lr: Learning, model: str, episode: float, strict: bool = False) -> float:
    """Red's evasion level for ``model`` in effect at ``episode`` (0.0 before the first logged level).

    ``strict``: the level logged *before* ``episode``, i.e. the one red used while the samples of an update
    at ``episode`` were collected (a level logged at the same episode is red's reaction to that window).
    """
    ep = lr.evasion["episode"]
    e = lr.evasion[(lr.evasion["model"] == model) & ((ep < episode) if strict else (ep <= episode))]
    return float(e["level"].iloc[-1]) if not e.empty else 0.0


def state_at(lr: Learning, model: str, episode: float) -> tuple[int, dict]:
    """``(version, evaluation)`` of ``model``'s detector at ``episode``: the last update's ``after`` at or
    before it, or the first update's ``before`` (the pretrained model) when no update has happened yet."""
    u = lr.updates[lr.updates["model"] == model]
    done = u[u["episode"] <= episode]
    if not done.empty:
        r = done.iloc[-1]
        return int(r["version"]), r["after"]
    if not u.empty:
        r = u.iloc[0]
        return int(r["version"]) - 1, r["before"]
    return 0, {}


def n_malicious_eval(evaluation: dict) -> int | None:
    """How many malicious held-out rows an evaluation's recall rests on (``n_eval.malicious``), if logged."""
    n = (evaluation.get("n_eval") or {}).get("malicious") if evaluation else None
    return int(n) if n else None


def update_level(lr: Learning, r) -> float:
    """The evasion level red used while an update's samples were collected (logged ``red_level`` if present)."""
    lvl = r.get("red_level")
    if lvl is not None and not pd.isna(lvl):
        return float(lvl)
    return evasion_at(lr, r["model"], r["episode"], strict=True)


def arms_race(lr: Learning, model: str) -> pd.DataFrame:
    """Step series for one sensor: red's evasion level and the detector's recall *at red's current level*.

    Points come from every logged event. A detector update contributes two points at the same episode
    (before and after, at the level red was using), so the recovery shows as a vertical jump. A red evasion
    reading contributes red's (possibly new) level with the recall the arena logged for it (``recall_at_level``)
    or, for runs without that field, the recall read off the detector's latest evaluation.
    Columns: ``episode, evasion, score, version, event, n`` (``n`` = malicious test rows behind the recall).
    """
    metric = detector_metric(lr)
    ev = lr.evasion[lr.evasion["model"] == model]
    up = lr.updates[lr.updates["model"] == model]
    times = sorted(set(ev["episode"]) | set(up["episode"]))
    rows = []
    for t in times:
        hit = up[up["episode"] == t]
        if not hit.empty:
            r = hit.iloc[-1]
            s0 = update_level(lr, r)
            rows.append([t, s0, at_level(r["before"], s0, metric), int(r["version"]) - 1, "before",
                         n_malicious_eval(r["before"])])  # fmt: skip
            rows.append([t, s0, at_level(r["after"], s0, metric), int(r["version"]), "after",
                         n_malicious_eval(r["after"])])  # fmt: skip
        e = ev[ev["episode"] == t]
        if not e.empty:
            row = e.iloc[-1]
            s = float(row["level"])
            v, state = state_at(lr, model, t)
            score = row["recall"] if metric == "recall" and pd.notna(row["recall"]) else None
            if score is None:
                score = at_level(state, s, metric) if state else None
            ver = int(row["version"]) if pd.notna(row["version"]) else v
            rows.append([t, s, score, ver, "evasion", n_malicious_eval(state)])
    return pd.DataFrame(rows, columns=["episode", "evasion", "score", "version", "event", "n"])


def escalations(levels: list[float]) -> int:
    """How many separate times red escalated: runs of rising levels, each counted once."""
    n, rising = 0, False
    for a, b in itertools.pairwise(levels):
        if b > a + 1e-9:
            if not rising:
                n += 1
            rising = True
        elif b < a - 1e-9:
            rising = False
    return n


def _dips(ev: pd.DataFrame, min_drop: float = 0.1) -> list[tuple[float, float]]:
    """Per escalation that hurt the detector: (recall right after red rose, best recall before red backed off
    or rose again). An escalation counts only when recall fell by at least ``min_drop`` with it."""
    out, low, high, prev, prev_sc = [], None, None, None, None
    for lvl, sc in zip(ev["evasion"], ev["score"], strict=True):
        if prev is not None and lvl > prev + 1e-9:
            if low is not None:
                out.append((low, high))
                low = high = None
            if prev_sc is not None and sc <= prev_sc - min_drop:
                low = high = sc
        elif low is not None:
            if lvl < prev - 1e-9:
                out.append((low, high))
                low = high = None
            else:
                high = max(high, sc)
        prev, prev_sc = lvl, sc
    if low is not None:
        out.append((low, high))
    return out


def sensor_story(lr: Learning, model: str) -> dict:
    """Classify one sensor's arms race from the logged series and describe it in one honest sentence.

    ``kind``: ``"cycles"`` (red escalates repeatedly and the detector recovers), ``"stuck"`` (red sits at its top
    level and recall there stays low), ``"noisy"`` (cycles, but on a small evaluation set) or ``"quiet"``.
    """
    word = "recall" if detector_metric(lr) == "recall" else "AUC"
    ar = arms_race(lr, model).dropna(subset=["score"])
    ev = ar[ar["event"] == "evasion"]
    name = model.capitalize()
    if ev.empty:
        return {"kind": "quiet", "model": model, "text": f"{name}: no evasion readings logged."}
    levels = ev["evasion"].tolist()
    top = max(levels)
    n_esc = escalations(levels)
    share_top = sum(abs(x - top) < 1e-9 for x in levels) / len(levels)
    at_top = ev[(ev["evasion"] - top).abs() < 1e-9]["score"].astype(float)
    ns = [int(x) for x in ar["n"].dropna()]
    n_small = min(ns) if ns else None
    u = lr.updates[lr.updates["model"] == model]
    mal = u["n_malicious"].dropna().astype(float)
    others = lr.updates[lr.updates["model"] != model]["n_malicious"].dropna().astype(float)
    ratchet = ""
    ls = landscape(lr, model, "recall" if word == "recall" else "auc")
    if ls["levels"]:
        r0, r1 = ls["z"][-1][0], ls["z"][-1][-1]
        if r0 is not None and r1 is not None and r1 - r0 >= 0.1:
            ratchet = (
                f"; at evasion {ls['levels'][-1]:.1f} its {word} ratcheted up from {_pct(r0)} to {_pct(r1)}"
            )
    noise = ""
    if n_small is not None and n_small < 60:
        half = 100 * 1.96 * math.sqrt(0.25 / n_small)
        noise = (f" Each reading rests on only {n_small} malicious test rows (about ±{half:.0f} pts), so read the "
                 "trend, not single points.")  # fmt: skip
    if share_top >= 0.6 and len(at_top) and float(at_top.max()) < 0.5:
        few = ""
        if len(mal) and len(others) and mal.mean() < 0.5 * others.mean():
            few = (f": it gets only about {mal.mean():.0f} confirmed malicious examples per update "
                   f"(vs about {others.mean():.0f} for the other sensors), too few to learn the disguise")  # fmt: skip
        text = (f"{name} does not oscillate. Red sits at evasion {top:.1f} for {share_top:.0%} of training and "
                f"{word} there stays at {_pct(float(at_top.min()))}–{_pct(float(at_top.max()))}; red's disguised "
                f"{model} still evades{few}." + noise)  # fmt: skip
        return {"kind": "stuck", "model": model, "text": text}
    pairs = [(lo, hi) for lo, hi in _dips(ev) if hi - lo >= 0.05]
    if len(pairs) >= 2:
        n_esc = len(pairs)
        lo, hi = [p[0] for p in pairs], [p[1] for p in pairs]
        text = (f"{name}: {n_esc} rounds of back-and-forth. Each time red raised its evasion, {word} at its level "
                f"dipped to {_pct(min(lo))}–{_pct(max(lo))} and recovered to {_pct(min(hi))}–{_pct(max(hi))} after "
                f"retraining{ratchet}." + noise)  # fmt: skip
        return {"kind": "noisy" if noise else "cycles", "model": model, "text": text}
    text = (f"{name}: red escalated {n_esc} time{'s' if n_esc != 1 else ''}; {word} at its level ended at "
            f"{_pct(float(ev['score'].iloc[-1]))}{ratchet}." + noise)  # fmt: skip
    return {"kind": "noisy" if noise else "quiet", "model": model, "text": text}


def headline_model(lr: Learning) -> str | None:
    """The sensor whose arms race is clearest: clean cycles first, then the most escalations."""
    best = None
    for m in lr.models:
        ar = arms_race(lr, m)
        key = (
            sensor_story(lr, m)["kind"] == "cycles",
            escalations(ar[ar["event"] == "evasion"]["evasion"].tolist()),
        )
        if best is None or key > best[0]:
            best = (key, m)
    return best[1] if best else None


def landscape(lr: Learning, model: str, metric: str = "auc") -> dict:
    """Detector ``metric`` over (training game × evasion level), one column per detector version.

    The first column is the pretrained model (the first update's ``before``); each further column is one
    update's ``after``. Returns ``{"episodes", "versions", "levels", "z"}`` with ``z[level_i][version_i]``.
    """
    u = lr.updates[lr.updates["model"] == model]
    if u.empty:
        return {"episodes": [], "versions": [], "levels": [], "z": []}
    cols = [(0, int(u.iloc[0]["version"]) - 1, u.iloc[0]["before"])]
    cols += [(int(r.episode), int(r.version), r.after) for r in u.itertuples(index=False)]
    keys = sorted({k for _, _, e in cols for k in (e.get(f"{metric}_by_level") or {})}, key=float)
    z = [[(e.get(f"{metric}_by_level") or {}).get(k) for _, _, e in cols] for k in keys]
    return {"episodes": [c[0] for c in cols], "versions": [c[1] for c in cols], "levels": [float(k) for k in keys],
            "z": z}  # fmt: skip


def version_table(lr: Learning, model: str) -> pd.DataFrame:
    """One row per detector version of ``model`` with the forgetting check and the evasive-traffic gain."""
    u = lr.updates[lr.updates["model"] == model]
    cols = ["version", "episode", "n_new", "n_replay", "loss_before", "loss_after", "clean_before",
            "clean_after", "clean_vs_pretrained", "red_level", "evasive_before", "evasive_after"]  # fmt: skip
    if u.empty:
        return pd.DataFrame(columns=cols)
    clean0 = u.iloc[0]["clean_before"]
    rows = []
    for r in u.itertuples(index=False):
        s = update_level(lr, r._asdict())
        rows.append([int(r.version), int(r.episode), r.n_new, r.n_replay, r.loss_before, r.loss_after,
                     r.clean_before, r.clean_after,
                     None if clean0 is None or r.clean_after is None else float(r.clean_after) - float(clean0),
                     s, at_level(r.before, s, "auc"), at_level(r.after, s, "auc")])  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def current_evasion(lr: Learning) -> dict[str, float]:
    if lr.evasion.empty:
        return {}
    last = lr.evasion.sort_values("episode").groupby("model").tail(1)
    return {r.model: float(r.level) for r in last.itertuples(index=False)}


def first_evasion(lr: Learning) -> dict[str, float]:
    if lr.evasion.empty:
        return {}
    first = lr.evasion.sort_values("episode").groupby("model").head(1)
    return {r.model: float(r.level) for r in first.itertuples(index=False)}


# ============================================================================================== agents


def mix_matrix(lr: Learning, side: str) -> tuple[list[str], list[int], list[list[float]]]:
    """``(actions, episodes, share[action][episode])`` for one side's action mix over training."""
    m = lr.mix[lr.mix["side"] == side]
    if m.empty:
        return [], [], []
    piv = m.pivot_table(index="action", columns="episode", values="share", aggfunc="last").fillna(0.0)
    order = piv.iloc[:, -1].sort_values(ascending=False).index  # most used at the end of training on top
    piv = piv.loc[order]
    return list(piv.index), [int(c) for c in piv.columns], piv.to_numpy().tolist()


def probe_ids(lr: Learning, side: str) -> list[str]:
    p = lr.probes[lr.probes["side"] == side]
    return list(dict.fromkeys(p["probe_id"]))


def probe_matrix(lr: Learning, side: str, probe_id: str) -> dict:
    """Q-values of one probe situation across checkpoints: ``{"description", "actions", "checkpoints", "q",
    "chosen"}`` with ``q[action][checkpoint]`` (``None`` where an action wasn't scored)."""
    p = lr.probes[(lr.probes["side"] == side) & (lr.probes["probe_id"] == probe_id)].sort_values(
        "after_episode"
    )
    if p.empty:
        return {"description": "", "actions": [], "checkpoints": [], "q": [], "chosen": []}
    actions = list(dict.fromkeys(a for q in p["q"] for a in q))
    last_q = p.iloc[-1]["q"]
    if len(actions) > 10:  # DQN probes log the top 8 moves per checkpoint: keep the final top 8 plus every choice
        keep = set(sorted(last_q, key=lambda a: -(last_q[a] if last_q[a] is not None else -math.inf))[:8])
        keep |= {c for c in p["chosen"] if c}
        actions = [a for a in actions if a in keep]
    actions.sort(key=lambda a: -(last_q.get(a) if last_q.get(a) is not None else -math.inf))
    return {
        "description": p.iloc[-1]["description"],
        "actions": actions,
        "checkpoints": [int(c) for c in p["after_episode"]],
        "q": [[q.get(a) for q in p["q"]] for a in actions],
        "chosen": list(p["chosen"]),
    }


def probe_change(pm: dict) -> dict:
    """How the preferred action moved across checkpoints.

    ``first`` is the choice at the first checkpoint where the agent had any values for the situation (all-zero
    Q-values are an untrained tie, not a preference); ``changed`` means the choice differed at some checkpoint;
    ``settled_at`` is where the final preference took hold for good.
    """
    ch, cps = pm["chosen"], pm["checkpoints"]
    if not ch:
        return {"first": None, "last": None, "settled_at": None, "changed": False, "switches": 0}
    trained = [i for i, col in enumerate(zip(*pm["q"], strict=False)) if any((v or 0) != 0 for v in col)]
    i0 = trained[0] if trained else 0
    settled = len(ch) - 1
    while settled > i0 and ch[settled - 1] == ch[-1]:
        settled -= 1
    switches = sum(a != b for a, b in itertools.pairwise(ch[i0:]))
    return {"first": ch[i0], "last": ch[-1], "settled_at": cps[settled], "changed": switches > 0,
            "switches": switches}  # fmt: skip


# ============================================================================================== takeaways


def move_label(key) -> str:
    """``"monitor→15"`` -> ``"monitor → host 15"``; a plain action id just loses its underscores."""
    a, sep, t = str(key).partition("→")
    return a.replace("_", " ") + (f" → host {t}" if sep else "")


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def esc(s) -> str:
    return html.escape(str(s))


def arms_race_takeaway(lr: Learning, models: list[str] | None = None) -> str:
    """One sentence for the arms race: the clearest sensor's story (or the given sensor's)."""
    model = (models or [None])[0] if models and len(models) == 1 else headline_model(lr)
    if model is None:
        return "No detector updates or evasion changes were logged."
    return sensor_story(lr, model)["text"]


def landscape_takeaway(lr: Learning, model: str) -> str:
    ls = landscape(lr, model, "auc")
    if not ls["levels"]:
        return f"The {model} detector has not been updated yet."
    top = len(ls["levels"]) - 1
    hi0, hi1 = ls["z"][top][0], ls["z"][top][-1]
    c0, c1 = ls["z"][0][0], ls["z"][0][-1]
    if hi0 is None or hi1 is None:
        return f"The {model} detector has {len(ls['versions']) - 1} logged updates."
    clean = f"; clean-traffic AUC moved {c0:.3f} → {c1:.3f}" if c0 is not None and c1 is not None else ""
    return (f"At evasion {ls['levels'][top]:.1f}, the {model} detector v{ls['versions'][-1]} scores AUC "
            f"{hi1:.2f}, {'up' if hi1 >= hi0 else 'down'} from {hi0:.2f} for the pretrained model{clean}.")  # fmt: skip


def versions_takeaway(lr: Learning, model: str, tolerance: float = 0.02) -> str:
    vt = version_table(lr, model)
    if vt.empty:
        return f"No {model} detector updates were logged."
    worst = vt["clean_vs_pretrained"].dropna()
    forgot = worst[worst < -tolerance]
    gain = (vt["evasive_after"] - vt["evasive_before"]).dropna()
    avg = (
        f"; each update added {gain.mean():+.3f} AUC on red's current evasion on average"
        if not gain.empty
        else ""
    )
    if forgot.empty:
        drop = max(0.0, -float(worst.min())) if not worst.empty else 0.0
        held = (f"never fell more than {drop:.3f} below the pretrained model" if drop > 0.0005
                else "never fell below the pretrained model")  # fmt: skip
        return f"{len(vt)} {model} updates, and clean-traffic AUC {held} (forgetting check passed){avg}."  # fmt: skip
    v = int(vt.loc[forgot.index[0], "version"])
    return (f"Forgetting check: from v{v}, clean-traffic AUC is more than {tolerance:.2f} below the pretrained "
            f"model ({float(forgot.min()):+.3f} at worst){avg}.")  # fmt: skip


def dqn_takeaway(lr: Learning) -> str:
    """Loss, value and memory over training for Q-network agents, in one sentence."""
    parts = []
    for side in ("red", "blue"):
        s = lr.stats[lr.stats["side"] == side].sort_values("episode")
        lo = s.dropna(subset=["loss"])
        if len(lo) < 2:
            continue
        k = max(1, len(lo) // 5)
        l0, l1 = float(lo["loss"].iloc[:k].mean()), float(lo["loss"].iloc[-k:].mean())
        gs = s["grad_steps"].dropna()
        steps = f" over {int(gs.iloc[-1]):,} gradient steps" if not gs.empty else ""
        trend = (f"loss fell from {l0:.3f} to {l1:.3f}" if l1 < l0 * 0.9 else
                 f"loss rose from {l0:.3f} to {l1:.3f}" if l1 > l0 * 1.1 else f"loss held near {l1:.3f}")  # fmt: skip
        q = s["q_mean"].dropna()
        qtxt = f", mean Q {float(q.iloc[0]):+.2f} → {float(q.iloc[-1]):+.2f}" if len(q) >= 2 else ""
        parts.append(f"{side}'s {trend}{steps}{qtxt}")
    if not parts:
        return "No Q-network statistics were logged."
    eps = lr.stats["epsilon"].dropna()
    tail = f"; exploration decayed to {_pct(float(eps.iloc[-1]))}" if not eps.empty else ""
    line = "; ".join(parts)
    return line[:1].upper() + line[1:] + tail + ". Loss is the Huber TD loss on replayed batches, so it moves with the targets too."


def agent_takeaway(lr: Learning) -> str:
    if lr.dqn:
        return dqn_takeaway(lr)
    parts = []
    for side in ("red", "blue"):
        s = lr.stats[lr.stats["side"] == side].dropna(subset=["td_error"])
        if len(s) < 2:
            continue
        td0, td1 = float(s["td_error"].iloc[0]), float(s["td_error"].iloc[-1])
        n1 = int(s["n_states"].iloc[-1]) if pd.notna(s["n_states"].iloc[-1]) else None
        fall = (
            f"TD error fell {(1 - td1 / td0):.0%}"
            if td0 > 0 and td1 < td0
            else f"TD error {td0:.3f} → {td1:.3f}"
        )
        parts.append(f"{side.capitalize()} learned {n1:,} states and its {fall}" if n1 else f"{side}: {fall}")
    if not parts:
        return "No agent statistics were logged."
    eps = lr.stats["epsilon"].dropna()
    tail = f", while exploration decayed to {_pct(float(eps.iloc[-1]))}" if not eps.empty else ""
    return "; ".join(parts) + tail + "."


def mix_takeaway(lr: Learning, side: str) -> str:
    actions, _, z = mix_matrix(lr, side)
    if not actions:
        return f"No action mix was logged for {side}."
    first = {a: z[i][0] for i, a in enumerate(actions)}
    last = {a: z[i][-1] for i, a in enumerate(actions)}
    grew = max(actions, key=lambda a: last[a] - first[a])
    shrank = min(actions, key=lambda a: last[a] - first[a])
    return (f"Learned {side} now spends {_pct(last[grew])} of its moves on {grew.replace('_', ' ')} "
            f"(from {_pct(first[grew])}) and {_pct(last[shrank])} on {shrank.replace('_', ' ')} "
            f"(from {_pct(first[shrank])}).")  # fmt: skip


def probes_takeaway(lr: Learning, side: str) -> str:
    ids = probe_ids(lr, side)
    if not ids:
        return f"No probe situations were logged for {side}."
    rows = [(probe_matrix(lr, side, pid), None) for pid in ids]
    rows = [(pm, probe_change(pm)) for pm, _ in rows]
    changed = [(pm, c) for pm, c in rows if c["changed"]]
    n = len(ids)
    if not changed:
        return (
            f"Learned {side} kept the same preferred action in all {n} probe situations across checkpoints."
        )
    moved = [(pm, c) for pm, c in changed if c["first"] != c["last"]]
    head = f"{len(changed)} of {n} {side} probe situations changed their preferred action at some checkpoint"
    if not moved:
        return head + ", but each came back to its original choice by the end."
    pm, c = max(moved, key=lambda t: t[1]["settled_at"] or 0)
    return (f"{head}; in “{esc(pm['description'])}” it moved from {esc(move_label(c['first']))} to {esc(move_label(c['last']))} "
            f"(settled by game {c['settled_at']:,}).")  # fmt: skip
