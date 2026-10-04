"""Generate the synthetic **v2** fixture run (adaptive detectors + learning telemetry).

    .venv/Scripts/python.exe tests/dashboard/fixtures/make_v2_fixture.py

Writes ``tests/dashboard/fixtures/runs_v2/20261003-101500-7/``. Everything is deterministic (seeded).

This is a *test fixture* for the dashboard, built against docs/contracts.md "Adaptive detectors & learning
telemetry (v2)" before the arena writes real v2 runs. The numbers follow plausible dynamics (detectors that
recover at red's current evasion level after each update, a red bandit that raises evasion when it keeps
getting caught, Q-tables that grow while TD error falls) but they are synthetic: the dashboard never generates
or decorates learning data itself; it only reads files like these.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

RUN_ID = "20261003-101500-7"
OUT = Path(__file__).parent / "runs_v2" / RUN_ID
EPISODES = 1500
EVAL_EVERY = 250
EVAL_N = 40
STATS_EVERY = 50
UPDATE_EVERY = 100
MIN_SAMPLES = 32
LEVELS = [round(0.1 * i, 1) for i in range(8)]  # 0.0 .. 0.7
MODELS = ("malware", "phishing", "network")
CHECKPOINTS = list(range(0, EPISODES + 1, EVAL_EVERY))
MAX_ROUNDS = 36

RED_ACTIONS = ["recon", "phish", "exploit", "escalate", "lateral_move", "exfiltrate", "wait"]
BLUE_ACTIONS = ["monitor", "patch", "isolate", "restore", "reset_credentials", "wait"]
ACTION_MODEL = {"phish": "phishing", "exploit": "network", "lateral_move": "network", "escalate": "malware",
                "exfiltrate": "network", "recon": "network"}  # fmt: skip
FEATURES = {
    "malware": ["entropy", "imports_count", "section_count", "virtual_size", "has_signature"],
    "phishing": ["url_length", "n_subdomains", "has_ip_host", "form_action_ext", "domain_age_days"],
    "network": ["src_bytes", "dst_bytes", "duration", "conn_count", "srv_error_rate"],
}
MITRE = {
    "recon": ("ATT&CK", "T1046", "Network Service Discovery", "Discovery"),
    "phish": ("ATT&CK", "T1566", "Phishing", "Initial Access"),
    "exploit": ("ATT&CK", "T1210", "Exploitation of Remote Services", "Lateral Movement"),
    "escalate": ("ATT&CK", "T1068", "Exploitation for Privilege Escalation", "Privilege Escalation"),
    "lateral_move": ("ATT&CK", "T1021", "Remote Services", "Lateral Movement"),
    "exfiltrate": ("ATT&CK", "T1041", "Exfiltration Over C2 Channel", "Exfiltration"),
    "monitor": ("D3FEND", "D3-NTA", "Network Traffic Analysis", "Detect"),
    "patch": ("D3FEND", "D3-SU", "Software Update", "Harden"),
    "isolate": ("D3FEND", "D3-NI", "Network Isolation", "Isolate"),
    "restore": ("D3FEND", "D3-RA", "Restore Access", "Restore"),
    "reset_credentials": ("D3FEND", "D3-CRO", "Credential Rotation", "Evict"),
}


def r3(x: float) -> float:
    return round(float(x), 3)


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------------------------- graph


def make_graph() -> dict:
    roles = ["dmz"] * 3 + ["workstation"] * 6 + ["server"] * 4 + ["server"]
    xs = {"dmz": 0.08, "workstation": 0.36, "server": 0.65}
    nodes, counts = [], {}
    for i, role in enumerate(roles):
        crown = i == len(roles) - 1
        k = counts.get(role, 0)
        counts[role] = k + 1
        n_role = roles.count(role) - (1 if role == "server" else 0)
        x = 0.92 if crown else xs[role]
        y = 0.5 if crown else (k + 0.5) / n_role
        nodes.append({"id": i, "role": role, "crown_jewel": crown, "x": round(x, 4), "y": round(y, 4)})
    edges = [[0, 1], [1, 2], [0, 3], [0, 4], [1, 5], [1, 6], [2, 7], [2, 8], [3, 4], [5, 6], [7, 8], [4, 5],
             [3, 9], [5, 10], [6, 10], [7, 11], [8, 12], [9, 10], [11, 12], [10, 13], [11, 13], [12, 13]]  # fmt: skip
    return {"nodes": nodes, "edges": edges}


GRAPH = make_graph()
ADJ: dict[int, set[int]] = {n["id"]: set() for n in GRAPH["nodes"]}
for u, v in GRAPH["edges"]:
    ADJ[u].add(v)
    ADJ[v].add(u)
CROWN = 13


def shortest_path(src: int, dst: int) -> list[int]:
    prev, frontier, seen = {}, [src], {src}
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(ADJ[u]):
                if v not in seen:
                    seen.add(v)
                    prev[v] = u
                    nxt.append(v)
        frontier = nxt
    path, cur = [dst], dst
    while cur != src:
        cur = prev[cur]
        path.append(cur)
    return path[::-1]


# ---------------------------------------------------------------------------------------------- detector dynamics


def base_auc(s: float) -> float:
    return 0.955 - 0.52 * s**1.35


class Detector:
    """Synthetic stand-in for AdaptiveDetector.evaluate(): coverage per evasion level grows near trained levels."""

    def __init__(self, model: str, rng: random.Random):
        self.model, self.rng = model, rng
        self.version = 0
        self.cover = {s: 0.0 for s in LEVELS}
        self.drift = 0.0

    def auc(self, s: float) -> float:
        b = base_auc(s)
        return clip(b + (0.95 - b) * 0.88 * self.cover[s] - self.drift * (1 - s), 0.5, 0.99)

    def recall(self, s: float) -> float:
        return clip(1.32 * self.auc(s) - 0.40, 0.02, 0.98)

    def evaluate(self) -> dict:
        return {
            "clean_auc": r3(self.auc(0.0)),
            "auc_by_level": {f"{s:.1f}": r3(self.auc(s)) for s in LEVELS},
            "recall_by_level": {f"{s:.1f}": r3(self.recall(s)) for s in LEVELS},
        }

    def update(self, level: float, n_new: int) -> tuple[dict, dict, float, float]:
        before = self.evaluate()
        loss_before = 0.18 + 0.55 * (1 - self.auc(level)) + self.rng.uniform(-0.02, 0.02)
        rate = min(1.0, n_new / 90)
        for s in LEVELS:
            gain = 0.5 * math.exp(-(((s - level) / 0.15) ** 2)) * rate
            self.cover[s] = self.cover[s] * (0.9 if gain < 0.05 else 1.0) + (1 - self.cover[s]) * gain
        # a little forgetting on clean traffic that replay mostly absorbs
        self.drift = clip(self.drift + self.rng.uniform(-0.004, 0.007), 0.0, 0.025)
        self.version += 1
        after = self.evaluate()
        loss_after = loss_before * self.rng.uniform(0.62, 0.78)
        return before, after, r3(loss_before), r3(loss_after)


# ---------------------------------------------------------------------------------------------- learning.jsonl


def skill(ep: float, side: str) -> float:
    """0..1 competence of the learned agent after ``ep`` training games."""
    k = 1 - math.exp(-ep / (520 if side == "red" else 640))
    return k


def epsilon_at(ep: float) -> float:
    return max(0.02, 1.0 - ep / (0.6 * EPISODES) * 0.98)


def normalise(d: dict) -> dict:
    t = sum(d.values())
    return {k: r3(v / t) for k, v in d.items()}


def red_mix(ep: float) -> dict:
    k = skill(ep, "red")
    target = {"recon": 0.08, "phish": 0.07, "exploit": 0.2, "escalate": 0.24, "lateral_move": 0.26,
              "exfiltrate": 0.1, "wait": 0.05}  # fmt: skip
    return normalise({a: (1 - k) / len(RED_ACTIONS) + k * target[a] for a in RED_ACTIONS})


def blue_mix(ep: float) -> dict:
    k = skill(ep, "blue")
    target = {"monitor": 0.42, "patch": 0.14, "isolate": 0.24, "restore": 0.1, "reset_credentials": 0.07,
              "wait": 0.03}  # fmt: skip
    return normalise({a: (1 - k) / len(BLUE_ACTIONS) + k * target[a] for a in BLUE_ACTIONS})


PROBES = {
    "red": [
        ("r1", "Foothold on a workstation, no privilege yet, nothing flagged", ["escalate", "lateral_move", "recon", "wait"], "escalate"),
        ("r2", "Admin on a workstation next to a server, network sensor hot", ["lateral_move", "exploit", "wait", "recon"], "wait"),
        ("r3", "Admin on the crown jewel", ["exfiltrate", "escalate", "wait", "lateral_move"], "exfiltrate"),
        ("r4", "Only foothold was just detected", ["lateral_move", "escalate", "wait", "phish"], "lateral_move"),
        ("r5", "No foothold, two DMZ hosts patched", ["phish", "exploit", "recon", "wait"], "phish"),
        ("r6", "Two footholds, one on a server, quiet sensors", ["exploit", "lateral_move", "escalate", "exfiltrate"], "exploit"),
    ],
    "blue": [
        ("b1", "One host above threshold, unconfirmed", ["monitor", "isolate", "patch", "wait"], "monitor"),
        ("b2", "Confirmed foothold on a workstation", ["isolate", "reset_credentials", "monitor", "restore"], "isolate"),
        ("b3", "Two footholds, one detected, server hot", ["isolate", "monitor", "patch", "restore"], "isolate"),
        ("b4", "Quiet network, two unpatched servers", ["patch", "monitor", "wait", "restore"], "patch"),
        ("b5", "Isolated host is clean again", ["restore", "monitor", "wait", "patch"], "restore"),
        ("b6", "Crown jewel score rising slowly", ["monitor", "isolate", "patch", "reset_credentials"], "monitor"),
    ],
}  # fmt: skip


def probe_q(rng: random.Random, ep: int, side: str, actions: list[str], best: str) -> dict:
    k = skill(ep, side)
    noise = 0.08 * (1 - k) + 0.01
    q = {}
    for i, a in enumerate(actions):
        learned = (0.55 if a == best else 0.32 - 0.07 * i) * k
        q[a] = r3(learned + rng.gauss(0, noise) + 0.02)
    return q


def make_learning(rng: random.Random) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    dets = {m: Detector(m, random.Random(11 + MODELS.index(m))) for m in MODELS}
    evasion = {m: 0.0 for m in MODELS}
    buffers = {m: 0 for m in MODELS}
    reveal_rate = {"malware": 0.42, "phishing": 0.22, "network": 0.62}  # revealed rows per game, per model
    history = {"versions": {}, "evasion": {}}
    for ep in range(STATS_EVERY, EPISODES + 1, STATS_EVERY):
        kb = skill(ep, "blue")
        for m in MODELS:
            buffers[m] += int(STATS_EVERY * reveal_rate[m] * (0.45 + 0.55 * kb) * rng.uniform(0.8, 1.2))
        if ep % UPDATE_EVERY == 0:
            for m in MODELS:
                if buffers[m] < MIN_SAMPLES:
                    continue
                n_new = min(buffers[m], 140)
                before, after, lb, la = dets[m].update(evasion[m], n_new)
                rows.append({"kind": "detector_update", "episode": ep, "model": m, "version": dets[m].version,
                             "n_new": n_new, "n_replay": n_new, "loss_before": lb, "loss_after": la,
                             "seconds": r3(rng.uniform(0.35, 0.8)), "before": before, "after": after})  # fmt: skip
                buffers[m] = 0
        # red bandit on outcomes: caught -> more evasive; too slow -> back off (evasion costs success)
        for m in MODELS:
            s = evasion[m]
            caught = dets[m].recall(s)
            if caught > 0.66 and s < 0.7 and rng.random() < 0.55:
                s += 0.1  # caught too often: blend further toward benign
            elif (caught < 0.5 or s >= 0.5) and s > 0.1 and rng.random() < 0.3:
                s -= 0.1  # evasion costs success, so back off when it isn't needed
            evasion[m] = round(clip(s, 0.0, 0.7), 1)
        rows.append({"kind": "red_evasion", "episode": ep, "levels": dict(evasion)})
        for side in ("red", "blue"):
            k = skill(ep, side)
            n_states = (
                int((340 if side == "red" else 160) * (1 - math.exp(-ep / 420)) + rng.uniform(-3, 3)) + 8
            )
            rows.append({"kind": "agent_stats", "episode": ep, "side": side, "n_states": n_states,
                         "mean_abs_q": r3(0.05 + 0.3 * k + rng.uniform(-0.01, 0.01)),
                         "td_error": r3(0.012 + 0.11 * math.exp(-ep / 380) * rng.uniform(0.85, 1.15)),
                         "epsilon": r3(epsilon_at(ep)),
                         "action_mix": red_mix(ep) if side == "red" else blue_mix(ep)})  # fmt: skip
        if ep % EVAL_EVERY == 0 or ep == STATS_EVERY:
            history["versions"][ep] = {m: dets[m].version for m in MODELS}
            history["evasion"][ep] = dict(evasion)
        history.setdefault("recall", {})[ep] = {m: dets[m].recall(evasion[m]) for m in MODELS}
    for ckpt in CHECKPOINTS:
        for side, probes in PROBES.items():
            for pid, desc, actions, best in probes:
                q = probe_q(rng, ckpt, side, actions, best)
                rows.append({"kind": "probe", "after_episode": ckpt, "side": side, "probe_id": pid,
                             "description": desc, "q": q, "chosen": max(q, key=q.get)})  # fmt: skip
    rows.sort(key=lambda r: (r.get("episode", r.get("after_episode")), r["kind"] != "detector_update"))
    history["versions"][0] = {m: 0 for m in MODELS}
    history["evasion"][0] = {m: 0.0 for m in MODELS}
    return rows, history


def state_at(history: dict, key: str, ep: int) -> dict:
    eps = [e for e in history[key] if e <= ep]
    return history[key][max(eps)] if eps else history[key][min(history[key])]


# ---------------------------------------------------------------------------------------------- game simulator


def blank_states(rng: random.Random) -> list[dict]:
    return [{"id": n["id"], "compromised": False, "detected": False, "isolated": False, "patched": False,
             "privilege": 0, "scores": {m: round(rng.uniform(0.0, 0.06), 2) for m in MODELS}}
            for n in GRAPH["nodes"]]  # fmt: skip


def simulate(
    episode: int, after: int | None, matchup: str, phase: str, seed: int, history: dict, probe: bool = False
) -> list[dict]:
    """One game as turn records. Learned sides play with checkpoint-dependent competence."""
    rng = random.Random(seed)
    ref = after if after is not None else episode
    versions = state_at(history, "versions", ref)
    evas = state_at(history, "evasion", ref)
    red_agent = (
        "learned" if matchup in ("red_learned_vs_blue_baseline", "learned_vs_learned") else "heuristic"
    )
    blue_agent = (
        "learned" if matchup in ("blue_learned_vs_red_baseline", "learned_vs_learned") else "heuristic"
    )
    kr = skill(ref, "red") if red_agent == "learned" else 0.55
    kb = skill(ref, "blue") if blue_agent == "learned" else 0.5
    det_recall = {m: clip(1.32 * (0.955 - 0.52 * evas[m] ** 1.35) - 0.4 + 0.25 * min(1, versions[m] / 6))
                  for m in MODELS}  # fmt: skip
    states = blank_states(rng)
    footholds: list[int] = []
    turns: list[dict] = []
    winner = None
    t = 0
    for rnd in range(MAX_ROUNDS):
        for actor in ("red", "blue"):
            eps_now = 0.0 if phase == "eval" else epsilon_at(episode)
            explored = rng.random() < eps_now * 0.5
            rec = {"run_id": RUN_ID, "episode": episode, "turn": t, "actor": actor, "phase": phase,
                   "matchup": matchup, "agent": red_agent if actor == "red" else blue_agent,
                   "epsilon": r3(eps_now), "explored": explored, "detector_versions": dict(versions),
                   "classifier_inputs": [], "source": None, "target": None, "success": False,
                   "reward": -0.005}  # fmt: skip
            if after is not None:
                rec["after_episode"] = after
            if probe:
                rec["probe_game"] = True
            if actor == "red":
                action, src, tgt, ok = red_move(rng, states, footholds, kr, evas, explored)
            else:
                action, src, tgt, ok = blue_move(rng, states, footholds, kb, explored)
            rec.update({"action_id": action, "source": src, "target": tgt, "success": ok})
            actions = RED_ACTIONS if actor == "red" else BLUE_ACTIONS
            k = kr if actor == "red" else kb
            dv = {a: r3(rng.uniform(-0.05, 0.15) * (1 - k) + 0.1 * k * rng.random()) for a in actions}
            dv[action] = r3(max(dv.values()) + (0.02 if not explored else -0.03) + 0.2 * k)
            rec["decision_values"] = dv
            if actor == "red" and action in ACTION_MODEL and tgt is not None:
                m = ACTION_MODEL[action]
                s = evas[m]
                loud = {"recon": 0.3, "phish": 0.8, "exploit": 0.7, "escalate": 0.6, "lateral_move": 0.45,
                        "exfiltrate": 0.9}[action]  # fmt: skip
                score = clip(loud * det_recall[m] + rng.uniform(-0.12, 0.12), 0.01, 0.99)
                states[tgt]["scores"][m] = round(max(states[tgt]["scores"][m], score), 2)
                row = [r3(rng.gauss(0.6 * (1 - s), 0.4)) for _ in FEATURES[m]]
                rec["classifier_inputs"] = [{"node": tgt, "model": m, "row": row, "score": r3(score),
                                             "evasion": s, "version": versions[m]}]  # fmt: skip
                if ok and score > 0.5 and states[tgt]["compromised"] and rng.random() < 0.35 + 0.4 * kb:
                    states[tgt]["detected"] = True
            if actor == "red" and action == "exfiltrate" and ok:
                winner = "red"
                rec["reward"] = 1.0
            rec["node_states"] = [json.loads(json.dumps(s)) for s in states]
            rec["done"] = winner is not None
            rec["winner"] = winner
            turns.append(rec)
            t += 1
            if winner:
                return turns
    turns[-1]["done"] = True
    turns[-1]["winner"] = winner = "blue"
    return turns


def red_move(rng, states, footholds, k, evas, explored):
    live = [f for f in footholds if not states[f]["isolated"] and states[f]["compromised"]]
    footholds[:] = live

    def p_cost(m):  # evasion lowers success: p *= 1 - evasion_cost * s
        return 1 - 0.35 * evas[m]

    best = min(live, key=lambda f: len(shortest_path(f, CROWN))) if live else None
    if explored or rng.random() > 0.35 + 0.6 * k:
        action = rng.choice(["recon", "wait", "phish", "escalate"])
    elif best is None:
        action = "phish"
    elif best == CROWN:
        action = "exfiltrate" if states[best]["privilege"] >= 2 else "escalate"
    elif states[best]["privilege"] < 1:
        action = "escalate"
    else:
        nxt = shortest_path(best, CROWN)[1]
        action = (
            "exploit" if GRAPH["nodes"][nxt]["role"] == "server" and rng.random() < 0.5 else "lateral_move"
        )
    if action == "phish":
        tgt = rng.choice([3, 4, 5, 6, 7, 8])
        ok = not live and not states[tgt]["isolated"] and rng.random() < 0.5 * p_cost("phishing")
        if ok:
            states[tgt]["compromised"] = True
            footholds.append(tgt)
        return "phish", None, tgt, ok
    if best is None:
        return (
            ("recon", None, rng.choice([0, 1, 2]), True) if action == "recon" else ("wait", None, None, False)
        )
    src = best
    if action in ("lateral_move", "exploit"):
        tgt = shortest_path(src, CROWN)[1]
        if states[tgt]["isolated"]:
            return "wait", None, None, False
        p = (
            (0.8 if action == "lateral_move" else 0.6)
            * (0.5 if states[tgt]["patched"] else 1)
            * p_cost("network")
        )
        ok = rng.random() < p and states[src]["privilege"] >= 1
        if ok:
            states[tgt]["compromised"] = True
            states[tgt]["privilege"] = max(states[tgt]["privilege"], 1 if tgt == CROWN else 0)
            if tgt not in footholds:
                footholds.append(tgt)
        return action, src, tgt, ok
    if action == "escalate":
        ok = rng.random() < 0.7 * p_cost("malware")
        if ok:
            states[src]["privilege"] = min(2, states[src]["privilege"] + 1)
        return "escalate", src, src, ok
    if action == "exfiltrate":
        ok = rng.random() < 0.75 * p_cost("network")
        return "exfiltrate", src, src, ok
    if action == "recon":
        return "recon", src, rng.choice(sorted(ADJ[src])), True
    return "wait", None, None, False


def blue_move(rng, states, footholds, k, explored):
    detected = [s["id"] for s in states if s["detected"] and s["compromised"] and not s["isolated"]]
    hot = sorted((s for s in states if not s["isolated"]), key=lambda s: -max(s["scores"].values()))
    if explored or rng.random() > 0.3 + 0.65 * k:
        a = rng.choice(["monitor", "patch", "wait"])
        tgt = rng.randrange(len(states)) if a != "wait" else None
    elif detected:
        a, tgt = "isolate", detected[0]
    elif hot and max(hot[0]["scores"].values()) > 0.5:
        a, tgt = "monitor", hot[0]["id"]
    else:
        unpatched = [s["id"] for s in states if not s["patched"] and s["id"] >= 9]
        a, tgt = (
            ("patch", rng.choice(unpatched))
            if unpatched and rng.random() < 0.5
            else ("monitor", hot[0]["id"])
        )
    s = states[tgt] if tgt is not None else None
    ok = False
    if a == "monitor" and s is not None:
        ok = s["compromised"] and max(s["scores"].values()) > 0.45
        if ok:
            s["detected"] = True
    elif a == "isolate" and s is not None:
        ok = s["compromised"]
        s["isolated"] = True
        s["compromised"] = False if ok else s["compromised"]
        if tgt in footholds:
            footholds.remove(tgt)
    elif a == "patch" and s is not None:
        ok = not s["patched"]
        s["patched"] = True
    return a, None, tgt, ok


# ---------------------------------------------------------------------------------------------- enrichment


def enrich(turns: list[dict], rng: random.Random) -> list[dict]:
    out = []
    caught: dict[str, int] = {}
    for t in turns:
        t = json.loads(json.dumps(t))
        a = t["action_id"]
        shap = []
        for ci in t["classifier_inputs"]:
            names = FEATURES[ci["model"]]
            feats = [{"name": n, "value": v, "shap": r3(rng.gauss(0, 0.08)), "raw": r3(abs(v) * 100)}
                     for n, v in zip(names, ci["row"], strict=True)]  # fmt: skip
            feats.sort(key=lambda f: -abs(f["shap"]))
            base = 0.31
            shap.append({"node": ci["node"], "model": ci["model"], "base_value": base, "output": ci["score"],
                         "additivity_error": 0.0, "top_features": feats[:4]})  # fmt: skip
        m = MITRE.get(a)
        t["mitre"] = (
            None if m is None or (t["actor"] == "red" and a == "wait")
            else {"framework": m[0], "version": "v16" if m[0] == "ATT&CK" else "1.0", "technique_id": m[1],
                  "technique_name": m[2], "tactic": m[3]}
        )  # fmt: skip
        t["shap"] = shap or []
        tgt = t.get("target")
        if t["actor"] == "red":
            line = f"Red chooses {a.replace('_', ' ')}" + (f" on host {tgt}" if tgt is not None else "") + "."
            for ci in t["classifier_inputs"]:
                if ci["score"] > 0.5:
                    caught[ci["model"]] = caught.get(ci["model"], 0) + 1
                line += (f" Red's {ci['model']} evasion is {ci['evasion']:.1f}; detector v{ci['version']} "
                         f"scored the activity {ci['score']:.2f}"
                         + (", so blue is likely to notice." if ci["score"] > 0.5 else ", which stays under the threshold."))  # fmt: skip
        else:
            line = (
                f"Blue chooses {a.replace('_', ' ')}" + (f" on host {tgt}" if tgt is not None else "") + "."
            )
        t["rationale"] = line
        t["rationale_meta"] = {"source": "template", "model": None, "input_tokens": 0, "output_tokens": 0}
        out.append(t)
    return out


# ---------------------------------------------------------------------------------------------- main


def main() -> None:
    rng = random.Random(7)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "agents" / "checkpoints").mkdir(parents=True)
    (OUT / "detectors").mkdir()
    learning, history = make_learning(rng)

    summary = []
    for ep in range(EPISODES):
        h2h = 0.25 + 0.2 * skill(ep, "red") - 0.1 * skill(ep, "blue")
        winner = "red" if rng.random() < h2h else "blue"
        turns = int(rng.uniform(18, 40) if winner == "red" else rng.uniform(30, 72))
        summary.append({"kind": "episode", "episode": ep, "winner": winner, "turns": turns,
                        "red_return": 1.0 if winner == "red" else -1.0,
                        "blue_return": 1.0 if winner == "blue" else -1.0,
                        "epsilon": r3(epsilon_at(ep)), "matchup": "learned_vs_learned",
                        "logged": ep == 700})  # fmt: skip
    games: list[list[dict]] = []
    for ep in (700,):
        games.append(simulate(ep, None, "learned_vs_learned", "train", 1000 + ep, history))
    nxt = EPISODES
    for ckpt in CHECKPOINTS:
        for side, matchup in (
            ("red", "red_learned_vs_blue_baseline"),
            ("blue", "blue_learned_vs_red_baseline"),
        ):
            first = nxt
            k = skill(ckpt, side)
            rate = (0.08 + 0.5 * k) if side == "red" else (0.24 + 0.42 * k)
            wins = round(rate * EVAL_N + rng.uniform(-2, 2))
            red_rate = wins / EVAL_N if side == "red" else 1 - wins / EVAL_N
            summary.append({"kind": "eval", "after_episode": ckpt, "matchup": matchup, "n": EVAL_N,
                            "red_win_rate": r3(red_rate), "blue_win_rate": r3(1 - red_rate),
                            "episodes": [first, first + EVAL_N - 1], "logged_n": 2})  # fmt: skip
            # probe game: the same env seed at every checkpoint
            games.append(simulate(first, ckpt, matchup, "eval", 105 if side == "red" else 106, history,
                                  probe=True))  # fmt: skip
            if ckpt == CHECKPOINTS[-1]:
                games.append(simulate(first + 1, ckpt, matchup, "eval", 9000 + first, history))
            nxt += EVAL_N
    summary.sort(key=lambda r: (r.get("episode", r.get("after_episode", 0)), r["kind"]))

    final_red = [r for r in summary if r["kind"] == "eval"][-2]
    final_blue = [r for r in summary if r["kind"] == "eval"][-1]
    config = {"run_id": RUN_ID, "seed": 7, "episodes": EPISODES, "label": "synthetic v2 fixture",
              "init_from": None, "adaptive": True, "n_nodes": len(GRAPH["nodes"]), "crown_jewel": CROWN,
              "params": {"episodes": EPISODES, "eval_every": EVAL_EVERY, "eval_n": EVAL_N, "adaptive": True,
                         "detector_lr": 1e-4, "detector_update_every": UPDATE_EVERY,
                         "detector_min_samples": MIN_SAMPLES, "evasion_cost": 0.35, "red_evasion_lr": 0.1,
                         "stats_every": STATS_EVERY},
              "env": {"max_rounds": MAX_ROUNDS, "detect_threshold": 0.5}}  # fmt: skip
    progress = {"status": "done", "phase": "done", "episode": EPISODES, "episodes": EPISODES,
                "started": "2026-10-03T10:15:00", "updated": "2026-10-03T10:21:42",
                "last_eval": {"after_episode": EPISODES, "red_win_rate": final_red["red_win_rate"],
                              "blue_win_rate": final_blue["blue_win_rate"]}, "error": None}  # fmt: skip

    def dump(name, obj):
        (OUT / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")

    def dump_lines(name, rows):
        with open(OUT / name, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    dump("graph.json", GRAPH)
    dump("config.json", config)
    dump("progress.json", progress)
    dump_lines("summary.jsonl", summary)
    dump_lines("learning.jsonl", learning)
    dump_lines("episodes.jsonl", [t for g in games for t in g])
    to_enrich = [g for g in games if g[0].get("probe_game") and g[0]["after_episode"] in (0, EPISODES)]
    to_enrich += [g for g in games if not g[0].get("probe_game") and g[0]["phase"] == "eval"]
    dump_lines("episodes_enriched.jsonl", [t for g in to_enrich for t in enrich(g, rng)])
    for ckpt in CHECKPOINTS:
        for side in ("red", "blue"):
            st = [
                r
                for r in learning
                if r["kind"] == "agent_stats" and r["side"] == side and r["episode"] <= max(ckpt, 50)
            ]
            n = st[-1]["n_states"] if st else 8
            q = {f"s{i}": [0.0] for i in range(n)}
            dump(f"agents/checkpoints/{side}_{ckpt:05d}.json", {"side": side, "kind": "tabular_q", "q": q})
    for side in ("red", "blue"):
        shutil.copy(
            OUT / "agents" / "checkpoints" / f"{side}_{EPISODES:05d}.json", OUT / "agents" / f"{side}.json"
        )
    for r in learning:
        if r["kind"] == "detector_update":
            (OUT / "detectors" / f"{r['model']}_v{r['version']}.keras").write_bytes(b"")
    n_up = sum(r["kind"] == "detector_update" for r in learning)
    print(f"wrote {OUT} ({len(games)} games, {n_up} detector updates)")


if __name__ == "__main__":
    main()
