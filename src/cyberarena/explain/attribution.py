"""Integrated gradients on the v4 DQN agents' Q-networks: why this move rather than a typical alternative.

For a turn played by a DQN agent (``candidates`` and ``chosen_features`` logged), the chosen move's feature vector
``x`` is attributed against a **baseline** ``b`` = the mean feature vector of the turn's logged candidate set (the
top-12 moves by Q). Attributions sum to ``Q(x) - Q(b)`` (completeness); the error of the Riemann sum is stored.

**Checkpoint.** The Q-network that made the decision: ``agents/checkpoints/{side}_qnet_<after>.keras`` with the
largest ``after`` not later than the turn's training point (eval turns: ``after_episode``; training turns: the
episode number), or the final ``agents/{side}_qnet.keras`` when the point is at/after the end of training and no
checkpoint sits exactly there. Each model file is loaded once.

**Baseline reconstruction.** Turn records hold only the chosen move's features, so the other candidates' vectors are
rebuilt from the pre-move state (the previous turn record's ``node_states``) plus ``graph.json``:
- global ``g_*`` features are shared by every candidate of a turn, so they equal the chosen move's;
- the action one-hot, ``t_none``, role one-hot, hops / degree, isolated / patched / compromised / privilege /
  detected flags, the three detector scores and their max, ``t_closer``, ``t_nbr_detected`` and the source-host
  ``s_*`` terms are recomputed exactly (checked against the chosen move: ``baseline_info.check_error``);
- a candidate on the same target (source) as the chosen move copies its target (source) block exactly;
- what the log cannot reveal (red ``t_recon``, ``t_open_nbrs``, ``s_recon``, ``p_success`` of a different move;
  blue ``t_confirmed``) is imputed with the chosen move's value, which gives that column zero attribution. Imputed
  columns are listed in ``baseline_info.imputed``.
If the reconstruction check fails (``check_error`` > 0.02), the baseline falls back to all zeros.

The path integral is exact for the ReLU MLP (split at every ReLU switch along the path), with a 32/64-step
midpoint Riemann sum as the fallback for other activations. Gradients come from a float64 NumPy mirror of the Dense/ReLU MLP (validated against Keras at load, else
``tf.GradientTape``). Results are cached per (run, checkpoint, feature-vector hash) in
``explain_cache/agent_ig.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cyberarena.explain.shap_values import ShapCache

IG_VERSION = 1
STEP_SCHEDULE = (32, 64, 128, 256)  # 32-64 normally; more only if completeness is still off by > 1%
COMPLETENESS_TOL = 0.01  # relative to |Q(x) - Q(b)|
COMPLETENESS_ABS = 1e-4  # ... or absolute, when Q(x) ~ Q(b)
CHECK_TOL = 0.02
TOP_K = 8
MODELS = ("malware", "phishing", "network")
STATIC_COLUMNS = frozenset({"t_none", "t_dmz", "t_workstation", "t_server", "t_crown", "t_dist", "t_degree",
                            "s_has", "s_dist"})  # topology-only columns: valid to check on any state
_CKPT_RE = re.compile(r"^(red|blue)_qnet_(\d+)\.keras$")


# ------------------------------------------------------------------------------------------- checkpoints


def training_point(turn: dict) -> int:
    """Episode count the deciding network had been trained for (eval: ``after_episode``; train: ``episode``)."""
    if turn.get("phase") == "eval" and turn.get("after_episode") is not None:
        return int(turn["after_episode"])
    return int(turn["episode"])


def list_checkpoints(agents_dir: Path, side: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    ck = Path(agents_dir) / "checkpoints"
    if ck.is_dir():
        for p in ck.iterdir():
            m = _CKPT_RE.match(p.name)
            if m and m.group(1) == side:
                out[int(m.group(2))] = p
    return out


def select_checkpoint(agents_dir: Path, side: str, point: int, total_episodes: int | None = None
                      ) -> tuple[Path, int | None] | None:  # fmt: skip
    """(model path, checkpoint ``after`` or None for the final model), closest to and not after ``point``."""
    ckpts = list_checkpoints(agents_dir, side)
    final = Path(agents_dir) / f"{side}_qnet.keras"
    if point in ckpts:
        return ckpts[point], point
    if final.exists() and (total_episodes is not None and point >= total_episodes or not ckpts):
        return final, None
    eligible = [a for a in ckpts if a <= point]
    if eligible:
        a = max(eligible)
        return ckpts[a], a
    return None


# ------------------------------------------------------------------------------------------------ Q-net


class QNet:
    """Scalar Q(x) and dQ/dx for a saved Keras Q-network (NumPy mirror of Dense layers, tf fallback)."""

    _ACT = frozenset({"relu", "linear", "tanh", "sigmoid"})

    def __init__(self, model, layers: list[tuple[np.ndarray, np.ndarray, str]] | None = None):
        self.model = model
        self.layers = layers
        self.n_in = int(layers[0][0].shape[0]) if layers else int(model.inputs[0].shape[-1])

    @classmethod
    def load(cls, path: Path) -> QNet:
        import keras

        model = keras.models.load_model(path, compile=False)
        return cls.from_keras(model)

    @classmethod
    def from_keras(cls, model) -> QNet:
        layers = []
        for layer in model.layers:
            if type(layer).__name__ != "Dense":
                layers = None
                break
            act = layer.get_config().get("activation")
            if act not in cls._ACT:
                layers = None
                break
            w, b = layer.get_weights()
            layers.append((w.astype(np.float64), b.astype(np.float64), act))
        net = cls(model, layers)
        if layers is not None:  # validate the mirror
            x = np.random.default_rng(0).uniform(0, 1, size=(16, net.n_in))
            ref = np.asarray(model(x.astype(np.float32)), dtype=np.float64).reshape(-1)
            if np.max(np.abs(ref - net.q(x))) > 1e-4:
                net.layers = None
        return net

    @staticmethod
    def _act(z: np.ndarray, act: str) -> tuple[np.ndarray, np.ndarray]:
        if act == "relu":
            return np.maximum(z, 0.0), (z > 0).astype(np.float64)
        if act == "tanh":
            a = np.tanh(z)
            return a, 1.0 - a * a
        if act == "sigmoid":
            a = 1.0 / (1.0 + np.exp(-z))
            return a, a * (1.0 - a)
        return z, np.ones_like(z)

    def q(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.layers is None:
            return np.asarray(self.model(X.astype(np.float32)), dtype=np.float64).reshape(-1)
        h = X
        for w, b, act in self.layers:
            h, _ = self._act(h @ w + b, act)
        return h.reshape(-1)

    def grad(self, X: np.ndarray) -> np.ndarray:
        """dQ/dx for each row of X, shape (n, d)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.layers is None:
            import tensorflow as tf

            xt = tf.constant(X.astype(np.float32))
            with tf.GradientTape() as tape:
                tape.watch(xt)
                y = tf.reduce_sum(self.model(xt))
            return np.asarray(tape.gradient(y, xt), dtype=np.float64)
        h, derivs = X, []
        for w, b, act in self.layers:
            h, d = self._act(h @ w + b, act)
            derivs.append(d)
        g = np.ones((X.shape[0], 1))
        for (w, _, _), d in zip(reversed(self.layers), reversed(derivs), strict=True):
            g = (g * d) @ w.T
        return g


def path_breakpoints(net: QNet, x: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Every alpha in [0, 1] where a ReLU unit switches along b + alpha (x - b), or None if the net isn't
    piecewise linear (non-ReLU hidden activations / no NumPy mirror). Between breakpoints the gradient is constant."""
    if net.layers is None or any(act not in ("relu", "linear") for _, _, act in net.layers):
        return None
    delta = x - b
    bps = np.array([0.0, 1.0])
    for li in range(len(net.layers) - 1):
        h = b[None, :] + bps[:, None] * delta[None, :]
        for w, bias, act in net.layers[:li]:
            h, _ = QNet._act(h @ w + bias, act)
        w, bias, _ = net.layers[li]
        z = h @ w + bias  # (n_bps, units), linear in alpha inside each segment
        z0, z1 = z[:-1], z[1:]
        cross = (z0 * z1) < 0
        seg, unit = np.nonzero(cross)
        a0, a1 = bps[seg], bps[seg + 1]
        frac = z0[seg, unit] / (z0[seg, unit] - z1[seg, unit])
        bps = np.unique(np.concatenate([bps, a0 + (a1 - a0) * frac]))
    return bps


def integrated_gradients(net: QNet, x: np.ndarray, baseline: np.ndarray,
                         schedule: tuple[int, ...] = STEP_SCHEDULE, exact: bool = True) -> dict:  # fmt: skip
    """IG along the straight path baseline -> x.

    For a ReLU MLP the path integral is computed exactly: the path is split at every ReLU switch and the (constant)
    gradient of each linear piece is weighted by its length (``integration = "exact_piecewise_linear"``).
    Otherwise (or ``exact=False``) a midpoint Riemann sum with 32, then 64 (then up to 256) steps is refined until
    the completeness error is within 1% of |Q(x) - Q(b)| (``integration = "riemann_midpoint"``).
    """
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    qx, qb = net.q(np.stack([x, b]))
    delta = x - b
    target = float(qx - qb)

    def finish(attr: np.ndarray, **kw) -> dict:
        err = abs(float(attr.sum()) - target)
        return {"attr": attr, "abs": err, "rel": err / max(abs(target), 1e-9), **kw}

    bps = path_breakpoints(net, x, b) if exact else None
    if bps is not None:
        lengths = np.diff(bps)
        mids = (bps[:-1] + bps[1:]) / 2
        g = net.grad(b[None, :] + mids[:, None] * delta[None, :])
        res = finish((lengths[:, None] * g).sum(axis=0) * delta, steps=None, segments=len(lengths),
                     integration="exact_piecewise_linear")  # fmt: skip
    else:
        res = None
        for m in schedule:
            alphas = (np.arange(m) + 0.5) / m
            g = net.grad(b[None, :] + alphas[:, None] * delta[None, :])
            res = finish(g.mean(axis=0) * delta, steps=m, segments=None, integration="riemann_midpoint")
            if res["rel"] <= COMPLETENESS_TOL or res["abs"] <= COMPLETENESS_ABS:
                break
    return {"q": float(qx), "q_baseline": float(qb), **res}


# --------------------------------------------------------------------------------------- graph & baseline


@dataclass
class GraphInfo:
    roles: dict[int, str]
    crown: int | None
    adj: dict[int, set[int]]
    dist: dict[int, int]
    max_dist: int
    max_degree: int

    @classmethod
    def from_graph(cls, graph: dict) -> GraphInfo:
        roles = {n["id"]: n["role"] for n in graph["nodes"]}
        crown = next((n["id"] for n in graph["nodes"] if n.get("crown_jewel")), None)
        adj: dict[int, set[int]] = {i: set() for i in roles}
        for u, v in graph.get("edges", []):
            adj[u].add(v)
            adj[v].add(u)
        dist: dict[int, int] = {}
        if crown is not None:
            dist[crown] = 0
            q = deque([crown])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
        max_dist = max(dist.values(), default=1) or 1
        for i in roles:  # unreachable hosts sit at the cap
            dist.setdefault(i, max_dist)
        max_degree = max((len(a) for a in adj.values()), default=1) or 1
        return cls(roles, crown, adj, dist, max_dist, max_degree)


@dataclass
class Reconstructor:
    """Rebuilds candidate feature vectors for one turn (see module docstring)."""

    names: tuple[str, ...]
    side: str
    g: GraphInfo
    imputed: set = field(default_factory=set)

    def _target_block(self, t: int | None, state: dict[int, dict]) -> dict[str, float] | None:
        """Recomputable t_* values; None entries mean "unknown from the log"."""
        out: dict[str, float | None] = {n: 0.0 for n in self.names if n.startswith("t_")}
        if t is None:
            out["t_none"] = 1.0
            return out
        ns = state.get(t)
        if ns is None:
            return None
        role = self.g.roles.get(t)
        crown = t == self.g.crown  # the crown jewel's role one-hot is t_crown alone
        out.update({"t_dmz": float(role == "dmz" and not crown), "t_workstation": float(role == "workstation" and not crown),
                    "t_server": float(role == "server" and not crown), "t_crown": float(crown)})  # fmt: skip
        out["t_dist"] = min(self.g.dist[t], self.g.max_dist) / self.g.max_dist
        out["t_degree"] = len(self.g.adj[t]) / self.g.max_degree
        out["t_isolated"] = float(ns["isolated"])
        out["t_patched"] = float(ns["patched"])
        scores = ns.get("scores") or {}
        for m in MODELS:
            out[f"t_score_{m}"] = float(scores.get(m, 0.0))
        out["t_score_max"] = max((float(scores.get(m, 0.0)) for m in MODELS), default=0.0)
        if self.side == "red":
            out["t_compromised"] = float(ns["compromised"])
            out["t_privilege"] = ns["privilege"] / 2.0
            fh = [k for k, v in state.items() if v["compromised"] and not v["isolated"]]
            front = min(self.g.dist[k] for k in fh) if fh else self.g.max_dist
            out["t_closer"] = float(self.g.dist[t] < front)
            out["t_recon"] = None
            out["t_open_nbrs"] = None
        else:
            out["t_detected"] = float(ns["detected"])
            nd = sum(bool(state[k]["detected"] and not state[k]["isolated"]) for k in self.g.adj[t] if k in state)
            out["t_nbr_detected"] = min(nd, 4) / 4.0
            out["t_confirmed"] = None
        return {k: v for k, v in out.items() if k in self.names}

    def _source_block(self, s: int | None, state: dict[int, dict]) -> dict[str, float] | None:
        out: dict[str, float | None] = {n: 0.0 for n in self.names if n.startswith("s_")}
        if not out or s is None:
            return out
        ns = state.get(s)
        if ns is None:
            return None
        out["s_has"] = 1.0
        out["s_privilege"] = ns["privilege"] / 2.0
        out["s_score_max"] = max((float((ns.get("scores") or {}).get(m, 0.0)) for m in MODELS), default=0.0)
        out["s_dist"] = min(self.g.dist[s], self.g.max_dist) / self.g.max_dist
        out["s_recon"] = None
        return {k: v for k, v in out.items() if k in self.names}

    def vector(self, cand: dict, chosen: dict, chosen_move: tuple, state: dict[int, dict],
               shortcut: bool = True) -> tuple[dict[str, float], list[str]]:  # fmt: skip
        """(feature dict, names of recomputed columns). ``shortcut=False`` recomputes even same-host blocks."""
        vec = dict(chosen)
        recomputed: list[str] = []
        a, s, t = cand.get("action"), cand.get("source"), cand.get("target")
        for n in self.names:
            if n.startswith("a_"):
                vec[n] = float(n == f"a_{a}")
        same_t = shortcut and t == chosen_move[2]
        same_s = shortcut and s == chosen_move[1]
        for same, block in ((same_t, None if same_t else self._target_block(t, state)),
                            (same_s, None if same_s else self._source_block(s, state))):  # fmt: skip
            if same:
                continue
            if block is None:  # host missing from the state: keep the chosen values
                continue
            for k, v in block.items():
                if v is None:
                    if vec.get(k) is not None and k in chosen:
                        self.imputed.add(k)
                    continue
                vec[k] = v
                recomputed.append(k)
        if "p_success" in self.names and (a, s, t) != chosen_move:
            self.imputed.add("p_success")
        return vec, recomputed


def vec_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.round(np.asarray(a, dtype=np.float64), 5).tobytes())
    return h.hexdigest()[:24]


def _move(c: dict) -> tuple:
    return (c.get("action"), c.get("source"), c.get("target"))


def chosen_and_runner_up(turn: dict) -> tuple[dict | None, dict | None]:
    cands = turn.get("candidates") or []
    me = (turn.get("action_id"), turn.get("source"), turn.get("target"))
    chosen = next((c for c in cands if _move(c) == me), None)
    others = sorted((c for c in cands if _move(c) != me), key=lambda c: -float(c.get("q") or 0.0))
    return chosen, (others[0] if others else None)


# ------------------------------------------------------------------------------------------------ service


@dataclass
class IGStats:
    turns: int = 0
    computed: int = 0
    cache_hits: int = 0
    nulls: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    load_seconds: float = 0.0
    models_loaded: int = 0
    max_completeness_rel: float = 0.0
    max_completeness_abs: float = 0.0
    max_q_mismatch: float = 0.0
    max_check_error: float = 0.0
    steps: dict[str, int] = field(default_factory=dict)
    baselines: dict[str, int] = field(default_factory=dict)

    def null(self, why: str) -> None:
        self.nulls[why] = self.nulls.get(why, 0) + 1


class AttributionService:
    """Per-run IG explainer. ``explain_turn(turn, prev_turn)`` returns ``agent_attribution`` or None."""

    def __init__(self, run_dir: Path, graph: dict, cfg: dict, cache_path: Path | None = None,
                 loader=None, top_k: int = TOP_K):  # fmt: skip
        self.run_dir = Path(run_dir)
        self.agents_dir = self.run_dir / "agents"
        self.cfg = cfg or {}
        self.g = GraphInfo.from_graph(graph)
        self.cache = ShapCache(cache_path)
        self.loader = loader or QNet.load
        self.top_k = top_k
        self.stats = IGStats()
        self._nets: dict[Path, QNet] = {}
        self._names: dict[str, tuple[str, ...] | None] = {}
        agents = self.cfg.get("agents") or {}
        self.kind = agents.get("type") or agents.get("kind")
        self.total_episodes = self.cfg.get("episodes")

    def feature_names(self, side: str) -> tuple[str, ...] | None:
        if side not in self._names:
            names = None
            meta = self.agents_dir / f"{side}_qnet.json"
            if meta.exists():
                names = json.loads(meta.read_text(encoding="utf-8")).get("feature_names")
            if not names:
                names = ((self.cfg.get("agents") or {}).get("feature_names") or {}).get(side)
            self._names[side] = tuple(names) if names else None
        return self._names[side]

    def net(self, path: Path) -> QNet:
        if path not in self._nets:
            t0 = time.perf_counter()
            self._nets[path] = self.loader(path)
            self.stats.load_seconds += time.perf_counter() - t0
            self.stats.models_loaded += 1
        return self._nets[path]

    def _baseline(self, turn: dict, prev: dict | None, names: tuple[str, ...], chosen: dict
                  ) -> tuple[np.ndarray, str, dict]:  # fmt: skip
        cands = turn.get("candidates") or []
        pre_turn = prev if prev is not None and prev.get("episode") == turn.get("episode") else None
        states = (pre_turn or turn).get("node_states") or []
        state = {ns["id"]: ns for ns in states}
        rec = Reconstructor(names, turn["actor"], self.g)
        me = (turn.get("action_id"), turn.get("source"), turn.get("target"))
        # self-check: rebuild the chosen move from the state alone and compare the recomputed columns
        self_vec, cols = rec.vector({"action": me[0], "source": me[1], "target": me[2]}, chosen, me, state,
                                    shortcut=False)  # fmt: skip
        if pre_turn is None:  # first turn: only the post-move state exists, and the move changed its own target
            cols = [c for c in cols if c in STATIC_COLUMNS or not c.startswith(("t_", "s_"))]
        check = max((abs(self_vec[c] - chosen[c]) for c in cols), default=0.0)
        rec.imputed.clear()
        vecs = [rec.vector(c, chosen, me, state)[0] for c in cands]
        info = {"n_candidates": len(cands), "candidate_set": f"top-{len(cands)} logged candidates",
                "pre_state": "previous_turn" if pre_turn is not None else "this_turn_post_state",
                "imputed": sorted(rec.imputed), "check_error": round(float(check), 5)}  # fmt: skip
        self.stats.max_check_error = max(self.stats.max_check_error, float(check))
        if not vecs:
            return np.zeros(len(names)), "zeros", info | {"fallback_reason": "no candidates"}
        if check > CHECK_TOL:
            return np.zeros(len(names)), "zeros", info | {"fallback_reason": f"reconstruction check {check:.3f}"}
        mean = np.mean([[v[n] for n in names] for v in vecs], axis=0)
        return mean, "candidate_mean", info

    def explain_turn(self, turn: dict, prev: dict | None = None) -> dict | None:
        self.stats.turns += 1
        side = turn.get("actor")
        if turn.get("agent", "learned") != "learned" or not turn.get("candidates"):
            self.stats.null("not a DQN decision")
            return None
        feats = turn.get("chosen_features")
        names = self.feature_names(side)
        if not feats or names is None or any(n not in feats for n in names):
            self.stats.null("chosen_features missing")
            return None
        sel = select_checkpoint(self.agents_dir, side, training_point(turn), self.total_episodes)
        if sel is None:
            self.stats.null("no checkpoint")
            return None
        path, after = sel
        x = np.array([float(feats[n]) for n in names])
        baseline, kind, info = self._baseline(turn, prev, names, feats)
        rel_ckpt = path.relative_to(self.run_dir).as_posix()
        key = f"ig{IG_VERSION}|{self.run_dir.name}|{rel_ckpt}|{vec_hash(x, baseline)}"
        res = self.cache.get(key)
        if res is None:
            net = self.net(path)
            t0 = time.perf_counter()
            ig = integrated_gradients(net, x, baseline)
            attr = ig["attr"]
            order = np.argsort(-np.abs(attr), kind="stable")[: self.top_k]
            res = {
                "q": round(ig["q"], 5), "q_baseline": round(ig["q_baseline"], 5),
                "integration": ig["integration"], "steps": ig["steps"], "segments": ig["segments"],
                "completeness_error": round(ig["abs"], 7), "completeness_rel": round(ig["rel"], 6),
                "attribution_sum": round(float(attr.sum()), 6),
                "top_features": [{"name": names[i], "value": round(float(x[i]), 4),
                                  "baseline": round(float(baseline[i]), 4), "attribution": round(float(attr[i]), 5)}
                                 for i in order if attr[i] != 0.0],
            }  # fmt: skip
            self.stats.seconds += time.perf_counter() - t0
            self.stats.computed += 1
            self.cache.put(key, res)
        else:
            self.stats.cache_hits += 1
        s = self.stats
        s.max_completeness_rel = max(s.max_completeness_rel, res["completeness_rel"])
        s.max_completeness_abs = max(s.max_completeness_abs, res["completeness_error"])
        how = res["integration"] if res["steps"] is None else f"riemann_{res['steps']}"
        s.steps[how] = s.steps.get(how, 0) + 1
        s.baselines[kind] = s.baselines.get(kind, 0) + 1
        chosen, runner = chosen_and_runner_up(turn)
        q_logged = None if chosen is None else chosen.get("q")
        if q_logged is not None:
            s.max_q_mismatch = max(s.max_q_mismatch, abs(float(q_logged) - res["q"]))
        out = {"method": "integrated_gradients", "baseline": kind, "baseline_info": info, **res,
               "q_logged": q_logged, "checkpoint": rel_ckpt, "checkpoint_after": after,
               "runner_up": None if runner is None else {k: runner.get(k) for k in ("action", "source", "target", "q")}}  # fmt: skip
        if runner is not None and q_logged is not None:
            out["margin"] = round(float(q_logged) - float(runner["q"]), 5)
        return out
