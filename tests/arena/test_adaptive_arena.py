"""v2 arena: adaptive detectors, red evasion, learning.jsonl, probes (docs/contracts.md, "Adaptive detectors & ...").

Everything runs on stub classifiers and the numpy ``StubAdaptiveDetector`` (no TensorFlow).
"""

import itertools
import json

import numpy as np
import pytest

from cyberarena.arena.actions import ACTION_IDS, BlueAction, RedAction
from cyberarena.arena.adaptation import DetectorTrainer, EvasionBandit
from cyberarena.arena.env import EVASION_LEVELS, MODELS, CyberArenaEnv
from cyberarena.arena.params import spec
from cyberarena.arena.probes import PROBES
from cyberarena.arena.stub import StubAdaptiveDetector, stub_classifiers, stub_detectors
from cyberarena.arena.train import build_parser, main

TINY = ["--episodes", "24", "--seed", "4", "--eval-every", "12", "--eval-n", "3", "--eval-log-n", "1",
        "--log-every", "6", "--detector-update-every", "8", "--detector-min-samples", "8", "--stats-every", "6",
        "--adaptive-pool-size", "64", "--stub", "--quiet"]  # fmt: skip


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return main([*TINY, "--runs-dir", str(tmp_path_factory.mktemp("runs"))])


def make_env(**kw):
    return CyberArenaEnv(graph_seed=3, detectors=stub_detectors(), adaptive_pool_size=64, **kw)


# ------------------------------------------------------------------------------------------ learning.jsonl


def test_learning_jsonl_kinds_and_schema(run):
    rows = read_jsonl(run / "learning.jsonl")
    kinds = {r["kind"] for r in rows}
    assert kinds == {"detector_update", "red_evasion", "agent_stats", "probe"}

    for r in (r for r in rows if r["kind"] == "detector_update"):
        assert {"episode", "model", "version", "n_new", "n_replay", "loss_before", "loss_after", "before",
                "after"} <= set(r)  # fmt: skip
        assert r["model"] in MODELS and r["version"] >= 1 and r["episode"] % 8 == 0
        for k in ("before", "after"):
            assert set(r[k]["auc_by_level"]) == {f"{s:.1f}" for s in EVASION_LEVELS}
            assert "clean_auc" in r[k] and "recall_by_level" in r[k]
        assert (run / "detectors" / f"{r['model']}_v{r['version']}.keras").exists()

    ev = [r for r in rows if r["kind"] == "red_evasion"]
    assert ev[0]["episode"] == 0 and [r["episode"] for r in ev[1:]] == [6, 12, 18, 24]
    for r in ev:
        assert set(r["levels"]) == set(MODELS) and all(v in EVASION_LEVELS for v in r["levels"].values())

    st = [r for r in rows if r["kind"] == "agent_stats"]
    assert [(r["episode"], r["side"]) for r in st[:2]] == [(6, "red"), (6, "blue")]
    for r in st:
        assert {"n_states", "mean_abs_q", "td_error", "epsilon", "action_mix"} <= set(r)
        assert set(r["action_mix"]) <= set(ACTION_IDS[r["side"]])
        if r["action_mix"]:
            assert sum(r["action_mix"].values()) == pytest.approx(1.0, abs=1e-3)

    pr = [r for r in rows if r["kind"] == "probe"]
    assert {r["after_episode"] for r in pr} == {0, 12, 24}
    for side in ("red", "blue"):
        ids = [r["probe_id"] for r in pr if r["side"] == side and r["after_episode"] == 0]
        assert 6 <= len(ids) <= 10 and len(set(ids)) == len(ids)
    for r in pr:
        assert r["description"] and r["chosen"] in r["q"]
        assert {k.split("→")[0] for k in r["q"]} <= set(ACTION_IDS[r["side"]])  # v4: "<action>→<target>" keys


def test_turn_record_v2_fields(run):
    recs = read_jsonl(run / "episodes.jsonl")
    assert recs
    for r in recs:
        assert set(r["detector_versions"]) == set(MODELS)
        assert isinstance(r["probe_game"], bool)
        for c in r["classifier_inputs"]:
            assert c["evasion"] in EVASION_LEVELS and c["version"] == r["detector_versions"][c["model"]]
    assert max(max(r["detector_versions"].values()) for r in recs) >= 1


def test_config_records_adaptation(run):
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["adaptive"] is True
    assert cfg["params"]["detector_update_every"] == 8 and cfg["params"]["evasion_cost"] == 0.25
    assert cfg["adaptation"]["final"]["detector_versions"]
    assert (run / "agents" / "red_evasion.json").exists()


def test_probe_game_logged_every_checkpoint(run):
    recs = read_jsonl(run / "episodes.jsonl")
    evals = [r for r in read_jsonl(run / "summary.jsonl") if r["kind"] == "eval"]
    probe_eps = {r["episode"] for r in recs if r["probe_game"]}
    assert probe_eps == {r["probe_episode"] for r in evals}
    for r in recs:
        if r["probe_game"]:
            assert r["phase"] == "eval" and "after_episode" in r


# ------------------------------------------------------------------------------------------ env mechanics


def test_no_labels_without_blue_action():
    env = make_env()
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    while not env.done:
        if env.actor == "red":
            a = int(rng.choice(env.valid_actions("red")))
            env.step(a)
        else:  # blue only watches / hardens: no isolate, restore or reset, and no host is ever confirmed
            ok = [a for a in env.valid_actions("blue") if a in (BlueAction.WAIT, BlueAction.PATCH,
                                                                 BlueAction.MONITOR)]  # fmt: skip
            env.step(int(rng.choice(ok)))
        assert env.revealed == []


@pytest.mark.parametrize("forensics", [False, True])
def test_isolate_reveals_that_hosts_rows(forensics):
    from cyberarena.arena.env import ArenaConfig

    env = make_env(config=ArenaConfig(phish_forensics=forensics))
    env.reset(seed=1)
    start = int(env.footholds()[0])
    # v4 phish_forensics: the phishing campaign behind red's beachhead (start host + other recipients)
    lure = [m for box in env.mailbox for m in box]
    assert len(env.mailbox[start]) == int(forensics) and all(m[:2] == ("phishing", 1) for m in lure)
    assert len(lure) == (env.cfg.phish_campaign_size if forensics else 0)
    env.step((int(RedAction.ESCALATE), start, start))
    trail = list(env.trail[start])
    assert any(t[1] == 1 for t in trail) or trail  # baseline rows + the escalate row
    env.detected[start] = True
    env.step((int(BlueAction.ISOLATE), None, start))
    # proving the phished beachhead sweeps every mailbox for the campaign
    assert env.revealed[:len(trail)] == trail and len(env.trail[start]) == 0
    assert sorted(env.revealed[len(trail):]) == sorted(m for m in lure if m not in trail)
    assert env.confirmed[start]


def test_monitor_reveals_only_on_confirmed_host():
    env = make_env()
    env.reset(seed=2)
    env.step(int(RedAction.WAIT))
    t = int(np.flatnonzero(~env.compromised)[0])
    env.step((int(BlueAction.MONITOR), None, t))
    assert env.revealed == []
    env.reset(seed=2)
    env.confirmed[t] = True
    env.step(int(RedAction.WAIT))
    env.detected[t] = False
    env.step((int(BlueAction.MONITOR), None, t))
    assert env.revealed and {r[0] for r in env.revealed} == set(MODELS)


def _exploit_success_rate(level_idx: int, cost: float, n: int = 400) -> float:
    env = make_env(evasion_cost=cost)
    wins = 0
    for i in range(n):
        env.reset(seed=i)
        env.evasion_idx[:] = level_idx
        dmz = [c for c in env.candidates("red", int(RedAction.EXPLOIT)) if c[0] is None]
        _, _, _, _, info = env.step((int(RedAction.EXPLOIT), None, dmz[0][1]))
        wins += info["success"]
    return wins / n


def test_evasion_lowers_success():
    loud = _exploit_success_rate(0, cost=0.8)
    quiet = _exploit_success_rate(len(EVASION_LEVELS) - 1, cost=0.8)
    assert loud == pytest.approx(0.45, abs=0.07)
    assert quiet == pytest.approx(0.45 * (1 - 0.8 * 0.7), abs=0.07)
    assert _exploit_success_rate(len(EVASION_LEVELS) - 1, cost=0.0) == loud  # no cost -> same odds, same draws


def test_evasive_rows_score_lower_and_are_logged():
    env = make_env()
    means = [env.pool.scores[("network", 1, k)].mean() for k in range(len(EVASION_LEVELS))]
    assert all(a > b for a, b in itertools.pairwise(means))
    env.reset(seed=0)
    env.evasion_idx[:] = 5
    s = int(env.footholds()[0])
    _, _, _, _, info = env.step((int(RedAction.ESCALATE), s, s))
    for c in info["classifier_inputs"]:
        assert c["evasion"] == (EVASION_LEVELS[5] if c["label"] == 1 else 0.0)


def test_pools_rescored_after_update(tmp_path):
    dets = stub_detectors()
    env = CyberArenaEnv(graph_seed=3, detectors=dets, adaptive_pool_size=64)
    trainer = DetectorTrainer(dets, env, tmp_path, min_samples=8)
    before = {k: v.copy() for k, v in env.pool.scores.items()}
    rows_before = {k: v.copy() for k, v in env.pool.rows.items()}
    n = 64
    trainer.add_revealed([("network", 1, 6, i) for i in range(n)] + [("network", 0, 0, i) for i in range(n)])
    out = trainer.maybe_update(100, {m: 0.6 for m in MODELS})
    assert [r["model"] for r in out] == ["network"] and out[0]["version"] == 1
    assert env.pool.version == {"malware": 0, "phishing": 0, "network": 1}
    for k in env.pool.scores:
        assert np.array_equal(env.pool.rows[k], rows_before[k])  # rows fixed, only scores refreshed
        if k[0] == "network":
            assert not np.allclose(env.pool.scores[k], before[k])
            np.testing.assert_allclose(env.pool.scores[k], dets["network"].predict_proba(env.pool.rows[k]), rtol=1e-6)
        else:
            assert np.array_equal(env.pool.scores[k], before[k])
    # the update learned the s=0.6 rows: they now score higher than before
    assert env.pool.scores[("network", 1, 6)].mean() > before[("network", 1, 6)].mean()
    assert (tmp_path / "network_v1.keras").exists()
    assert trainer.counts("network") == (0, 0)  # buffer consumed


def test_min_samples_gates_update(tmp_path):
    dets = stub_detectors()
    env = CyberArenaEnv(graph_seed=3, detectors=dets, adaptive_pool_size=64)
    trainer = DetectorTrainer(dets, env, tmp_path, min_samples=32)
    trainer.add_revealed([("malware", 0, 0, i) for i in range(40)])  # only benign rows
    assert trainer.maybe_update(10, {m: 0.0 for m in MODELS}) == []
    assert dets["malware"].n_updates_called == 0


def test_bandit_only_moves_on_outcomes():
    b = EvasionBandit(seed=0, lr=0.5, explore=0.0, detect_penalty=0.1)
    assert b.levels() == {m: 0.0 for m in MODELS}
    quiet = {m: {"n_act": 0, "n_leak": 0, "progress": 0.0, "caught": 0} for m in MODELS}
    b.update(b.choose(), quiet)  # red never touched a sensor -> nothing learned
    assert b.levels() == {m: 0.0 for m in MODELS}
    caught = {**quiet, "network": {"n_act": 4, "n_leak": 0, "progress": 0.05, "caught": 4}}
    b.update(b.choose(), caught)
    assert b.levels()["network"] > 0.0 and b.levels()["malware"] == 0.0


# ------------------------------------------------------------------------------------------ run-level


def _strip(rows):
    out = []
    for r in rows:
        r = dict(r)
        r.pop("run_id", None)
        r.pop("seconds", None)
        out.append(r)
    return out


def test_determinism(tmp_path):
    a = main([*TINY, "--runs-dir", str(tmp_path / "a")])
    b = main([*TINY, "--runs-dir", str(tmp_path / "b")])
    for f in ("summary.jsonl", "learning.jsonl", "episodes.jsonl"):
        assert _strip(read_jsonl(a / f)) == _strip(read_jsonl(b / f)), f


def _probe_games(run):
    games: dict[tuple[str, int], list] = {}
    for r in read_jsonl(run / "episodes.jsonl"):
        if r["probe_game"]:
            key = r["matchup"] + (f"@{r['evasion']}" if "evasion" in r else "")
            games.setdefault((key, r["after_episode"]), []).append(r)
    return games


def _same_until_choices_differ(g1, g2, env_keys):
    """Turn by turn the env outcome matches as long as every choice so far matched; returns #turns compared."""
    n = 0
    for r1, r2 in zip(g1, g2, strict=False):
        assert (r1["turn"], r1["actor"]) == (r2["turn"], r2["actor"])
        if (r1["action_id"], r1["source"], r1["target"]) != (r2["action_id"], r2["source"], r2["target"]):
            break
        for k in env_keys:
            assert r1[k] == r2[k], (k, r1["turn"])
        n += 1
    return n


def test_probe_game_identical_except_agent_choices(tmp_path):
    run = main([*TINY, "--runs-dir", str(tmp_path), "--no-adaptive"])
    games = _probe_games(run)
    assert len(games) == 12  # (2 matchups + 2 disguised-attacker levels) x 3 checkpoints
    compared = 0
    for matchup in ("red_learned_vs_blue_baseline", "blue_learned_vs_red_baseline"):
        ckpts = sorted(k for k in games if k[0] == matchup)
        first = games[ckpts[0]]
        assert first[0]["episode"] != games[ckpts[1]][0]["episode"]
        for k in ckpts[1:]:
            compared += _same_until_choices_differ(first, games[k], ("success", "node_states", "classifier_inputs",
                                                                     "reward", "done", "winner"))  # fmt: skip
    # the scripted red's opening (fixed RNG) is the same at every checkpoint, so there is always a shared prefix
    assert compared >= 2


def test_probe_game_adaptive_same_rows_new_scores(run):
    """Adaptive: same seed and same emitted rows while choices match; only detector scores may differ."""
    games = _probe_games(run)
    for matchup in ("red_learned_vs_blue_baseline", "blue_learned_vs_red_baseline"):
        ckpts = sorted(k for k in games if k[0] == matchup)
        assert len(ckpts) == 3
        # the starting foothold is the same; red's first move may already have added one (v4 DQN red exploits)
        start = []
        for k in ckpts:
            r = games[k][0]
            comp = [n["compromised"] for n in r["node_states"]]
            if r["actor"] == "red" and r["success"] and r["action_id"] in ("exploit", "phish", "lateral_move"):
                comp[r["target"]] = False
            start.append(comp)
        assert all(s == start[0] for s in start)
        g0, g1 = games[ckpts[0]], games[ckpts[1]]
        r0, r1 = g0[0], g1[0]
        if (r0["action_id"], r0["target"], r0["red_evasion"]) == (r1["action_id"], r1["target"], r1["red_evasion"]):
            assert [c["row"] for c in r0["classifier_inputs"]] == [c["row"] for c in r1["classifier_inputs"]]


def test_no_adaptive_has_no_v2_learning(tmp_path):
    run = main([*TINY, "--runs-dir", str(tmp_path), "--no-adaptive"])
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["adaptive"] is False and "adaptation" not in cfg
    assert not (run / "detectors").exists()
    kinds = {r["kind"] for r in read_jsonl(run / "learning.jsonl")}
    assert kinds == {"agent_stats", "probe"}
    for r in read_jsonl(run / "episodes.jsonl"):
        assert set(r["detector_versions"].values()) == {0}
        if r["matchup"] != "blue_learned_vs_red_evasive":  # disguised-attacker probe games use fixed levels
            assert all(c["evasion"] == 0.0 for c in r["classifier_inputs"])


def test_warm_start_restores_detectors_and_evasion(run, tmp_path):
    src_cfg = json.loads((run / "config.json").read_text())
    final = src_cfg["adaptation"]["final"]
    new = main([*TINY, "--runs-dir", str(tmp_path), "--init-from", str(run), "--eps-start", "0.3"])
    cfg = json.loads((new / "config.json").read_text())
    restored = {m: v for m, v in final["detector_versions"].items() if v > 0}
    assert cfg["init_from"]["detectors"] == restored
    assert cfg["init_from"]["red_evasion"] == final["red_evasion"]
    for m, v in restored.items():
        assert (new / "detectors" / f"{m}_v{v}.keras").exists()  # copied so turn records resolve
    first_eval = next(r for r in read_jsonl(new / "episodes.jsonl") if r["phase"] == "eval")
    assert all(first_eval["detector_versions"][m] == v for m, v in restored.items())
    ev0 = next(r for r in read_jsonl(new / "learning.jsonl") if r["kind"] == "red_evasion")
    assert ev0["levels"] == final["red_evasion"]


def test_warm_start_blue_only_skips_red_evasion(run, tmp_path):
    new = main([*TINY, "--runs-dir", str(tmp_path), "--init-from", str(run), "--init-side", "blue"])
    cfg = json.loads((new / "config.json").read_text())
    assert cfg["init_from"]["red_evasion"] is None and cfg["init_from"]["detectors"]


def test_warm_start_from_v1_run(tmp_path):
    old = main([*TINY, "--runs-dir", str(tmp_path / "old"), "--no-adaptive"])
    new = main([*TINY, "--runs-dir", str(tmp_path / "new"), "--init-from", str(old)])
    cfg = json.loads((new / "config.json").read_text())
    assert cfg["init_from"]["detectors"] == {} and cfg["init_from"]["red_evasion"] is None


# ------------------------------------------------------------------------------------------ params


def test_adaptation_param_group():
    groups = {g["id"]: g for g in spec()["groups"]}
    keys = {p["key"]: p for p in groups["adaptation"]["params"]}
    assert {"adaptive", "detector_lr", "detector_update_every", "detector_min_samples", "evasion_cost",
            "red_evasion_lr"} <= set(keys)  # fmt: skip
    assert keys["adaptive"]["type"] == "bool" and keys["adaptive"]["default"] is True
    assert keys["adaptive"]["flag"] == "--adaptive" and keys["adaptive"]["flag_false"] == "--no-adaptive"
    assert keys["detector_update_every"]["default"] == 100 and keys["detector_min_samples"]["default"] == 32
    assert all(p["help"] for p in keys.values())
    assert build_parser().parse_args(["--no-adaptive"]).adaptive is False
    assert build_parser().parse_args(["--adaptive"]).adaptive is True


def test_probe_states_are_valid_feature_tuples():
    for side, probes in PROBES.items():
        assert 6 <= len(probes) <= 10
        n = 7 if side == "red" else 5
        for pid, desc, state, actions in probes:
            assert len(state) == n and desc and set(actions) <= set(ACTION_IDS[side]), pid


def test_stub_detector_roundtrip(tmp_path):
    d = StubAdaptiveDetector("network")
    X = np.concatenate([d.evasive_rows(1, 0.5, 32, 0), d.sample(0, 32, 1)])
    d.update(X, np.r_[np.ones(32), np.zeros(32)])
    d.save(tmp_path / "network_v1.keras")
    e = StubAdaptiveDetector.load(tmp_path / "network_v1.keras")
    assert e.version == 1
    np.testing.assert_allclose(e.predict_proba(X), d.predict_proba(X))
    assert set(stub_classifiers()) == set(MODELS)
