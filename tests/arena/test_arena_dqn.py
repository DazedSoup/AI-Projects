"""v4 DQN agents (docs/contracts.md, "TensorFlow agents (v4)" + "Disjoint row partitions").

Stub classifiers / stub adaptive detectors throughout; the Q-networks themselves are real Keras models.
"""

import hashlib
import json

import numpy as np
import pytest

from cyberarena.arena.actions import ACTION_IDS
from cyberarena.arena.agents import QAgent, make_baseline
from cyberarena.arena.dqn import ARROW, DQNAgent, DQNConfig, NumpyMLP, parse_hidden
from cyberarena.arena.env import CyberArenaEnv
from cyberarena.arena.features import FEATURE_NAMES, N_FEATURES
from cyberarena.arena.params import spec
from cyberarena.arena.probes import SNAPSHOT_PROBES, build_probe_state
from cyberarena.arena.stub import StubAdaptiveDetector, stub_classifiers, stub_detectors
from cyberarena.arena.train import main, play_episode

TINY = ["--episodes", "16", "--seed", "4", "--eval-every", "8", "--eval-n", "3", "--eval-log-n", "2",
        "--log-every", "4", "--detector-update-every", "8", "--detector-min-samples", "8", "--stats-every", "8",
        "--adaptive-pool-size", "64", "--eval-evasive-n", "3", "--stub", "--quiet",
        "--replay-size", "1000", "--batch-size", "16"]  # fmt: skip
SMALL = DQNConfig(hidden=(16,), replay_size=200, batch_size=8, target_sync=3, train_every=1, learn_start=16)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return main([*TINY, "--runs-dir", str(tmp_path_factory.mktemp("runs"))])


def stub_env(seed=3):
    return CyberArenaEnv(graph_seed=seed, classifiers=stub_classifiers())


def train_some(agent, games=6, seed=0):
    env = stub_env()
    other = make_baseline("blue" if agent.side == "red" else "red", "heuristic", 1, 0.3)
    agents = {agent.side: agent, other.side: other}
    for g in range(games):
        play_episode(env, agents, seed + g, learn={agent.side}, explore=True)
    return env


# ------------------------------------------------------------------------------------------ network


def test_numpy_mirror_matches_keras():
    agent = DQNAgent("red", seed=1, cfg=SMALL)
    X = np.random.default_rng(0).random((50, N_FEATURES["red"]), dtype=np.float32)
    np.testing.assert_allclose(agent.net.online(X), agent.net.keras_q(X), atol=1e-5)
    train_some(agent, games=4)
    assert agent.grad_steps > 0
    np.testing.assert_allclose(agent.net.online(X), agent.net.keras_q(X), atol=1e-5)  # synced after training


def test_mirror_forward_is_relu_mlp():
    rng = np.random.default_rng(1)
    w = [rng.normal(size=(3, 4)), rng.normal(size=4), rng.normal(size=(4, 1)), rng.normal(size=1)]
    x = rng.normal(size=(2, 3))
    want = (np.maximum(x @ w[0] + w[1], 0) @ w[2] + w[3])[:, 0]
    np.testing.assert_allclose(NumpyMLP(w)(x), want, rtol=1e-5)


def test_target_network_syncs_every_target_sync_steps():
    agent = DQNAgent("blue", seed=2, cfg=SMALL)
    env = stub_env()
    agents = {"red": make_baseline("red", "heuristic", 1, 0.3), "blue": agent}
    synced_at = []
    for g in range(8):
        play_episode(env, agents, g, learn={"blue"}, explore=True)
        same = all(np.array_equal(a, b) for a, b in zip(agent.net.target.w, agent.net.online.w, strict=True))
        synced_at.append((agent.grad_steps, same))
    assert agent.grad_steps >= 6
    for steps, same in synced_at:
        if same and steps:
            assert steps % SMALL.target_sync == 0 or steps < SMALL.target_sync
    assert any(not same for _, same in synced_at)  # in between syncs the online net has moved on


def test_replay_ring_buffer_and_n_step_returns():
    cfg = DQNConfig(hidden=(8,), replay_size=4, n_step=2, gamma=0.5, learn_start=10**6)
    agent = DQNAgent("red", seed=0, cfg=cfg)
    nxt = np.ones((3, N_FEATURES["red"]), np.float32)
    from cyberarena.arena.dqn import _Pending

    agent._traj.extend([_Pending(np.full(N_FEATURES["red"], 1, np.float32), 1.0),
                        _Pending(np.full(N_FEATURES["red"], 2, np.float32), 2.0)])  # fmt: skip
    agent._commit(nxt)  # G = 1 + 0.5 * 2, bootstrap gamma^2
    assert agent.buffer.g[0] == pytest.approx(2.0) and agent.buffer.disc[0] == pytest.approx(0.25)
    agent.end_episode()  # terminal: G = 2, no bootstrap
    assert agent.buffer.g[1] == pytest.approx(2.0) and agent.buffer.disc[1] == 0.0 and agent.buffer.nxt[1] is None
    for i in range(5):
        agent.buffer.add(np.zeros(N_FEATURES["red"]), float(i), 0.0, None)
    assert len(agent.buffer) == 4 and agent.buffer.added == 7


def test_double_dqn_targets():
    agent = DQNAgent("red", seed=3, cfg=SMALL)
    train_some(agent, games=3)
    agent.net.target = NumpyMLP([w * 0.5 for w in agent.net.online.w])  # make the target net differ
    idx = np.array([i for i in range(len(agent.buffer)) if agent.buffer.nxt[i] is not None][:5])
    y = agent.targets(idx)
    for j, i in enumerate(idx):
        X = agent.buffer.nxt[i]
        a = int(np.argmax(agent.net.online(X)))  # online net chooses ...
        want = agent.buffer.g[i] + agent.buffer.disc[i] * agent.net.target(X)[a]  # ... target net values
        assert y[j] == pytest.approx(want, rel=1e-5)


def test_parse_hidden():
    assert parse_hidden("64,64") == (64, 64) and parse_hidden("32") == (32,)
    for bad in ("", "0", "64,64,64,64", "x"):
        with pytest.raises(ValueError):
            parse_hidden(bad)


def test_save_load_roundtrip(tmp_path):
    agent = DQNAgent("blue", seed=4, cfg=SMALL)
    train_some(agent, games=3)
    agent.save(tmp_path / "blue_qnet")
    again = DQNAgent.load(tmp_path / "blue_qnet")
    X = np.random.default_rng(1).random((20, N_FEATURES["blue"]), dtype=np.float32)
    np.testing.assert_allclose(again.net.online(X), agent.net.online(X), atol=1e-6)
    meta = DQNAgent.read_meta(tmp_path / "blue_qnet")
    assert meta["feature_names"] == list(FEATURE_NAMES["blue"]) and meta["grad_steps"] == agent.grad_steps
    assert again.cfg.hidden == SMALL.hidden


# ------------------------------------------------------------------------------------------ training run


def test_run_files_and_config(run):
    agents = run / "agents"
    for side in ("red", "blue"):
        assert (agents / f"{side}_qnet.keras").is_file() and (agents / f"{side}_qnet.json").is_file()
        for after in (0, 8, 16):
            assert (agents / "checkpoints" / f"{side}_qnet_{after:05d}.keras").is_file()
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["agents"]["type"] == "dqn" and cfg["agents"]["double_dqn"]
    assert cfg["agents"]["feature_names"] == {s: list(v) for s, v in FEATURE_NAMES.items()}
    assert cfg["params"]["agent"] == "dqn" and cfg["params"]["replay_size"] == 1000


def test_candidate_log_schema(run):
    recs = read_jsonl(run / "episodes.jsonl")
    learned = [r for r in recs if r["agent"] == "learned"]
    assert learned
    for r in recs:
        if r["agent"] != "learned":
            assert r["candidates"] is None and r["chosen_features"] is None
            assert set(r["decision_values"]) <= set(ACTION_IDS[r["actor"]])
    for r in learned:
        dv, cands, feats = r["decision_values"], r["candidates"], r["chosen_features"]
        assert 1 <= len(dv) <= 8 and 1 <= len(cands) <= 12
        for k in dv:
            a, _, t = k.partition(ARROW)
            assert a in ACTION_IDS[r["actor"]] and (t == "" or t.isdigit())
        qs = [c["q"] for c in cands]
        assert qs == sorted(qs, reverse=True)
        assert list(dv.values()) == sorted(dv.values(), reverse=True)
        for c in cands:
            assert set(c) == {"action", "source", "target", "q"} and c["action"] in ACTION_IDS[r["actor"]]
        assert list(feats) == list(FEATURE_NAMES[r["actor"]])
        assert feats[f"a_{r['action_id']}"] == 1.0
        if not r["explored"]:  # greedy: the chosen move is the top candidate
            top = cands[0]
            assert (top["action"], top["source"], top["target"]) == (r["action_id"], r["source"], r["target"])


def test_agent_stats_and_probe_rows(run):
    rows = read_jsonl(run / "learning.jsonl")
    st = [r for r in rows if r["kind"] == "agent_stats"]
    assert st
    for r in st:
        assert {"loss", "q_mean", "replay_size", "grad_steps", "td_error", "action_mix"} <= set(r)
        assert r["replay_size"] >= 0 and r["grad_steps"] >= 0
    assert max(r["grad_steps"] for r in st) > 0
    pr = [r for r in rows if r["kind"] == "probe"]
    assert {r["after_episode"] for r in pr} == {0, 8, 16}
    for side in ("red", "blue"):
        ids = [r["probe_id"] for r in pr if r["side"] == side and r["after_episode"] == 0]
        assert ids == [p for p, _, _ in SNAPSHOT_PROBES[side]]
    for r in pr:
        assert r["chosen"] in r["q"] and len(r["q"]) <= 8 and r["description"]
        assert {k.partition(ARROW)[0] for k in r["q"]} <= set(ACTION_IDS[r["side"]])
        assert {"compromised", "detected", "scores", "turn"} <= set(r["state"])
    # same probe, same state at every checkpoint
    p1 = [r["state"] for r in pr if r["probe_id"] == "b4"]
    assert all(s == p1[0] for s in p1)


def test_probe_states_are_playable():
    env = CyberArenaEnv(graph_seed=11, classifiers=stub_classifiers())
    for side, probes in SNAPSHOT_PROBES.items():
        assert 6 <= len(probes) <= 10
        for pid, desc, _ in probes:
            build_probe_state(env, side, pid)
            assert env.actor == side and desc
            assert env.all_candidates(side)
            assert len(env.footholds()) > 0  # the game is not over in any probe


def test_dqn_run_is_reproducible(run, tmp_path):
    other = main([*TINY, "--runs-dir", str(tmp_path)])
    assert read_jsonl(other / "summary.jsonl") == read_jsonl(run / "summary.jsonl")


def test_warm_start_dqn(run, tmp_path, monkeypatch):
    from cyberarena.arena import train

    saved = {s: DQNAgent.load_weights(run / "agents" / f"{s}_qnet") for s in ("red", "blue")}
    seen = {}
    real_play = train.play_episode

    def spy(env, agents, *a, **k):
        for side, ag in agents.items():
            if isinstance(ag, DQNAgent) and side not in seen:
                seen[side] = [w.copy() for w in ag.net.model.get_weights()]
        return real_play(env, agents, *a, **k)

    monkeypatch.setattr(train, "play_episode", spy)
    new = main([*TINY, "--runs-dir", str(tmp_path), "--init-from", str(run), "--init-side", "red",
                "--n-nodes", "19"])  # fmt: skip
    assert all(np.array_equal(a, b) for a, b in zip(seen["red"], saved["red"], strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(seen["blue"], saved["blue"], strict=True))
    cfg = json.loads((new / "config.json").read_text())
    assert cfg["init_from"]["side"] == "red"


def test_init_from_wrong_agent_type_exits_2(run, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main([*TINY, "--runs-dir", str(tmp_path), "--init-from", str(run), "--agent", "tabular"])
    assert e.value.code == 2 and "--agent dqn" in capsys.readouterr().err


def test_param_spec_has_agent_knobs():
    params = {p["key"]: p for g in spec()["groups"] for p in g["params"]}
    assert params["agent"]["choices"] == ["dqn", "tabular"] and params["agent"]["default"] == "dqn"
    for k in ("dqn_lr", "dqn_hidden", "replay_size", "batch_size", "target_sync", "train_every", "n_step",
              "dqn_gamma"):  # fmt: skip
        assert params[k]["advanced"] and params[k]["help"] and params[k]["target"] == "cli"
    assert "agent" in params and not params["agent"].get("advanced")


def test_bad_hidden_exits_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main([*TINY, "--runs-dir", str(tmp_path), "--dqn-hidden", "64,abc"])
    assert e.value.code == 2


# ------------------------------------------------------------------------------------------ tabular == v3

# Digests of v3 (pre-v4 code) runs with these args: summary rows and every logged move (ignoring the
# disguised-attacker eval, whose pools v4 moved to the arena_eval partition).
GOLDEN_ARGS = ["--episodes", "40", "--seed", "4", "--eval-every", "20", "--eval-n", "4", "--eval-log-n", "2",
               "--log-every", "5", "--detector-update-every", "10", "--detector-min-samples", "8",
               "--stats-every", "10", "--adaptive-pool-size", "64", "--stub", "--quiet", "--eval-evasive-n", "5",
               "--rules", "cheap-isolation", "--agent", "tabular",
               "--env-json", '{"phish_forensics": false}']  # fmt: skip
GOLDEN = {"adaptive": "c634311471ff8627a4c0bd1c0d88cbf3e7c1116c747b318b10f8843c40781f27",
          "frozen": "e608c596ec28f92ea6d7b7a818781be135df957a22bfa92b7ed436e306804753"}


def digest(run_dir):
    h = hashlib.sha256()
    for r in read_jsonl(run_dir / "summary.jsonl"):
        if r.get("matchup") != "blue_learned_vs_red_evasive":
            h.update(json.dumps(r, sort_keys=True).encode())
    for r in read_jsonl(run_dir / "episodes.jsonl"):
        if r.get("matchup") != "blue_learned_vs_red_evasive":
            keep = {k: r[k] for k in ("episode", "turn", "actor", "action_id", "source", "target", "success",
                                      "winner")}  # fmt: skip
            h.update(json.dumps(keep, sort_keys=True).encode())
    return h.hexdigest()


@pytest.mark.parametrize("cond", ["adaptive", "frozen"])
def test_tabular_reproduces_v3(cond, tmp_path):
    extra = ["--no-adaptive"] if cond == "frozen" else []
    run_dir = main([*GOLDEN_ARGS, *extra, "--runs-dir", str(tmp_path)])
    assert digest(run_dir) == GOLDEN[cond]
    assert isinstance(QAgent.load(run_dir / "agents" / "red.json"), QAgent)
    assert json.loads((run_dir / "config.json").read_text())["agents"]["type"] == "tabular"


# ------------------------------------------------------------------------------------------ partitions


def test_pools_use_partitions():
    dets = stub_detectors()
    train_env = CyberArenaEnv(graph_seed=2, detectors=dets, adaptive_pool_size=32)
    assert train_env.pool.partition == "arena_train"
    assert all(set(d.partitions_used) == {"arena_train"} for d in dets.values())
    eval_dets = stub_detectors()
    eval_env = CyberArenaEnv(graph_seed=2, detectors=eval_dets, adaptive_pool_size=32, pool_seed=99,
                             partition="arena_eval")  # fmt: skip
    assert eval_env.pool.partition == "arena_eval"
    assert all(set(d.partitions_used) == {"arena_eval"} for d in eval_dets.values())
    for key, rows in train_env.pool.rows.items():  # disjoint rows
        a = {r.tobytes() for r in rows}
        assert not a & {r.tobytes() for r in eval_env.pool.rows[key]}


class NoPartitionDetector(StubAdaptiveDetector):
    """A v3-style detector without ``pool_rows``: the arena falls back to ``evasive_rows`` / ``sample``."""

    pool_rows = None

    def __getattribute__(self, name):
        if name in ("pool_rows", "partition_sizes"):
            raise AttributeError(name)
        return super().__getattribute__(name)


def test_detectors_without_partitions_fall_back():
    dets = {m: NoPartitionDetector(m) for m in ("malware", "phishing", "network")}
    env = CyberArenaEnv(graph_seed=2, detectors=dets, adaptive_pool_size=16)
    assert env.pool.partition is None and env.pool.size("network", 1, 3) == 16


def test_run_records_partitions(run):
    cfg = json.loads((run / "config.json").read_text())
    assert set(cfg["partition_sizes"]) == {"malware", "phishing", "network"}
    assert set(cfg["partition_sizes"]["network"]) == {"metric", "arena_train", "arena_eval"}
    assert cfg["pools"]["train_partition"] == "arena_train" and cfg["pools"]["eval_partition"] == "arena_eval"


# ------------------------------------------------------------------------------------------ phishing forensics


def test_campaign_sweep_reveals_all_lures():
    from cyberarena.arena.actions import BlueAction, RedAction
    from cyberarena.arena.env import ArenaConfig

    env = CyberArenaEnv(ArenaConfig(phish_campaign_size=3), graph_seed=3, detectors=stub_detectors(),
                        adaptive_pool_size=64)  # fmt: skip
    env.reset(seed=5)
    start = int(env.footholds()[0])
    lures = [m for box in env.mailbox for m in box]
    assert len(lures) == 3 and sum(len(b) > 0 for b in env.mailbox) == 3 and env.mailbox[start]
    env.step(int(RedAction.RECON))
    env.detected[start] = True
    env.step((int(BlueAction.ISOLATE), None, start))  # proves the phished beachhead -> campaign sweep
    assert all(m in env.revealed for m in lures)
    assert not any(env.mailbox)


def test_forensics_never_changes_game_play(tmp_path):
    """Frozen detectors: phishing forensics only adds labels, so the games are move-for-move identical."""
    base = [*GOLDEN_ARGS[:-2], "--no-adaptive", "--eval-evasion-levels", ""]
    on = main([*base, "--runs-dir", str(tmp_path / "on")])
    off = main([*base, "--env-json", '{"phish_forensics": false}', "--runs-dir", str(tmp_path / "off")])
    assert digest(on) == digest(off)
