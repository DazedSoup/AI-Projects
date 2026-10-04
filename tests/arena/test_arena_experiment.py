"""Multi-seed experiments (docs/contracts.md, "Experiments & rigor (v3)" -> "Multi-seed experiments").

Aggregate maths against hand-computed values, the manifest lifecycle with a fake train command, a real
``--stub`` experiment, ``--aggregate`` recomputation and cancelling.
"""

import json
import math
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cyberarena.arena.experiment import (
    aggregate,
    count_cycles,
    describe,
    learned_win_rate,
    main,
    metric_key,
    paired_contrast,
    parse_seeds,
)

WIN = sys.platform == "win32"
STUB_TRAIN = ["--stub", "--quiet", "--eval-every", "6", "--eval-n", "3", "--eval-log-n", "0", "--eval-evasive-n", "4",
              "--log-every", "50", "--detector-update-every", "4", "--detector-min-samples", "8",
              "--stats-every", "3", "--adaptive-pool-size", "64"]  # fmt: skip


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ statistics


def test_describe_t_interval_hand_computed():
    d = describe([0.6, 0.7, 0.8])
    # mean 0.7, sd 0.1 (ddof 1), t(0.975, df=2) = 4.302653 -> half width 4.302653 * 0.1 / sqrt(3) = 0.248414
    assert d["mean"] == pytest.approx(0.7) and d["sd"] == pytest.approx(0.1) and d["n"] == 3
    assert d["ci95"] == pytest.approx([0.7 - 0.248414, 0.7 + 0.248414], abs=1e-4)
    assert d["per_seed"] == [0.6, 0.7, 0.8]
    one = describe([0.5])
    assert one["mean"] == 0.5 and one["sd"] is None and one["ci95"] is None


def test_paired_contrast_hand_computed():
    c = paired_contrast([0.5, 0.6, 0.9], [0.4, 0.4, 0.6])
    # d = [0.1, 0.2, 0.3]: mean 0.2, sd 0.1, se 0.1/sqrt(3) = 0.057735, t = 3.464102 (df 2)
    # df=2 has a closed form: two-sided p = 1 - t / sqrt(t^2 + 2) = 1 - 3.464102 / sqrt(14) = 0.074180
    assert c["diff_mean"] == pytest.approx(0.2) and c["n"] == 3
    assert c["t"] == pytest.approx(3.4641, abs=1e-4)
    assert c["p_value"] == pytest.approx(1 - 3.464102 / math.sqrt(14), abs=1e-5)
    assert c["ci95"] == pytest.approx([0.2 - 0.248414, 0.2 + 0.248414], abs=1e-4)
    # pairing matters: the same numbers unpaired would not give sd 0.1
    assert paired_contrast([0.3, 0.3], [0.1, 0.1])["p_value"] == 0.0  # constant non-zero difference
    assert paired_contrast([0.3], [0.1])["p_value"] is None


def test_count_cycles_definition():
    assert count_cycles([0.0, 0.3, 0.1]) == 1
    assert count_cycles([0.0, 0.1, 0.0, 0.1, 0.0]) == 0  # one-step jitter is not a cycle
    assert count_cycles([0.0, 0.4, 0.1, 0.5, 0.2]) == 2
    assert count_cycles([0.0, 0.2, 0.4, 0.6]) == 0  # still rising: not a completed cycle
    assert count_cycles([0.7, 0.5, 0.3, 0.6, 0.4]) == 1  # starts high: the first fall is not a cycle
    assert count_cycles([]) == 0


def test_metric_keys_and_learned_side():
    assert metric_key({"matchup": "blue_learned_vs_red_evasive", "evasion": 0.7}) == "blue_learned_vs_red_evasive@0.7"
    assert metric_key({"matchup": "blue_learned_vs_red_baseline"}) == "blue_learned_vs_red_baseline"
    assert learned_win_rate({"matchup": "red_learned_vs_blue_baseline", "red_win_rate": 0.6, "blue_win_rate": 0.4}) == 0.6
    assert parse_seeds("1-3,7") == [1, 2, 3, 7]
    with pytest.raises(ValueError):
        parse_seeds("3-1")


# ------------------------------------------------------------------------------------------ fake train

FAKE_TRAIN = textwrap.dedent('''
    """Fake cyberarena.arena.train: writes a run dir with synthetic eval rows; seed 2 crashes."""
    import argparse, json, sys, time
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int); p.add_argument("--episodes", type=int); p.add_argument("--label")
    p.add_argument("--runs-dir"); p.add_argument("--no-adaptive", action="store_true")
    args, extra = p.parse_known_args()
    cond = "frozen" if args.no_adaptive else "adaptive"
    run = Path(args.runs_dir) / f"{cond}-{args.seed}"
    run.mkdir(parents=True)
    print(f"RUN_DIR {run.resolve()}", flush=True)
    man = Path(args.runs_dir) / "experiments" / "fake" / "manifest.json"
    seen = json.loads(man.read_text(encoding="utf-8")) if man.exists() else None
    (run / "argv.json").write_text(json.dumps({"argv": sys.argv[1:], "label": args.label, "extra": extra,
                                               "manifest_runs": len(seen["runs"]) if seen else None}))
    if args.seed == 2 and cond == "frozen":
        print("boom: simulated crash", flush=True)
        sys.exit(3)
    bonus = 0.2 if cond == "adaptive" else 0.0
    rows = []
    for a in (0, 50, 100):
        # learned blue vs evasive red @0.7: frozen 0.3 + 0.01*seed, adaptive adds 0.2 after episode 0
        v = 0.3 + 0.01 * args.seed + (bonus if a else 0.0)
        rows.append({"kind": "eval", "after_episode": a, "matchup": "blue_learned_vs_red_evasive", "evasion": 0.7,
                     "n": 10, "red_win_rate": round(1 - v, 4), "blue_win_rate": round(v, 4)})
        rows.append({"kind": "eval", "after_episode": a, "matchup": "red_learned_vs_blue_baseline", "n": 10,
                     "red_win_rate": 0.5, "blue_win_rate": 0.5})
    with open(run / "summary.jsonl", "w") as f:
        f.write("\\n".join(json.dumps(r) for r in rows) + "\\n")
    lv = [0.0, 0.3, 0.1, 0.4, 0.1] if cond == "adaptive" else []
    with open(run / "learning.jsonl", "w") as f:
        for i, x in enumerate(lv):
            f.write(json.dumps({"kind": "red_evasion", "episode": i * 10, "levels": {"network": x}}) + "\\n")
        if cond == "adaptive":
            for ep, r0, r1 in ((20, 0.1, 0.2), (40, 0.3, 0.6)):
                f.write(json.dumps({"kind": "detector_update", "episode": ep, "model": "network", "version": ep // 20,
                                    "before": {"recall_by_level": {"0.7": r0}},
                                    "after": {"recall_by_level": {"0.7": r1}}}) + "\\n")
    time.sleep(0.3)
''')


@pytest.fixture(scope="module")
def fake_exp(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp")
    script = base / "fake_train.py"
    script.write_text(FAKE_TRAIN, encoding="utf-8")
    rc = main(["--name", "fake", "--seeds", "1-3", "--conditions", "adaptive,frozen", "--episodes", "100",
               "--jobs", "3", "--runs-dir", str(base), "--train-cmd", json.dumps([sys.executable, str(script)]),
               "--quiet", "--", "--eval-n", "9"])  # fmt: skip
    return base / "experiments" / "fake", rc


def test_manifest_lifecycle(fake_exp):
    exp, rc = fake_exp
    assert rc == 1  # one run failed, the others finished
    man = read(exp / "manifest.json")
    assert {"name", "created", "episodes", "seeds", "conditions", "extra_args", "runs"} <= set(man)
    assert man["name"] == "fake" and man["seeds"] == [1, 2, 3] and man["conditions"] == ["adaptive", "frozen"]
    assert man["extra_args"] == ["--eval-n", "9"] and man["episodes"] == 100 and man["status"] == "done"
    by = {(r["seed"], r["condition"]): r for r in man["runs"]}
    assert len(by) == 6
    for (seed, cond), r in by.items():
        assert {"seed", "condition", "run_dir", "status"} <= set(r)
        assert r["run_dir"] and r["run_dir"].endswith(f"{cond}-{seed}")
        argv = read(exp.parent.parent / f"{cond}-{seed}" / "argv.json")
        assert argv["label"] == f"fake · {cond} · seed {seed}"
        assert argv["extra"] == ["--eval-n", "9"]
        assert argv["manifest_runs"] == 6  # manifest written before any run started
        assert ("--no-adaptive" in argv["argv"]) == (cond == "frozen")
        if (seed, cond) == (2, "frozen"):
            assert r["status"] == "error" and r["returncode"] == 3 and "boom" in r["error"]
        else:
            assert r["status"] == "done" and r["returncode"] == 0
    assert (exp / "logs" / "frozen_seed2.log").read_text().strip().endswith("boom: simulated crash")


def test_aggregate_values(fake_exp):
    exp, _ = fake_exp
    agg = read(exp / "aggregate.json")
    assert agg["n_seeds"] == 3 and agg["runs_used"] == {"adaptive": [1, 2, 3], "frozen": [1, 3]}
    assert agg["runs_failed"] == [{"seed": 2, "condition": "frozen", "status": "error", "error": "boom: simulated crash"}]
    k = "blue_learned_vs_red_evasive@0.7"
    curve = agg["curves"]["adaptive"][k]
    assert [c["after_episode"] for c in curve] == [0, 50, 100]
    assert curve[1]["per_seed"] == [0.51, 0.52, 0.53] and curve[1]["mean"] == pytest.approx(0.52)
    assert curve[1]["sd"] == pytest.approx(0.01)
    assert agg["final"]["frozen"][k]["per_seed"] == [0.31, 0.33]
    # late_mean: checkpoints >= 50 -> adaptive seed s = 0.5 + 0.01 s
    assert agg["late_mean"]["adaptive"][k]["per_seed"] == [0.51, 0.52, 0.53]
    assert agg["late_mean"]["adaptive"]["red_learned_vs_blue_baseline"]["mean"] == 0.5
    c = next(c for c in agg["contrasts"] if c["metric"] == f"{k} late_mean")
    assert agg["contrasts"][0] is not None and agg["contrasts"][0]["metric"] == f"{k} late_mean"
    assert c["a"] == "adaptive" and c["b"] == "frozen" and c["paired_by_seed"] is True
    assert c["seeds"] == [1, 3] and c["diff_mean"] == pytest.approx(0.2) and c["p_value"] == 0.0
    arms = agg["arms_race"]["network"]
    assert arms["cycles_mean"] == 2  # 0 -> 0.3 -> 0.1 -> 0.4 -> 0.1
    assert arms["recall_at_0.7_first"] == 0.1 and arms["recall_at_0.7_last"] == 0.6


def test_aggregate_recomputes_from_runs(fake_exp):
    exp, _ = fake_exp
    before = read(exp / "aggregate.json")
    (exp / "aggregate.json").unlink()
    assert main(["--aggregate", str(exp), "--quiet"]) == 0
    after = read(exp / "aggregate.json")
    before.pop("generated"), after.pop("generated")
    assert after == before
    assert aggregate(exp)["contrasts"] == before["contrasts"]


def test_existing_experiment_name_refused(fake_exp, capsys):
    exp, _ = fake_exp
    rc = main(["--name", "fake", "--seeds", "1", "--runs-dir", str(exp.parent.parent), "--quiet"])
    assert rc == 2 and "already exists" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------ real train (stub)


def test_real_stub_experiment(tmp_path):
    rc = main(["--name", "stub", "--seeds", "1-2", "--conditions", "adaptive,frozen", "--episodes", "12",
               "--runs-dir", str(tmp_path), "--quiet", "--", *STUB_TRAIN])  # fmt: skip
    assert rc == 0
    exp = tmp_path / "experiments" / "stub"
    man = read(exp / "manifest.json")
    assert all(r["status"] == "done" for r in man["runs"])
    for r in man["runs"]:
        cfg = read(Path(r["run_dir"]) / "config.json")
        assert cfg["seed"] == r["seed"] and cfg["adaptive"] == (r["condition"] == "adaptive")
        assert cfg["label"] == f"stub · {r['condition']} · seed {r['seed']}"
    agg = read(exp / "aggregate.json")
    metrics = [c["metric"] for c in agg["contrasts"]]
    assert metrics[:3] == ["blue_learned_vs_red_evasive@0.7 late_mean", "blue_learned_vs_red_evasive@0.4 late_mean",
                           "blue_learned_vs_red_baseline late_mean"]  # fmt: skip
    for cond in ("adaptive", "frozen"):
        assert [p["after_episode"] for p in agg["curves"][cond]["blue_learned_vs_red_evasive@0.7"]] == [0, 6, 12]
    assert set(agg["arms_race"]) == {"malware", "phishing", "network"}
    for v in agg["arms_race"].values():
        assert {"cycles_mean", "recall_at_0.7_first", "recall_at_0.7_last"} <= set(v)


# ------------------------------------------------------------------------------------------ cancel


def test_cancel_stops_all_children(tmp_path):
    cmd = [sys.executable, "-m", "cyberarena.arena.experiment", "--name", "c", "--seeds", "1-2",
           "--conditions", "adaptive,frozen", "--episodes", "20000", "--jobs", "4", "--runs-dir", str(tmp_path),
           "--quiet", "--", *STUB_TRAIN]  # fmt: skip
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)
    man_path = tmp_path / "experiments" / "c" / "manifest.json"
    deadline = time.time() + 60
    while time.time() < deadline:  # every child running and past setup
        try:
            runs = read(man_path)["runs"]
            if all(r["run_dir"] for r in runs) and all(
                    read(Path(r["run_dir"]) / "progress.json")["phase"] in ("train", "eval")
                    for r in runs):
                break
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("children never started")
    proc.send_signal(signal.CTRL_BREAK_EVENT if WIN else signal.SIGTERM)
    _, err = proc.communicate(timeout=60)
    assert proc.returncode == 130, err
    man = read(man_path)
    assert man["status"] == "cancelled"
    for r in man["runs"]:
        assert r["status"] == "error" and r["error"] == "cancelled"
        assert r["returncode"] == 130  # train itself handled the break and exited cleanly
        prog = read(Path(r["run_dir"]) / "progress.json")
        assert prog["status"] == "error" and prog["error"] == "cancelled"
    assert not (man_path.parent / "aggregate.json").exists()
