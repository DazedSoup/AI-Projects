"""Rule-set presets (--rules): v5 defaults make isolating a clean host costly; cheap-isolation restores v4."""

import json

import pytest

from cyberarena.arena.env import ArenaConfig
from cyberarena.arena.params import RULE_PRESETS, ParamError, parse_env_json, spec
from cyberarena.arena.train import main

TINY = ["--episodes", "12", "--seed", "4", "--eval-every", "6", "--eval-n", "3", "--eval-log-n", "1",
        "--log-every", "6", "--detector-update-every", "6", "--detector-min-samples", "8", "--stats-every", "6",
        "--adaptive-pool-size", "64", "--eval-evasive-n", "2", "--stub", "--quiet", "--agent", "tabular"]  # fmt: skip


def test_defaults_and_presets():
    assert (ArenaConfig().r_isolate_false, ArenaConfig().r_isolated_upkeep) == (0.3, 0.03)
    cheap = parse_env_json(None, "cheap-isolation")
    assert (cheap.r_isolate_false, cheap.r_isolated_upkeep) == (0.1, 0.01)
    assert parse_env_json(None).to_dict() == ArenaConfig().to_dict()
    # --env-json still overrides the preset
    assert parse_env_json('{"r_isolate_false": 0.2}', "cheap-isolation").r_isolate_false == 0.2
    with pytest.raises(ParamError):
        parse_env_json(None, "nope")


def test_rules_param_in_spec():
    p = next(p for g in spec()["groups"] for p in g["params"] if p["key"] == "rules")
    assert p["default"] == "default" and p["choices"] == list(RULE_PRESETS) and p["flag"] == "--rules"
    assert "0.1 / 0.01" in p["help"]


def read(run, f):
    return [json.loads(x) for x in (run / f).read_text(encoding="utf-8").splitlines()]


def test_cheap_isolation_equals_explicit_old_rewards(tmp_path):
    a = main([*TINY, "--rules", "cheap-isolation", "--runs-dir", str(tmp_path / "a")])
    b = main([*TINY, "--env-json", '{"r_isolate_false": 0.1, "r_isolated_upkeep": 0.01}',
              "--runs-dir", str(tmp_path / "b")])  # fmt: skip
    strip = lambda rs: [{k: v for k, v in r.items() if k not in ("run_id", "seconds")} for r in rs]
    for f in ("summary.jsonl", "episodes.jsonl"):
        assert strip(read(a, f)) == strip(read(b, f)), f
    env = json.loads((a / "config.json").read_text())["env"]
    assert (env["r_isolate_false"], env["r_isolated_upkeep"]) == (0.1, 0.01)
