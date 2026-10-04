import json

from cyberarena import config, pipeline


def _fake_repo(tmp_path, monkeypatch, *, data=False, models=False):
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "artifacts" / "models")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    for flag, folder, ext in ((data, config.DATA_PROCESSED, "npz"), (models, config.MODELS_DIR, "keras")):
        if flag:
            folder.mkdir(parents=True)
            for m in config.CLASSIFIERS:
                (folder / f"{m}.{ext}").write_bytes(b"x")


def _make_run(runs, name, label, status="done", enriched=False):
    d = runs / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"label": label}))
    (d / "progress.json").write_text(json.dumps({"status": status}))
    if enriched:
        (d / "episodes_enriched.jsonl").write_text("{}\n")
    return d


def test_fresh_clone_runs_every_step_in_order(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    calls = []

    def runner(cmd):
        calls.append(cmd)
        if "cyberarena.arena.train" in cmd:  # the reference step produces the run the enrich step needs
            _make_run(config.RUNS_DIR, "20261003-000000-7", "reference")
        return 0

    assert pipeline.run([], runner=runner) == 0
    modules = [c[2] for c in calls]
    assert modules == ["cyberarena.ml.datasets", "cyberarena.ml.train", "cyberarena.arena.train",
                       "cyberarena.explain.enrich", "cyberarena.arena.experiment"]  # fmt: skip
    enrich = calls[3]
    assert enrich[-1].endswith("20261003-000000-7")
    assert all("--online" not in c for c in calls)


def test_finished_steps_are_skipped(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch, data=True, models=True)
    _make_run(config.RUNS_DIR, "20261003-000000-7", "reference", enriched=True)
    (config.RUNS_DIR / "experiments" / "main").mkdir(parents=True)
    (config.RUNS_DIR / "experiments" / "main" / "aggregate.json").write_text("{}")
    calls = []
    assert pipeline.run([], runner=lambda c: calls.append(c) or 0) == 0
    assert calls == []


def test_unfinished_or_mislabelled_runs_are_not_reused(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    _make_run(config.RUNS_DIR, "20261003-000001-7", "reference", status="running")
    _make_run(config.RUNS_DIR, "20261003-000002-7", "something else")
    assert pipeline.find_run("reference", config.RUNS_DIR) is None


def test_failed_step_stops_with_its_exit_code(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    calls = []
    assert pipeline.run([], runner=lambda c: calls.append(c) or 3) == 3
    assert len(calls) == 1


def test_quick_dry_run_and_only(tmp_path, monkeypatch, capsys):
    _fake_repo(tmp_path, monkeypatch)
    assert pipeline.run(["--quick", "--dry-run", "--only", "reference,experiment"], runner=lambda c: 1 / 0) == 0
    out = capsys.readouterr().out
    assert "--episodes 300" in out and "--seeds 1-2" in out and "datasets" not in out
    assert pipeline.run(["--only", "bogus"]) == 2
