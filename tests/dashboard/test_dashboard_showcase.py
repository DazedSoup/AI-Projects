"""Public showcase (docs/contracts.md, "Public showcase (v6)"): the export, label resolution, the public-mode guards,
every page rendered from an export with nothing else on disk, no writes, and the admin publish toggles.

The export is made once per module from a copy of the v4 fixture (plus a fake experiment seed run and files that
must never be exported: agent weights, detectors, caches, logs)."""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cyberarena import config
from cyberarena.dashboard import lab, lab_runner
from cyberarena.dashboard import loaders as L
from cyberarena.dashboard import showcase as SC

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
import streamlit as st

APP = Path(config.__file__).parent / "dashboard" / "app.py"
FIX = Path(__file__).parent / "fixtures"
REF, TAB = "20261003-110349-7", "20261003-110348-7"  # labels "reference" and "tabular comparison"
SEED = "20261003-110349-1"  # main experiment, adaptive seed 1 (manifest run_dir 'runs-not-in-fixture\\...')
EXPS = ["main", "main-cheap-isolation", "diag-detA-evB", "diag-detA-evOff", "diag-detF-evB", "diag-detF-evOff"]
PUBLIC_PAGES = ("overview_page.py", "replay_page.py", "learning_page.py", "evidence_page.py")
RUN_FILES_ALLOWED = set(SC.RUN_FILES) | {L.EPISODES_FILE}


def _make_source(root: Path) -> Path:
    src = root / "runs"
    shutil.copytree(FIX / "runs_v4", src)
    for rid in (REF, TAB):  # things the export must skip
        (src / rid / "detectors").mkdir(exist_ok=True)
        (src / rid / "detectors" / "malware_v3.keras").write_bytes(b"x" * 100)
        (src / rid / "explain_cache").mkdir(exist_ok=True)
        (src / rid / "explain_cache" / "agent_ig.jsonl").write_text("{}\n", encoding="utf-8")
        (src / rid / "agents" / "checkpoints").mkdir(parents=True, exist_ok=True)
        (src / rid / "agents" / "checkpoints" / "red_qnet_1000.keras").write_bytes(b"w")
        (src / rid / "lab_enrich.log").write_text("log\n", encoding="utf-8")
    seed = src / SEED
    seed.mkdir()
    cfg = json.loads((src / REF / "config.json").read_text(encoding="utf-8"))
    cfg["label"] = "main · adaptive · seed 1"
    (seed / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    shutil.copyfile(src / REF / "summary.jsonl", seed / "summary.jsonl")
    shutil.copyfile(src / REF / "episodes.jsonl", seed / "episodes.jsonl")  # never copied for experiment runs
    shutil.copytree(src / REF / "agents", seed / "agents")
    (src / "diag-crosseval" / "main_seed1.json.log").write_text("log\n", encoding="utf-8")
    _plant_paths(src)
    return src


HOME = str(Path.home())
OTHER_DRIVE = r"D:\data\private"


def _plant_paths(src: Path) -> None:
    """Absolute paths of the kinds real runs record: manifests' absolute run_dir, a repo path, a home path,
    another drive and a UNC share, in JSON files and in a JSONL row."""
    for man_path in (src / "experiments").glob("*/manifest.json"):
        man = json.loads(man_path.read_text(encoding="utf-8"))
        for r in man.get("runs") or []:
            if r.get("run_dir"):
                r["run_dir"] = str(src / L.path_name(r["run_dir"]))
        man["log_dir"] = str(man_path.parent / "logs")
        man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    cfg_path = src / REF / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["init_from"] = {"run_dir": str(src / TAB), "side": "both"}
    cfg["data_dir"] = str(Path(config.ROOT) / "data" / "processed")
    cfg["cache"] = [str(Path.home() / ".cache" / "x.npz"), OTHER_DRIVE + r"\rows.csv", r"\\fileserver\share\a.json"]
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    explain = {"run": str(src / REF), "cache": str(src / REF / "explain_cache"), "model": "/home/someone/models/x.keras"}
    (src / REF / "explain_summary.json").write_text(json.dumps(explain), encoding="utf-8")
    summ = src / TAB / "summary.jsonl"
    rows = summ.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["source"] = str(src / TAB / "episodes.jsonl")
    summ.write_text("\n".join([json.dumps(first), *rows[1:]]) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """``(source runs dir, showcase dir)``: one export for the whole module."""
    root = tmp_path_factory.mktemp("sc")
    src = _make_source(root)
    out = root / "showcase"
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "PUBLIC", False)
    import tempfile

    mp.setattr(tempfile, "tempdir", str(root / "tmp"))
    (root / "tmp").mkdir()
    try:
        rc = SC.main(["--runs-dir", str(src), "--runs", "reference,tabular comparison", "--experiments", ",".join(EXPS),
                      "--out", str(out)])  # fmt: skip
    finally:
        mp.undo()
    assert rc == 0
    return src, out


def _tree(d: Path) -> dict[str, tuple]:
    """Every file and folder under ``d`` with size and mtime: a snapshot to prove nothing was written."""
    out = {}
    for p in sorted(d.rglob("*")):
        s = p.stat()
        out[p.relative_to(d).as_posix()] = (p.is_dir(), s.st_size if p.is_file() else 0, s.st_mtime_ns)
    return out


# ============================================================================================== export


def test_dashboard_showcase_export_copies_only_what_the_dashboard_reads(exported):
    _, out = exported
    man = json.loads((out / SC.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert set(man) == {"created", "runs", "experiments", "default_run", "bytes"}
    assert man["runs"] == [REF, TAB] and man["experiments"] == EXPS and man["default_run"] == REF
    assert man["bytes"] == sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    for rid in (REF, TAB):
        names = {p.relative_to(out / rid).as_posix() for p in (out / rid).rglob("*")}
        assert names <= RUN_FILES_ALLOWED, names - RUN_FILES_ALLOWED  # no agents/, detectors/, caches, logs
        assert {"config.json", "graph.json", "summary.jsonl", "learning.jsonl", "episodes.jsonl"} <= names
    assert {p.name for p in (out / SEED).iterdir()} == {"config.json", "summary.jsonl"}  # experiment run
    for e in EXPS:
        assert {p.name for p in (out / "experiments" / e).iterdir()} == {"manifest.json", "aggregate.json"}
    assert not (out / "experiments" / "smoke-dqn").exists()  # not selected
    # cross-evaluation files about the main-cheap-isolation runs; never their logs
    assert sorted(p.name for p in (out / "diag-crosseval").iterdir()) == ["main_seed1.json", "main_seed2.json"]
    assert not any(p.suffix in (".keras", ".log") for p in out.rglob("*"))
    assert not list(out.parent.glob(".showcase.build-*"))  # the build folder was swapped in


def test_dashboard_showcase_trimmed_log_keeps_narrated_and_probe_games_in_order(exported):
    src, out = exported
    for rid in (REF, TAB):
        full = L.build_episode_index(src / rid, disk_cache=False)
        cut = L.build_episode_index(out / rid, disk_cache=False)
        enr = set(L.load_enriched(src / rid))
        cat = L.episode_catalog(L.load_enriched(src / rid), *L.load_summary(src / rid), full.meta)
        probes = {int(e) for e in L.probe_games(cat)["episode"]}
        assert set(cut.episodes) == (enr | probes) & set(full.episodes)
        assert probes and set(cut.episodes) < set(full.episodes)
        for ep in cut.episodes:  # the byte-offset index works on the trimmed file: same records
            assert L.read_episode(cut, ep) == L.read_episode(full, ep)
        orig = (src / rid / L.EPISODES_FILE).read_bytes().splitlines(keepends=True)
        kept = (out / rid / L.EPISODES_FILE).read_bytes().splitlines(keepends=True)
        it = iter(orig)
        assert all(any(line == o for o in it) for line in kept)  # a subsequence: original line order, exact bytes
    # the compare view's probe games survive with their metadata
    cut = L.build_episode_index(out / REF, disk_cache=False)
    cat = L.episode_catalog(L.load_enriched(out / REF), *L.load_summary(out / REF), cut.meta)
    assert not L.probe_games(cat).empty
    assert set(cat["kind"]) <= {"narrated", "probe"}  # training/eval games the summary lists are dropped


def test_dashboard_showcase_max_mb_fails_without_replacing(exported, tmp_path, monkeypatch, capsys):
    src, _ = exported
    monkeypatch.setattr(config, "PUBLIC", False)
    dst = tmp_path / "showcase"
    assert SC.main(["--runs-dir", str(src), "--runs", REF, "--out", str(dst)]) == 0
    before = _tree(dst)
    assert SC.main(["--runs-dir", str(src), "--runs", REF, "--out", str(dst), "--max-mb", "0.01"]) == 1
    assert "over --max-mb" in capsys.readouterr().err
    assert _tree(dst) == before
    assert not list(tmp_path.glob(".showcase.*"))
    # a folder that isn't a showcase is never replaced
    other = tmp_path / "mine"
    other.mkdir()
    (other / "keep.txt").write_text("x", encoding="utf-8")
    assert SC.main(["--runs-dir", str(src), "--runs", REF, "--out", str(other)]) == 2
    assert (other / "keep.txt").exists()
    assert SC.main(["--runs-dir", str(src), "--runs", REF, "--out", str(src)]) == 2  # never the runs folder


def test_dashboard_showcase_dry_run_writes_nothing(exported, tmp_path, monkeypatch, capsys):
    src, _ = exported
    monkeypatch.setattr(config, "PUBLIC", False)
    assert SC.main(["--runs-dir", str(src), "--runs", REF, "--out", str(tmp_path / "x"), "--dry-run"]) == 0
    assert not (tmp_path / "x").exists()
    assert "trimmed" in capsys.readouterr().out


# ============================================================================================== resolution


def _run(src: Path, rid: str, label: str, status: str = "done") -> None:
    d = src / rid
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"label": label}), encoding="utf-8")
    (d / "progress.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    shutil.copyfile(FIX / "runs_v4" / REF / "summary.jsonl", d / "summary.jsonl")


def test_dashboard_showcase_label_resolution(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "PUBLIC", False)
    src = tmp_path / "runs"
    shutil.copytree(FIX / "runs_v4", src)
    _run(src, "20261003-120000-7", "reference", "running")  # newer but unfinished: skipped
    _run(src, "20261001-000000-7", "Reference")  # older finished duplicate label
    _run(src, "20261002-000000-7", "tabular comparison (cheap isolation)")
    assert SC.resolve_run(src, "reference") == REF  # newest *finished* run with that label
    assert SC.resolve_run(src, " REFERENCE ") == REF  # case and spaces don't matter
    assert SC.resolve_run(src, TAB) == TAB  # run ids pass through
    assert SC.resolve_run(src, "tabular comparison") == TAB  # an exact label beats a longer one it prefixes
    assert SC.resolve_run(src, "tabular comparison (cheap") == "20261002-000000-7"  # one label by prefix
    with pytest.raises(SC.ResolveError, match="ambiguous"):
        SC.resolve_run(src, "tabular")
    with pytest.raises(SC.ResolveError, match="no run"):
        SC.resolve_run(src, "nope")
    _run(src, "20261003-130000-7", "only running", "running")
    with pytest.raises(SC.ResolveError, match="no finished run"):
        SC.resolve_run(src, "only running")
    assert SC.resolve_experiment(src, "main") == "main"
    assert SC.resolve_experiment(src, "MAIN") == "main"
    with pytest.raises(SC.ResolveError):
        SC.resolve_experiment(src, "nope")
    out = tmp_path / "showcase"
    for args in (["--runs", "tabular"], ["--runs", "nope"], ["--experiments", "nope"], ["--runs", ""]):
        assert SC.main(["--runs-dir", str(src), "--out", str(out), *args]) == 2, args
    err = capsys.readouterr().err
    assert "ambiguous" in err and "no run with id or label 'nope'" in err and "no experiment 'nope'" in err
    assert not out.exists()


def test_dashboard_showcase_cli_reads_the_saved_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC", False)
    src = tmp_path / "runs"
    shutil.copytree(FIX / "runs_v4", src)
    out = tmp_path / "showcase"
    assert SC.main(["--runs-dir", str(src), "--out", str(out)]) == 2  # nothing selected yet
    SC.write_selection(src, {"runs": [TAB, REF], "experiments": ["main"], "default_run": TAB})
    assert SC.main(["--runs-dir", str(src), "--out", str(out)]) == 0
    man = json.loads((out / SC.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert man["runs"] == [TAB, REF] and man["experiments"] == ["main"] and man["default_run"] == TAB
    assert SC.main(["--runs-dir", str(src), "--out", str(out), "--runs", "reference", "--save-selection"]) == 0
    assert SC.read_selection(src) == {"runs": [REF], "experiments": [], "default_run": REF}


def test_dashboard_showcase_ignored_by_run_listings(tmp_path):
    src = tmp_path / "runs"
    shutil.copytree(FIX / "runs_v4", src)
    (src / SC.SELECTION_FILE).write_text("{}", encoding="utf-8")
    shutil.copytree(src / REF, src / "showcase" / REF)  # an export kept inside runs/
    (src / "showcase" / "summary.jsonl").write_text("", encoding="utf-8")  # even if it looked like a run
    (src / "showcase" / "config.json").write_text("{}", encoding="utf-8")
    names = {p.name for p in L.list_runs(src)} | {p.name for p in lab.list_run_dirs(src)}
    assert "showcase" not in names and SC.SELECTION_FILE not in names
    assert {REF, TAB} <= names


# ============================================================================================== public guards


class _Recorder:
    """``subprocess`` stand-in that fails the test if anything is started."""

    TimeoutExpired = subprocess.TimeoutExpired
    DEVNULL = subprocess.DEVNULL

    def __init__(self):
        self.calls = []

    def run(self, cmd, **kw):
        self.calls.append(cmd)
        raise AssertionError(f"subprocess.run in public mode: {cmd}")

    def Popen(self, cmd, **kw):
        self.calls.append(cmd)
        raise AssertionError(f"subprocess.Popen in public mode: {cmd}")


def test_dashboard_showcase_public_guards_raise(tmp_path, monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(lab, "subprocess", rec)
    monkeypatch.setattr(lab_runner, "subprocess", rec)
    monkeypatch.setattr(config, "PUBLIC", True)
    runs = tmp_path / "runs"
    runs.mkdir()
    job = {"token": "t", "job_file": str(runs / ".lab" / "t.json"), "pid": os.getpid(), "train_pid": None,
           "run_dir": None, "train_cmd": ["python", "-c", "pass"]}  # fmt: skip
    calls = {
        "load_param_spec": lambda: lab.load_param_spec(),
        "process_cmdline": lambda: lab.process_cmdline(os.getpid()),
        "kill_tree": lambda: lab.kill_tree(os.getpid()),
        "launch_job": lambda: lab.launch_job(runs, ["--episodes", "10"]),
        "launch_experiment": lambda: lab.launch_experiment(runs, "x", ["python", "-m", "x"]),
        "request_stop": lambda: lab.request_stop(job),
        "stop_job": lambda: lab.stop_job(job),
        "stop_job_force": lambda: lab.stop_job(job, force=True),
        "save_job": lambda: lab.save_job(dict(job)),
        "write_json_atomic": lambda: lab.write_json_atomic(runs / "x.json", {}),
        "run_showcase_export": lambda: lab.run_showcase_export(),
        "runner_run": lambda: lab_runner.run(Path(job["job_file"])),
        "runner_experiment": lambda: lab_runner.run_experiment(dict(job), Path(job["job_file"])),
        "runner_logged": lambda: lab_runner._run_logged(["python"], runs / "l.log", dict(job), None),
        "runner_cancel": lambda: lab_runner._cancel(None),
        "runner_main": lambda: lab_runner.main(["--job", job["job_file"]]),
        "write_selection": lambda: SC.write_selection(runs, {"runs": ["a"]}),
        "set_published": lambda: SC.set_published(runs, "runs", "a", True),
        "set_default_run": lambda: SC.set_default_run(runs, "a"),
    }
    for name, fn in calls.items():
        with pytest.raises(lab.PublicModeError, match="public showcase") as ei:
            fn()
        assert "CYBERARENA_PUBLIC" in str(ei.value), name
    assert SC.main(["--runs-dir", str(runs), "--runs", "a", "--out", str(tmp_path / "o")]) == 2
    assert rec.calls == []
    assert list(runs.rglob("*")) == []  # nothing was written
    monkeypatch.setattr(config, "PUBLIC", False)  # the same calls work again for the admin
    lab.write_json_atomic(runs / "x.json", {"a": 1})
    assert json.loads((runs / "x.json").read_text(encoding="utf-8")) == {"a": 1}


# ============================================================================================== public pages


@pytest.fixture
def public(exported, monkeypatch, tmp_path):
    """The dashboard as hosted: ``CYBERARENA_PUBLIC=1`` and ``CYBERARENA_RUNS_DIR=<export>`` read by
    ``cyberarena.config`` itself; the source runs folder is moved away so only the export exists."""
    src, out = exported
    monkeypatch.setenv("CYBERARENA_PUBLIC", "1")
    monkeypatch.setenv("CYBERARENA_RUNS_DIR", str(out))
    monkeypatch.setenv("CYBERARENA_ROOT", str(tmp_path / "no-root"))  # no repo runs/ to fall back on
    importlib.reload(config)
    assert config.PUBLIC is True and Path(config.RUNS_DIR) == out
    monkeypatch.setattr(lab, "subprocess", _Recorder())
    hidden = src.with_name("runs-hidden")
    os.replace(src, hidden)
    import tempfile

    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    st.cache_data.clear()
    yield out
    st.cache_data.clear()
    os.replace(hidden, src)
    monkeypatch.delenv("CYBERARENA_PUBLIC")
    monkeypatch.delenv("CYBERARENA_RUNS_DIR")
    monkeypatch.delenv("CYBERARENA_ROOT")
    importlib.reload(config)


def _ok(app):
    assert not app.exception, [e.value for e in app.exception]


def _html(app) -> str:
    return " ".join(str(e.value) for e in app.get("html"))


def _app(page: str = "overview_page.py", **state):
    app = AppTest.from_file(str(APP), default_timeout=120)
    for k, v in state.items():
        app.session_state[k] = v
    app.run()
    _ok(app)
    if page != "overview_page.py":
        app.switch_page(page).run()
        _ok(app)
    return app


@pytest.mark.parametrize("page", PUBLIC_PAGES)
@pytest.mark.parametrize("run_id", [REF, TAB])
def test_dashboard_showcase_public_page_renders(public, page, run_id):
    app = _app(page, run=public / run_id)
    h = _html(app)
    assert "ca-page-head" in h
    assert "Public showcase" in h and SC_GITHUB in h
    assert "Traceback" not in h


SC_GITHUB = "https://github.com/DazedSoup/ai-solutions"


def test_dashboard_showcase_public_defaults_and_picker(public):
    app = _app()
    assert app.session_state["run"] == public / REF  # MANIFEST default_run
    picker = app.sidebar.selectbox(key="run")
    assert len(picker.options) == 2  # the published runs only: experiment seed runs carry just a summary
    assert all(SEED not in str(o) for o in picker.options)
    assert "Showcase copy" in " ".join(str(c.value) for c in app.sidebar.caption)


def test_dashboard_showcase_public_lab_absent(public, monkeypatch):
    app = _app()
    with pytest.raises(ValueError, match="Known pages") as e:  # not registered: nothing to switch to
        app.switch_page("lab_page.py")
    assert "lab_page" not in str(e.value).split("Known pages")[1]
    assert not [t for t in app.toggle if str(t.key).startswith(("lab_", "ev_publish"))]
    # admin mode registers it (same app, same data)
    monkeypatch.setattr(config, "PUBLIC", False)
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.switch_page("lab_page.py")


def test_dashboard_showcase_public_replay_compare_and_missing_games(public):
    app = _app("replay_page.py", run=public / REF)
    h = _html(app)
    assert "Game" in h
    kinds = app.selectbox(key="rp_kind").options
    assert all(("Narrated" in k) or ("Probe" in k) for k in kinds), kinds
    app.segmented_control(key="rp_mode").set_value("Same game, earlier agent").run()
    _ok(app)
    assert "After" in _html(app)
    app = _app("replay_page.py", run=public / TAB)  # no narration: probe games only
    assert app.selectbox(key="rp_kind").options


V1 = "20260101-000000-1"  # predates learning.jsonl: Learning falls back to agents/checkpoints Q-tables


@pytest.fixture
def public_v1(tmp_path, monkeypatch):
    """An export of a v1 run whose Q-table checkpoints exist at the source but never reach the showcase."""
    src = tmp_path / "runs"
    shutil.copytree(FIX / "runs", src)
    ck = src / V1 / "agents" / "checkpoints"
    ck.mkdir(parents=True)
    for side in ("red", "blue"):
        (ck / f"{side}_100.json").write_text(json.dumps({"q": {"s1": [0.0], "s2": [1.0]}}), encoding="utf-8")
    assert not L.checkpoint_q_states(src / V1).empty  # the admin sees Q-table growth from these
    monkeypatch.setattr(config, "PUBLIC", False)
    out = tmp_path / "showcase"
    assert SC.main(["--runs-dir", str(src), "--runs", V1, "--out", str(out)]) == 0
    assert not (out / V1 / "agents").exists()
    shutil.rmtree(src)  # nothing but the export on disk
    monkeypatch.setattr(config, "PUBLIC", True)
    monkeypatch.setattr(config, "RUNS_DIR", out)
    monkeypatch.setattr(lab, "subprocess", _Recorder())
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    st.cache_data.clear()
    yield out
    st.cache_data.clear()


def test_dashboard_showcase_public_v1_run_without_checkpoints(public_v1):
    before = _tree(public_v1)
    app = _app()
    assert app.session_state["run"] == public_v1 / V1
    assert "Q-table checkpoints not included in the public showcase" in _html(app)
    app.switch_page("learning_page.py").run()
    _ok(app)
    assert "Agent checkpoints not in the showcase" in _html(app)
    for page in ("replay_page.py", "evidence_page.py"):
        app.switch_page(page).run()
        _ok(app)
    assert "No experiments in this showcase" in _html(app)
    assert _tree(public_v1) == before


def test_dashboard_showcase_public_evidence_every_experiment(public):
    app = _app("evidence_page.py")
    _ok(app)
    h = _html(app)
    assert "What we learned along the way" in h  # main links the previous-rules result and the 2×2 diagnosis
    assert not [t for t in app.toggle if str(t.key).startswith("ev_publish")]  # admin only
    opts = app.selectbox(key="ev_exp").options
    assert len(opts) == len(EXPS) + 1  # six experiments plus the diag-* factorial
    for o in list(opts):
        app.selectbox(key="ev_exp").select(o).run()
        _ok(app)
        assert "ca-page-head" in _html(app)


def test_dashboard_showcase_public_pass_writes_nothing(public, tmp_path):
    """A full pass over every page and view leaves the showcase folder and the temp dir untouched."""
    import tempfile

    tmp = Path(tempfile.gettempdir())
    before, tmp_before = _tree(public), _tree(tmp)
    for run_id in (REF, TAB):
        app = _app(run=public / run_id)
        for page in PUBLIC_PAGES[1:]:
            app.switch_page(page).run()
            _ok(app)
        app.switch_page("replay_page.py").run()
        app.segmented_control(key="rp_mode").set_value("Same game, earlier agent").run()
        _ok(app)
        app.switch_page("evidence_page.py").run()
        for o in list(app.selectbox(key="ev_exp").options):
            app.selectbox(key="ev_exp").select(o).run()
            _ok(app)
    assert _tree(public) == before
    assert _tree(tmp) == tmp_before  # the log index lived in memory only


# ============================================================================================== admin publish


class _SpecOnly:
    TimeoutExpired = subprocess.TimeoutExpired
    DEVNULL = subprocess.DEVNULL

    def __init__(self):
        self.runs = []

    def run(self, cmd, **kw):
        self.runs.append(cmd)
        if SC_MODULE in cmd:
            return subprocess.CompletedProcess(cmd, 0, "wrote showcase (3 files, 0.01 MB)", "")
        return subprocess.CompletedProcess(cmd, 0, (FIX / "param_spec_v4.json").read_text(encoding="utf-8"), "")

    def Popen(self, cmd, **kw):  # pragma: no cover
        raise AssertionError("nothing may be launched")


SC_MODULE = "cyberarena.showcase"


@pytest.fixture
def admin(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    shutil.copytree(FIX / "runs_v4", runs)
    monkeypatch.setattr(config, "PUBLIC", False)
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    fake = _SpecOnly()
    monkeypatch.setattr(lab, "subprocess", fake)
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    st.cache_data.clear()
    yield runs, fake
    st.cache_data.clear()


def _sel(runs: Path) -> dict:
    return json.loads((runs / SC.SELECTION_FILE).read_text(encoding="utf-8"))


def test_dashboard_showcase_lab_publish_toggles_write_selection(admin):
    runs, _ = admin
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.switch_page("lab_page.py").run()
    _ok(app)
    toggles = {t.key: t for t in app.toggle if str(t.key).startswith("lab_publish:")}
    assert toggles, "the history's default selection shows Publish toggles"
    key = min(toggles)
    rid = key.split(":", 1)[1]
    app.toggle(key=key).set_value(True).run()
    _ok(app)
    assert _sel(runs) == {"runs": [rid], "experiments": [], "default_run": rid}
    assert app.toggle(key=key).value is True
    assert "Showcase" in _html(app) and "Estimated export" in _html(app)
    app.toggle(key=key).set_value(False).run()
    _ok(app)
    assert _sel(runs) == {"runs": [], "experiments": [], "default_run": None}
    assert not list(runs.glob(".*.tmp"))  # atomic write leaves no temp file


def test_dashboard_showcase_lab_export_button_runs_the_cli(admin):
    runs, fake = admin
    SC.write_selection(runs, {"runs": [REF], "experiments": ["main"]})
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.switch_page("lab_page.py").run()
    _ok(app)
    assert "1 runs · 1 experiments" in _html(app)
    app.button(key="lab_showcase_export").click().run()
    _ok(app)
    cmd = next(c for c in fake.runs if SC_MODULE in c)
    assert cmd[-2:] == ["-m", SC_MODULE]
    assert any("Exported the showcase" in str(s.value) for s in app.success)
    assert any("wrote showcase" in str(c.value) for c in app.code)


def test_dashboard_showcase_evidence_publish_toggle(admin):
    runs, _ = admin
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.switch_page("evidence_page.py").run()
    _ok(app)
    app.toggle(key="ev_publish:main").set_value(True).run()
    _ok(app)
    assert _sel(runs)["experiments"] == ["main"]
    fam = "family:diag"
    app.selectbox(key="ev_exp").select(fam).run()
    _ok(app)
    app.toggle(key=f"ev_publish:{fam}").set_value(True).run()
    _ok(app)
    exps = _sel(runs)["experiments"]
    assert exps[0] == "main" and {"diag-detA-evB", "diag-detA-evOff", "diag-detF-evB", "diag-detF-evOff"} <= set(exps)
    app.selectbox(key="ev_exp").select("main").run()
    app.toggle(key="ev_publish:main").set_value(False).run()
    _ok(app)
    assert "main" not in _sel(runs)["experiments"]


def test_dashboard_showcase_selection_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC", False)
    runs = tmp_path / "runs"
    runs.mkdir()
    assert SC.read_selection(runs) == {"runs": [], "experiments": [], "default_run": None}
    SC.set_published(runs, "runs", "a", True)
    SC.set_published(runs, "runs", "b", True)
    SC.set_published(runs, "experiments", "main", True)
    SC.set_default_run(runs, "b")
    assert _sel(runs) == {"runs": ["a", "b"], "experiments": ["main"], "default_run": "b"}
    SC.set_published(runs, "runs", "b", False)  # removing the default falls back to the first run
    assert _sel(runs) == {"runs": ["a"], "experiments": ["main"], "default_run": "a"}
    (runs / SC.SELECTION_FILE).write_text("not json", encoding="utf-8")
    assert SC.read_selection(runs)["runs"] == []


DRIVE_PATH = re.compile(rb"(?<![A-Za-z0-9])[A-Za-z]:(?:\\|/)")


def leaks(out: Path, extra: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """Every (file, needle) in ``out`` naming the repo root, the home dir (raw, forward-slash or JSON-escaped,
    any case) or a drive-letter path."""
    needles = set()
    for n in (str(Path(config.ROOT)), HOME, *extra):
        for form in (n, n.replace("\\", "/"), json.dumps(n)[1:-1]):
            needles.add(form.lower().encode("utf-8"))
    hits = []
    for f in out.rglob("*"):
        if f.is_file():
            data = f.read_bytes()
            low = data.lower()
            rel = f.relative_to(out).as_posix()
            hits += [(rel, n.decode()) for n in needles if n in low]
            hits += [(rel, m.group(0).decode()) for m in DRIVE_PATH.finditer(data)]
    return hits


def test_dashboard_showcase_export_leaks_no_local_paths(exported):
    src, out = exported
    assert leaks(src, (str(src),)), "the planted source paths must be found by the grep"
    assert leaks(out, (str(src), OTHER_DRIVE)) == []
    # rewritten as asked: runs-relative, repo-relative, else the basename
    man = json.loads((out / "experiments" / "main" / "manifest.json").read_text(encoding="utf-8"))
    assert SEED in {r["run_dir"] for r in man["runs"]}
    assert man["log_dir"] == "experiments/main/logs"
    cfg = json.loads((out / REF / "config.json").read_text(encoding="utf-8"))
    assert cfg["init_from"]["run_dir"] == TAB
    assert cfg["data_dir"] == "data/processed"
    assert cfg["cache"] == ["x.npz", "rows.csv", "a.json"]
    ex = json.loads((out / REF / "explain_summary.json").read_text(encoding="utf-8"))
    assert ex == {"run": REF, "cache": f"{REF}/explain_cache", "model": "x.keras"}
    first = json.loads((out / TAB / "summary.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["source"] == f"{TAB}/episodes.jsonl"
    assert L.load_summary(out / TAB)[1].shape == L.load_summary(src / TAB)[1].shape
    # Evidence still finds the experiment's runs from the scrubbed manifest
    from cyberarena.dashboard import evidence as E

    e = next(x for x in E.list_experiments(out) if x["name"] == "main")
    assert E.experiment_rules(e) is not None
    prog = E.run_progress(man, out)
    assert SEED in set(prog["run_id"])
    assert E.run_dir_of(SEED, out) == out / SEED and (out / SEED / "config.json").exists()
