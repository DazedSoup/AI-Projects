import json

import pytest

from cyberarena import config, deploy


def _fake_root(tmp_path):
    root = tmp_path / "repo"
    pkg = root / "src" / "cyberarena"
    (pkg / "dashboard" / "__pycache__").mkdir(parents=True)
    (pkg / "arena").mkdir()
    for f in deploy.PACKAGE_FILES:
        (pkg / f).write_text("# stub\n")
    (pkg / "dashboard" / "app.py").write_text("# app\n")
    (pkg / "dashboard" / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"x")
    (pkg / "arena" / "train.py").write_text("# training code must not be deployed\n")
    (root / ".streamlit").mkdir()
    (root / ".streamlit" / "config.toml").write_text("[theme]\n")
    (root / "requirements-dashboard.txt").write_text("streamlit==1.64.0\n")
    (root / "LICENSE").write_text("MIT\n")
    showcase = root / "showcase"
    (showcase / "20261003-110349-7").mkdir(parents=True)
    (showcase / "20261003-110349-7" / "summary.jsonl").write_text("{}\n")
    (showcase / "MANIFEST.json").write_text(json.dumps({"runs": ["20261003-110349-7"], "experiments": ["main"]}))
    return root, showcase


def test_streamlit_bundle_is_self_configuring_and_dashboard_only(tmp_path):
    root, showcase = _fake_root(tmp_path)
    out = tmp_path / "bundle"
    info = deploy.assemble(out, showcase, root=root)  # default target: streamlit
    assert info["target"] == "streamlit"
    assert (out / "PUBLIC_SHOWCASE").exists()
    assert (out / "src" / "cyberarena" / "dashboard" / "app.py").exists()
    assert not (out / "src" / "cyberarena" / "arena").exists()
    assert not list(out.rglob("__pycache__"))
    assert not (out / "Dockerfile").exists() and not (out / ".gitattributes").exists()  # no Docker, no LFS
    assert deploy.MAIN_FILE in (out / "README.md").read_text(encoding="utf-8")
    assert (out / "requirements.txt").read_text() == "streamlit==1.64.0\n"


def test_hf_docker_bundle(tmp_path):
    root, showcase = _fake_root(tmp_path)
    out = tmp_path / "space"
    info = deploy.assemble(out, showcase, root=root, target="hf-docker")

    assert (out / "src" / "cyberarena" / "dashboard" / "app.py").exists()
    assert not (out / "src" / "cyberarena" / "arena").exists()
    assert not list(out.rglob("__pycache__"))
    assert (out / "showcase" / "20261003-110349-7" / "summary.jsonl").exists()
    assert (out / "requirements.txt").read_text() == "streamlit==1.64.0\n"
    docker = (out / "Dockerfile").read_text()
    assert "CYBERARENA_PUBLIC=1" in docker and "CYBERARENA_RUNS_DIR=/home/user/app/showcase" in docker
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n") and "sdk: docker" in readme and f"app_port: {deploy.SPACE_PORT}" in readme
    assert "showcase/** filter=lfs" in (out / ".gitattributes").read_text()
    assert info["manifest"]["runs"] == ["20261003-110349-7"]


def test_assemble_requires_an_exported_showcase(tmp_path):
    root, showcase = _fake_root(tmp_path)
    (showcase / "MANIFEST.json").unlink()
    with pytest.raises(FileNotFoundError, match="cyberarena.showcase"):
        deploy.assemble(tmp_path / "space", showcase, root=root)


def test_public_marker_switches_config(monkeypatch, tmp_path):
    import importlib

    (tmp_path / "PUBLIC_SHOWCASE").write_text("x")
    monkeypatch.setenv("CYBERARENA_ROOT", str(tmp_path))
    monkeypatch.delenv("CYBERARENA_PUBLIC", raising=False)
    monkeypatch.delenv("CYBERARENA_RUNS_DIR", raising=False)
    try:
        cfg = importlib.reload(config)
        assert cfg.PUBLIC is True and cfg.RUNS_DIR == tmp_path / "showcase"
        monkeypatch.setenv("CYBERARENA_PUBLIC", "0")  # an explicit env setting still wins
        assert importlib.reload(config).PUBLIC is False
    finally:
        monkeypatch.delenv("CYBERARENA_ROOT")
        monkeypatch.delenv("CYBERARENA_PUBLIC")
        importlib.reload(config)
    assert config.PUBLIC is False


def test_config_env_switches(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("CYBERARENA_PUBLIC", "1")
    monkeypatch.setenv("CYBERARENA_RUNS_DIR", str(tmp_path / "showcase"))
    try:
        cfg = importlib.reload(config)
        assert cfg.PUBLIC is True
        assert cfg.RUNS_DIR == tmp_path / "showcase"
    finally:
        monkeypatch.delenv("CYBERARENA_PUBLIC")
        monkeypatch.delenv("CYBERARENA_RUNS_DIR")
        importlib.reload(config)
    assert config.PUBLIC is False


def test_reassemble_keeps_the_space_git_checkout(tmp_path):
    root, showcase = _fake_root(tmp_path)
    out = tmp_path / "space"
    deploy.assemble(out, showcase, root=root)
    (out / ".git").mkdir()
    (out / ".git" / "config").write_text("[remote]\n")
    (out / "stale.txt").write_text("from an older deploy\n")
    deploy.assemble(out, showcase, root=root)
    assert (out / ".git" / "config").read_text() == "[remote]\n"
    assert not (out / "stale.txt").exists()
    assert (out / "PUBLIC_SHOWCASE").exists()


def test_rebuild_keeps_an_edited_readme(tmp_path):
    root, showcase = _fake_root(tmp_path)
    out = tmp_path / "bundle"
    deploy.assemble(out, showcase, root=root)
    assert deploy.MAIN_FILE in (out / "README.md").read_text(encoding="utf-8")  # default on first build
    (out / "README.md").write_text("")  # the owner empties it on GitHub
    deploy.assemble(out, showcase, root=root)
    assert (out / "README.md").read_text() == ""
