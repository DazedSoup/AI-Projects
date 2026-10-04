import importlib

from cyberarena import config


def test_subpackages_import():
    for sub in ("ml", "arena", "explain", "dashboard"):
        importlib.import_module(f"cyberarena.{sub}")


def test_paths_rooted_in_repo():
    assert (config.ROOT / "pyproject.toml").exists()
    assert config.CLASSIFIERS == ("malware", "phishing", "network")
