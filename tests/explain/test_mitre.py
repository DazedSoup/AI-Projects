import pytest

from cyberarena.arena.actions import BLUE_ACTION_IDS, RED_ACTION_IDS
from cyberarena.explain import mitre


def test_every_action_has_a_table_entry():
    assert mitre.coverage() == {"red": [], "blue": []}
    assert set(mitre.RED_ATTACK) == set(RED_ACTION_IDS)
    assert set(mitre.BLUE_D3FEND) == set(BLUE_ACTION_IDS)


@pytest.mark.parametrize("action_id", [a for a in RED_ACTION_IDS if a != "wait"])
def test_every_red_move_gets_an_attack_technique(action_id):
    tag = mitre.tag("red", action_id, "dmz")
    assert tag["framework"] == "ATT&CK" and tag["version"] == mitre.ATTACK_VERSION
    assert tag["technique_id"].startswith("T") and tag["technique_name"] and tag["tactic"]


def test_expected_red_ids():
    ids = {a: mitre.tag("red", a, "dmz")["technique_id"] for a in RED_ACTION_IDS if a != "wait"}
    assert ids == {"recon": "T1046", "phish": "T1566", "exploit": "T1190", "escalate": "T1068",
                   "lateral_move": "T1021", "exfiltrate": "T1041"}  # fmt: skip
    assert mitre.tag("red", "wait") is None


def test_exploit_depends_on_target_exposure():
    assert mitre.tag("red", "exploit", "dmz")["technique_id"] == "T1190"
    assert mitre.tag("red", "exploit", "server")["technique_id"] == "T1210"
    assert mitre.tag("red", "exploit")["technique_id"] == "T1190"


def test_blue_tags_are_d3fend_or_null():
    for a in BLUE_ACTION_IDS:
        tag = mitre.tag("blue", a, "server")
        assert tag is None or (tag["framework"] == "D3FEND" and tag["technique_id"].startswith("D3-"))
    assert mitre.tag("blue", "isolate")["technique_id"] == "D3-NI"


def test_unknown_actions_raise():
    with pytest.raises(KeyError):
        mitre.tag("red", "teleport")
    with pytest.raises(ValueError):
        mitre.tag("green", "wait")


def test_table_lists_every_action():
    rows = mitre.table()
    assert {(r["actor"], r["action_id"]) for r in rows} >= {("red", a) for a in RED_ACTION_IDS}
