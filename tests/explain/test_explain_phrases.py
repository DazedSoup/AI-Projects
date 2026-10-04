"""Feature-name -> plain-English phrase table and the "why this host" rationale for DQN moves."""

from __future__ import annotations

import pytest
from explain_fixtures import GRAPH, make_turn

from cyberarena.arena.features import FEATURE_NAMES
from cyberarena.explain.enrich import make_facts
from cyberarena.explain.narrate import build_prompt, template_rationale
from cyberarena.explain.phrases import PHRASES, PhraseContext, describe


@pytest.mark.parametrize("side", ["red", "blue"])
def test_every_feature_name_has_a_phrase(side):
    missing = [n for n in FEATURE_NAMES[side] if n not in PHRASES]
    assert not missing, f"no plain-English phrase for {side} features {missing}"


@pytest.mark.parametrize("side", ["red", "blue"])
@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_every_phrase_renders_without_placeholders(side, value):
    for n in FEATURE_NAMES[side]:
        ctx = PhraseContext(target=13, source=12, t_hops=1, s_hops=2, max_dist=4, role="server")
        text = describe(n, value, ctx)
        assert text and "{" not in text and "}" not in text, (n, text)
        assert n not in text, f"{n} fell back to its raw name"


def test_example_phrases():
    ctx = PhraseContext(target=13, t_hops=1, max_dist=4)
    assert describe("t_dist", 0.25, ctx) == "host 13 is one hop from the crown jewel"
    assert describe("t_score_network", 0.08, ctx) == "its network score is low (0.08)"  # host already named
    assert describe("a_patch", 0.0) == "patches rate lower here"
    assert describe("t_workstation", 0.0, PhraseContext(target=4, role="server")) == \
        "host 4 is a server rather than a workstation"  # fmt: skip


def _dqn_facts(**attr_kw):
    cands = [{"action": "lateral_move", "source": 1, "target": 2, "q": 0.62},
             {"action": "lateral_move", "source": 1, "target": 0, "q": 0.48}]  # fmt: skip
    attribution = {
        "method": "integrated_gradients", "baseline": "candidate_mean", "q": 0.62, "checkpoint": "x",
        "top_features": [{"name": "t_dist", "value": 1 / 3, "baseline": 0.67, "attribution": 0.09},
                         {"name": "t_score_network", "value": 0.08, "baseline": 0.3, "attribution": 0.04},
                         {"name": "t_degree", "value": 1.0, "baseline": 0.5, "attribution": -0.05}],
    }  # fmt: skip
    attribution.update(attr_kw)
    turn = make_turn(candidates=cands, chosen_features={"t_dist": 1 / 3}, agent_attribution=attribution,
                     classifier_inputs=[])  # fmt: skip
    return make_facts(turn, None, None, GRAPH, {"env": {"max_rounds": 36}}, None, attribution)


def test_why_this_host_rationale():
    text = template_rationale(_dqn_facts())
    first, rest = text.split(". ", 1)
    assert first == "Red moves laterally from host 1 to host 2 (server) and succeeds"
    assert rest.startswith("It picks host 2 over host 0 (Q 0.62 vs 0.48), mainly because host 2 is one hop from the "
                           "crown jewel and its network score is low (0.08)")  # fmt: skip
    assert "degree" not in rest and "connected" not in rest  # negative attributions are not reasons


def test_why_rationale_is_deterministic_and_prompt_has_the_facts():
    f = _dqn_facts()
    assert template_rationale(f) == template_rationale(_dqn_facts())
    p = build_prompt(f)
    assert "Runner-up: moving laterally to host 0, Q 0.480 (chosen Q 0.620, margin +0.140)" in p
    assert "integrated gradients" in p and "host 2 is one hop from the crown jewel (+0.090)" in p
    assert "rather than the runner-up" in p


def test_exploration_pick_is_named():
    f = _dqn_facts()
    f.turn["explored"] = True
    f.turn["action_id"], f.turn["target"] = "lateral_move", 0
    assert "exploration pick" in template_rationale(f)


def test_adaptive_prompt_forbids_claiming_adaptation_helps():
    turn = make_turn(classifier_inputs=[{"node": 2, "model": "network", "row": [0.1], "score": 0.3,
                                         "evasion": 0.7, "version": 3}])  # fmt: skip
    adaptation = {"evasion": {"network": 0.7}, "detector_versions": {"network": 3}, "updated_since_last_move": []}
    f = make_facts(turn, None, None, GRAPH, {}, adaptation)
    p = build_prompt(f)
    assert "do not claim that detector adaptation helps blue win" in p
    for word in ("helps blue", "helping blue", "thanks to adaptation"):
        assert word not in template_rationale(f).lower()
