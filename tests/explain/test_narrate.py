"""Prompt builder, offline template and response parsing, with a mocked Anthropic client (no network)."""

from types import SimpleNamespace

import pytest
from explain_fixtures import GRAPH, SHAP_ENTRY, make_turn

from cyberarena.explain import mitre, narrate
from cyberarena.explain.narrate import (
    MissingCredentialsError,
    Narrator,
    TurnFacts,
    build_prompt,
    clean_text,
    parse_response,
    template_rationale,
)


def facts(**kw) -> TurnFacts:
    turn = make_turn(**kw)
    roles = {n["id"]: n["role"] for n in GRAPH["nodes"]}
    tag = mitre.tag(turn["actor"], turn["action_id"], roles.get(turn["target"]))
    return TurnFacts(turn=turn, roles=roles, crown_jewel=3, shap=[SHAP_ENTRY], mitre=tag, max_rounds=36)


class FakeMessages:
    def __init__(self, reply="Red pivots to the server because its network score is low (0.21). Extra. More.",
                 stop_reason="end_turn"):  # fmt: skip
        self.calls = []
        self.reply = reply
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = [SimpleNamespace(type="text", text=self.reply)] if self.reply is not None else []
        return SimpleNamespace(content=content, stop_reason=self.stop_reason,
                               stop_details=SimpleNamespace(category="cyber", explanation="x"),
                               usage=SimpleNamespace(input_tokens=420, output_tokens=38))  # fmt: skip


def fake_client(**kw):
    return SimpleNamespace(messages=FakeMessages(**kw))


# ---------------------------------------------------------------------------------------------- prompt


def test_prompt_contains_structured_facts():
    p = build_prompt(facts())
    assert "abstract game" in p
    assert "Actor: RED, learned Q-learning policy." in p
    assert "lateral_move - red moves laterally from host 1 (workstation) to host 2 (server): success" in p
    assert "ATT&CK T1021 Remote Services (Lateral Movement)" in p
    assert "lateral_move 1.42 (chosen), escalate 0.97, wait 0.1" in p
    assert "Target host 2 (server) after the move" in p and "network 0.21" in p
    assert "src_bytes -0.120" in p and "baseline 0.31" in p
    assert "1/4 hosts compromised" in p and "crown jewel host 3 is not compromised" in p
    assert p.endswith("explain why red made this move.")


def test_prompt_for_blue_baseline_and_exploration():
    p = build_prompt(facts(actor="blue", action_id="isolate", source=None, agent="baseline", explored=True,
                           decision_values={"monitor": 3.0, "isolate": 1.0, "wait": 0.0, "patch": 2.0}))  # fmt: skip
    assert "Actor: BLUE, scripted heuristic baseline." in p
    assert "heuristic priorities" in p
    assert "isolate 1 (chosen)" in p  # chosen value appended even when outside the top 3
    assert "exploration/noise pick" in p
    assert "D3FEND D3-NI Network Isolation" in p


def test_prompt_is_stable_for_cache_keys():
    assert build_prompt(facts()) == build_prompt(facts())


# -------------------------------------------------------------------------------------------- template


def test_template_is_deterministic_and_specific():
    a, b = template_rationale(facts()), template_rationale(facts())
    assert a == b
    assert a.startswith("Red moves laterally from host 1 to host 2 (server) and succeeds.")
    assert "Q-values (1.42 vs escalate 0.97)" in a and "src_bytes" in a


def test_template_covers_blue_moves_and_ties():
    t = template_rationale(facts(actor="blue", action_id="monitor", success=False,
                                 decision_values={"monitor": 0.0, "wait": 0.0}))  # fmt: skip
    assert t.startswith("Blue monitors host 2 (server) and finds nothing new.")
    assert "tie-break" in t
    end = template_rationale(facts(done=True, winner="red", action_id="exfiltrate", target=3))
    assert "crown jewel" in end and end.endswith("red wins.")


def test_offline_narrator_needs_no_client_or_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    n = Narrator(offline=True)
    text, meta = n.narrate_one(facts())
    assert meta == {"source": "template"} and text == template_rationale(facts())


# --------------------------------------------------------------------------------------- online, mocked


def test_online_call_shape_parse_and_cache(tmp_path):
    client = fake_client()
    n = Narrator(offline=False, model="claude-haiku-4-5", client=client, cache_path=tmp_path / "n.jsonl",
                 concurrency=1, rpm=0)  # fmt: skip
    text, meta = n.narrate_one(facts())
    assert text == "Red pivots to the server because its network score is low (0.21). Extra."
    assert meta["source"] == "claude" and meta["input_tokens"] == 420
    (call,) = client.messages.calls
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == narrate.SYSTEM_PROMPT
    assert call["messages"] == [{"role": "user", "content": build_prompt(facts())}]
    assert call["max_tokens"] == 200 and "output_config" not in call and "temperature" not in call

    again, meta2 = n.narrate_one(facts())
    assert again == text and meta2["source"] == "cache" and len(client.messages.calls) == 1
    # a fresh narrator reads the disk cache and makes no call
    client2 = fake_client()
    n2 = Narrator(offline=False, model="claude-haiku-4-5", client=client2, cache_path=tmp_path / "n.jsonl")
    assert n2.narrate_one(facts())[0] == text and client2.messages.calls == []


def test_refusal_falls_back_to_template_and_is_labelled():
    n = Narrator(offline=False, client=fake_client(reply=None, stop_reason="refusal"), rpm=0)
    text, meta = n.narrate_one(facts())
    assert text == template_rationale(facts())
    assert meta["source"] == "template:refusal" and meta["refusal_category"] == "cyber"
    assert n.stats.refusals == 1


def test_narrate_many_preserves_order_with_threads():
    n = Narrator(offline=False, client=fake_client(), concurrency=3, rpm=0)
    fs = [facts(turn=i) for i in range(6)]
    out = n.narrate_many(fs)
    assert len(out) == 6 and n.stats.api_calls == 6


def test_newer_models_keep_thinking_on_with_low_effort():
    client = fake_client()
    Narrator(offline=False, model="claude-sonnet-5-5", client=client, rpm=0).narrate_one(facts())
    call = client.messages.calls[0]
    assert call["output_config"] == {"effort": "low"} and "thinking" not in call


def test_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY is not set.*--offline"):
        Narrator(offline=False)


def test_model_resolution(monkeypatch):
    monkeypatch.delenv(narrate.MODEL_ENV, raising=False)
    assert narrate.resolve_model() == "claude-haiku-4-5"
    monkeypatch.setenv(narrate.MODEL_ENV, "claude-sonnet-5-5")
    assert narrate.resolve_model() == "claude-sonnet-5-5"
    assert narrate.resolve_model("claude-opus-5-5") == "claude-opus-5-5"


# ---------------------------------------------------------------------------------------------- parsing


def test_parse_response_variants():
    msg = SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1, output_tokens=2),
                          content=[SimpleNamespace(type="thinking", thinking=""),
                                   SimpleNamespace(type="text", text='  "Rationale: Blue isolates host 7.\n'
                                                                     'Its score is 0.9."  ')])  # fmt: skip
    text, meta = parse_response(msg)
    assert text == "Blue isolates host 7. Its score is 0.9."
    assert meta == {"stop_reason": "end_turn", "input_tokens": 1, "output_tokens": 2}
    empty = SimpleNamespace(stop_reason="max_tokens", usage=None, content=[])
    assert parse_response(empty)[0] is None


def test_clean_text_caps_length():
    assert clean_text("a " * 400).endswith("…")
    assert clean_text("One. Two. Three.") == "One. Two."


def test_cost_estimate_scales_with_calls():
    one = narrate.estimate_cost(["x" * 350], "claude-haiku-4-5")
    two = narrate.estimate_cost(["x" * 350] * 2, "claude-haiku-4-5")
    assert two["calls"] == 2 and two["usd"] == pytest.approx(2 * one["usd"], rel=0.01)
