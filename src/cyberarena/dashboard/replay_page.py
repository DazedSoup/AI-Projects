"""Replay: step through one game on the 3D network, or compare the same probe game at two checkpoints.

Reads ``episodes_enriched.jsonl`` (narrated games) and ``episodes.jsonl`` through a byte-offset index; never
runs agents or calls APIs. Other pages open a game here by setting ``st.session_state["_open_episode"]``.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from cyberarena.dashboard import charts, common
from cyberarena.dashboard import loaders as L
from cyberarena.dashboard import theme as T

ss = st.session_state
SPEEDS = {"0.5×": 1.2, "1×": 0.6, "2×": 0.25}
SINGLE, COMPARE = "Replay a game", "Same game, earlier agent"
SOURCE = {
    "claude": "Narrated by Claude",
    "template": "Narrated offline (template)",
    "mixed": "Narrated: Claude + template",
    "unknown": "Narrated",
    "none": "Not narrated",
}
WHY_WIN = {
    "red": "red exfiltrated data from the crown jewel",
    "blue": "blue held the network until the turn limit",
}

ss.setdefault("turn", 0)
ss.setdefault("playing", False)


def _goto(t: int) -> None:
    ss.turn = t
    ss.playing = False


def _step(delta: int, n: int) -> None:
    ss.turn = max(0, min(n - 1, ss.turn + delta))
    ss.playing = False


def _toggle_play(n: int) -> None:
    if not ss.playing and ss.turn >= n - 1:
        ss.turn = 0
    ss.playing = not ss.playing


def _reset_turn() -> None:
    ss.turn = 0
    ss.playing = False


run = common.current_run()
T.page_header(
    "Replay", "Step through a game move by move: the network after each move, what each side chose and why."
)

enr = common.enriched(run)
eps, evals = common.summary(run)
with st.spinner("Indexing the game log (first time only)…"):
    meta = common.index_meta(run)
catalog = L.episode_catalog(enr, eps, evals, meta)
graph = common.graph(run)

opened = ss.pop("_open_episode", None)
if opened is not None and opened in set(catalog["episode"]):
    ss["rp_mode"] = SINGLE
    ss["rp_kind"] = catalog.loc[catalog["episode"] == opened, "kind"].iloc[0]
    ss["episode"] = opened
    _reset_turn()

if catalog.empty:
    T.empty_state("No games to replay", "This run has no logged games yet.")
    st.stop()

kinds = [k for k in L.GAME_KINDS if (catalog["kind"] == k).any()]
counts = catalog["kind"].value_counts().to_dict()
probes = L.probe_games(catalog)
if ss.get("rp_kind") not in kinds:
    ss["rp_kind"] = kinds[0]

# ------------------------------------------------------------------------------------------------ toolbar
with common.toolbar("replay"):
    mode = st.segmented_control(
        "View", [SINGLE, COMPARE], key="rp_mode", default=SINGLE, required=True,
        help="Same game, earlier agent: the fixed-seed probe game, played by two checkpoints side by side.",
    )  # fmt: skip
    if mode == SINGLE:
        kind = st.selectbox(
            "Game type", kinds, key="rp_kind", format_func=lambda k: f"{L.GAME_KINDS[k]} · {counts.get(k, 0)}",
            on_change=lambda: ss.pop("episode", None), width=200,
            help="Narrated games have rationales, SHAP features and MITRE tags. Probe games replay the same "
            "fixed seed at every checkpoint. Evaluation games are played against a scripted opponent.",
        )  # fmt: skip
        sub = catalog[catalog["kind"] == kind]
        labels = dict(zip(sub["episode"].astype(int), sub["label"], strict=True))
        if ss.get("episode") not in labels:
            ss["episode"] = next(iter(labels))
        episode = st.selectbox(
            "Game", list(labels), format_func=labels.get, key="episode", on_change=_reset_turn
        )
    elif not probes.empty:
        order = list(L.EVAL_MATCHUPS) + [L.HEAD_TO_HEAD]
        matchups = sorted(dict.fromkeys(probes["probe_key"]),
                          key=lambda x: (order.index(x.split("@")[0]) if x.split("@")[0] in order else 99, x))  # fmt: skip
        if ss.get("cmp_matchup") not in matchups:
            ss.pop("cmp_matchup", None)
        mu = st.selectbox("Matchup", matchups, format_func=L.matchup_label, key="cmp_matchup", width=300)
        pm = probes[probes["probe_key"] == mu]
        cps = [int(c) for c in pm["after_episode"]]
        a = st.selectbox(
            "Earlier agent", cps, index=0, key="cmp_a", format_func=lambda c: f"after {c:,} games"
        )
        b = st.selectbox("Later agent", cps, index=len(cps) - 1, key="cmp_b",
                         format_func=lambda c: f"after {c:,} games")  # fmt: skip
    view = st.segmented_control("Network", ["3D", "2D"], key="rp_view", default="3D", required=True)
if mode == SINGLE:
    T.caption(
        f"Reads {'narrated games from <code>episodes_enriched.jsonl</code>' if kind == 'narrated' else 'turn records from <code>episodes.jsonl</code>'}"
        f" in <code>runs/{T.esc(run.name)}/</code>. Pick a game type, then a game; labels say who won, how fast, and "
        "which agents played."
    )
if graph is None:
    T.empty_state(
        "Network layout missing", "<code>graph.json</code> is missing for this run, so it can't be drawn."
    )
    st.stop()


# ================================================================================================ helpers


def outcome(t: dict) -> tuple[str, str]:
    if t["action_id"] == "wait":
        return "no-op", ""
    return ("succeeded", "ok") if t.get("success") else ("failed", "fail")


def mitre_chip(m: dict | None) -> str:
    if not m:
        return ""
    kind = "d3fend" if m.get("framework") == "D3FEND" else "attack"
    return T.chip(f"{m.get('technique_id', '')} {m.get('technique_name', '')}".strip(), kind)


def log_rows(turns: list[dict], upto: int) -> tuple[list[dict], int]:
    rows = []
    for t in reversed(turns[: upto + 1]):
        res, cls = outcome(t)
        why = t.get("rationale") or ("" if t.get("enriched") else "not narrated")
        sents = L.sentences(why) or [why]
        first, rest = sents[0], " ".join(sents[1:])
        tgt, src = t.get("target"), t.get("source")
        target = "—" if tgt is None else (f"{src} → {tgt}" if src is not None and src != tgt else str(tgt))
        chip = mitre_chip(t.get("mitre"))
        rows.append({
            "turn": str(t["turn"]),
            "side": f'<span class="actor" style="--ca-actor:{T.TEAM.get(t["actor"], "#888")}">{T.esc(t["actor"])}</span>',
            "action": T.esc(t["action_id"].replace("_", " ")) + (f" → {T.esc(target)}" if tgt is not None else "")
                      + (' <span class="ca-chip">exploring</span>' if t.get("explored") else "")
                      + f'<div style="font-size:11.5px;margin-top:2px"><span class="{cls}">{res}</span></div>'
                      + (f'<div style="margin-top:3px">{chip}</div>' if chip else ""),
            "why": (f'<details class="ca-more why"><summary>{T.esc(first)}</summary>{T.esc(rest)}</details>'
                    if rest else f'<span class="why">{T.esc(first)}</span>'),
        })  # fmt: skip
    return rows, 0


def first_turn(turns: list[dict], pred) -> int | None:
    for t in turns:
        if pred(t):
            return t["turn"]
    return None


def crown_turn(turns: list[dict]) -> int | None:
    crown = next((n["id"] for n in graph["nodes"] if n["crown_jewel"]), None)
    if crown is None:
        return None
    return first_turn(
        turns, lambda t: any(ns["id"] == crown and ns.get("compromised") for ns in t["node_states"])
    )


def network(turn: dict | None, key: str, height: int, compact: bool = False, ghosts: list[dict] | None = None) -> None:
    if view == "3D":
        T.chart(charts.host_network_3d(graph, turn, height=height, revision=f"{run.name}", ghosts=ghosts), key=key,
                three_d=True)  # fmt: skip
    else:
        T.chart(charts.host_network_2d(graph, turn, height=height, compact=compact, ghosts=ghosts), key=key)


# ================================================================================================ compare mode

if mode == COMPARE:
    if probes.empty:
        T.empty_state(
            "No probe games in this run",
            "Probe games are one fixed-seed game per matchup, replayed at every checkpoint so you can watch the same "
            "situation handled by an earlier and a later agent. Runs with adaptive learning log them "
            '(<code>"probe_game": true</code>); this run predates them.',
        )
        st.stop()
    T.caption(
        "Reads probe games from <code>episodes.jsonl</code>: the same environment seed played at every checkpoint, "
        "so any difference comes from what the agents learned."
    )
    games = {}
    for c in (a, b):
        ep = int(pm[pm["after_episode"] == c]["episode"].iloc[0])
        games[c] = (ep, common.game_turns(run, ep))
    n_max = max(len(t) for _, t in games.values())
    ss.setdefault("cmp_turn", 0)
    ss.cmp_turn = min(ss.cmp_turn, n_max - 1)
    with st.container(key="playbar-cmp", horizontal=True, vertical_alignment="center"):
        st.slider("Turn", 0, max(1, n_max - 1), key="cmp_turn", label_visibility="collapsed", format="Turn %d",
                  help="Both games advance together; a game that already ended stays on its final turn.")  # fmt: skip
    cols = st.columns(2, gap="medium")
    summary_bits = []
    if a == b:
        T.caption("Both sides show the same checkpoint; pick an earlier and a later agent to compare them.")
    for slot, (col, c) in enumerate(zip(cols, (a, b), strict=True)):
        ep, turns = games[c]
        if not turns:
            with col, common.card(f"cmp{slot}"):
                T.empty_state(f"Checkpoint {c:,}", "This probe game has no turn records.")
            continue
        idx = min(ss.cmp_turn, len(turns) - 1)
        t = turns[idx]
        last = turns[-1]
        win = last.get("winner")
        with col, common.card(f"cmp{slot}"):
            T.card_header(
                f"After {c:,} training games",
                f"<span class='ca-mono'>Game {ep}</span> · {T.esc(L.matchup_label(mu))}",
                right=T.badge(f"{win} wins in {len(turns)} turns" if win else f"{len(turns)} turns", win or "",
                              T.TEAM.get(win)),
            )  # fmt: skip
            network(t, key=f"cmp_net_{slot}", height=360, compact=True)
            ended = " · game over" if ss.cmp_turn >= len(turns) - 1 else ""
            T.caption(f"Turn {t['turn']}{ended}: <b style='color:{T.TEAM.get(t['actor'])}'>{t['actor']}</b> "
                      f"{T.esc(L.move_headline(t, graph))} — {outcome(t)[0]}.")  # fmt: skip
            ct = crown_turn(turns)
            summary_bits.append((c, win, len(turns), ct))
    st.html(T.state_key_html())
    if len(summary_bits) == 2:
        (c0, w0, n0, k0), (c1, w1, n1, k1) = summary_bits

        def desc(w, n, k):
            s = f"{w} won in {n} turns" if w else f"it lasted {n} turns"
            return s + (
                f", with red on the crown jewel by turn {k}"
                if k is not None
                else ", and red never reached the crown jewel"
            )

        T.takeaway(f"Same seed, same opponent: after {c0:,} games {desc(w0, n0, k0)}; after {c1:,} games "
                   f"{desc(w1, n1, k1)}.", learning=True)  # fmt: skip
    st.stop()


# ================================================================================================ single game

turns = common.game_turns(run, int(episode))
if not turns:
    T.empty_state(
        f"Game {episode} has no turn records", "The summary lists it, but the log doesn't contain it."
    )
    st.stop()
n = len(turns)
last = turns[-1]
row = catalog[catalog["episode"] == episode].iloc[0]

# ------------------------------------------------------------------------------------------------ game header
win = last.get("winner") if last.get("done") else None
src = L.rationale_source(turns)
ev = row.get("evasion") if "evasion" in row else None
head = [
    T.badge(f"{win} wins" if win else "unfinished", win or "", T.TEAM.get(win)),
    T.badge(f"{n} turns"),
    T.badge(L.matchup_label(last.get("matchup"), None if ev is None or pd.isna(ev) else ev)),
]
if last.get("after_episode") is not None:
    head.append(T.badge(f"checkpoint {int(last['after_episode']):,}"))
elif last.get("phase") == "train":
    head.append(T.badge("during training"))
if row["probe_game"]:
    head.append(T.badge("probe game", "learn", T.VIOLET))
head.append(T.badge(SOURCE[src]))
st.html(
    f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:4px 0 0">'
    f'<h2 style="margin:0;font-size:20px;font-weight:640;letter-spacing:-.01em">Game '
    f'<span class="ca-mono">{int(episode)}</span></h2><div class="ca-badges">{"".join(head)}</div></div>'
)

# keyboard: ← / → step, space plays or pauses, Home / End jump (ignored while typing or on a focused slider)
st.html(
    """<script>
(() => {
  const doc = window.parent && window.parent.document ? window.parent.document : document;
  if (doc.__caReplayKeys) return;
  doc.__caReplayKeys = true;
  doc.addEventListener('keydown', (e) => {
    const t = e.target;
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable ||
              (t.getAttribute && t.getAttribute('role') === 'slider'))) return;
    const map = {ArrowLeft: 'rp_prev', ArrowRight: 'rp_next', ' ': 'rp_play', Home: 'rp_first', End: 'rp_last'};
    const k = map[e.key];
    if (!k) return;
    const b = doc.querySelector('.st-key-' + k + ' button');
    if (!b) return;
    e.preventDefault();
    b.click();
  });
})();
</script>""",
    unsafe_allow_javascript=True,
)


def ghosts_for(turn: dict, k: int = 3) -> list[dict]:
    """The top-``k`` candidate moves the agent considered but didn't pick (v4 runs only)."""
    out = []
    for c in L.turn_candidates(turn):
        if c["chosen"]:
            continue
        rank = L.turn_candidates(turn).index(c) + 1
        out.append({**c, "actor": turn["actor"], "rank": rank, "label": L.move_phrase(c["action"], c["target"], graph)})
        if len(out) >= k:
            break
    return out


def runner_up_strip(cmp: dict, actor: str) -> None:
    margin = cmp["margin"]
    lead = "ahead of" if margin >= 0 else "behind"
    facts = "".join(
        f'<div class="ca-ru-fact"><span class="k">{T.esc(f["label"])}</span>'
        f'<span class="v"><b>{T.esc(f["chosen"])}</b><span class="vs">vs</span>{T.esc(f["runner_up"])}</span></div>'
        for f in cmp["facts"]
    ) or '<div class="ca-ru-fact"><span class="k">The two moves differ in nothing the log records.</span></div>'
    how = ("ranked by the agent's own attributions" if cmp["ranked_by_attribution"]
           else "from the logged state before the move")  # fmt: skip
    st.html(
        f'<div class="ca-ru" style="--ca-actor:{T.TEAM.get(actor, "#888")}">'
        f'<div class="ca-ru-head"><span class="ca-eyebrow" style="margin:0">Chosen vs runner-up</span>'
        f'<span class="ca-ru-margin">Q margin <b>{margin:+.3f}</b></span></div>'
        f'<div class="ca-ru-moves"><span class="c">{T.esc(cmp["chosen_label"])}</span>'
        f'<span class="ca-muted">{lead}</span><span class="r">{T.esc(cmp["runner_label"])}</span></div>'
        f'<div class="ca-ru-facts">{facts}</div><div class="ca-ru-note">Biggest differences, {how}.</div></div>'
    )


def why_card(turn: dict, idx: int) -> None:
    actor = turn["actor"]
    res, _ = outcome(turn)
    v4 = bool(L.turn_candidates(turn))
    tags = [T.badge(actor, actor, T.TEAM.get(actor)), T.badge(res)]
    if turn.get("explored"):
        tags.append(T.badge("exploration pick"))
    if turn.get("agent") in ("heuristic", "random"):
        tags.append(T.badge("scripted opponent"))
    chip = mitre_chip(turn.get("mitre"))
    T.card_header("Why this move", f"Turn {turn['turn']} of {n}")
    st.html(
        f'<div class="ca-why-head"><span class="ca-why-action">{T.esc(L.move_headline(turn, graph).capitalize())}</span>'
        f'<div class="ca-badges">{"".join(tags)}{chip}</div></div>'
    )
    if turn.get("rationale"):
        sents = L.sentences(turn["rationale"])
        lead, more = " ".join(sents[:2]), " ".join(sents[2:])
        extra_html = (f'<details class="ca-more"><summary>Full reasoning</summary><p>{T.esc(more)}</p></details>'
                      if more else "")  # fmt: skip
        st.html(f'<p class="ca-why-text">{T.esc(lead)}</p>{extra_html}')
    else:
        T.caption("This game isn't narrated, so there is no written rationale, attribution or SHAP breakdown. "
                  "Pick a game from <b>Narrated</b> to see them.")  # fmt: skip
    facts = L.adaptation_facts(turn)
    if facts:
        chips = []
        for m in facts["read"] or list(facts["evasion"]):
            if m in facts["evasion"] and actor == "red":
                chips.append(T.chip(f"{m} evasion {facts['evasion'][m]:.1f}", "learn"))
            if m in facts["versions"]:
                sc = facts["scores"].get(m)
                chips.append(T.chip(f"{m} detector v{facts['versions'][m]}" + (f" · scored {sc:.2f}" if sc is not None else ""),
                                    "learn"))  # fmt: skip
        for m in facts["retrained"]:
            chips.append(T.chip(f"{m} detector retrained since the last move", "learn strong"))
        for m, x in facts["caught_rate"].items():
            if m in (facts["read"] or facts["evasion"]):
                chips.append(T.chip(f"red caught on {m} {x:.0%} of the time lately"))
        fleet = " · ".join(f"{m} v{v}" for m, v in facts["versions"].items())
        if chips or fleet:
            st.html(
                '<div class="ca-adapt"><div style="display:flex;flex-direction:column;gap:6px">'
                '<div class="ca-eyebrow" style="color:#8E7CFF;margin:0">Adaptation this turn</div>'
                f'<div class="ca-badges">{"".join(chips) or T.esc("no detector read on this move")}</div>'
                + (f'<div style="opacity:.7;font-size:12px">Detectors in play: {T.esc(fleet)}</div>' if fleet else "")
                + "</div></div>"
            )  # fmt: skip
    if turn.get("done"):
        st.html(f'<div class="ca-adapt" style="background:transparent;border-color:{T.TEAM.get(win, "#888")}">'
                f"<b>Game over</b> — {T.esc(WHY_WIN.get(win, (win or '?') + ' wins'))}.</div>")  # fmt: skip
    if not turn.get("decision_values"):
        T.caption("No decision values were logged for this turn.")
        return
    if v4:
        web = L.decision_web_v4(turn, graph)
        T.chart(charts.decision_web_v4_figure(web, actor, height=440), key="rp_web")
        notes = [(f"Decision web: the centre is the chosen move, #{web['chosen_rank']} of the {web['n_candidates']} "
                 "moves logged; spokes go to the next-best moves, as thick as their Q-value relative to the best shown.")]  # fmt: skip
        if web["attribution"]:
            meta = web["attribution_meta"] or {}
            att = turn.get("agent_attribution") or {}
            ck = att.get("checkpoint_after")
            base = {"candidate_mean": "the average of the moves it considered", "zeros": "an all-zero input",
                    "mean": "the average input"}.get(str(meta.get("baseline")), "a baseline input")  # fmt: skip
            notes.append("Outer ring: the inputs that moved this move's value most, compared with " + base
                         + " (integrated gradients" + (f" on the checkpoint after {int(ck):,} games" if ck is not None else "")
                         + ").")  # fmt: skip
        elif turn.get("enriched"):
            notes.append("No agent attribution was logged for this turn yet.")
        if not web["chosen_is_best"]:
            notes.append("The chosen move was not the top-valued one: an exploration pick.")
        T.caption(" ".join(notes))
        before = turns[idx - 1]["node_states"] if idx > 0 else None
        cmp = L.chosen_vs_runner_up(turn, before, graph)
        if cmp:
            runner_up_strip(cmp, actor)
        if web["detector"]:
            st.html('<p class="ca-card-title" style="font-size:13px;margin-top:4px">Detector reading this turn (SHAP)</p>'
                    '<p class="ca-card-sub">What pushed the sensor\'s score: orange toward malicious, green toward '
                    "benign. Separate from the agent's own reasoning above.</p>")  # fmt: skip
            T.chart(charts.detector_shap_figure(web["detector"][:6]), key="rp_shap")
        with st.expander("Every move it weighed"):
            T.chart(charts.q_bar_figure(turn["decision_values"], L.move_key(turn["action_id"], turn.get("target")),
                                        actor, "Q-value"), key="rp_qbar")  # fmt: skip
            if web["attribution"]:
                st.dataframe(pd.DataFrame(web["attribution"])[["phrase", "name", "value", "attribution"]], hide_index=True,
                             column_config={"phrase": "Input", "name": "Feature",
                                            "value": st.column_config.NumberColumn("Value", format="%.3f"),
                                            "attribution": st.column_config.NumberColumn("Attribution", format="%+.4f")})  # fmt: skip
        return
    web = L.decision_web(turn)
    T.chart(charts.decision_web_figure(web, actor, height=360), key="rp_web")
    notes = []
    if web["value_kind"] == "Q-value":
        notes.append("Decision web: the centre is the chosen action type; spokes to the other options are as thick as "
                     "their learned value (Q-value) relative to the best. This agent picks an action type and a fixed "
                     "rule picks the host.")  # fmt: skip
    else:
        notes.append(f"{actor.capitalize()} is the scripted opponent here, so spokes show its fixed rule priorities, "
                     "not learned values.")  # fmt: skip
    if web["has_shap"]:
        notes.append("Outer ring: the detector features that moved its score most (SHAP).")
    if not web["chosen_is_best"]:
        notes.append("The chosen move was not the top-valued one: an exploration pick.")
    T.caption(" ".join(notes))
    with st.expander("Every option it weighed"):
        T.chart(charts.q_bar_figure(turn["decision_values"], turn["action_id"], actor, web["value_kind"]), key="rp_qbar")
        if web["features"]:
            df = pd.DataFrame(web["features"])[["model", "node", "name", "shap", "value", "raw"]]
            st.dataframe(df, hide_index=True, column_config={
                "model": "Detector", "node": "Host", "name": "Feature",
                "shap": st.column_config.NumberColumn("SHAP", format="%+.3f"),
                "value": st.column_config.NumberColumn("Scaled value", format="%.3f"), "raw": "Raw value"})  # fmt: skip


def game_view() -> None:
    """Playbar, network, reasoning and move log. A fragment: stepping and autoplay rerun only this part, so the
    page around it (catalog, pickers) isn't rebuilt on every frame."""
    playing = bool(ss.playing)
    speed_now = ss.get("rp_speed", "1×")
    if playing != ss.get("_frag_playing") or speed_now != ss.get("_frag_speed"):
        ss["_frag_playing"], ss["_frag_speed"] = playing, speed_now
        if ss.get("_frag_started"):
            st.rerun()  # the fragment's timer depends on play state and speed: re-register it
    ss["_frag_started"] = True
    if playing:
        now = time.monotonic()
        if now - ss.get("_tick", 0.0) >= SPEEDS[speed_now] * 0.8:
            if ss.turn < n - 1:
                ss.turn += 1
            else:
                ss.playing = False
            ss["_tick"] = now
    ss.turn = max(0, min(n - 1, ss.turn))
    idx = ss.turn
    turn = turns[idx]
    with st.container(key="playbar-main", horizontal=True, vertical_alignment="center", gap="small"):
        st.button(":material/first_page:", key="rp_first", on_click=_goto, args=(0,), help="First turn (Home)")
        st.button(":material/chevron_left:", key="rp_prev", on_click=_step, args=(-1, n), help="Previous turn (←)")
        st.button(":material/pause: Pause" if ss.playing else ":material/play_arrow: Play", key="rp_play",
                  on_click=_toggle_play, args=(n,), type="primary", width=104, help="Play or pause (space)")  # fmt: skip
        st.button(":material/chevron_right:", key="rp_next", on_click=_step, args=(1, n), help="Next turn (→)")
        st.button(":material/last_page:", key="rp_last", on_click=_goto, args=(n - 1,), help="Last turn (End)")
        if n > 1:
            st.slider("Turn", 0, n - 1, key="turn", label_visibility="collapsed", format="Turn %d",
                      help="Turn within the game (0-based). Red and blue alternate.")  # fmt: skip
        st.html(f'<div class="ca-mono" style="white-space:nowrap;font-size:12.5px;opacity:.75">{idx + 1} / {n}</div>',
                width="content")  # fmt: skip
        st.segmented_control("Speed", list(SPEEDS), key="rp_speed", default="1×", required=True,
                             label_visibility="collapsed")  # fmt: skip
    T.caption("Keys: <b>←</b> / <b>→</b> step · <b>space</b> play or pause · <b>Home</b> / <b>End</b> first or last turn.")
    ghosts = ghosts_for(turn)
    left, right = st.columns([1.2, 1], gap="medium")
    with left, common.card("network"):
        T.card_header(
            f"Network after turn {turn['turn']}",
            "Zones are stacked from the Internet-facing DMZ down to the crown jewel. Shape and colour show each "
            "host's state; the solid path is this turn's move (dashed if it failed)"
            + ("; faint numbered rings are the next-best moves the agent considered" if ghosts else "")
            + ". Drag to rotate.",
        )
        network(turn, key="rp_net", height=470, ghosts=ghosts)
        st.html(T.state_key_html())
    with right, common.card("why"):
        why_card(turn, idx)
    with left, common.card("log"):  # under the network, so the two columns stay balanced
        T.card_header("Move log", "Newest first, up to the current turn. Click a reason to read the rest of it.")
        rows, cur = log_rows(turns, idx)
        st.html(T.table([("turn:num", "Turn"), ("side", "Side"), ("action", "Move · result · technique"), ("why", "Why")],
                        rows, height=520, current=cur))  # fmt: skip


st.fragment(game_view, run_every=SPEEDS[ss.get("rp_speed", "1×")] if ss.playing else None)()
