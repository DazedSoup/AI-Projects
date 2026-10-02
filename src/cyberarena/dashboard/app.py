"""cyberarena replay dashboard (simulation only; reads run logs, never runs agents or calls APIs).

Launch:  .venv/Scripts/python.exe -m streamlit run src/cyberarena/dashboard/app.py
Runs are discovered under ``cyberarena.config.RUNS_DIR`` (override the root with ``CYBERARENA_ROOT``).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from cyberarena.config import RUNS_DIR
from cyberarena.dashboard import charts
from cyberarena.dashboard import loaders as L

st.set_page_config(page_title="cyberarena replay", layout="wide")

SPEEDS = {"slow": 1.2, "normal": 0.6, "fast": 0.25}
BADGE = {
    "claude": ("Rationale: written by Claude", "green"),
    "template": ("Rationale: offline template", "orange"),
    "mixed": ("Rationale: mixed Claude / template", "violet"),
    "unknown": ("Rationale: source unknown", "gray"),
    "none": ("Not enriched: no rationale / SHAP / MITRE", "gray"),
}

ss = st.session_state
ss.setdefault("turn", 0)
ss.setdefault("playing", False)


def _reset_turn() -> None:
    ss.turn = 0
    ss.playing = False


def _change_run() -> None:
    ss.pop("episode", None)
    _reset_turn()


def _step(delta: int, n: int) -> None:
    ss.turn = max(0, min(n - 1, ss.turn + delta))
    ss.playing = False


def _goto(t: int) -> None:
    ss.turn = t
    ss.playing = False


def _toggle_play(n: int) -> None:
    if not ss.playing and ss.turn >= n - 1:
        ss.turn = 0
    ss.playing = not ss.playing


# ------------------------------------------------------------------------------------------------ sidebar: data
st.sidebar.title("cyberarena")
st.sidebar.caption("Simulation-only red/blue replay")
runs = L.list_runs(RUNS_DIR)
if not runs:
    st.info(f"No runs found under `{RUNS_DIR}`. Train one first (Phase 3).")
    st.stop()
run_dir = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name, on_change=_change_run)
run_key = str(run_dir)

summary_eps, summary_evals = L.cached_summary(run_key)
enriched = L.cached_enriched(run_key)
catalog = L.episode_catalog(enriched, summary_eps, summary_evals)
if catalog.empty:
    st.warning("This run has no logged episodes.")
    st.stop()

labels = {int(r.episode): ("★ " if r.enriched else "") + r.label for r in catalog.itertuples(index=False)}
episode = st.sidebar.selectbox(
    f"Episode ({int(catalog['enriched'].sum())} enriched first, {len(catalog)} logged)",
    list(labels),
    format_func=labels.get,
    on_change=_reset_turn,
    key="episode",
)

if episode in enriched:
    turns = enriched[episode]
else:
    big = Path(run_dir) / L.EPISODES_FILE
    if not big.exists():
        st.warning(f"`{L.EPISODES_FILE}` is missing, so only enriched episodes can be replayed.")
        st.stop()
    stat = big.stat()
    with st.spinner("Indexing episodes.jsonl (first time only)…"):
        turns = L.cached_episode(run_key, int(episode), stat.st_size, stat.st_mtime_ns)
if not turns:
    st.warning(f"Episode {episode} has no turn records in this run.")
    st.stop()
n_turns = len(turns)

# ------------------------------------------------------------------------------------------------ sidebar: replay
if ss.pop("_advance", False):
    ss.turn = ss.turn + 1
ss.turn = max(0, min(n_turns - 1, ss.turn))

st.sidebar.divider()
st.sidebar.subheader("Replay")
if n_turns > 1:
    st.sidebar.slider("Turn", 0, n_turns - 1, key="turn", help="Turn index within the episode (0-based)")
c1, c2, c3, c4 = st.sidebar.columns(4)
c1.button("⏮", on_click=_goto, args=(0,), help="First turn", width="stretch")
c2.button("◀", on_click=_step, args=(-1, n_turns), help="Step back", width="stretch")
c3.button("▶", on_click=_step, args=(1, n_turns), help="Step forward", width="stretch")
c4.button("⏭", on_click=_goto, args=(n_turns - 1,), help="Last turn", width="stretch")
st.sidebar.button(
    "⏸ Pause" if ss.playing else "▶ Play",
    on_click=_toggle_play,
    args=(n_turns,),
    width="stretch",
    type="primary",
)
speed = st.sidebar.select_slider("Play speed", list(SPEEDS), value="normal")

idx = ss.turn
turn = turns[idx]
graph = L.cached_graph(run_key) if (Path(run_dir) / L.GRAPH_FILE).exists() else None

# ------------------------------------------------------------------------------------------------ header
last = turns[-1]
agents = {t["actor"]: t.get("agent") or "?" for t in turns}
src = L.rationale_source(turns)
st.title(f"Episode {episode}")
meta = [
    f"**{last.get('phase') or '?'}**"
    + (f" @ ep {last['after_episode']}" if last.get("after_episode") else ""),
    f"matchup `{last.get('matchup') or '?'}`",
    f"red: {agents.get('red', '?')} · blue: {agents.get('blue', '?')}",
    f"{n_turns} turns",
]
if last.get("done"):
    meta.append(f"winner: **{last.get('winner')}**")
st.markdown(" · ".join(meta))
label, colour = BADGE[src]
st.badge(label, color=colour)

actor = turn["actor"]
outcome = "no-op" if turn["action_id"] == "wait" else "succeeded" if turn.get("success") else "failed"
tgt = turn.get("target")
where = (
    ""
    if tgt is None
    else f" on host {tgt}"
    if turn.get("source") in (None, tgt)
    else (f" from host {turn['source']} to host {tgt}")
)
st.markdown(
    f"#### Turn {turn['turn']} · :{'red' if actor == 'red' else 'blue'}[{actor}] **{turn['action_id']}**"
    f"{where} — {outcome}" + (" · *exploration pick*" if turn.get("explored") else "")
)
if turn.get("done"):
    st.success(f"Game over: **{turn.get('winner')}** wins on this turn.")

# ------------------------------------------------------------------------------------------------ graph + web
left, right = st.columns([1.15, 1])
with left:
    st.subheader("Host graph")
    if graph is None:
        st.info("`graph.json` missing for this run.")
    else:
        st.plotly_chart(charts.host_graph(graph, turn), key="host_graph", config={"displayModeBar": False})
        st.caption(
            "State after this turn. Shape and colour both encode state; ★ = crown jewel (gold ring). "
            "Ring/arrow = this turn's move in the actor's colour (dotted arrow = failed)."
        )

with right:
    st.subheader("Decision web")
    web = L.decision_web(turn)
    if not turn.get("decision_values"):
        st.info("No decision values logged for this turn.")
    else:
        st.plotly_chart(
            charts.decision_web_figure(web, actor), key="decision_web", config={"displayModeBar": False}
        )
        notes = []
        if web["value_kind"] != "Q-value":
            notes.append(
                f"**{actor} is a {web['agent']} baseline:** spoke widths are *heuristic priorities*, "
                "not Q-values."
            )
        else:
            notes.append("Spoke width ∝ Q-value (min–max scaled over this turn's candidates).")
        if not web["chosen_is_best"]:
            notes.append("The chosen action was **not** the top-valued one (exploration / noise pick).")
        if web["has_shap"]:
            cls = "; ".join(
                f"{c['model']} on host {c['node']}: base {c['base_value']} → {c['output']}"
                for c in web["classifiers"]
            )
            notes.append(
                f"Outer ring: top SHAP features of the detector read this turn ({cls}); width ∝ |SHAP|."
            )
        elif not web["enriched"]:
            notes.append("Outer ring empty: this episode is **not enriched** (no SHAP).")
        else:
            notes.append("Outer ring empty: no detector was read on this turn.")
        st.caption(" ".join(notes))

# ------------------------------------------------------------------------------------------------ move log
st.subheader("Move log")
log = L.move_log(turns, idx).iloc[::-1].reset_index(drop=True)
event = st.dataframe(
    log,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key=f"log-{episode}-{idx}",
    height=min(38 + 35 * len(log), 330),
    column_config={
        "turn": st.column_config.NumberColumn(width="small"),
        "actor": st.column_config.TextColumn(width="small"),
        "ok": st.column_config.TextColumn(width="small"),
        "reward": st.column_config.NumberColumn(format="%.3f", width="small"),
        "rationale": st.column_config.TextColumn(width="large"),
    },
)
rows = getattr(getattr(event, "selection", None), "rows", None) or []
detail_pos = int(log.loc[rows[0], "turn"]) if rows else int(turn["turn"])
detail = next((t for t in turns if t["turn"] == detail_pos), turn)
with st.expander(
    f"Turn {detail['turn']} details — {detail['actor']} {detail['action_id']} "
    "(select a row above to inspect another turn)",
    expanded=True,
):
    dweb = L.decision_web(detail)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**Top {dweb['value_kind']}s**")
        if detail.get("decision_values"):
            st.plotly_chart(
                charts.q_bar_figure(
                    detail["decision_values"], detail["action_id"], detail["actor"], dweb["value_kind"]
                ),
                key="qbar",
                config={"displayModeBar": False},
            )
        else:
            st.caption("none logged")
        m = detail.get("mitre")
        if m:
            st.markdown(
                f"**{m.get('framework', '')} {m.get('version', '')}**: `{m.get('technique_id')}` "
                f"{m.get('technique_name')} — *{m.get('tactic')}*"
            )
        elif detail["enriched"]:
            st.caption("MITRE: no mapping for this action (e.g. red `wait`, some blue actions).")
        else:
            st.caption("MITRE: not enriched.")
    with d2:
        st.markdown("**Rationale**")
        if detail.get("rationale"):
            meta = detail.get("rationale_meta") or {}
            st.write(detail["rationale"])
            st.caption(
                f"source: {meta.get('source', 'unknown')}"
                + (f" · model {meta['model']}" if meta.get("model") else "")
            )
        else:
            st.caption("not enriched")
        st.markdown("**SHAP features**")
        if dweb["features"]:
            st.dataframe(
                pd.DataFrame(dweb["features"])[["model", "node", "name", "shap", "value", "raw"]],
                hide_index=True,
            )
        else:
            st.caption("none (not enriched, or no detector read on this turn)")

# ------------------------------------------------------------------------------------------------ win rate
st.subheader("Win rate over training")
curve = L.eval_curve(summary_evals)
if curve.empty and summary_eps.empty:
    st.info("`summary.jsonl` missing or empty.")
else:
    window = st.slider("Rolling window (head-to-head episodes)", 20, 300, 100, step=10, key="window")
    h2h = L.rolling_head_to_head(summary_eps, window)
    st.plotly_chart(
        charts.win_rate_figure(curve, h2h, window), key="win_rate", config={"displayModeBar": False}
    )
    if not curve.empty:
        n = int(curve["n"].iloc[0])
        half = round(100 * float(((curve["ci_high"] - curve["ci_low"]) / 2).max()))
        st.caption(
            f"Each eval point is the learned agent's win rate over n={n} greedy episodes against the "
            "heuristic baseline (same env seeds at every checkpoint), with a 95% Wilson interval. With n="
            f"{n} the interval is up to ±{half} points, so moves between neighbouring checkpoints that stay "
            "inside each other's bars are not distinguishable from noise. The dashed line is red's rolling "
            "win rate in training learned-vs-learned episodes (ε-greedy, so not directly comparable)."
        )
        with st.expander("Eval table"):
            st.dataframe(curve.drop(columns=["matchup"]), hide_index=True)

# ------------------------------------------------------------------------------------------------ autoplay
if ss.playing:
    if ss.turn < n_turns - 1:
        time.sleep(SPEEDS[speed])
        ss["_advance"] = True
        st.rerun()
    else:
        ss.playing = False
        st.rerun()
