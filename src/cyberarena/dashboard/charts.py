"""Plotly figure builders. Pure functions of loader output; no Streamlit calls here.

Palette: one set of mid-luminance hexes chosen to read on both the light and dark Streamlit themes
(figure backgrounds are transparent and text inherits the Streamlit theme). Team colours (red/blue) were run
through the dataviz palette validator in both modes. Node states use five colours *plus* five marker
symbols, because no five-hue set clears colour-blind separation on its own; the symbol is the primary cue.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from cyberarena.dashboard.loaders import NODE_STATES, node_state

TEAM = {"red": "#e05252", "blue": "#3380dd"}
NEUTRAL = "#898781"  # muted ink, legible on both surfaces
EDGE = "rgba(137,135,129,0.45)"
GRID = "rgba(137,135,129,0.25)"
STATE_COLOR = {
    "clean": "#9a9890",
    "patched": "#1a9e8f",
    "compromised": "#e0682e",
    "detected": "#e3a008",
    "isolated": "#8a7be0",
}
STATE_SYMBOL = {
    "clean": "circle",
    "patched": "hexagon",
    "compromised": "diamond",
    "detected": "triangle-up",
    "isolated": "square",
}
CROWN = "#c98500"
SHAP_POS = "#e0682e"  # pushes the classifier towards "malicious"
SHAP_NEG = "#1a9e8f"  # pushes towards "benign"

_BASE_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
    "hoverlabel": {"align": "left"},
}


def _hidden_axes(fig: go.Figure, xr=None, yr=None) -> None:
    ax = {"visible": False, "showgrid": False, "zeroline": False, "fixedrange": True}
    fig.update_xaxes(**ax, range=xr)
    fig.update_yaxes(**ax, range=yr)


# ------------------------------------------------------------------------------------------------ host graph


def host_graph(graph: dict, turn: dict | None) -> go.Figure:
    """Hosts at fixed positions, coloured+shaped by state after ``turn``; the move of ``turn`` highlighted."""
    pos = {n["id"]: (n["x"], n["y"]) for n in graph["nodes"]}
    states = {ns["id"]: ns for ns in (turn or {}).get("node_states", [])}
    fig = go.Figure()

    ex, ey = [], []
    for u, v in graph["edges"]:
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig.add_trace(
        go.Scatter(
            x=ex, y=ey, mode="lines", line={"color": EDGE, "width": 1.5}, hoverinfo="skip", showlegend=False
        )
    )

    # current move
    src, tgt = (turn or {}).get("source"), (turn or {}).get("target")
    actor = (turn or {}).get("actor", "red")
    colour = TEAM.get(actor, NEUTRAL)
    if turn and tgt is not None and tgt in pos:
        if src is not None and src in pos and src != tgt:
            (x0, y0), (x1, y1) = pos[src], pos[tgt]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line={"color": colour, "width": 5, "dash": "solid" if turn.get("success") else "dot"},
                )
            )
            fig.add_annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.4,
                arrowwidth=2.5,
                arrowcolor=colour,
                standoff=14,
                text="",
            )
        fig.add_trace(
            go.Scatter(
                x=[pos[tgt][0]],
                y=[pos[tgt][1]],
                mode="markers",
                hoverinfo="skip",
                name=f"{actor} move target",
                marker={
                    "symbol": "circle-open",
                    "size": 40,
                    "color": colour,
                    "line": {"width": 3, "color": colour},
                },
            )
        )

    # nodes: one trace per state so the legend doubles as a key (symbol + colour)
    by_state: dict[str, list[dict]] = {s: [] for s in NODE_STATES}
    for n in graph["nodes"]:
        ns = states.get(n["id"], {})
        by_state[node_state(ns)].append({**n, "ns": ns})
    for s in NODE_STATES:
        nodes = by_state[s]
        if not nodes:
            continue
        hover = []
        for n in nodes:
            ns = n["ns"]
            flags = [k for k in ("compromised", "detected", "isolated", "patched") if ns.get(k)]
            scores = ns.get("scores") or {}
            sc = "<br>".join(f"{m}: {v:.2f}" for m, v in scores.items())
            hover.append(
                f"<b>host {n['id']}</b> · {n['role']}{' · CROWN JEWEL' if n['crown_jewel'] else ''}"
                f"<br>state: {s}{' (' + ', '.join(flags) + ')' if flags else ''}"
                f"<br>red privilege: {ns.get('privilege', 0)}<br><i>detector scores</i><br>{sc}"
            )
        fig.add_trace(
            go.Scatter(
                x=[pos[n["id"]][0] for n in nodes],
                y=[pos[n["id"]][1] for n in nodes],
                mode="markers+text",
                name=s,
                text=[f"{n['id']}" + (" ★" if n["crown_jewel"] else "") for n in nodes],
                textposition="top center",
                hovertext=hover,
                hoverinfo="text",
                marker={
                    "symbol": STATE_SYMBOL[s],
                    "size": [30 if n["crown_jewel"] else 22 for n in nodes],
                    "color": STATE_COLOR[s],
                    "line": {
                        "width": [4 if n["crown_jewel"] else 1 for n in nodes],
                        "color": [CROWN if n["crown_jewel"] else "rgba(0,0,0,0.35)" for n in nodes],
                    },
                },
            )
        )
    for role, x in _role_columns(graph).items():
        fig.add_annotation(x=x, y=1.06, text=role, showarrow=False, font={"color": NEUTRAL, "size": 11})
    fig.update_layout(**_BASE_LAYOUT, height=440, legend={"orientation": "h", "y": -0.02, "x": 0})
    _hidden_axes(fig, xr=[-0.03, 1.03], yr=[0.0, 1.1])
    return fig


def _role_columns(graph: dict) -> dict[str, float]:
    xs: dict[str, list[float]] = {}
    for n in graph["nodes"]:
        role = "crown jewel" if n["crown_jewel"] else n["role"]
        xs.setdefault(role, []).append(n["x"])
    return {r: sum(v) / len(v) for r, v in xs.items()}


# ------------------------------------------------------------------------------------------------ decision web


def decision_web_figure(web: dict, actor: str) -> go.Figure:
    """Centre = chosen action; ring 1 = other candidates (width ∝ value); ring 2 = SHAP features (∝ |SHAP|)."""
    fig = go.Figure()
    colour = TEAM.get(actor, NEUTRAL)
    kind = web["value_kind"]

    def spoke(x, y, weight, col, opacity=0.75, stop=0.0):
        # stop: data-unit gap before the end point so a spoke doesn't run through a hollow label circle
        r = math.hypot(x, y) or 1.0
        k = (r - stop) / r
        fig.add_trace(
            go.Scatter(
                x=[0, x * k],
                y=[0, y * k],
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
                opacity=opacity,
                line={"color": col, "width": 1 + 9 * weight},
            )
        )

    acts = web["actions"]
    ax, ay, at, ah = [], [], [], []
    for i, a in enumerate(acts):
        ang = math.pi / 2 + 2 * math.pi * i / max(1, len(acts))
        x, y = 1.2 * math.cos(ang), 1.2 * math.sin(ang)
        spoke(x, y, a["weight"], colour, stop=0.3)
        ax.append(x)
        ay.append(y)
        at.append(f"{a['action']}<br>{a['value']:.3g}")
        ah.append(f"<b>{a['action']}</b><br>{kind}: {a['value']:.4g}<br>relative weight: {a['weight']:.2f}")
    if acts:
        fig.add_trace(
            go.Scatter(
                x=ax,
                y=ay,
                mode="markers+text",
                text=at,
                textposition="middle center",
                hovertext=ah,
                hoverinfo="text",
                name=f"candidate actions ({kind})",
                textfont={"size": 11},
                marker={"size": 54, "color": "rgba(0,0,0,0)", "line": {"color": colour, "width": 2}},
            )
        )

    feats = web["features"]
    fx, fy, ft, fh, fc = [], [], [], [], []
    for i, f in enumerate(feats):
        ang = math.pi / 2 + math.pi / max(1, len(feats)) + 2 * math.pi * i / max(1, len(feats))
        x, y = 2.3 * math.cos(ang), 2.3 * math.sin(ang)
        col = SHAP_POS if f["shap"] >= 0 else SHAP_NEG
        spoke(x, y, f["weight"], col, opacity=0.55)
        fx.append(x)
        fy.append(y)
        fc.append(col)
        name = f["name"] if len(f["name"]) <= 22 else f["name"][:21] + "…"
        ft.append(f"{name}<br>{f['shap']:+.3f}")
        raw = f" (raw {f['raw']})" if f.get("raw") is not None else ""
        fh.append(
            f"<b>{f['name']}</b><br>{f['model']} detector on host {f['node']}<br>SHAP {f['shap']:+.4f}"
            f"<br>scaled value {f['value']}{raw}"
        )
    if feats:
        fig.add_trace(
            go.Scatter(
                x=fx,
                y=fy,
                mode="markers+text",
                text=ft,
                hovertext=fh,
                hoverinfo="text",
                textposition=["top center" if y >= 0 else "bottom center" for y in fy],
                textfont={"size": 10},
                showlegend=False,
                marker={"size": 12, "color": fc, "line": {"width": 1, "color": "rgba(0,0,0,0.3)"}},
            )
        )
        for col, lab in ((SHAP_POS, "SHAP > 0 (towards malicious)"), (SHAP_NEG, "SHAP < 0 (towards benign)")):
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="markers", name=lab, marker={"size": 10, "color": col})
            )

    cv = web["chosen_value"]
    centre = f"<b>{web['chosen']}</b>" + (f"<br>{cv:.3g}" if cv is not None else "")
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers+text",
            text=[centre],
            textposition="middle center",
            textfont={"color": "#ffffff", "size": 12},
            name="chosen action",
            hovertext=[f"chosen: {web['chosen']}<br>{kind}: {cv}"],
            hoverinfo="text",
            marker={"size": 74, "color": colour, "line": {"width": 0}},
        )
    )
    fig.update_layout(**_BASE_LAYOUT, height=440, legend={"orientation": "h", "y": -0.02, "x": 0})
    _hidden_axes(fig, xr=[-3.0, 3.0], yr=[-2.9, 2.9])
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ------------------------------------------------------------------------------------------------ win rate


def win_rate_figure(curve: pd.DataFrame, h2h: pd.DataFrame, window: int) -> go.Figure:
    """Eval points (learned side's win rate) with 95% Wilson error bars + rolling head-to-head red win rate."""
    fig = go.Figure()
    fig.add_hline(y=0.5, line={"color": GRID, "width": 1, "dash": "dot"})
    if not h2h.empty:
        fig.add_trace(
            go.Scatter(
                x=h2h["episode"],
                y=h2h["red_win_rate"],
                mode="lines",
                name=f"Head-to-head (learned vs learned): red win rate, rolling {window}",
                line={"color": NEUTRAL, "width": 2, "dash": "dash"},
                hovertemplate="training ep %{x}<br>red wins %{y:.0%} of last "
                f"{window} head-to-head episodes<extra></extra>",
            )
        )
    for side, sub in curve.groupby("side", sort=False):
        sub = sub.sort_values("after_episode")
        col = TEAM[side]
        fig.add_trace(
            go.Scatter(
                x=sub["after_episode"],
                y=sub["win_rate"],
                mode="lines+markers",
                name=sub["label"].iloc[0] + f": {side} win rate (eval, n={int(sub['n'].iloc[0])})",
                line={"color": col, "width": 2},
                marker={"size": 9, "color": col},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": sub["ci_high"] - sub["win_rate"],
                    "arrayminus": sub["win_rate"] - sub["ci_low"],
                    "color": col,
                    "thickness": 1.5,
                    "width": 5,
                },
                customdata=sub[["wins", "n", "ci_low", "ci_high"]].to_numpy(),
                hovertemplate="after ep %{x}<br>win rate %{y:.0%} (%{customdata[0]}/%{customdata[1]})"
                "<br>95% CI %{customdata[2]:.0%}–%{customdata[3]:.0%}<extra>" + side + "</extra>",
            )
        )
    fig.update_layout(
        **{**_BASE_LAYOUT, "margin": {"l": 60, "r": 10, "t": 10, "b": 10}},
        height=400,
        hovermode="closest",
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor=GRID, zeroline=False, title="win rate")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, title="training episode")
    return fig


def q_bar_figure(values: dict, chosen: str, actor: str, kind: str) -> go.Figure:
    items = sorted(values.items(), key=lambda kv: kv[1])
    colour = TEAM.get(actor, NEUTRAL)
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items],
            y=[k for k, _ in items],
            orientation="h",
            marker={"color": [colour if k == chosen else NEUTRAL for k, _ in items]},
            hovertemplate="%{y}: %{x:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        **{**_BASE_LAYOUT, "margin": {"l": 10, "r": 10, "t": 10, "b": 10}},
        height=40 + 26 * len(items),
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, title=kind, zeroline=True, zerolinecolor=GRID)
    return fig


# ------------------------------------------------------------------------------------------------ Lab: compare runs

# Categorical run colours (dataviz reference order, slot 2 taken from its dark step so one list passes the
# validator on both the light and the dark surface). Colour follows the run (its position in the run history),
# never its rank in the selection. Markers repeat the identity for colour-blind readers.
RUN_COLORS = ["#2a78d6", "#d95926", "#1baf7a", "#c98500", "#d55181", "#008300", "#9085e9"]
RUN_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "star", "hexagon"]
COMPARE_PANELS = (
    ("red", "Learned red vs baseline blue — red win rate"),
    ("blue", "Learned blue vs baseline red — blue win rate"),
)


def run_style(slot: int) -> tuple[str, str]:
    return RUN_COLORS[slot % len(RUN_COLORS)], RUN_SYMBOLS[slot % len(RUN_SYMBOLS)]


def compare_figure(runs: list[dict]) -> go.Figure:
    """Overlay eval curves of several runs. ``runs``: ``[{"name", "curve" (eval_curve frame), "slot"}]``.

    Two panels on one shared win-rate axis (one per learned side), each point with its 95% Wilson interval.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        subplot_titles=[t for _, t in COMPARE_PANELS],
    )  # fmt: skip
    for col_i, (side, _) in enumerate(COMPARE_PANELS, start=1):
        fig.add_hline(y=0.5, line={"color": GRID, "width": 1, "dash": "dot"}, row=1, col=col_i)
    for r in runs:
        colour, symbol = run_style(r["slot"])
        curve = r["curve"]
        for col_i, (side, _) in enumerate(COMPARE_PANELS, start=1):
            sub = curve[curve["side"] == side].sort_values("after_episode") if not curve.empty else curve
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["after_episode"],
                    y=sub["win_rate"],
                    mode="lines+markers",
                    name=r["name"],
                    legendgroup=r["name"],
                    showlegend=col_i == 1 or curve[curve["side"] == "red"].empty,
                    line={"color": colour, "width": 2},
                    marker={"size": 9, "color": colour, "symbol": symbol},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": sub["ci_high"] - sub["win_rate"],
                        "arrayminus": sub["win_rate"] - sub["ci_low"],
                        "color": colour,
                        "thickness": 1.2,
                        "width": 4,
                    },
                    customdata=sub[["wins", "n", "ci_low", "ci_high"]].to_numpy(),
                    hovertemplate="after ep %{x}<br>"
                    + side
                    + " win rate %{y:.0%} (%{customdata[0]}/%{customdata[1]})"
                    "<br>95% CI %{customdata[2]:.0%}–%{customdata[3]:.0%}<extra>" + r["name"] + "</extra>",
                ),
                row=1,
                col=col_i,
            )
    fig.update_layout(
        **{**_BASE_LAYOUT, "margin": {"l": 60, "r": 10, "t": 40, "b": 10}},
        height=420,
        hovermode="closest",
        legend={"orientation": "h", "y": -0.2, "x": 0},
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="win rate (eval vs baseline)", row=1, col=1)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, title="training episode")
    return fig
