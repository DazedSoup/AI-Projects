"""Plotly figure builders. Pure functions of loader/learning output; the only Streamlit-aware bit is
:func:`theme.mode`, which picks the light or dark template.

Colour roles (see ``theme.py``): team red/blue for anything that belongs to a side, violet only for learning
and adaptation (detector versions, recall at red's level, the learning landscape), neutral grey for context.
Node states pair colour with shape everywhere (3D and 2D use the same symbols).
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cyberarena.dashboard import learning as LL
from cyberarena.dashboard import theme as T
from cyberarena.dashboard.loaders import NODE_STATES, node_state, wilson_interval

TEAM = T.TEAM
CROWN = "#D9A33A"
SHAP_POS = "#E8853A"  # pushes the detector towards "malicious"
SHAP_NEG = "#199E70"  # pushes towards "benign"

ZONES = [  # (zone id, label, z height in the 3D view)
    ("internet", "Internet", 4.0),
    ("dmz", "DMZ", 3.0),
    ("workstation", "Workstations", 2.0),
    ("server", "Servers", 1.0),
    ("crown", "Crown jewel", 0.0),
]
ZONE_Z = {z: h for z, _, h in ZONES}
INTERNET = -1  # pseudo node: where phishing comes from (outside the simulated network)


def zone_of(n: dict) -> str:
    if n.get("crown_jewel"):
        return "crown"
    role = str(n.get("role", "")).lower()
    if role in ("dmz", "internet", "edge", "gateway"):
        return "dmz"
    if role.startswith("work") or role in ("host", "client", "endpoint"):
        return "workstation"
    return "server"


# ============================================================================================== host network


def layout_3d(graph: dict) -> dict[int, tuple[float, float, float]]:
    """Stacked zone layers: each zone is a ring at its own height (Internet/DMZ on top, crown jewel at the
    bottom). Within a ring, hosts keep the order of their fixed 2D ``y`` so neighbours stay neighbours."""
    by_zone: dict[str, list[dict]] = {}
    for n in graph["nodes"]:
        by_zone.setdefault(zone_of(n), []).append(n)
    pos: dict[int, tuple[float, float, float]] = {INTERNET: (0.0, 0.0, ZONE_Z["internet"])}
    for zone, nodes in by_zone.items():
        nodes = sorted(nodes, key=lambda n: (n["y"], n["id"]))
        k = len(nodes)
        r = 0.0 if k == 1 else min(1.25, 0.42 + 0.11 * k)
        for i, n in enumerate(nodes):
            a = -math.pi / 2 + 2 * math.pi * i / max(1, k) + (0.35 if zone == "workstation" else 0.0)
            pos[n["id"]] = (r * math.cos(a), r * math.sin(a), ZONE_Z[zone])
    return pos


def layout_2d(graph: dict) -> dict[int, tuple[float, float]]:
    pos = {n["id"]: (n["x"], n["y"]) for n in graph["nodes"]}
    xs = [p[0] for p in pos.values()] or [0.0]
    pos[INTERNET] = (min(xs) - 0.11, 0.5)
    return pos


def _move(turn: dict | None, nodes: set[int]) -> tuple[int | None, int | None]:
    """``(source, target)`` of this turn's move. Phishing has no source: it comes from the Internet."""
    if not turn:
        return None, None
    src, tgt = turn.get("source"), turn.get("target")
    if tgt is None or tgt not in nodes:
        return None, None
    if src is None and turn.get("actor") == "red" and turn.get("action_id") == "phish":
        src = INTERNET
    if src is not None and src != INTERNET and src not in nodes:
        src = None
    return src, tgt


def _hover(n: dict, ns: dict, state: str) -> str:
    flags = [k for k in ("compromised", "detected", "isolated", "patched") if ns.get(k)]
    scores = ns.get("scores") or {}
    sc = "<br>".join(f"{m}: {v:.2f}" for m, v in scores.items())
    zone = {"crown": "crown jewel", "dmz": "DMZ", "workstation": "workstation", "server": "server"}[
        zone_of(n)
    ]
    return (
        f"<b>Host {n['id']}</b> · {zone}<br>{T.STATE_LABEL[state]}"
        + (f" ({', '.join(flags)})" if flags and flags != [state] else "")
        + f"<br>red privilege {ns.get('privilege', 0)}"
        + (f"<br><span style='opacity:.7'>detector scores</span><br>{sc}" if sc else "")
    )


def _disk(r: float, z: float, n: int = 48) -> tuple[list, list, list, list, list, list]:
    xs, ys, zs = [0.0], [0.0], [z]
    for i in range(n):
        a = 2 * math.pi * i / n
        xs.append(r * math.cos(a))
        ys.append(r * math.sin(a))
        zs.append(z)
    i_, j_, k_ = [], [], []
    for i in range(1, n + 1):
        i_.append(0)
        j_.append(i)
        k_.append(i % n + 1)
    return xs, ys, zs, i_, j_, k_


def _ghost_ends(g: dict, nodes: set[int]) -> tuple[int | None, int | None]:
    return _move({"source": g.get("source"), "target": g.get("target"), "actor": g.get("actor"),
                  "action_id": g.get("action")}, nodes)  # fmt: skip


def host_network_3d(graph: dict, turn: dict | None, height: int = 540, revision: str = "net",
                    ghosts: list[dict] | None = None) -> go.Figure:  # fmt: skip
    """``ghosts``: other candidate moves the agent considered (``{action, source, target, actor, rank, q}``),
    drawn as faint dashed paths and rings so the choice can be seen against its alternatives."""
    m = T.mode()
    pos = layout_3d(graph)
    cols = T.state_colors(m)
    states = {ns["id"]: ns for ns in (turn or {}).get("node_states", [])}
    fig = go.Figure()

    # zone plates + labels
    by_zone: dict[str, list[int]] = {}
    for n in graph["nodes"]:
        by_zone.setdefault(zone_of(n), []).append(n["id"])
    plate = "rgba(138,147,166,0.07)" if m == "dark" else "rgba(70,80,100,0.06)"
    for zone, label, z in ZONES:
        if zone == "internet":
            continue
        ids = by_zone.get(zone, [])
        if not ids:
            continue
        r = max(math.hypot(pos[i][0], pos[i][1]) for i in ids) + 0.32
        xs, ys, zs, i_, j_, k_ = _disk(r, z - 0.02)
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=i_, j=j_, k=k_, color=plate, opacity=1.0,
                                hoverinfo="skip", showscale=False, flatshading=True, lighting={"ambient": 1.0}))  # fmt: skip
        fig.add_trace(go.Scatter3d(x=[r + 0.08], y=[0], z=[z], mode="text", text=[label], hoverinfo="skip",
                                   textposition="middle right", showlegend=False,
                                   textfont={"size": 11, "color": T.tok("muted", m), "family": T.FONT_UI}))  # fmt: skip

    # edges: live links, then links to isolated hosts (dotted, fainter)
    live, cut = ([], [], []), ([], [], [])
    for u, v in graph["edges"]:
        iso = states.get(u, {}).get("isolated") or states.get(v, {}).get("isolated")
        tgt = cut if iso else live
        for axis in range(3):
            tgt[axis].extend([pos[u][axis], pos[v][axis], None])
    edge_c = "rgba(138,147,166,0.38)" if m == "dark" else "rgba(90,100,120,0.32)"
    fig.add_trace(go.Scatter3d(x=live[0], y=live[1], z=live[2], mode="lines", hoverinfo="skip",
                               showlegend=False, line={"color": edge_c, "width": 2}))  # fmt: skip
    if cut[0]:
        fig.add_trace(go.Scatter3d(x=cut[0], y=cut[1], z=cut[2], mode="lines", hoverinfo="skip", showlegend=False,
                                   line={"color": edge_c, "width": 1.5, "dash": "dot"}))  # fmt: skip
    # Internet anchor (only drawn when a move starts there)
    ids = {n["id"] for n in graph["nodes"]}
    src, tgt = _move(turn, ids)
    if src == INTERNET:
        ix, iy, iz = pos[INTERNET]
        fig.add_trace(go.Scatter3d(x=[ix], y=[iy], z=[iz], mode="markers+text", text=["Internet"],
                                   textposition="top center", hovertext=["Internet (outside the network)"],
                                   hoverinfo="text", showlegend=False,
                                   textfont={"size": 11, "color": T.tok("muted", m)},
                                   marker={"size": 5, "color": T.tok("neutral", m), "symbol": "circle-open"}))  # fmt: skip

    # candidate moves the agent considered but didn't pick: faint, dashed, numbered by rank
    for g in ghosts or []:
        gs, gt = _ghost_ends(g, ids)
        if gt is None:
            continue
        gc = _rgba(TEAM.get(g.get("actor", "red"), T.tok("neutral", m)), 0.55)
        if gs is not None and gs != gt:
            if gs == INTERNET and src != INTERNET:
                ix, iy, iz = pos[INTERNET]
                fig.add_trace(go.Scatter3d(x=[ix], y=[iy], z=[iz], mode="markers", hoverinfo="skip", showlegend=False,
                                           marker={"size": 4, "color": T.tok("neutral", m), "symbol": "circle-open"}))  # fmt: skip
            (x0, y0, z0), (x1, y1, z1) = pos[gs], pos[gt]
            fig.add_trace(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines", hoverinfo="skip",
                                       showlegend=False, line={"color": gc, "width": 3.5, "dash": "dash"}))  # fmt: skip
        x1, y1, z1 = pos[gt]
        fig.add_trace(go.Scatter3d(x=[x1], y=[y1], z=[z1], mode="markers+text", showlegend=False,
                                   text=[f"#{g.get('rank', '')}"], textposition="bottom center",
                                   textfont={"size": 10, "color": gc, "family": T.FONT_MONO},
                                   hovertext=[f"considered: {g.get('label', '')}<br>Q {g.get('q', 0):.3f} (rank {g.get('rank')})"],
                                   hoverinfo="text",
                                   marker={"size": 20, "color": "rgba(0,0,0,0)", "symbol": "circle-open",
                                           "line": {"width": 2, "color": gc}}))  # fmt: skip

    # current move: highlighted path + cone at the target
    if tgt is not None:
        actor = turn.get("actor", "red")
        c = TEAM.get(actor, T.tok("neutral", m))
        ok = bool(turn.get("success"))
        if src is not None and src != tgt:
            (x0, y0, z0), (x1, y1, z1) = pos[src], pos[tgt]
            fig.add_trace(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines", hoverinfo="skip",
                                       showlegend=False,
                                       line={"color": c, "width": 9 if ok else 6, "dash": "solid" if ok else "dash"}))  # fmt: skip
            d = math.dist((x0, y0, z0), (x1, y1, z1)) or 1.0
            fig.add_trace(go.Cone(x=[x0 + (x1 - x0) * 0.86], y=[y0 + (y1 - y0) * 0.86], z=[z0 + (z1 - z0) * 0.86],
                                  u=[(x1 - x0) / d], v=[(y1 - y0) / d], w=[(z1 - z0) / d], anchor="tip",
                                  sizemode="absolute", sizeref=0.22, showscale=False, hoverinfo="skip",
                                  colorscale=[[0, c], [1, c]]))  # fmt: skip
        x1, y1, z1 = pos[tgt]
        fig.add_trace(go.Scatter3d(x=[x1], y=[y1], z=[z1], mode="markers", hoverinfo="skip", showlegend=False,
                                   marker={"size": 26, "color": c, "opacity": 0.25, "symbol": "circle"}))  # fmt: skip

    # hosts: one trace per state (shape + colour)
    by_state: dict[str, list[dict]] = {s: [] for s in NODE_STATES}
    for n in graph["nodes"]:
        ns = states.get(n["id"], {})
        by_state[node_state(ns)].append({**n, "ns": ns})
    ring = T.tok("bg", m)
    for s in NODE_STATES:
        nodes = by_state[s]
        if not nodes:
            continue
        fig.add_trace(go.Scatter3d(
            x=[pos[n["id"]][0] for n in nodes], y=[pos[n["id"]][1] for n in nodes],
            z=[pos[n["id"]][2] for n in nodes], mode="markers+text", name=T.STATE_LABEL[s],
            text=[str(n["id"]) for n in nodes], textposition="top center",
            textfont={"size": 10, "color": T.tok("text2", m), "family": T.FONT_MONO},
            hovertext=[_hover(n, n["ns"], s) for n in nodes], hoverinfo="text",
            marker={"symbol": T.STATE_SYMBOL[s], "size": [15 if n["crown_jewel"] else 11 for n in nodes],
                    "color": cols[s], "line": {"width": 2 if s != "isolated" else 3,
                                               "color": cols[s] if s == "isolated" else ring}},
        ))  # fmt: skip
    crown = [n for n in graph["nodes"] if n["crown_jewel"]]
    if crown:
        fig.add_trace(go.Scatter3d(x=[pos[n["id"]][0] for n in crown], y=[pos[n["id"]][1] for n in crown],
                                   z=[pos[n["id"]][2] for n in crown], mode="markers", hoverinfo="skip",
                                   showlegend=False,
                                   marker={"symbol": "circle-open", "size": 20, "color": CROWN,
                                           "line": {"width": 3, "color": CROWN}}))  # fmt: skip
    hidden = {"visible": False, "showbackground": False, "showgrid": False, "zeroline": False, "title": ""}
    T.style(fig, height=height, m=m)
    fig.update_layout(
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        uirevision=revision,
        scene={"xaxis": hidden, "yaxis": hidden, "zaxis": hidden, "aspectmode": "manual",
               "aspectratio": {"x": 1.0, "y": 1.0, "z": 1.05}, "dragmode": "turntable",
               "camera": {"eye": {"x": 0.98, "y": -1.05, "z": 0.36}, "center": {"x": 0.06, "y": 0, "z": -0.02},
                          "up": {"x": 0, "y": 0, "z": 1}}},
    )  # fmt: skip
    return fig


def host_network_2d(graph: dict, turn: dict | None, height: int = 460, compact: bool = False,
                    ghosts: list[dict] | None = None) -> go.Figure:  # fmt: skip
    """The same network flat: zones as columns (Internet → DMZ → workstations → servers → crown jewel)."""
    m = T.mode()
    pos = layout_2d(graph)
    cols = T.state_colors(m)
    states = {ns["id"]: ns for ns in (turn or {}).get("node_states", [])}
    fig = go.Figure()
    edge_c = "rgba(138,147,166,0.40)" if m == "dark" else "rgba(90,100,120,0.35)"
    ex, ey, cx, cy = [], [], [], []
    for u, v in graph["edges"]:
        iso = states.get(u, {}).get("isolated") or states.get(v, {}).get("isolated")
        xs, ys = (cx, cy) if iso else (ex, ey)
        xs += [pos[u][0], pos[v][0], None]
        ys += [pos[u][1], pos[v][1], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line={"color": edge_c, "width": 1.2}, hoverinfo="skip",
                             showlegend=False))  # fmt: skip
    if cx:
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", hoverinfo="skip", showlegend=False,
                                 line={"color": edge_c, "width": 1, "dash": "dot"}))  # fmt: skip
    ids = {n["id"] for n in graph["nodes"]}
    src, tgt = _move(turn, ids)
    for g in ghosts or []:
        gs, gt = _ghost_ends(g, ids)
        if gt is None:
            continue
        gc = _rgba(TEAM.get(g.get("actor", "red"), T.tok("neutral", m)), 0.5)
        if gs is not None and gs != gt:
            (x0, y0), (x1, y1) = pos[gs], pos[gt]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                     line={"color": gc, "width": 1.5, "dash": "dash"}))  # fmt: skip
        fig.add_trace(go.Scatter(x=[pos[gt][0]], y=[pos[gt][1]], mode="markers+text", showlegend=False,
                                 text=[f"#{g.get('rank', '')}"], textposition="bottom center",
                                 textfont={"size": 9.5, "color": gc, "family": T.FONT_MONO},
                                 hovertext=[f"considered: {g.get('label', '')}<br>Q {g.get('q', 0):.3f}"], hoverinfo="text",
                                 marker={"size": 24 if not compact else 18, "color": "rgba(0,0,0,0)",
                                         "line": {"width": 1.5, "color": gc}}))  # fmt: skip
    if tgt is not None:
        c = TEAM.get(turn.get("actor", "red"), T.tok("neutral", m))
        ok = bool(turn.get("success"))
        if src == INTERNET:
            ix, iy = pos[INTERNET]
            fig.add_trace(go.Scatter(x=[ix], y=[iy], mode="markers+text", text=["Internet"], hoverinfo="skip",
                                     textposition="bottom center", showlegend=False,
                                     textfont={"size": 10, "color": T.tok("muted", m)},
                                     marker={"size": 9, "symbol": "circle-open", "color": T.tok("neutral", m),
                                             "line": {"width": 1.5}}))  # fmt: skip
        if src is not None and src != tgt:
            (x0, y0), (x1, y1) = pos[src], pos[tgt]
            fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="",
                               showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=3 if ok else 2,
                               arrowcolor=c, standoff=11 if not compact else 7, startstandoff=6, opacity=1 if ok else 0.75)  # fmt: skip
        fig.add_trace(go.Scatter(x=[pos[tgt][0]], y=[pos[tgt][1]], mode="markers", hoverinfo="skip",
                                 showlegend=False,
                                 marker={"size": 30 if not compact else 20, "color": c, "opacity": 0.2}))  # fmt: skip
    ring = T.tok("bg", m)
    by_state: dict[str, list[dict]] = {s: [] for s in NODE_STATES}
    for n in graph["nodes"]:
        ns = states.get(n["id"], {})
        by_state[node_state(ns)].append({**n, "ns": ns})
    for s in NODE_STATES:
        nodes = by_state[s]
        if not nodes:
            continue
        size = [(19 if n["crown_jewel"] else 14) * (0.72 if compact else 1) for n in nodes]
        fig.add_trace(go.Scatter(
            x=[pos[n["id"]][0] for n in nodes], y=[pos[n["id"]][1] for n in nodes],
            mode="markers" if compact else "markers+text", name=T.STATE_LABEL[s],
            text=[str(n["id"]) for n in nodes], textposition="top center",
            textfont={"size": 10, "family": T.FONT_MONO, "color": T.tok("text2", m)},
            hovertext=[_hover(n, n["ns"], s) for n in nodes], hoverinfo="text",
            marker={"symbol": T.STATE_SYMBOL[s] if T.STATE_SYMBOL[s] != "x" else "x-thin-open", "size": size,
                    "color": cols[s], "line": {"width": 2.5 if s in ("isolated", "detected") else 2,
                                               "color": cols[s] if s in ("isolated", "detected") else ring}},
        ))  # fmt: skip
    crown = [n for n in graph["nodes"] if n["crown_jewel"]]
    if crown:
        fig.add_trace(go.Scatter(x=[pos[n["id"]][0] for n in crown], y=[pos[n["id"]][1] for n in crown],
                                 mode="markers", hoverinfo="skip", showlegend=False,
                                 marker={"symbol": "circle-open", "size": 30 if not compact else 22, "color": CROWN,
                                         "line": {"width": 2.5}}))  # fmt: skip
    if not compact:
        xs: dict[str, list[float]] = {}
        for n in graph["nodes"]:
            xs.setdefault(zone_of(n), []).append(n["x"])
        names = {z: lab for z, lab, _ in ZONES}
        for z, v in xs.items():
            fig.add_annotation(x=sum(v) / len(v), y=1.07, text=names[z].upper(), showarrow=False,
                               font={"size": 10, "color": T.tok("muted", m)}, yanchor="bottom")  # fmt: skip
    T.style(fig, height=height, m=m)
    xs_all = [p[0] for p in pos.values()]
    ax = {"visible": False, "showgrid": False, "zeroline": False, "fixedrange": True}
    fig.update_xaxes(**ax, range=[min(xs_all) - 0.04, max(xs_all) + 0.05])
    fig.update_yaxes(**ax, range=[-0.06, 1.14])
    fig.update_layout(showlegend=False, margin={"l": 4, "r": 4, "t": 4, "b": 4})
    return fig


# ============================================================================================== decision web


def decision_web_figure(web: dict, actor: str, height: int = 380) -> go.Figure:
    """Centre = chosen action; ring 1 = other candidates (spoke width ∝ value); ring 2 = SHAP features."""
    m = T.mode()
    fig = go.Figure()
    colour = TEAM.get(actor, T.tok("neutral", m))
    kind = web["value_kind"]

    def spoke(x, y, weight, col, opacity=0.7, stop=0.0):
        r = math.hypot(x, y) or 1.0
        k = (r - stop) / r
        fig.add_trace(go.Scatter(x=[0, x * k], y=[0, y * k], mode="lines", hoverinfo="skip", showlegend=False,
                                 opacity=opacity, line={"color": col, "width": 1 + 7 * weight}))  # fmt: skip

    acts = web["actions"]
    ax, ay, at, ah = [], [], [], []
    for i, a in enumerate(acts):
        ang = math.pi / 2 + 2 * math.pi * (i + 0.5) / max(1, len(acts))
        x, y = 1.5 * math.cos(ang), 1.5 * math.sin(ang)
        spoke(x, y, a["weight"], colour, stop=0.28, opacity=0.35 + 0.45 * a["weight"])
        ax.append(x)
        ay.append(y)
        at.append(f"{a['action'].replace('_', ' ')}<br><span style='font-size:10px'>{a['value']:.3g}</span>")
        ah.append(f"<b>{a['action']}</b><br>{kind}: {a['value']:.4g}<br>relative to best: {a['weight']:.2f}")
    if acts:
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers+text", text=at, textposition="middle center",
                                 hovertext=ah, hoverinfo="text", name=f"Other actions ({kind})",
                                 textfont={"size": 9.5, "color": T.tok("text2", m)},
                                 marker={"size": 48, "color": T.tok("surface", m),
                                         "line": {"color": colour, "width": 1.5}}))  # fmt: skip
    feats = web["features"]
    fx, fy, ft, fh, fc = [], [], [], [], []
    for i, f in enumerate(feats):
        ang = math.pi / 2 + 2 * math.pi * i / max(1, len(feats))
        x, y = 2.75 * math.cos(ang), 2.75 * math.sin(ang)
        col = SHAP_POS if f["shap"] >= 0 else SHAP_NEG
        spoke(x, y, f["weight"], col, opacity=0.5)
        fx.append(x)
        fy.append(y)
        fc.append(col)
        name = f["name"] if len(f["name"]) <= 20 else f["name"][:19] + "…"
        ft.append(f"{name} {f['shap']:+.2f}")
        raw = f" (raw {f['raw']})" if f.get("raw") is not None else ""
        fh.append(f"<b>{f['name']}</b><br>{f['model']} detector on host {f['node']}<br>SHAP {f['shap']:+.4f}"
                  f"<br>scaled value {f['value']}{raw}")  # fmt: skip
    if feats:
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="markers+text", text=ft, hovertext=fh, hoverinfo="text",
                                 textposition=[("top " if y >= 0 else "bottom ") + ("center" if abs(x) < 0.8 else "right" if x > 0 else "left")
                                               for x, y in zip(fx, fy, strict=True)],
                                 textfont={"size": 10, "color": T.tok("muted", m)}, showlegend=False,
                                 marker={"size": 10, "color": fc, "line": {"width": 2, "color": T.tok("bg", m)}}))  # fmt: skip
        for col, lab in (
            (SHAP_POS, "Feature pushes toward malicious"),
            (SHAP_NEG, "Feature pushes toward benign"),
        ):
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="markers", name=lab, marker={"size": 9, "color": col})
            )
    cv = web["chosen_value"]
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Chosen action", hoverinfo="text",
                             hovertext=[f"<b>chosen: {web['chosen']}</b><br>{kind}: {cv}"],
                             marker={"size": 70, "color": colour, "line": {"width": 0}}))  # fmt: skip
    fig.add_annotation(x=0, y=0, showarrow=False, font={"color": "#FFFFFF", "size": 12, "family": T.FONT_UI},
                       text=f"<b>{web['chosen'].replace('_', ' ')}</b>" + (f"<br>{cv:.3g}" if cv is not None else ""))  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(
        legend={"y": -0.02, "yanchor": "top", "font": {"size": 11}}, margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    ax_ = {"visible": False, "showgrid": False, "zeroline": False, "fixedrange": True}
    lim = 3.25 if feats else 2.05
    fig.update_xaxes(**ax_, range=[-lim - 0.9, lim + 0.9])
    fig.update_yaxes(**ax_, range=[-lim, lim], scaleanchor="x", scaleratio=1)
    return fig


def q_bar_figure(values: dict, chosen: str, actor: str, kind: str) -> go.Figure:
    m = T.mode()
    items = sorted(values.items(), key=lambda kv: kv[1])
    colour = TEAM.get(actor, T.tok("neutral", m))
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k.replace("_", " ").replace("→", " → ") for k, _ in items], orientation="h",
        marker={"color": [colour if k == chosen else T.tok("neutral_dim", m) for k, _ in items],
                "cornerradius": 4},
        hovertemplate="%{y}: %{x:.4g}<extra></extra>", width=0.62,
    ))  # fmt: skip
    T.style(fig, height=40 + 26 * len(items), m=m)
    fig.update_xaxes(title=kind, zeroline=True, zerolinecolor=T.tok("axis", m))
    fig.update_yaxes(showline=False, tickfont={"family": T.FONT_UI, "size": 11.5})
    return fig


# ============================================================================================== win rate & games


def _band(fig, x, lo, hi, colour, name, row=None, col=None):
    rgba = _rgba(colour, 0.12)
    kw = {"row": row, "col": col} if row else {}
    fig.add_trace(go.Scatter(x=list(x), y=list(hi), mode="lines", line={"width": 0}, hoverinfo="skip",
                             showlegend=False, legendgroup=name), **kw)  # fmt: skip
    fig.add_trace(go.Scatter(x=list(x), y=list(lo), mode="lines", line={"width": 0}, fill="tonexty",
                             fillcolor=rgba, hoverinfo="skip", showlegend=False, legendgroup=name), **kw)  # fmt: skip


def _rgba(hex_colour: str, a: float) -> str:
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def win_rate_figure(curve: pd.DataFrame, h2h: pd.DataFrame, window: int, height: int = 360) -> go.Figure:
    """Each learned side's eval win rate against its scripted opponent, with a 95% Wilson band, plus the
    rolling head-to-head red win rate from training games."""
    m = T.mode()
    fig = go.Figure()
    fig.add_hline(y=0.5, line={"color": T.tok("axis", m), "width": 1})
    if not h2h.empty:
        fig.add_trace(go.Scatter(x=h2h["episode"], y=h2h["red_win_rate"], mode="lines",
                                 name=f"Head-to-head: red's share of the last {window} training games",
                                 line={"color": T.tok("neutral", m), "width": 1.25}, opacity=0.7, legendrank=3,
                                 hovertemplate="%{y:.0%} red wins (head-to-head, rolling)<extra></extra>"))  # fmt: skip
    ends = {s: float(g.sort_values("after_episode")["win_rate"].iloc[-1]) for s, g in curve.groupby("side")}
    close = len(ends) == 2 and abs(ends.get("red", 0) - ends.get("blue", 0)) < 0.07
    for side, sub in curve.groupby("side", sort=False):
        sub = sub.sort_values("after_episode")
        c = TEAM[side]
        name = sub["label"].iloc[0]
        other = ends.get("blue" if side == "red" else "red", -1)
        yshift = (
            0 if not close else (8 if ends[side] >= other and side == "red" or ends[side] > other else -8)
        )
        _band(fig, sub["after_episode"], sub["ci_low"], sub["ci_high"], c, name)
        fig.add_trace(go.Scatter(
            x=sub["after_episode"], y=sub["win_rate"], mode="lines+markers", name=name, legendgroup=name,
            legendrank=1 if side == "red" else 2,
            line={"color": c, "width": 2}, marker={"size": 8, "color": c, "line": {"width": 2, "color": T.tok("bg", m)}},
            customdata=sub[["wins", "n", "ci_low", "ci_high"]].to_numpy(),
            hovertemplate="%{y:.0%} (%{customdata[0]}/%{customdata[1]} games) · 95% CI %{customdata[2]:.0%}–"
                          "%{customdata[3]:.0%}<extra>" + name + "</extra>",
        ))  # fmt: skip
        last = sub.iloc[-1]
        fig.add_annotation(x=last["after_episode"], y=last["win_rate"], text=f"{last['win_rate']:.0%}",
                           showarrow=False, xanchor="left", xshift=8, yshift=yshift,
                           font={"size": 12, "color": T.tok("text", m),
                                                                           "family": T.FONT_MONO})  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(hovermode="x unified", margin={"l": 8, "r": 40, "t": 36, "b": 8})
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", title="win rate", dtick=0.25)
    fig.update_xaxes(title="training games", showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor=T.tok("axis", m), spikedash="solid")  # fmt: skip
    return fig


def game_length_figure(episodes: pd.DataFrame, window: int = 100, height: int = 300) -> go.Figure:
    """Rolling median length of training games, split by who won (``window`` = that side's last N wins)."""
    m = T.mode()
    fig = go.Figure()
    if not episodes.empty:
        eps = episodes.dropna(subset=["turns"])
        for side in ("red", "blue"):
            sub = eps[eps["winner"] == side]
            if len(sub) < 5:
                continue
            med = sub["turns"].astype(float).rolling(window, min_periods=max(5, window // 4)).median()
            fig.add_trace(go.Scatter(x=sub["episode"], y=med, mode="lines", name=f"{side.capitalize()} wins",
                                     line={"color": TEAM[side], "width": 2},
                                     hovertemplate="median %{y:.0f} turns<extra>" + side + " wins</extra>"))  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(hovermode="x unified", margin={"l": 8, "r": 8, "t": 36, "b": 8})
    fig.update_yaxes(title="turns per game (median)", rangemode="tozero")
    fig.update_xaxes(title="training games")
    return fig


# ============================================================================================== learning


def arms_race_figure(
    lr: LL.Learning, models: list[str], height: int = 420, compact: bool = False
) -> go.Figure:
    """Small multiples, one column per sensor. Top row: the detector's recall at red's *current* evasion level
    (violet; diamonds = detector updates). Bottom row: red's evasion level (red). Same x across all panels."""
    m = T.mode()
    metric = LL.detector_metric(lr)
    word = "Recall" if metric == "recall" else "AUC"
    n = max(1, len(models))
    fig = make_subplots(rows=2, cols=n, shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.05,
                        row_heights=[0.62, 0.38], column_titles=None if compact else [f"{x} sensor" for x in models])  # fmt: skip
    for j, model in enumerate(models, start=1):
        ar = LL.arms_race(lr, model)
        if ar.empty:
            continue
        ups = ar[ar["event"] == "after"]
        for ep in ups["episode"]:
            for r in (1, 2):
                fig.add_vline(x=ep, line={"color": _rgba(T.VIOLET, 0.16), "width": 1}, row=r, col=j)
        sc = ar.dropna(subset=["score"])
        if metric == "recall" and sc["n"].notna().all() and not sc.empty:
            # 95% Wilson band: how much each recall reading can move given the test rows behind it
            ci = [
                wilson_interval(round(float(v) * int(n)), int(n))
                for v, n in zip(sc["score"], sc["n"], strict=True)
            ]
            for k, (ys, fill) in enumerate((([c[1] for c in ci], None), ([c[0] for c in ci], "tonexty"))):
                fig.add_trace(go.Scatter(x=sc["episode"], y=ys, mode="lines", line={"width": 0, "shape": "hv"},
                                         fill=fill, fillcolor=_rgba(T.VIOLET, 0.13), hoverinfo="skip",
                                         showlegend=j == 1 and k == 1, legendgroup="ci",
                                         name="95% interval (test-set size)"), row=1, col=j)  # fmt: skip
        fig.add_trace(go.Scatter(x=sc["episode"], y=sc["score"], mode="lines", line={"color": T.VIOLET, "width": 2,
                                 "shape": "hv"}, name=f"Detector {word.lower()} at red's current level",
                                 legendgroup="score", showlegend=j == 1,
                                 customdata=sc[["evasion", "version"]].to_numpy(),
                                 hovertemplate=f"{word} %{{y:.0%}} · detector v%{{customdata[1]}} · red evasion "
                                               "%{customdata[0]:.1f}<extra>" + model + "</extra>"),
                      row=1, col=j)  # fmt: skip
        fig.add_trace(go.Scatter(x=ups["episode"], y=ups["score"], mode="markers", name="Detector update",
                                 legendgroup="upd", showlegend=j == 1,
                                 marker={"symbol": "diamond", "size": 8 if not compact else 7, "color": T.VIOLET,
                                         "line": {"width": 1.5, "color": T.tok("bg", m)}},
                                 customdata=ups[["version"]].to_numpy(),
                                 hovertemplate="update → v%{customdata[0]} · " + word.lower() + " %{y:.0%}<extra>"
                                 + model + "</extra>"), row=1, col=j)  # fmt: skip
        ev = ar.drop_duplicates(subset=["episode"], keep="last")
        fig.add_trace(go.Scatter(x=ev["episode"], y=ev["evasion"], mode="lines", name="Red evasion level",
                                 legendgroup="ev", showlegend=j == 1, fill="tozeroy", fillcolor=_rgba(T.RED, 0.10),
                                 line={"color": T.RED, "width": 2, "shape": "hv"},
                                 hovertemplate="red evasion %{y:.1f}<extra>" + model + "</extra>"),
                      row=2, col=j)  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(hovermode="x unified", margin={"l": 8, "r": 8, "t": 44 if not compact else 30, "b": 8},
                      legend={"y": 1.06 if not compact else 1.1})  # fmt: skip
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", dtick=0.25, row=1)
    fig.update_yaxes(range=[0, 0.78], dtick=0.2, tickformat=".1f", row=2)
    fig.update_yaxes(title=f"{word} at red's level", row=1, col=1)
    fig.update_yaxes(title="red evasion", row=2, col=1)
    for j in range(1, n + 1):
        fig.update_xaxes(title="training games" if not compact else None, row=2, col=j)
    for a in fig.layout.annotations:  # column titles
        a.font = {"size": 12, "color": T.tok("text2", m), "family": T.FONT_UI}
    return fig


def landscape_figure(ls: dict, red_path: list[tuple[int, float, float]] | None = None, three_d: bool = True,
                     height: int = 480, revision: str = "ls") -> go.Figure:  # fmt: skip
    """Detector AUC over (training game × evasion level), one column per detector version. ``red_path``:
    ``[(episode, red level, AUC there)]`` drawn on top so you can see where red was operating."""
    m = T.mode()
    cs = T.colorscale("violet", m)
    x, y, z = ls["episodes"], ls["levels"], ls["z"]
    vers = ls["versions"]
    custom = [[f"v{v}" for v in vers] for _ in y]
    hover = "game %{x:,}<br>evasion %{y:.1f}<br>AUC %{z:.3f} · %{customdata}<extra></extra>"
    flat = [v for row in z for v in row if v is not None]
    zlo = max(0.5, math.floor((min(flat) if flat else 0.5) * 20) / 20)
    if three_d:
        fig = go.Figure(go.Surface(x=x, y=y, z=z, colorscale=cs, cmin=zlo, cmax=1.0, customdata=custom,
                                   hovertemplate=hover, opacity=0.96,
                                   contours={"z": {"show": True, "usecolormap": False, "color": _rgba("#FFFFFF", 0.25)
                                                   if m == "dark" else _rgba("#0E1320", 0.18), "width": 1,
                                                   "start": 0.5, "end": 1.0, "size": 0.05}},
                                   colorbar={"title": {"text": "AUC", "side": "top"}, "len": 0.6, "thickness": 10,
                                             "x": 1.0, "tickformat": ".2f"},
                                   lighting={"ambient": 0.85, "diffuse": 0.3, "specular": 0.05}))  # fmt: skip
        if red_path:
            fig.add_trace(go.Scatter3d(x=[p[0] for p in red_path], y=[p[1] for p in red_path],
                                       z=[p[2] + 0.015 for p in red_path], mode="lines+markers",
                                       name="Red's evasion level at each update",
                                       line={"color": T.RED, "width": 5}, marker={"size": 3.5, "color": T.RED},
                                       hovertemplate="game %{x:,} · red at %{y:.1f} · AUC %{z:.3f}<extra></extra>"))  # fmt: skip
        T.style(fig, height=height, m=m)
        fig.update_layout(
            uirevision=revision, margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend={"y": 0.98, "x": 0.01},
            scene={"xaxis": {"title": {"text": "training games"}, "autorange": "reversed"}, "yaxis": {"title": {"text": "red evasion"},
                   "dtick": 0.1}, "zaxis": {"title": {"text": "AUC"}, "range": [zlo, 1.0], "tickformat": ".2f"},
                   "aspectmode": "manual", "aspectratio": {"x": 1.6, "y": 1.0, "z": 0.75},
                   "camera": {"eye": {"x": 0.45, "y": 1.75, "z": 0.72}, "center": {"x": 0, "y": 0, "z": -0.12}}},
        )  # fmt: skip
        return fig
    fig = go.Figure(go.Heatmap(x=x, y=y, z=z, colorscale=cs, zmin=zlo, zmax=1.0, customdata=custom,
                               hovertemplate=hover.replace("%{z:.3f}", "%{z:.3f}"), xgap=2, ygap=2,
                               colorbar={"title": {"text": "AUC", "side": "top"}, "thickness": 10, "len": 0.8,
                                         "tickformat": ".2f"}))  # fmt: skip
    if red_path:
        fig.add_trace(go.Scatter(x=[p[0] for p in red_path], y=[p[1] for p in red_path], mode="lines+markers",
                                 name="Red's evasion level at each update", line={"color": T.RED, "width": 2,
                                 "shape": "hv"}, marker={"size": 8, "color": T.RED,
                                 "line": {"width": 2, "color": T.tok("bg", m)}},
                                 hovertemplate="game %{x:,} · red at %{y:.1f}<extra></extra>"))  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(margin={"l": 8, "r": 8, "t": 36, "b": 8})
    fig.update_xaxes(
        title="training games (one column per detector version)", showgrid=False, type="category"
    )
    fig.update_yaxes(title="red evasion level", dtick=0.1, showgrid=False)
    return fig


def agent_figure(stats: pd.DataFrame, qstates: pd.DataFrame | None = None, height: int = 280) -> go.Figure:
    """Three panels, each its own measure: Q-table size, TD error, exploration rate; red and blue lines."""
    m = T.mode()
    titles = ["Q-table size (states seen)", "TD error (lower = steadier values)", "Exploration rate ε"]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=titles)
    for side in ("red", "blue"):
        s = stats[stats["side"] == side] if not stats.empty else stats
        c = TEAM[side]
        name = f"Learned {side}"
        if not s.empty:
            for col, field, fmt in ((1, "n_states", ",.0f"), (2, "td_error", ".3f"), (3, "epsilon", ".0%")):
                fig.add_trace(go.Scatter(x=s["episode"], y=s[field], mode="lines", name=name, legendgroup=side,
                                         showlegend=col == 1, line={"color": c, "width": 2},
                                         hovertemplate="%{y:" + fmt + "}<extra>" + name + "</extra>"),
                              row=1, col=col)  # fmt: skip
        elif qstates is not None and not qstates.empty:
            q = qstates[qstates["side"] == side]
            fig.add_trace(go.Scatter(x=q["after_episode"], y=q["n_states"], mode="lines+markers", name=name,
                                     legendgroup=side, line={"color": c, "width": 2},
                                     marker={"size": 7, "color": c, "line": {"width": 2, "color": T.tok("bg", m)}},
                                     hovertemplate="%{y:,} states<extra>" + name + "</extra>"), row=1, col=1)  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(hovermode="x unified", margin={"l": 8, "r": 8, "t": 54, "b": 8}, legend={"y": 1.2})
    fig.update_yaxes(rangemode="tozero")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02], row=1, col=3)
    fig.update_xaxes(title="training games")
    for a in fig.layout.annotations:
        a.font = {"size": 12, "color": T.tok("text2", m), "family": T.FONT_UI}
        a.x = a.x - 0.0
    return fig


def mix_figure(
    actions: list[str], episodes: list[int], z: list[list[float]], side: str, height: int = 260
) -> go.Figure:
    """Share of moves per action over training (heatmap in the side's own hue)."""
    m = T.mode()
    zmax = max((max(r) for r in z), default=1.0)
    fig = go.Figure(go.Heatmap(x=episodes, y=[a.replace("_", " ") for a in actions], z=z,
                               colorscale=T.colorscale(side, m), zmin=0, zmax=zmax, xgap=1, ygap=2,
                               hovertemplate="game %{x:,}<br>%{y}: %{z:.0%} of moves<extra></extra>",
                               colorbar={"tickformat": ".0%", "thickness": 8, "len": 0.9, "outlinewidth": 0}))  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(margin={"l": 8, "r": 8, "t": 8, "b": 8})
    fig.update_xaxes(title="training games", showgrid=False, showline=False)
    fig.update_yaxes(showgrid=False, showline=False, autorange="reversed",
                     tickfont={"family": T.FONT_UI, "size": 11.5})  # fmt: skip
    return fig


def probe_figure(pm: dict, side: str, height: int = 190) -> go.Figure:
    """One probe situation: Q-value per action (rows) at each checkpoint (columns); ● = the action chosen."""
    m = T.mode()
    vals = [v for row in pm["q"] for v in row if v is not None]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    xs = [f"{c:,}" for c in pm["checkpoints"]]
    ys = [LL.move_label(a) for a in pm["actions"]]
    fig = go.Figure(go.Heatmap(x=xs, y=ys, z=pm["q"], colorscale=T.colorscale(side, m), zmin=lo, zmax=hi,
                               xgap=2, ygap=2, showscale=False,
                               hovertemplate="checkpoint %{x}<br>%{y}: Q %{z:.3f}<extra></extra>"))  # fmt: skip
    cx = [xs[i] for i, a in enumerate(pm["chosen"]) if a in pm["actions"]]
    cy = [LL.move_label(pm["chosen"][i]) for i, a in enumerate(pm["chosen"]) if a in pm["actions"]]
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", hoverinfo="skip", showlegend=False,
                             marker={"symbol": "circle", "size": 7, "color": T.tok("text", m),
                                     "line": {"width": 1.5, "color": T.tok("bg", m)}}))  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(margin={"l": 4, "r": 4, "t": 4, "b": 4})
    fig.update_xaxes(showgrid=False, showline=False, type="category", tickfont={"size": 10})
    fig.update_yaxes(showgrid=False, showline=False, autorange="reversed", type="category",
                     tickfont={"family": T.FONT_UI, "size": 11})  # fmt: skip
    return fig


def adaptive_vs_frozen_figure(adaptive: pd.DataFrame, frozen: pd.DataFrame, side: str = "blue",
                              height: int = 320) -> go.Figure:  # fmt: skip
    """One learned side's eval win rate in an adaptive-detector run (violet) and a frozen-detector run (grey)."""
    m = T.mode()
    fig = go.Figure()
    fig.add_hline(y=0.5, line={"color": T.tok("axis", m), "width": 1})
    for curve, name, c, sym in ((frozen, "Frozen detectors", T.tok("neutral", m), "square"),
                                (adaptive, "Adaptive detectors", T.VIOLET, "circle")):  # fmt: skip
        sub = curve[curve["side"] == side].sort_values("after_episode") if not curve.empty else curve
        if sub.empty:
            continue
        _band(fig, sub["after_episode"], sub["ci_low"], sub["ci_high"], c, name)
        fig.add_trace(go.Scatter(
            x=sub["after_episode"], y=sub["win_rate"], mode="lines+markers", name=name, legendgroup=name,
            line={"color": c, "width": 2}, marker={"size": 8, "color": c, "symbol": sym,
                                                   "line": {"width": 2, "color": T.tok("bg", m)}},
            customdata=sub[["wins", "n", "ci_low", "ci_high"]].to_numpy(),
            hovertemplate="%{y:.0%} (%{customdata[0]}/%{customdata[1]}) · 95% CI %{customdata[2]:.0%}–"
                          "%{customdata[3]:.0%}<extra>" + name + "</extra>",
        ))  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(hovermode="x unified", margin={"l": 8, "r": 8, "t": 36, "b": 8})
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", dtick=0.25, title=f"learned {side}'s win rate")
    fig.update_xaxes(title="training games")
    return fig


# ============================================================================================== Lab: compare runs

# Categorical run colours (dataviz reference order; slot 2 from its dark step so the list passes the validator
# on both surfaces). Colour follows the run (its position in the run history), never its rank in the selection.
RUN_COLORS = ["#2a78d6", "#d95926", "#1baf7a", "#c98500", "#d55181", "#008300", "#9085e9"]
RUN_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "star", "hexagon"]
COMPARE_PANELS = (
    ("red", "Learned red vs scripted blue — red's win rate"),
    ("blue", "Learned blue vs scripted red — blue's win rate"),
)


def run_style(slot: int) -> tuple[str, str]:
    return RUN_COLORS[slot % len(RUN_COLORS)], RUN_SYMBOLS[slot % len(RUN_SYMBOLS)]


def compare_figure(runs: list[dict], height: int = 380) -> go.Figure:
    """Overlay eval curves of several runs. ``runs``: ``[{"name", "curve" (eval_curve frame), "slot"}]``."""
    m = T.mode()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[t for _, t in COMPARE_PANELS])  # fmt: skip
    for col_i in (1, 2):
        fig.add_hline(y=0.5, line={"color": T.tok("axis", m), "width": 1}, row=1, col=col_i)
    for r in runs:
        colour, symbol = run_style(r["slot"])
        curve = r["curve"]
        for col_i, (side, _) in enumerate(COMPARE_PANELS, start=1):
            sub = curve[curve["side"] == side].sort_values("after_episode") if not curve.empty else curve
            if sub.empty:
                continue
            _band(
                fig, sub["after_episode"], sub["ci_low"], sub["ci_high"], colour, r["name"], row=1, col=col_i
            )
            fig.add_trace(go.Scatter(
                x=sub["after_episode"], y=sub["win_rate"], mode="lines+markers", name=r["name"],
                legendgroup=r["name"], showlegend=col_i == 1 or curve[curve["side"] == "red"].empty,
                line={"color": colour, "width": 2},
                marker={"size": 8, "color": colour, "symbol": symbol, "line": {"width": 1.5, "color": T.tok("bg", m)}},
                customdata=sub[["wins", "n", "ci_low", "ci_high"]].to_numpy(),
                hovertemplate="after game %{x:,}<br>" + side + " wins %{y:.0%} (%{customdata[0]}/%{customdata[1]})"
                "<br>95% CI %{customdata[2]:.0%}–%{customdata[3]:.0%}<extra>" + r["name"] + "</extra>",
            ), row=1, col=col_i)  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(margin={"l": 8, "r": 8, "t": 56, "b": 8}, legend={"y": 1.16})
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", dtick=0.25)
    fig.update_yaxes(title="win rate vs scripted opponent", row=1, col=1)
    fig.update_xaxes(title="training games")
    for a in fig.layout.annotations:
        a.font = {"size": 12, "color": T.tok("text2", m), "family": T.FONT_UI}
    return fig


# kept for callers that still use the old names
host_graph = host_network_2d


# ============================================================================================== Evidence (experiments)


def condition_color(cond: str, slot: int, m: str | None = None) -> str:
    """Adaptive = learning violet, frozen = neutral grey, anything else a categorical run colour."""
    if cond == "adaptive":
        return T.VIOLET
    if cond == "frozen":
        return T.tok("neutral", m)
    return RUN_COLORS[slot % len(RUN_COLORS)]


def contrast_dots_figure(c: dict, height: int = 96, span: float | None = None) -> go.Figure:
    """One contrast on a points axis: grey zero line, the 95% CI as a bar, the mean as a diamond, and one dot
    per seed (the paired per-seed differences). Colour encodes the direction only when it is significant."""
    m = T.mode()
    d = [x for x in (c.get("per_seed_diff") or []) if x is not None]
    mean = c.get("diff_mean") or 0.0
    lo, hi = (c.get("ci95") or [None, None])[:2]
    sig = c.get("_sig", False)
    col = (T.tok("good", m) if mean > 0 else T.RED) if sig else T.tok("text2", m)
    fig = go.Figure()
    fig.add_vline(x=0, line={"color": T.tok("axis", m), "width": 1.5})
    if lo is not None and hi is not None:
        fig.add_trace(go.Scatter(x=[100 * lo, 100 * hi], y=[0, 0], mode="lines", hoverinfo="skip", showlegend=False,
                                 line={"color": _rgba(col if col.startswith("#") else "#8A93A6", 0.45), "width": 10}))  # fmt: skip
    seeds = c.get("seeds") or list(range(1, len(d) + 1))
    jitter = [((i % 3) - 1) * 0.18 for i in range(len(d))]
    fig.add_trace(go.Scatter(x=[100 * x for x in d], y=jitter, mode="markers", showlegend=False,
                             marker={"size": 8, "color": T.tok("surface", m), "line": {"width": 1.75, "color": col}},
                             customdata=list(seeds)[: len(d)],
                             hovertemplate="seed %{customdata}: %{x:+.0f} pts<extra></extra>"))  # fmt: skip
    fig.add_trace(go.Scatter(x=[100 * mean], y=[0], mode="markers", showlegend=False,
                             marker={"symbol": "diamond", "size": 13, "color": col,
                                     "line": {"width": 1.5, "color": T.tok("bg", m)}},
                             hovertemplate="mean %{x:+.1f} pts<extra></extra>"))  # fmt: skip
    T.style(fig, height=height, m=m)
    lim = span or max([abs(100 * x) for x in d + [lo or 0, hi or 0, mean]] + [10]) * 1.12
    fig.update_xaxes(range=[-lim, lim], ticksuffix=" pts", zeroline=False, showgrid=True, nticks=7,
                     tickfont={"size": 10})  # fmt: skip
    fig.update_yaxes(visible=False, range=[-0.6, 0.6], fixedrange=True)
    fig.update_layout(margin={"l": 4, "r": 4, "t": 4, "b": 4}, hovermode="closest")
    return fig


def multiseed_figure(cf: pd.DataFrame, matchup: str, conditions: list[str], height: int = 300,
                     show_seeds: bool = True, labels: dict[str, str] | None = None) -> go.Figure:  # fmt: skip
    """Mean across seeds (line), 95% t-interval (band) and each seed (thin line) per condition, for one matchup."""
    m = T.mode()
    fig = go.Figure()
    fig.add_hline(y=0.5, line={"color": T.tok("axis", m), "width": 1})
    sub = cf[cf["matchup"] == matchup]
    for slot, cond in enumerate(conditions):
        s = sub[sub["condition"] == cond].sort_values("after_episode")
        if s.empty:
            continue
        c = condition_color(cond, slot, m)
        name = (labels or {}).get(cond, cond)
        if s["lo"].notna().all():
            _band(fig, s["after_episode"], s["lo"], s["hi"], c, name)
        if show_seeds:
            seeds = sorted({x for ss in s["seeds"] for x in ss})
            for seed in seeds:
                xs, ys = [], []
                for r in s.itertuples(index=False):
                    if seed in r.seeds:
                        xs.append(r.after_episode)
                        ys.append(r.per_seed[list(r.seeds).index(seed)])
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": c, "width": 0.9}, opacity=0.38,
                                         showlegend=False, legendgroup=name, hoverinfo="skip"))  # fmt: skip
        fig.add_trace(go.Scatter(
            x=s["after_episode"], y=s["mean"], mode="lines+markers", name=name, legendgroup=name,
            line={"color": c, "width": 2.4}, marker={"size": 6, "color": c, "line": {"width": 1.5, "color": T.tok("bg", m)}},
            customdata=s[["lo", "hi"]].to_numpy(),
            hovertemplate="%{y:.0%} mean · 95% CI %{customdata[0]:.0%}–%{customdata[1]:.0%}<extra>" + name + "</extra>",
        ))  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(hovermode="x unified", margin={"l": 8, "r": 8, "t": 30, "b": 8})
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", dtick=0.25, title="learned side's win rate")
    fig.update_xaxes(title="training games")
    return fig


def recall_dumbbell_figure(rows: list[dict], height: int = 220) -> go.Figure:
    """Detector recall on red disguised at 0.7: pretrained (open circle) → after the last retrain (filled), one
    thin line per seed and a bold one for the mean, one row per detector."""
    m = T.mode()
    fig = go.Figure()
    ys = [r["model"] for r in rows]
    for i, r in enumerate(rows):
        n = len(r["per_first"])
        for k, (a, b) in enumerate(zip(r["per_first"], r["per_last"], strict=True)):
            if a is None or b is None:
                continue
            off = (k - (n - 1) / 2) * 0.09
            fig.add_trace(go.Scatter(x=[a, b], y=[i + off, i + off], mode="lines", hoverinfo="skip", showlegend=False,
                                     line={"color": _rgba(T.VIOLET, 0.35), "width": 1.2}))  # fmt: skip
            fig.add_trace(go.Scatter(x=[b], y=[i + off], mode="markers", showlegend=False,
                                     marker={"size": 5, "color": _rgba(T.VIOLET, 0.6)},
                                     hovertemplate=f"seed {r['seeds'][k]}: {a:.0%} → {b:.0%}<extra>{r['model']}</extra>"))  # fmt: skip
        if r["first"] is not None and r["last"] is not None:
            fig.add_trace(go.Scatter(x=[r["first"], r["last"]], y=[i, i], mode="lines", hoverinfo="skip",
                                     showlegend=False, line={"color": T.VIOLET, "width": 3}))  # fmt: skip
            fig.add_trace(go.Scatter(x=[r["first"]], y=[i], mode="markers", name="Pretrained (v0)", showlegend=i == 0,
                                     marker={"size": 12, "color": T.tok("surface", m), "line": {"width": 2.5, "color": T.VIOLET}},
                                     hovertemplate="pretrained %{x:.0%}<extra>" + r["model"] + "</extra>"))  # fmt: skip
            fig.add_trace(go.Scatter(x=[r["last"]], y=[i], mode="markers+text", name="After the last retrain",
                                     showlegend=i == 0, text=[f"{r['last']:.0%}"], textposition="middle right",
                                     textfont={"family": T.FONT_MONO, "size": 11, "color": T.tok("text", m)},
                                     marker={"size": 12, "color": T.VIOLET, "line": {"width": 2, "color": T.tok("bg", m)}},
                                     hovertemplate="after retraining %{x:.0%}<extra>" + r["model"] + "</extra>"))  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_xaxes(range=[0, 1.0], tickformat=".0%", dtick=0.25, title="recall on red disguised at 0.7")
    fig.update_yaxes(tickvals=list(range(len(ys))), ticktext=ys, showgrid=False,
                     tickfont={"family": T.FONT_UI, "size": 12}, autorange="reversed")  # fmt: skip
    fig.update_layout(margin={"l": 8, "r": 30, "t": 30, "b": 8}, hovermode="closest")
    return fig


# ============================================================================================== decision web v4

ATTR_UP = "#2BB673"  # input feature raised the chosen move's value
ATTR_DOWN = "#E8853A"  # input feature lowered it


def _wrap(text: str, width: int = 16) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines[:3])


def decision_web_v4_figure(web: dict, actor: str, height: int = 400) -> go.Figure:
    """Centre: the chosen concrete move. Ring 1: the other top candidate moves, spoke width and dot size ∝ Q
    relative to the best shown; the runner-up gets a solid outline and a label. Ring 2: the agent's own most
    attributed input features (integrated gradients), dot size ∝ |attribution|, coloured by whether they raised or
    lowered the chosen move's value."""
    m = T.mode()
    fig = go.Figure()
    colour = TEAM.get(actor, T.tok("neutral", m))
    moves = web["moves"]
    feats = web.get("attribution") or []
    r1, r2 = 1.55, 3.1

    def spoke(x, y, weight, col, opacity=0.7, stop=0.0, dash="solid", start=0.0):
        r = math.hypot(x, y) or 1.0
        k0, k = start / r, (r - stop) / r
        fig.add_trace(go.Scatter(x=[x * k0, x * k], y=[y * k0, y * k], mode="lines", hoverinfo="skip", showlegend=False,
                                 opacity=opacity, line={"color": col, "width": 1 + 6 * weight, "dash": dash}))  # fmt: skip

    def outward(x, y) -> str:
        return ("top " if y > 0.35 else "bottom " if y < -0.35 else "middle ") + (
            "center" if abs(x) < 0.5 else "right" if x > 0 else "left")

    xs, ys, txt, hov, sz, lw, lc, pos = [], [], [], [], [], [], [], []
    for i, mv in enumerate(moves):
        ang = math.pi / 2 + 2 * math.pi * (i + 0.5) / max(1, len(moves))
        x, y = r1 * math.cos(ang), r1 * math.sin(ang)
        spoke(x, y, mv["weight"], colour, stop=0.1, opacity=0.3 + 0.45 * mv["weight"])
        xs.append(x)
        ys.append(y)
        tag = "<b>runner-up</b> · " if mv["runner_up"] else ""
        txt.append(f"{tag}{mv['label']}<br><span style='font-size:9.5px;opacity:.75'>Q {mv['q']:.3f}</span>")
        hov.append(f"<b>#{mv['rank']} {mv['long']}</b><br>Q-value {mv['q']:.4f}<br>relative to the best shown: "
                   f"{mv['weight']:.2f}")  # fmt: skip
        sz.append(12 + 12 * mv["weight"])
        lw.append(3 if mv["runner_up"] else 1.5)
        lc.append(colour if mv["runner_up"] else _rgba(colour, 0.6))
        pos.append(outward(x, y))
    if moves:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=txt, textposition=pos,
                                 hovertext=hov, hoverinfo="text", showlegend=False,
                                 textfont={"size": 10.5, "color": T.tok("text2", m), "family": T.FONT_UI},
                                 marker={"size": sz, "color": T.tok("surface", m), "line": {"color": lc, "width": lw}}))  # fmt: skip
    fx, fy, ft, fh, fc, fpos, fsz = [], [], [], [], [], [], []
    for i, f in enumerate(feats):
        ang = math.pi / 2 + 2 * math.pi * i / max(1, len(feats)) + math.pi / max(2, len(feats))
        x, y = r2 * math.cos(ang), r2 * math.sin(ang)
        col = ATTR_UP if f["attribution"] >= 0 else ATTR_DOWN
        # no spoke: the outer ring would cross the move labels; dot size carries |attribution| instead
        fx.append(x)
        fy.append(y)
        fc.append(col)
        ft.append(f"{_wrap(f['phrase'], 15)} <b>{f['attribution']:+.2f}</b>")
        val = f" (value {f['value']:.3g})" if isinstance(f.get("value"), (int, float)) else ""
        fh.append(f"<b>{f['phrase']}</b><br><span style='opacity:.7'>{f['name']}</span>{val}<br>"
                  f"attribution {f['attribution']:+.4f} to Q")  # fmt: skip
        fpos.append("top center" if y >= 0 else "bottom center")
        fsz.append(8 + 10 * f["weight"])
    if feats:
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="markers+text", text=ft, hovertext=fh, hoverinfo="text",
                                 textposition=fpos, textfont={"size": 10, "color": T.tok("muted", m)}, showlegend=False,
                                 marker={"size": fsz, "color": fc, "line": {"width": 2, "color": T.tok("bg", m)}}))  # fmt: skip
        for col, lab in ((ATTR_UP, "Input raised this move's value"), (ATTR_DOWN, "Input lowered it")):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=lab, marker={"size": 9, "color": col}))
    cv = web.get("chosen_value")
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", hoverinfo="text", showlegend=False,
                             hovertext=[f"<b>chosen: {web['chosen_long']}</b><br>Q-value {cv:.4f}" if cv is not None
                                        else f"<b>chosen: {web['chosen_long']}</b>"],
                             marker={"size": 72, "color": colour, "line": {"width": 0}}))  # fmt: skip
    fig.add_annotation(x=0, y=0, showarrow=False, font={"color": "#FFFFFF", "size": 11.5, "family": T.FONT_UI},
                       text=f"<b>{_wrap(web['chosen_label'], 12)}</b>" + (f"<br>Q {cv:.3f}" if cv is not None else ""))  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(legend={"y": -0.02, "yanchor": "top", "font": {"size": 11}}, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    ax_ = {"visible": False, "showgrid": False, "zeroline": False, "fixedrange": True}
    lim = 3.75 if feats else 2.4
    fig.update_xaxes(**ax_, range=[-lim - 0.9, lim + 0.9])
    fig.update_yaxes(**ax_, range=[-lim, lim], scaleanchor="x", scaleratio=1)
    return fig


def detector_shap_figure(features: list[dict], height: int | None = None) -> go.Figure:
    """Compact SHAP bars for the detector reading(s) of this turn: orange pushes the score toward malicious,
    green toward benign."""
    m = T.mode()
    feats = sorted(features, key=lambda f: abs(f["shap"]))
    labels = [f"{f['name'][:28]} · {f.get('model', '')}" for f in feats]
    fig = go.Figure(go.Bar(
        x=[f["shap"] for f in feats], y=labels, orientation="h", width=0.6,
        marker={"color": [SHAP_POS if f["shap"] >= 0 else SHAP_NEG for f in feats], "cornerradius": 3},
        customdata=[[f.get("node"), f.get("value"), f.get("raw")] for f in feats],
        hovertemplate="%{y}<br>SHAP %{x:+.3f} · host %{customdata[0]}<br>scaled %{customdata[1]} · raw %{customdata[2]}"
                      "<extra></extra>",
    ))  # fmt: skip
    T.style(fig, height=height or 44 + 22 * len(feats), m=m)
    fig.update_xaxes(title="← benign · SHAP · malicious →", zeroline=True, zerolinecolor=T.tok("axis", m),
                     tickformat="+.2f")  # fmt: skip
    fig.update_yaxes(showline=False, tickfont={"family": T.FONT_MONO, "size": 10.5})
    fig.update_layout(margin={"l": 4, "r": 8, "t": 4, "b": 4})
    return fig


DQN_PANELS = (  # (field, title, hover format)
    ("loss", "Training loss (Huber)", ".4f"),
    ("td_error", "TD error", ".3f"),
    ("q_mean", "Mean Q of chosen moves", "+.3f"),
    ("replay_size", "Replay memory (transitions)", ",.0f"),
    ("grad_steps", "Gradient steps", ",.0f"),
    ("epsilon", "Exploration rate ε", ".0%"),
)


def dqn_agent_figure(stats: pd.DataFrame, height: int = 420) -> go.Figure:
    """Q-network learning stats over training, small multiples (each panel its own measure and scale), red and
    blue lines in every panel."""
    m = T.mode()
    panels = [p for p in DQN_PANELS if p[0] in stats and stats[p[0]].notna().any()]
    cols = 3
    rows = max(1, math.ceil(len(panels) / cols))
    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.07, vertical_spacing=0.2,
                        subplot_titles=[t for _, t, _ in panels])  # fmt: skip
    for side in ("red", "blue"):
        s = stats[stats["side"] == side].sort_values("episode") if not stats.empty else stats
        if s.empty:
            continue
        c = TEAM[side]
        name = f"Learned {side}"
        for i, (field, _, fmt) in enumerate(panels):
            r, k = divmod(i, cols)
            sub = s.dropna(subset=[field])
            fig.add_trace(go.Scatter(x=sub["episode"], y=sub[field], mode="lines", name=name, legendgroup=side,
                                     showlegend=i == 0, line={"color": c, "width": 2},
                                     hovertemplate="%{y:" + fmt + "}<extra>" + name + "</extra>"), row=r + 1, col=k + 1)  # fmt: skip
    T.style(fig, height=height, m=m)
    fig.update_layout(hovermode="x unified", margin={"l": 8, "r": 8, "t": 54, "b": 8}, legend={"y": 1.14})
    for i, (field, _, _) in enumerate(panels):
        r, k = divmod(i, cols)
        if field == "epsilon":
            fig.update_yaxes(tickformat=".0%", range=[0, 1.02], row=r + 1, col=k + 1)
        elif field in ("replay_size", "grad_steps"):
            fig.update_yaxes(tickformat="~s", rangemode="tozero", row=r + 1, col=k + 1)
        elif field == "q_mean":
            fig.update_yaxes(tickformat="+.2f", row=r + 1, col=k + 1)
        else:
            fig.update_yaxes(rangemode="tozero", row=r + 1, col=k + 1)
        if r == rows - 1:
            fig.update_xaxes(title="training games", row=r + 1, col=k + 1)
    for a in fig.layout.annotations:
        a.font = {"size": 12, "color": T.tok("text2", m), "family": T.FONT_UI}
    return fig
