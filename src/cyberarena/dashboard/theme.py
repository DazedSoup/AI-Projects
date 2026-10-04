"""Design system for the dashboard (docs/contracts.md, "Dashboard design system"). The only place with CSS.

* **Tokens** (:data:`TOKENS`): 8 px spacing, 10 px card radius, 8 px control radius, one neutral ramp per mode,
  team red/blue, and violet reserved for learning/adaptation visuals.
* **CSS** (:func:`inject`): fonts, cards, metric tiles, toolbars, badges, chips, tables and widget sizing. Widgets
  are styled through ``data-testid`` / ARIA role attributes and the ``st-key-<key>`` class Streamlit puts
  on keyed containers, never through generated class names. Surfaces and ink are written with
  ``color-mix(currentColor …)`` so the same CSS is right in the light and the dark theme, including right after a
  theme switch.
* **Plotly** (:func:`register_templates`, :func:`style`): two templates (``cyberarena_dark`` / ``_light``)
  registered once; every figure goes through :func:`style`, so fonts, grid, hover and margins match.

Colour choices were checked with the dataviz palette validator: team red/blue/violet pass on both surfaces
except violet↔blue, which must never be two series told apart by colour alone (the learning accent is never
plotted next to team blue). Node states use three hues plus two neutrals, always paired with a marker shape.
"""

from __future__ import annotations

import html
from collections.abc import Iterable

import plotly.graph_objects as go
import plotly.io as pio

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

# ============================================================================================== tokens

RED = "#E5484D"
BLUE = "#3E8BFF"
VIOLET = "#8E7CFF"  # learning / adaptation only
SPACE = 8  # px scale unit
RADIUS_CARD = 10
RADIUS_CONTROL = 8
FONT_UI = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
FONT_MONO = "'JetBrains Mono', Consolas, ui-monospace, SFMono-Regular, monospace"

TOKENS = {
    "dark": {
        "bg": "#0B0E14",
        "surface": "#121722",
        "surface2": "#19202C",
        "raised": "#1E2633",
        "border": "#262E3B",
        "text": "#E8EBF1",
        "text2": "#A7AFBF",
        "muted": "#7D8597",
        "grid": "#1D2430",
        "axis": "#2C3442",
        "neutral": "#8A93A6",  # clean hosts, head-to-head line, unchosen bars
        "neutral_dim": "#4C5567",
        "compromised": RED,
        "detected": "#C98500",
        "patched": "#199E70",
        "good": "#2BB673",
        "warn": "#E8A33A",
    },
    "light": {
        "bg": "#F4F5F8",
        "surface": "#FFFFFF",
        "surface2": "#F0F2F6",
        "raised": "#FFFFFF",
        "border": "#D8DCE4",
        "text": "#0E1320",
        "text2": "#4A5264",
        "muted": "#687083",
        "grid": "#E8EBF0",
        "axis": "#CDD2DB",
        "neutral": "#8A93A6",
        "neutral_dim": "#B4BAC6",
        "compromised": RED,
        "detected": "#EDA100",
        "patched": "#1BAF7A",
        "good": "#1E8A54",
        "warn": "#B36A00",
    },
}

TEAM = {"red": RED, "blue": BLUE}
STATE_SYMBOL = {  # 3D-safe symbols (scatter3d has no triangles), reused in 2D so the key never changes
    "clean": "circle",
    "patched": "square",
    "compromised": "diamond",
    "detected": "x",
    "isolated": "square-open",
}
STATE_LABEL = {
    "clean": "Clean",
    "patched": "Patched",
    "compromised": "Compromised",
    "detected": "Detected",
    "isolated": "Isolated",
}

# sequential ramps (one hue, light→dark on light; dark→light on dark so "more" is always more contrast)
_VIOLET_RAMP = ["#ECE9FF", "#D3CCFF", "#B6AAFF", "#9C8BFF", "#8E7CFF", "#6F5CE6", "#5442C2", "#3D2E96"]
_RED_RAMP = ["#FDECEC", "#F9C9CA", "#F3A2A4", "#EC7A7D", "#E5484D", "#C7363B", "#A2292D", "#7A1E21"]
_BLUE_RAMP = ["#E7F0FF", "#C3D9FF", "#9CC0FF", "#6EA4FF", "#3E8BFF", "#2C6FD9", "#2056AD", "#173F80"]
RAMPS = {"violet": _VIOLET_RAMP, "red": _RED_RAMP, "blue": _BLUE_RAMP}


def mode() -> str:
    """``"dark"`` or ``"light"``: the viewer's Streamlit theme (dark when unknown, the configured base)."""
    if st is None:
        return "dark"
    try:
        t = st.context.theme.type
    except Exception:  # noqa: BLE001 - no script context (tests, scripts)
        t = None
    return "light" if t == "light" else "dark"


def tok(name: str, m: str | None = None) -> str:
    return TOKENS[m or mode()][name]


def state_colors(m: str | None = None) -> dict[str, str]:
    t = TOKENS[m or mode()]
    return {
        "clean": t["neutral"],
        "patched": t["patched"],
        "compromised": t["compromised"],
        "detected": t["detected"],
        "isolated": t["neutral"],
    }


def colorscale(hue: str, m: str | None = None) -> list[list]:
    """A Plotly colourscale for a sequential ramp; low values recede toward the surface in either mode."""
    ramp = RAMPS[hue]
    # light: pale -> deep; dark: near-surface -> bright (more = more contrast against the surface)
    steps = (
        ramp
        if (m or mode()) == "light"
        else ["#1C2030", ramp[7], ramp[6], ramp[5], ramp[4], ramp[3], ramp[2]]
    )
    n = len(steps) - 1
    return [[i / n, c] for i, c in enumerate(steps)]


# ============================================================================================== plotly


def _template(m: str) -> go.layout.Template:
    t = TOKENS[m]
    axis = {
        "gridcolor": t["grid"],
        "linecolor": t["axis"],
        "zeroline": False,
        "showline": True,
        "ticks": "",
        "tickfont": {"family": FONT_MONO, "size": 11, "color": t["muted"]},
        "title": {"font": {"family": FONT_UI, "size": 12, "color": t["text2"]}, "standoff": 10},
        "automargin": True,
    }
    scene_axis = {
        "backgroundcolor": "rgba(0,0,0,0)",
        "gridcolor": t["grid"],
        "zerolinecolor": t["axis"],
        "showbackground": False,
        "tickfont": {"family": FONT_MONO, "size": 10, "color": t["muted"]},
        "title": {"font": {"family": FONT_UI, "size": 11, "color": t["text2"]}},
    }
    return go.layout.Template(
        layout={
            "font": {"family": FONT_UI, "size": 12, "color": t["text2"]},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "colorway": [BLUE, RED, t["neutral"], VIOLET],
            "margin": {"l": 8, "r": 8, "t": 8, "b": 8, "pad": 0},
            "xaxis": axis,
            "yaxis": axis,
            "scene": {"xaxis": scene_axis, "yaxis": scene_axis, "zaxis": scene_axis},
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.0,
                "xanchor": "left",
                "x": 0,
                "font": {"size": 12, "color": t["text2"]},
                "bgcolor": "rgba(0,0,0,0)",
                "itemclick": "toggle",
                "itemdoubleclick": "toggleothers",
            },
            "hoverlabel": {
                "bgcolor": t["raised"],
                "bordercolor": t["border"],
                "font": {"family": FONT_UI, "size": 12, "color": t["text"]},
                "align": "left",
            },
            "hovermode": "closest",
            "colorscale": {"sequential": colorscale("violet", m)},
            "coloraxis": {
                "colorbar": {
                    "outlinewidth": 0,
                    "thickness": 10,
                    "tickfont": {"family": FONT_MONO, "size": 10, "color": t["muted"]},
                }
            },
            "annotationdefaults": {"font": {"family": FONT_UI, "size": 11, "color": t["muted"]}},
        }
    )


_REGISTERED = False


def register_templates() -> None:
    """Register ``cyberarena_dark`` and ``cyberarena_light`` with plotly.io (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    for m in ("dark", "light"):
        pio.templates[f"cyberarena_{m}"] = _template(m)
    _REGISTERED = True


def style(fig: go.Figure, height: int | None = None, m: str | None = None, **layout) -> go.Figure:
    """Apply the registered template for the current mode, plus per-figure layout overrides."""
    register_templates()
    fig.update_layout(template=f"cyberarena_{m or mode()}", **layout)
    if height:
        fig.update_layout(height=height)
    return fig


PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}
PLOTLY_CONFIG_3D = {"displayModeBar": "hover", "displaylogo": False, "responsive": True,
                    "modeBarButtonsToRemove": ["toImage", "resetCameraLastSave3d"]}  # fmt: skip


def chart(fig: go.Figure, key: str, three_d: bool = False) -> None:
    """``st.plotly_chart`` with the house config (no modebar on 2D, theme handled by our template)."""
    st.plotly_chart(fig, key=key, theme=None, config=PLOTLY_CONFIG_3D if three_d else PLOTLY_CONFIG)


# ============================================================================================== CSS

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --ca-red: #E5484D; --ca-blue: #3E8BFF; --ca-violet: #8E7CFF;
  --ca-s1: 8px; --ca-s2: 16px; --ca-s3: 24px; --ca-s4: 32px;
  --ca-r-card: 10px; --ca-r-ctl: 8px; --ca-ctl-h: 36px;
  --ca-mono: 'JetBrains Mono', Consolas, ui-monospace, SFMono-Regular, monospace;
  --ca-ui: Inter, 'Segoe UI', system-ui, -apple-system, sans-serif;
}
/* surfaces and ink derive from the theme's text colour, so they are right in both modes */
.stApp { --ca-ink2: color-mix(in srgb, currentColor 70%, transparent);
         --ca-muted: color-mix(in srgb, currentColor 54%, transparent);
         --ca-line: color-mix(in srgb, currentColor 10%, transparent);
         --ca-line2: color-mix(in srgb, currentColor 16%, transparent);
         --ca-tint: color-mix(in srgb, currentColor 3.5%, transparent);
         --ca-tint2: color-mix(in srgb, currentColor 6%, transparent); }

html, body, .stApp { font-feature-settings: "cv11", "ss01"; -webkit-font-smoothing: antialiased; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1440px; padding: 3.25rem 2rem 4rem; }
[data-testid="stSidebarContent"] { padding-top: 0.5rem; }
[data-testid="stSidebar"] { width: 340px !important; min-width: 340px !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] input { font-size: 13px; }
[data-testid="stCaptionContainer"] code, .ca-caption code, .ca-card-sub code, .ca-page-head code, .ca-empty code {
  color: inherit; background: var(--ca-tint2); border-radius: 4px; padding: 1px 5px; font-size: .9em;
  font-family: var(--ca-mono); }

/* headings */
h1 { letter-spacing: -0.022em; line-height: 1.15 !important; padding: 0 0 4px !important; }
h2, h3 { letter-spacing: -0.012em; }
[data-testid="stHeadingWithActionElements"] a { display: none !important; }

/* top navigation */
[data-testid="stTopNavLink"], [data-testid="stSidebarNavLink"] { border-radius: var(--ca-r-ctl); }

/* captions & labels */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--ca-muted); }
[data-testid="stWidgetLabel"] p { font-size: 12px !important; font-weight: 550; letter-spacing: .01em;
                                   color: var(--ca-ink2); }

/* ---------------------------------------------------------------- controls: one height, one radius */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-tertiary"] {
  min-height: var(--ca-ctl-h); height: var(--ca-ctl-h); padding: 0 14px; border-radius: var(--ca-r-ctl);
  font-weight: 550; font-size: 13.5px; white-space: nowrap; transition: background .12s, border-color .12s;
}
[data-testid="stBaseButton-secondary"] { background: var(--ca-tint2); border-color: var(--ca-line2); }
[data-testid="stBaseButton-secondary"]:hover { background: color-mix(in srgb, currentColor 10%, transparent); }
[data-testid="stBaseButton-primary"] { box-shadow: 0 1px 0 rgba(0,0,0,.18); }
[data-testid="stButton"] button p, [data-testid="stPageLink"] p { font-weight: 550; }
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"] {
  min-height: var(--ca-ctl-h); border-radius: var(--ca-r-ctl);
}
[data-testid="stButtonGroup"] button { min-height: 32px; border-radius: 7px; font-weight: 550; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--ca-mono); font-size: 11.5px; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { font-family: var(--ca-mono);
  font-size: 11px; color: var(--ca-muted); }

/* tabs */
[data-testid="stTabs"] [role="tablist"] { gap: 4px; border-bottom: 1px solid var(--ca-line); }
[data-testid="stTabs"] [role="tab"] { padding: 8px 12px; border-radius: 8px 8px 0 0; }
[data-testid="stTabs"] [role="tab"] p { font-weight: 550; font-size: 13.5px; }

/* expander as a quiet disclosure */
[data-testid="stExpander"] details { border-radius: var(--ca-r-card); border-color: var(--ca-line); }
[data-testid="stExpander"] summary p { font-weight: 550; }

/* ---------------------------------------------------------------- cards and toolbars (keyed containers) */
[class*="st-key-card"] { background: var(--ca-card, var(--ca-tint)); box-shadow: var(--ca-shadow, none); border: 1px solid var(--ca-line); border-radius: var(--ca-r-card);
                         padding: 18px 20px 16px; gap: 12px; }
[class*="st-key-toolbar"] { background: var(--ca-tint); border: 1px solid var(--ca-line); border-radius: var(--ca-r-card);
                            padding: 12px 16px; gap: 12px; align-items: flex-end; }
[class*="st-key-toolbar"] [data-testid="stWidgetLabel"] { min-height: 18px; }
[class*="st-key-playbar"] { align-items: center; gap: 8px; }
[class*="st-key-playbar"] [data-testid="stSlider"] { padding: 0 8px; }
[class*="st-key-tiles"] { gap: 12px; }

.ca-card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.ca-card-title { font-size: 15px; font-weight: 620; letter-spacing: -0.01em; margin: 0; }
.ca-card-sub { font-size: 12.5px; color: var(--ca-muted); margin: 2px 0 0; line-height: 1.45; }
.ca-eyebrow { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
              color: var(--ca-muted); margin: 0 0 2px; }
.ca-takeaway { display: flex; gap: 10px; align-items: baseline; border-top: 1px solid var(--ca-line);
               padding-top: 10px; margin-top: 2px; font-size: 13.5px; line-height: 1.5; }
.ca-takeaway::before { content: ""; flex: 0 0 6px; height: 6px; border-radius: 50%; background: currentColor;
                       opacity: .5; transform: translateY(-2px); }
.ca-takeaway.learn::before { background: var(--ca-violet); opacity: 1; }
.ca-notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px 24px; }
.ca-notes p { margin: 0; font-size: 12.5px; line-height: 1.55; color: var(--ca-ink2); padding-left: 16px;
              border-left: 2px solid var(--ca-line2); }
.ca-caption { font-size: 12.5px; color: var(--ca-muted); line-height: 1.5; margin: 0; }

/* page header */
.ca-page-head { margin: 0 0 4px; }
.ca-page-head h1 { font-size: 26px; font-weight: 650; letter-spacing: -0.022em; margin: 0; padding: 0; }
.ca-page-head p { margin: 4px 0 0; color: var(--ca-muted); font-size: 13.5px; max-width: 900px; }

/* metric tiles */
.ca-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.ca-tile { background: var(--ca-card, var(--ca-tint)); box-shadow: var(--ca-shadow, none); border: 1px solid var(--ca-line); border-radius: var(--ca-r-card);
           padding: 14px 16px 14px; position: relative; overflow: hidden; min-height: 112px; }
.ca-tile .k { font-size: 12px; font-weight: 550; color: var(--ca-ink2); display: flex; gap: 8px; align-items: center; }
.ca-tile .k .dot { width: 8px; height: 8px; border-radius: 2px; flex: 0 0 8px; }
.ca-tile .v { font-family: var(--ca-ui); font-variant-numeric: tabular-nums; font-size: 30px; font-weight: 620;
              letter-spacing: -0.02em; line-height: 1.1; margin: 8px 0 6px; }
.ca-tile .v small { font-size: 14px; font-weight: 500; color: var(--ca-muted); margin-left: 4px; letter-spacing: 0; }
.ca-tile .d { font-family: var(--ca-mono); font-size: 11.5px; color: var(--ca-muted); }
.ca-tile .d .up { color: #2BB673; } .ca-tile .d .down { color: #E8853A; }
.ca-tile.learn { border-color: color-mix(in srgb, var(--ca-violet) 35%, transparent); }
.ca-tile.empty .v { color: var(--ca-muted); }

/* st.metric (Lab live panel) styled like the tiles */
[data-testid="stMetric"] { background: var(--ca-card, var(--ca-tint)); border: 1px solid var(--ca-line); border-radius: var(--ca-r-card);
                           padding: 12px 14px; }
[data-testid="stMetricLabel"] p { font-size: 12px !important; font-weight: 550; color: var(--ca-ink2); }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-size: 22px !important; }

/* badges & chips */
.ca-badges { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.ca-badge { display: inline-flex; align-items: center; gap: 6px; height: 24px; padding: 0 9px; border-radius: 999px;
            font-size: 12px; font-weight: 550; background: var(--ca-tint2); border: 1px solid var(--ca-line);
            white-space: nowrap; }
.ca-badge .sw { width: 7px; height: 7px; border-radius: 50%; }
.ca-badge.red { background: color-mix(in srgb, var(--ca-red) 14%, transparent); border-color: color-mix(in srgb, var(--ca-red) 40%, transparent); }
.ca-badge.blue { background: color-mix(in srgb, var(--ca-blue) 14%, transparent); border-color: color-mix(in srgb, var(--ca-blue) 40%, transparent); }
.ca-badge.learn { background: color-mix(in srgb, var(--ca-violet) 14%, transparent); border-color: color-mix(in srgb, var(--ca-violet) 45%, transparent); }
.ca-chip { display: inline-flex; align-items: center; gap: 4px; padding: 1px 7px; border-radius: 5px;
           font-family: var(--ca-mono); font-size: 11px; font-weight: 500; background: var(--ca-tint2);
           border: 1px solid var(--ca-line); white-space: nowrap; }
.ca-chip.attack { border-color: color-mix(in srgb, var(--ca-red) 45%, transparent); }
.ca-chip.d3fend { border-color: color-mix(in srgb, var(--ca-blue) 45%, transparent); }
.ca-chip.learn { border-color: color-mix(in srgb, var(--ca-violet) 45%, transparent);
                background: color-mix(in srgb, var(--ca-violet) 10%, transparent); }
.ca-chip.strong { background: color-mix(in srgb, var(--ca-violet) 28%, transparent); font-weight: 600; }
details.ca-more summary { cursor: pointer; list-style: none; }
details.ca-more summary::-webkit-details-marker { display: none; }
details.ca-more > summary::after { content: " more"; font-size: 11.5px; color: var(--ca-muted); }
details.ca-more[open] > summary::after { content: ""; }
details.ca-more p { margin: 6px 0 0; font-size: 13.5px; line-height: 1.6; color: var(--ca-ink2); }
.ca-mono, .ca-id { font-family: var(--ca-mono); font-variant-numeric: tabular-nums; }
.ca-id { font-size: .92em; padding: 0 4px; border-radius: 4px; background: var(--ca-tint2); }

/* tables */
.ca-table-wrap { max-height: var(--ca-table-h, 360px); overflow: auto; border: 1px solid var(--ca-line);
                 border-radius: var(--ca-r-ctl); }
.ca-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }
.ca-table th { position: sticky; top: 0; z-index: 1; text-align: left; font-size: 11px; font-weight: 600;
               letter-spacing: .06em; text-transform: uppercase; color: var(--ca-muted); padding: 8px 12px;
               background: var(--ca-table-head, #121722); border-bottom: 1px solid var(--ca-line); white-space: nowrap; }
.ca-table td { padding: 8px 12px; border-bottom: 1px solid var(--ca-line); vertical-align: top; line-height: 1.45; }
.ca-table tr:last-child td { border-bottom: 0; }
.ca-table td.num { font-family: var(--ca-mono); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.ca-table td.mono { font-family: var(--ca-mono); white-space: nowrap; }
.ca-table tr.current td { background: color-mix(in srgb, currentColor 6%, transparent); }
.ca-table tr.current td:first-child { box-shadow: inset 2px 0 0 currentColor; }
.ca-table .actor { display: inline-flex; align-items: center; gap: 6px; font-weight: 550; text-transform: capitalize; }
.ca-table .actor::before { content: ""; width: 8px; height: 8px; border-radius: 2px; background: var(--ca-actor); }
.ca-table .ok { color: #2BB673; } .ca-table .fail { color: var(--ca-muted); }
.ca-table .why { color: var(--ca-ink2); max-width: 560px; }
.ca-table td:has(.why) { min-width: 240px; }
.ca-flag { font-family: var(--ca-mono); font-size: 11.5px; }
.ca-flag.good { color: #2BB673; } .ca-flag.warn { color: #E8853A; }

/* why-this-move */
.ca-why-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.ca-why-action { font-size: 17px; font-weight: 640; letter-spacing: -0.01em; }
.ca-why-text { font-size: 14px; line-height: 1.6; margin: 0; }
.ca-adapt { display: flex; gap: 8px; align-items: center; font-size: 12.5px; padding: 8px 10px;
            border-radius: var(--ca-r-ctl); background: color-mix(in srgb, var(--ca-violet) 10%, transparent);
            border: 1px solid color-mix(in srgb, var(--ca-violet) 32%, transparent); }
.ca-adapt b { font-weight: 600; }

/* empty state */
.ca-empty { border: 1px dashed var(--ca-line2); border-radius: var(--ca-r-card); padding: 28px 24px; text-align: center; }
.ca-empty .t { font-weight: 600; font-size: 14.5px; margin: 0 0 4px; }
.ca-empty .b { color: var(--ca-muted); font-size: 13px; margin: 0 auto; max-width: 560px; line-height: 1.55; }

/* legend keys drawn in HTML (state key under the network) */
.ca-key { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--ca-ink2); }
.ca-key span { display: inline-flex; align-items: center; gap: 6px; }
.ca-key svg { flex: 0 0 auto; }

/* run summary in the sidebar */
.ca-run { font-size: 12.5px; line-height: 1.55; color: var(--ca-ink2); }
.ca-run dt { color: var(--ca-muted); font-size: 11px; letter-spacing: .06em; text-transform: uppercase; margin-top: 8px; }
.ca-run dd { margin: 0; font-family: var(--ca-mono); font-size: 12px; }
.ca-brand { display: flex; align-items: center; gap: 10px; margin: 2px 0 6px; }
.ca-brand .mark { width: 26px; height: 26px; border-radius: 7px; display: grid; place-items: center;
                  background: linear-gradient(135deg, var(--ca-red), var(--ca-blue)); color: #fff; font-weight: 700; font-size: 13px; }
.ca-brand .name { font-weight: 650; font-size: 15px; letter-spacing: -0.01em; }
.ca-brand .tag { font-size: 11px; color: var(--ca-muted); }
.ca-public { font-size: 12px; line-height: 1.5; color: var(--ca-ink2); border: 1px solid var(--ca-line2);
             border-radius: 8px; padding: 8px 10px; margin: 4px 0 8px; background: var(--ca-tint); }
.ca-public b { color: inherit; font-weight: 620; }
.ca-public a { display: inline-block; margin-top: 4px; font-weight: 560; }

/* charts sit directly on the card: Streamlit paints the theme background into the SVG, undo that */
[data-testid="stPlotlyChart"] .main-svg { background: transparent !important; }
[data-testid="stPlotlyChart"] .bglayer rect.bg { fill-opacity: 0 !important; }
[data-testid="stPlotlyChart"] .gl-container { background: transparent !important; }

/* Plotly 2D chrome follows the live theme (SVG text; 3D/WebGL text comes from the template) */
[data-testid="stPlotlyChart"] .xtick text, [data-testid="stPlotlyChart"] .ytick text,
[data-testid="stPlotlyChart"] .cbaxis text { fill: var(--ca-muted) !important; }
[data-testid="stPlotlyChart"] .legendtext, [data-testid="stPlotlyChart"] .xtitle,
[data-testid="stPlotlyChart"] .ytitle, [data-testid="stPlotlyChart"] .g-gtitle text,
[data-testid="stPlotlyChart"] .textpoint text { fill: var(--ca-ink2) !important; }
[data-testid="stPlotlyChart"] .gridlayer path { stroke: var(--ca-line) !important; }
[data-testid="stPlotlyChart"] .zerolinelayer path { stroke: var(--ca-line2) !important; }
[data-testid="stPlotlyChart"] path.xlines-above, [data-testid="stPlotlyChart"] path.ylines-above { stroke: var(--ca-line2) !important; }
/* evidence: stat cards, did-it-work rows */
.ca-section { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--ca-muted);
              margin: 6px 0 -4px; }
.ca-stat .ca-eyebrow { margin: 0; }
.ca-stat-v { font-size: 34px; font-weight: 640; letter-spacing: -0.02em; line-height: 1.05; margin: 6px 0 4px;
             font-variant-numeric: tabular-nums; }
.ca-stat-v small { font-size: 14px; font-weight: 500; color: var(--ca-muted); letter-spacing: 0; }
.ca-stat-v.pos { color: var(--ca-good, #2BB673); } .ca-stat-v.neg { color: var(--ca-red); }
.ca-stat-sub { font-family: var(--ca-mono); font-size: 11.5px; color: var(--ca-muted); }
.ca-dw { display: flex; flex-direction: column; gap: 10px; }
.ca-dw-row { display: grid; grid-template-columns: 92px 1fr; gap: 16px; align-items: start; padding: 12px 0 2px;
             border-top: 1px solid var(--ca-line); }
.ca-dw-row:first-child { border-top: 0; padding-top: 2px; }
.ca-dw-ans { display: inline-flex; justify-content: center; align-items: center; height: 30px; border-radius: 8px;
             font-weight: 650; font-size: 13.5px; border: 1px solid var(--ca-line2); background: var(--ca-tint2); }
.ca-dw-ans.yes { color: var(--ca-good, #2BB673); border-color: color-mix(in srgb, var(--ca-good, #2BB673) 45%, transparent);
                 background: color-mix(in srgb, var(--ca-good, #2BB673) 12%, transparent); }
.ca-dw-ans.no { color: var(--ca-red); border-color: color-mix(in srgb, var(--ca-red) 45%, transparent);
                background: color-mix(in srgb, var(--ca-red) 12%, transparent); }
.ca-dw-row .q { margin: 0; font-size: 12px; color: var(--ca-muted); font-weight: 550; }
.ca-dw-row .h { margin: 2px 0 0; font-size: 16px; font-weight: 620; letter-spacing: -0.01em; }
.ca-dw-row .d { margin: 4px 0 0; font-size: 12.5px; color: var(--ca-ink2); line-height: 1.5; }
.ca-grid td.num { text-align: center; font-size: 15px; }
.ca-grid td.num .ca-caption { font-size: 11px; }
/* chosen vs runner-up strip (replay, v4 agents) */
.ca-ru { border: 1px solid var(--ca-line); border-left: 3px solid var(--ca-actor); border-radius: var(--ca-r-ctl);
         padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; background: var(--ca-tint); }
.ca-ru-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.ca-ru-margin { font-family: var(--ca-mono); font-size: 12px; color: var(--ca-ink2); }
.ca-ru-moves { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; font-size: 13.5px; }
.ca-ru-moves .c { font-weight: 620; } .ca-ru-moves .r { color: var(--ca-ink2); }
.ca-muted { color: var(--ca-muted); font-size: 12px; }
.ca-ru-facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 6px 14px; }
.ca-ru-fact { display: flex; flex-direction: column; gap: 1px; }
.ca-ru-fact .k { font-size: 11.5px; color: var(--ca-muted); }
.ca-ru-fact .v { font-family: var(--ca-mono); font-size: 12.5px; }
.ca-ru-fact .vs { color: var(--ca-muted); margin: 0 6px; font-size: 11px; }
.ca-ru-note { font-size: 11.5px; color: var(--ca-muted); }
/* overview hero + how-it-works flow */
.ca-hero { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 24px 32px; align-items: end;
           padding: 4px 0 6px; }
.ca-hero h1 { font-size: 30px !important; font-weight: 680; letter-spacing: -0.025em; margin: 2px 0 8px !important; }
.ca-hero .lead { font-size: 15px; line-height: 1.55; color: var(--ca-ink2); margin: 0; max-width: 680px; }
.ca-hero .hook { font-size: 14px; line-height: 1.55; margin: 10px 0 0; max-width: 680px; padding-left: 12px;
                 border-left: 2px solid var(--ca-violet); }
.ca-hero-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.ca-hero-stats .s { background: var(--ca-card, var(--ca-tint)); box-shadow: var(--ca-shadow, none);
                    border: 1px solid var(--ca-line); border-radius: var(--ca-r-card); padding: 12px 14px; }
.ca-hero-stats .k { font-size: 11.5px; font-weight: 550; color: var(--ca-ink2); line-height: 1.3; min-height: 30px; }
.ca-hero-stats .v { font-size: 26px; font-weight: 640; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
                    margin: 4px 0 2px; white-space: nowrap; }
.ca-hero-stats .d { font-size: 11px; color: var(--ca-muted); line-height: 1.4; }
@media (max-width: 1180px) { .ca-hero { grid-template-columns: 1fr; } }
.ca-flow { display: flex; align-items: stretch; gap: 6px; }
.ca-flow-step { flex: 1 1 0; display: flex; gap: 10px; padding: 10px 12px; border-radius: var(--ca-r-ctl);
                background: var(--ca-tint); border: 1px solid var(--ca-line); min-width: 0; }
.ca-flow-step .n { flex: 0 0 22px; height: 22px; border-radius: 6px; display: grid; place-items: center; font-size: 11.5px;
                   font-weight: 650; font-family: var(--ca-mono); background: var(--ca-tint2); border: 1px solid var(--ca-line2); }
.ca-flow-step .t { margin: 1px 0 2px; font-weight: 620; font-size: 13.5px; }
.ca-flow-step .b { margin: 0; font-size: 12px; line-height: 1.45; color: var(--ca-ink2); }
.ca-flow-arr { align-self: center; color: var(--ca-muted); font-size: 14px; }
@media (max-width: 1100px) { .ca-flow { flex-wrap: wrap; } .ca-flow-step { flex: 1 1 260px; } .ca-flow-arr { display: none; } }
.ca-dw-ans.suggestive { color: var(--ca-violet); border-color: color-mix(in srgb, var(--ca-violet) 45%, transparent);
                        background: color-mix(in srgb, var(--ca-violet) 10%, transparent); }
.ca-dw-ans.suggestive-negative { color: #E8853A; border-color: color-mix(in srgb, #E8853A 45%, transparent); }
.ca-lessons { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px 24px; }
.ca-lesson { padding-left: 14px; border-left: 2px solid var(--ca-line2); }
.ca-lesson .k { margin: 0; font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
                color: var(--ca-muted); }
.ca-lesson .v { margin: 4px 0 0; font-size: 13px; line-height: 1.55; color: var(--ca-ink2); }
/* laptop widths: a narrower sidebar leaves the two-column pages room */
@media (max-width: 1400px) {
  [data-testid="stSidebar"] { width: 288px !important; min-width: 288px !important; }
  [data-testid="stMainBlockContainer"] { padding-left: 1.25rem; padding-right: 1.25rem; }
}
"""


def inject() -> None:
    """Inject the stylesheet (call once per script run, before any page renders)."""
    register_templates()
    m = mode()
    t = TOKENS[m]
    shadow = "0 1px 2px rgba(15,23,42,.05), 0 1px 1px rgba(15,23,42,.03)" if m == "light" else "none"
    # mode-specific surfaces (the rest of the CSS derives from currentColor and needs no mode)
    st.html(f"<style>{_CSS}\n.stApp {{ --ca-card: {t['surface']}; --ca-shadow: {shadow}; --ca-good: {t['good']}; }}</style>")


# ============================================================================================== HTML components


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eb = f'<div class="ca-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.html(f'<div class="ca-page-head">{eb}<h1>{esc(title)}</h1>{sub}</div>')


def card_header(title: str, subtitle: str = "", right: str = "") -> None:
    """Title + one-line subtitle (plain-English caption of what the chart shows). ``right``: raw HTML."""
    sub = f'<p class="ca-card-sub">{subtitle}</p>' if subtitle else ""
    st.html(
        f'<div class="ca-card-head"><div><p class="ca-card-title">{esc(title)}</p>{sub}</div>'
        f"<div>{right}</div></div>"
    )


def takeaway(text: str, learning: bool = False) -> None:
    """The single computed takeaway line that ends every chart card."""
    st.html(f'<div class="ca-takeaway{" learn" if learning else ""}"><span>{text}</span></div>')


def caption(text: str) -> None:
    st.html(f'<p class="ca-caption">{text}</p>')


def badge(text: str, tone: str = "", swatch: str | None = None) -> str:
    sw = f'<span class="sw" style="background:{swatch}"></span>' if swatch else ""
    return f'<span class="ca-badge {tone}">{sw}{esc(text)}</span>'


def badges(items: Iterable[str]) -> None:
    st.html(f'<div class="ca-badges">{"".join(items)}</div>')


def chip(text: str, kind: str = "") -> str:
    return f'<span class="ca-chip {kind}">{esc(text)}</span>'


def tile(label: str, value: str, delta: str = "", swatch: str | None = None, unit: str = "",
         learning: bool = False, empty: bool = False, help: str = "") -> str:  # fmt: skip
    sw = f'<span class="dot" style="background:{swatch}"></span>' if swatch else ""
    cls = " ".join(c for c in ("ca-tile", "learn" if learning else "", "empty" if empty else "") if c)
    u = f"<small>{esc(unit)}</small>" if unit else ""
    t = f' title="{esc(help)}"' if help else ""
    return (
        f'<div class="{cls}"{t}><div class="k">{sw}{esc(label)}</div><div class="v">{esc(value)}{u}</div>'
        f'<div class="d">{delta}</div></div>'
    )


def tiles(items: Iterable[str]) -> None:
    st.html(f'<div class="ca-tiles">{"".join(items)}</div>')


def delta_html(diff: float | None, fmt: str, suffix: str, good_up: bool = True) -> str:
    if diff is None:
        return ""
    if abs(diff) < 1e-9:
        return f"no change {esc(suffix)}"
    up = diff > 0
    cls = "up" if up == good_up else "down"
    return f'<span class="{cls}">{"▲" if up else "▼"} {fmt.format(abs(diff))}</span> {esc(suffix)}'


def empty_state(title: str, body: str) -> None:
    st.html(f'<div class="ca-empty"><p class="t">{esc(title)}</p><p class="b">{body}</p></div>')


def table(columns: list[tuple[str, str]], rows: list[dict], height: int = 360, current: int | None = None,
          head_bg: str | None = None) -> str:  # fmt: skip
    """A styled HTML table. ``columns``: ``[(key, header)]``; a key ending in ``:num``/``:mono`` sets the cell class.
    Row dicts hold *already escaped* HTML per key. ``current``: row index to highlight."""
    hb = head_bg or tok("surface")
    ths = "".join(f"<th>{esc(h)}</th>" for _, h in columns)
    body = []
    for i, r in enumerate(rows):
        tds = []
        for k, _ in columns:
            name, _, cls = k.partition(":")
            tds.append(f'<td class="{cls}">{r.get(name, "")}</td>')
        body.append(f'<tr class="{"current" if i == current else ""}">{"".join(tds)}</tr>')
    return (
        f'<div class="ca-table-wrap" style="--ca-table-h:{height}px;--ca-table-head:{hb}">'
        f'<table class="ca-table"><thead><tr>{ths}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def state_key_html(m: str | None = None) -> str:
    """HTML legend for node states: shape + colour, drawn with CSS (the plot legend is hidden to save space)."""
    cols = state_colors(m)
    shape = {
        "circle": "width:9px;height:9px;border-radius:50%;background:{c}",
        "square": "width:9px;height:9px;border-radius:1px;background:{c}",
        "diamond": "width:8px;height:8px;background:{c};transform:rotate(45deg)",
        "square-open": "width:9px;height:9px;border-radius:1px;border:2px solid {c};box-sizing:border-box",
    }
    out = []
    for s, sym in STATE_SYMBOL.items():
        if sym == "x":
            mark = f'<i style="font-style:normal;font-weight:700;color:{cols[s]};font-size:13px;line-height:9px">✕</i>'
        else:
            mark = f'<i style="display:inline-block;{shape[sym].format(c=cols[s])}"></i>'
        out.append(f"<span>{mark}{STATE_LABEL[s]}</span>")
    out.append(
        '<span><i style="display:inline-block;width:12px;height:12px;border-radius:50%;'
        'border:2px solid #D9A33A;box-sizing:border-box"></i>Crown jewel</span>'
    )
    return f'<div class="ca-key">{"".join(out)}</div>'
