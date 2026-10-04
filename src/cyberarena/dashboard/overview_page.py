"""Overview: what the project is, its headline multi-seed numbers, how it works, then the selected run's
metrics and the three headline charts (reads small files only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cyberarena.dashboard import charts, common
from cyberarena.dashboard import evidence as E
from cyberarena.dashboard import learning as LL
from cyberarena.dashboard import loaders as L
from cyberarena.dashboard import theme as T

run = common.current_run()
info = common.info(run)
eps, evals = common.summary(run)
curve = L.eval_curve(evals)
lr = common.learning(run)
enr = common.enriched(run)


@st.cache_data(show_spinner=False, max_entries=4)
def headline(root: str, stamp) -> dict | None:
    """Headline numbers from the default experiment (``main`` when it has finished)."""
    e = E.default_experiment(E.list_experiments(Path(root)))
    if e is None or not e["aggregate"]:
        return None
    agg = E.load_aggregate(e["dir"]) or {}
    late = agg.get("late_mean") or {}
    blue = {c: (late.get(c) or {}).get("blue_learned_vs_red_baseline", {}).get("mean") for c in late}
    red = {c: (late.get(c) or {}).get("red_learned_vs_blue_baseline", {}).get("mean") for c in late}
    ar = E.arms_race_rows(agg)
    gain = sum(r["last"] - r["first"] for r in ar) / len(ar) if ar else None
    used = agg.get("runs_used") or {}
    n_runs = sum(len(v) for v in used.values())
    return {"name": e["name"], "n_seeds": agg.get("n_seeds"), "conditions": list(agg.get("conditions") or []),
            "episodes": agg.get("episodes"), "n_runs": n_runs, "blue": blue, "red": red, "gain": gain,
            "n_models": len(ar), "arms": ar, "did": E.did_it_work(agg)}  # fmt: skip


def _exp_stamp() -> tuple:
    root = common.runs_dir() / E.EXPERIMENTS_DIR
    if not root.is_dir():
        return ()
    return tuple(sorted((p.name, L.file_stamp(p / "aggregate.json")) for p in root.iterdir() if p.is_dir()))


hl = headline(str(common.runs_dir()), _exp_stamp())


def reference_blue() -> tuple[float, int, str] | None:
    """Learned blue's win rate against the scripted red at the reference run's final checkpoint."""
    runs = L.list_runs(common.runs_dir())
    if not runs:
        return None
    ref = common.default_run(runs)
    c = L.eval_curve(common.summary(ref)[1])
    b = c[c["side"] == "blue"].sort_values("after_episode") if not c.empty else c
    if b.empty:
        return None
    return float(b["win_rate"].iloc[-1]), int(b["after_episode"].iloc[-1]), common.info(ref)["label"] or ref.name


def hero_hook(did: dict[str, str]) -> str:
    """One computed sentence on what the main experiment found (never claims more than its answers)."""
    det, red, blue = did.get("detectors"), did.get("red"), did.get("blue")
    if det != "yes":
        return ""
    parts = ["retraining makes blue's detectors catch far more disguised attacks"]
    if red == "yes":
        parts.append("the attacker gains from the arms race too")
    blue_txt = {"yes": "and the defender wins significantly more against disguise",
                "no": "yet the defender wins significantly less",
                "suggestive": "while the defender's gain against disguise is suggestive but not established",
                "suggestive-negative": "while the defender may do slightly worse (not established)"}.get(
        blue or "", "while the defender shows no clear difference")  # fmt: skip
    return "The finding: " + "; ".join(parts) + ", " + blue_txt + "."


def hero() -> None:
    stats = []
    hook = ""
    ref = reference_blue()
    if ref is not None:
        rate, after, label = ref
        stats.append(("Learned blue beats scripted red", f"{rate:.0%}",
                      f"final checkpoint (after {after:,} games) of the {label} run"))  # fmt: skip
    if hl:
        ar = hl.get("arms") or []
        if ar:
            lo = min(r["last"] / r["first"] for r in ar if r["first"])
            hi = max(r["last"] / r["first"] for r in ar if r["first"])
            stats.append(("Detector recall on disguised red", f"{lo:.0f}–{hi:.0f}×" if round(hi) > round(lo) else f"{hi:.1f}×",
                          " · ".join(f"{r['model']} {E.pct(r['first'])}→{E.pct(r['last'])}" for r in ar)
                          + " after retraining"))  # fmt: skip
        if hl["n_seeds"]:
            stats.append(("Independent seeds", f"{hl['n_seeds']}",
                          (f"× {len(hl['conditions'])} conditions · {hl['n_runs'] * int(hl['episodes'] or 0):,} "
                          "training games")))  # fmt: skip
        hook = hero_hook({r["key"]: r["answer"] for r in hl["did"]})
    st.html(
        '<div class="ca-hero ca-page-head"><div class="ca-hero-text">'
        '<div class="ca-eyebrow">Simulation only · reinforcement learning · TensorFlow</div>'
        "<h1>Red vs blue, learned from scratch</h1>"
        '<p class="lead">Two neural-network agents, an attacker and a defender, learn by playing thousands of games on a '
        "simulated company network, while the defender's machine-learning detectors are retrained on what it catches "
        "and the attacker learns to disguise itself.</p>"
        + (f'<p class="hook">{T.esc(hook)}</p>' if hook else "")
        + "</div>"
        + ('<div class="ca-hero-stats">' + "".join(
            f'<div class="s"><div class="k">{T.esc(k)}</div><div class="v">{T.esc(v)}</div><div class="d">{T.esc(d)}</div></div>'
            for k, v, d in stats) + "</div>" if stats else "")
        + "</div>"
    )  # fmt: skip
    if hl:
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            if st.button("See the evidence", key="ov_to_evidence", icon=":material/fact_check:", type="primary"):
                st.switch_page(common.PAGES["evidence"])
            T.caption(f"Recall and seed numbers from the <code>{T.esc(hl['name'])}</code> experiment "
                      f"(<code>runs/experiments/{T.esc(hl['name'])}/aggregate.json</code>).")  # fmt: skip


FLOW = [
    ("Data", ("Three public datasets: Android app permissions (NATICUSdroid), phishing websites (UCI), network flows "
             "(NSL-KDD), split into disjoint partitions.")),
    ("Detectors", ("Three TensorFlow classifiers score every host's activity: malware, phishing, network. In adaptive "
                  "runs they are retrained on what blue confirms.")),
    ("Arena", ("A 12–16 host network from DMZ to crown jewel. Red phishes, exploits and moves laterally; blue monitors, "
              "isolates and patches. Red may blend its activity toward normal traffic.")),
    ("Agents", ("Each side is a Q-network that scores every legal concrete move (action and host) and learns from "
               "replayed experience; scripted opponents test them at every checkpoint.")),
    ("Explanations", ("SHAP for the detectors, integrated gradients for the agents' choices, MITRE ATT&CK / D3FEND tags "
                     "and a plain-English rationale for each move.")),
]


def how_it_works() -> None:
    steps = []
    for i, (t, body) in enumerate(FLOW, 1):
        steps.append(f'<div class="ca-flow-step"><div class="n">{i}</div><div><p class="t">{t}</p>'
                     f'<p class="b">{T.esc(body)}</p></div></div>')  # fmt: skip
    with common.card("how"):
        T.card_header("How it works", "Data → detectors → arena → agents → explanations. Every page reads the files "
                      "this pipeline writes; nothing here touches a real network.")  # fmt: skip
        st.html('<div class="ca-flow">' + '<div class="ca-flow-arr">→</div>'.join(steps) + "</div>")


hero()
how_it_works()

games = f"{info['games']:,} training games" if info["games"] else "training games"
kind = L.agent_kind(info)
st.html(
    f'<p class="ca-section">Selected run · {T.esc(info["label"] or run.name)} · {games}'
    + (f" · {T.esc(kind)}" if kind else "") + "</p>"
)


# ------------------------------------------------------------------------------------------------ hero tiles
def side_rates(side: str) -> tuple[float | None, float | None, int | None, int | None]:
    sub = curve[curve["side"] == side].sort_values("after_episode") if not curve.empty else curve
    if sub.empty:
        return None, None, None, None
    return (float(sub["win_rate"].iloc[0]), float(sub["win_rate"].iloc[-1]),
            int(sub["after_episode"].iloc[0]), int(sub["after_episode"].iloc[-1]))  # fmt: skip


tiles = []
for side, opp in (("red", "blue"), ("blue", "red")):
    first, last, a0, _ = side_rates(side)
    tiles.append(
        T.tile(
            f"Learned {side} vs scripted {opp}",
            "—" if last is None else f"{last:.0%}",
            T.delta_html(
                None if first is None else (last - first) * 100, "{:.0f} pts", f"since checkpoint {a0:,}"
            )
            if first is not None
            else "no evaluations yet",
            swatch=T.TEAM[side],
            empty=last is None,
            help="Win rate of the learned agent against the scripted opponent at the latest checkpoint.",
        )
    )
frozen = info.get("frozen")
why_empty = "detectors frozen in this run" if frozen else "this run predates adaptive learning"
if lr is not None and not (lr.updates.empty and lr.evasion.empty):
    n_up = len(lr.updates)
    latest = lr.updates.sort_values("episode").iloc[-1] if n_up else None
    tiles.append(
        T.tile("Detector updates", f"{n_up}",
               f"{len(set(lr.updates['model']))} sensors · latest {latest['model']} v{int(latest['version'])}"
               if latest is not None else "none yet", swatch=T.VIOLET, learning=True,
               help="Online fine-tunes of blue's TensorFlow detectors during training."))  # fmt: skip
    cur, first = LL.current_evasion(lr), LL.first_evasion(lr)
    if cur:
        top = max(cur, key=cur.get)
        rest = " · ".join(f"{m} {v:.1f}" for m, v in cur.items() if m != top)
        tiles.append(
            T.tile("Red evasion now", f"{cur[top]:.1f}", f"highest on {top}" + (f" · {rest}" if rest else ""),
                   unit=f"on {top}", swatch=T.RED, learning=True,
                   help="How far red blends its activity toward benign traffic, per sensor (0 = none, 1 = fully)."))  # fmt: skip
    else:
        tiles.append(T.tile("Red evasion now", "—", "no levels logged", empty=True, learning=True))
else:
    tiles.append(T.tile("Detector updates", "0" if frozen else "—", why_empty, empty=True, learning=True))
    tiles.append(T.tile("Red evasion now", "—", why_empty, empty=True, learning=True))

qs_now = qs_first = None
qs_src = ""
if lr is not None and lr.dqn:
    last = lr.stats.sort_values("episode").groupby("side").tail(1)
    steps = last["grad_steps"].dropna()
    mem = last["replay_size"].dropna()
    tiles.append(T.tile("Gradient steps", f"{int(steps.sum()):,}" if not steps.empty else "—",
                        f"red + blue Q-networks · replay memory {int(mem.max()):,} moves" if not mem.empty else "red + blue",
                        empty=steps.empty, help="Training updates of the two agents' Q-networks."))  # fmt: skip
elif lr is not None and not lr.stats.empty and lr.stats["n_states"].notna().any():
    last = lr.stats.sort_values("episode").groupby("side").tail(1)
    first = lr.stats.sort_values("episode").groupby("side").head(1)
    qs_now, qs_first = int(last["n_states"].sum()), int(first["n_states"].sum())
    qs_src = f"since game {int(first['episode'].min()):,}"
else:
    q = common.q_states(run)
    if not q.empty:
        qs_now = int(q[q["after_episode"] == q["after_episode"].max()]["n_states"].sum())
        qs_first = int(q[q["after_episode"] == q["after_episode"].min()]["n_states"].sum())
        qs_src = f"since checkpoint {int(q['after_episode'].min()):,}"
if not (lr is not None and lr.dqn):
    tiles.append(
        T.tile("Q-states learned", "—" if qs_now is None else f"{qs_now:,}",
               T.delta_html(None if qs_first is None else qs_now - qs_first, "{:,.0f}", qs_src) if qs_now is not None
               else ("Q-table checkpoints " + common.NOT_IN_SHOWCASE if common.public() else "no Q-tables saved"),
               empty=qs_now is None,
               help="Distinct situations in the red and blue Q-tables combined."))  # fmt: skip
T.tiles(tiles)

# ------------------------------------------------------------------------------------------------ charts
left, right = st.columns([1.45, 1], gap="medium")
with left, common.card("winrate"):
    T.card_header(
        "Win rate against scripted opponents",
        "At every checkpoint each learned agent plays fresh games against a scripted opponent. Bands are 95% "
        "confidence intervals; the grey line is red's share of recent learned-vs-learned training games.",
    )
    if curve.empty and eps.empty:
        T.empty_state("No evaluations yet", "This run has not written any summary rows.")
    else:
        h2h = L.rolling_head_to_head(eps, 100)
        T.chart(charts.win_rate_figure(curve, h2h, 100, height=340), key="ov_winrate")
        bits = []
        for side, opp in (("red", "blue"), ("blue", "red")):
            f, la, a0, a1 = side_rates(side)
            if la is None:
                continue
            sub = curve[(curve["side"] == side) & (curve["after_episode"] == a1)].iloc[0]
            half = (sub["ci_high"] - sub["ci_low"]) / 2
            bits.append(f"learned {side} wins {la:.0%} against scripted {opp} after {a1:,} games "
                        f"(±{100 * half:.0f} pts), from {f:.0%} at checkpoint {a0:,}")  # fmt: skip
        line = "; ".join(bits)
        T.takeaway(line[:1].upper() + line[1:] + "." if bits else "No checkpoint evaluations yet.")

with right, common.card("armsrace"):
    T.card_header(
        "Arms race",
        "Red raises its evasion when it keeps getting caught; blue's detector is retrained and recovers. "
        "Top: detector recall at red's current level. Bottom: red's evasion.",
    )
    if frozen:
        T.empty_state(
            "Detectors were frozen in this run",
            "It was trained with adaptive learning off, as a comparison: blue kept the pretrained detectors and red "
            "never disguised its activity. The Learning page compares it with an adaptive run.",
        )
    elif lr is None or lr.updates.empty and lr.evasion.empty:
        T.empty_state(
            "This run predates adaptive learning",
            "It has no <code>learning.jsonl</code>: detectors stayed fixed and red never adapted its evasion. "
            + ("Pick another run in the sidebar to see the arms race." if common.public()
               else "Train a new run with adaptive learning on to see the arms race here."),
        )
    else:
        # the sensor with the clearest back-and-forth tells the story best in a compact preview
        model = LL.headline_model(lr) or lr.models[0]
        T.chart(charts.arms_race_figure(lr, [model], height=320, compact=True), key="ov_arms")
        T.takeaway(LL.arms_race_takeaway(lr, [model]), learning=True)
        if st.button("Open the Learning page", key="ov_to_learning", icon=":material/arrow_forward:"):
            st.switch_page(common.PAGES["learning"])

left, right = st.columns([1.45, 1], gap="medium")
with left, common.card("length"):
    T.card_header(
        "How long games last",
        "Median length of the last 150 training games each side won. Shorter red wins mean red found a more "
        "direct path to the crown jewel.",
    )
    if eps.empty:
        T.empty_state("No training games logged", "summary.jsonl has no game rows.")
    else:
        T.chart(charts.game_length_figure(eps, 150, height=280), key="ov_length")
        msg = "Not enough games of each outcome for a trend yet."
        red = eps[eps["winner"] == "red"].dropna(subset=["turns"])
        if len(red) >= 40:
            k = max(20, len(red) // 5)
            a, b = red["turns"].iloc[:k].median(), red["turns"].iloc[-k:].median()
            msg = (f"Red's winning games went from a median of {a:.0f} turns early in training to {b:.0f} in the "
                   f"last {k} red wins.")  # fmt: skip
        T.takeaway(msg)


def best_replay() -> tuple[int, str] | None:
    """The most representative narrated game: a median-length win for learned red at the last checkpoint."""
    if not enr:
        return None
    cat = L.episode_catalog(enr, eps.iloc[0:0], evals.iloc[0:0])
    cat = cat[cat["enriched"]]
    if cat.empty:
        return None
    pref = cat[(cat["matchup"] == "red_learned_vs_blue_baseline") & (cat["winner"] == "red")]
    if not pref.empty:
        top = pref["after_episode"].max()
        pref = pref[pref["after_episode"] == top] if pd.notna(top) else pref
        pref = pref.iloc[(pref["turns"] - pref["turns"].median()).abs().argsort()]
        r = pref.iloc[0]
        why = "A typical win for learned red at the final checkpoint."
    else:
        r = cat.iloc[0]
        why = "The first narrated game of this run."
    return int(r["episode"]), why


with right, common.card("bestgame"):
    T.card_header(
        "Watch a representative game", "Step through it move by move, with the reasoning behind each move."
    )
    pick = best_replay()
    if pick is None:
        T.empty_state(
            "No narrated games yet",
            "Narrated games for this run are " + common.NOT_IN_SHOWCASE + "." if common.public()
            else "Run enrichment to narrate a selection of games, then replay them.",
        )  # fmt: skip
    else:
        ep, why = pick
        turns = enr[ep]
        last = turns[-1]
        g = common.graph(run)
        if g is not None:
            T.chart(charts.host_network_2d(g, last, height=200, compact=True), key="ov_thumb")
        T.badges([
            T.badge(f"{last.get('winner', '?')} wins", last.get("winner") or "", T.TEAM.get(last.get("winner"))),
            T.badge(f"{len(turns)} turns"),
            T.badge(L.matchup_label(last.get("matchup"))),
        ])  # fmt: skip
        T.caption(f"<span class='ca-mono'>Game {ep}</span> — {why}")
        if st.button("Open replay", key="ov_open_replay", type="primary", icon=":material/play_arrow:"):
            common.open_replay(ep)
