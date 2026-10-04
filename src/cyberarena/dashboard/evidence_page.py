"""Evidence: is this real? Multi-seed experiments from ``runs/experiments/<name>/{manifest,aggregate}.json``.

Every number comes from the arena's ``aggregate.json`` (means across seeds, 95% t-intervals, paired t-tests);
the page only words them. Verdicts are computed from those numbers and say "no significant difference" when
that is the result. Factorial families (``diag-*``: one single-condition experiment per cell) are combined
here from the cells' own per-seed values, paired by seed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cyberarena.dashboard import charts, common
from cyberarena.dashboard import evidence as E
from cyberarena.dashboard import loaders as L
from cyberarena.dashboard import showcase as SC
from cyberarena.dashboard import theme as T

ss = st.session_state
STATS = {"late_mean": "Second half of training", "final": "Final checkpoint"}
FAMILY = "family:"


@st.cache_data(show_spinner=False, max_entries=8)
def _experiments(root: str, stamp) -> list[dict]:
    return E.list_experiments(Path(root))


def experiments() -> list[dict]:
    root = common.runs_dir() / E.EXPERIMENTS_DIR
    stamp = tuple(sorted((p.name, L.file_stamp(p / "manifest.json"), L.file_stamp(p / "aggregate.json"))
                         for p in root.iterdir() if p.is_dir())) if root.is_dir() else ()  # fmt: skip
    return _experiments(str(common.runs_dir()), stamp)


@st.cache_data(show_spinner=False, max_entries=16)
def _aggregate(exp_dir: str, stamp) -> dict | None:
    return E.load_aggregate(Path(exp_dir))


def aggregate(e: dict) -> dict | None:
    return _aggregate(str(e["dir"]), L.file_stamp(Path(e["dir"]) / "aggregate.json"))


def to_lab() -> None:
    ss["lab_kind"] = "Experiment"
    st.switch_page(common.PAGES["lab"])


CLI = ("python -m cyberarena.arena.experiment --name main --seeds 1-5 --conditions adaptive,frozen "
       "--episodes 2000")  # fmt: skip

T.page_header(
    "Evidence",
    "Is it real? One training run can get lucky, so each question here is answered by repeating training with "
    "independent seeds and comparing conditions seed by seed. Verdicts are computed from the numbers.",
)

exps = experiments()
fams = E.factorial_families(exps)
if not exps:
    with common.card("ev-empty"):
        if common.public():
            T.empty_state("No experiments in this showcase", "The author hasn't published any multi-seed experiments.")
            st.stop()
        T.empty_state(
            "No experiments yet",
            "An experiment trains the same setup several times with different seeds, once per condition (for example "
            f"adaptive vs frozen detectors). Run one from the command line:<br><code>{T.esc(CLI)}</code><br>"
            "or set one up in the Simulation Lab. Results land here when every run has finished.",
        )
        if st.button("Set up an experiment in the Lab", key="ev_to_lab", type="primary", icon=":material/science:"):
            to_lab()
    st.stop()

# ------------------------------------------------------------------------------------------------ toolbar
options = [e["name"] for e in exps] + [FAMILY + f["name"] for f in fams]
by_name = {e["name"]: e for e in exps}
fam_by = {FAMILY + f["name"]: f for f in fams}
opened = ss.pop("_open_experiment", None)
if opened in options:
    ss["ev_exp"] = opened
if ss.get("ev_exp") not in options:
    d = E.default_experiment(exps)
    ss["ev_exp"] = d["name"] if d else options[0]


def _publish_names(o: str) -> list[str]:
    """Experiments behind one picker option (a factorial family publishes all its cells)."""
    return [c["exp"]["name"] for c in fam_by[o]["cells"]] if o in fam_by else [o]


def _on_publish(o: str, key: str) -> None:
    for name in _publish_names(o):
        SC.set_published(common.runs_dir(), "experiments", name, bool(ss[key]))


def publish_toggle(o: str) -> None:
    """Admin only: add or remove this experiment from the public showcase (``runs/.showcase.json``)."""
    names = _publish_names(o)
    published = set(SC.read_selection(common.runs_dir())["experiments"])
    key = f"ev_publish:{o}"
    ss[key] = all(n in published for n in names)  # always mirror the file (the Lab may have changed it)
    st.toggle("Publish", key=key, on_change=_on_publish, args=(o, key),
              help="Include this experiment in the public showcase (runs/.showcase.json). Export it from the "
              "Showcase panel in the Simulation Lab." + (" A factorial publishes all of its cells." if o in fam_by
                                                         else ""))  # fmt: skip


def exp_label(o: str) -> str:
    if o in fam_by:
        f = fam_by[o]
        return f"{f['name']}-* factorial · {len(f['cells'])} cells · {len(f['seeds'])} seeds"
    return E.experiment_label(by_name[o])


with common.toolbar("evidence"):
    choice = st.selectbox("Experiment", options, key="ev_exp", format_func=exp_label, width=420,
                          help="Folders under runs/experiments/. A factorial groups the diag-* experiments, one per "
                          "cell of the design.")  # fmt: skip
    stat = st.segmented_control("Measured over", list(STATS), key="ev_stat", default="late_mean", required=True,
                                format_func=STATS.get,
                                help="Second half of training: each seed's mean over checkpoints after half the games "
                                "(steadier). Final checkpoint: the last evaluation only.")  # fmt: skip
    if common.admin():
        publish_toggle(choice)


def stat_card(c: dict, key: str, span: float | None = None) -> None:
    """One contrast: the difference, its interval and p-value, the per-seed dots and the verdict."""
    sig = E.significant(c)
    d = c.get("diff_mean") or 0.0
    tone = ("pos" if d > 0 else "neg") if sig else "ns"
    ci = c.get("ci95") or [None, None]
    sub = []
    if c.get("a_mean") is not None:
        sub.append(f"{T.esc(E.condition_label(c['a']))} {E.pct(c['a_mean'])} · {T.esc(E.condition_label(c['b']))} "
                   f"{E.pct(c['b_mean'])}")  # fmt: skip
    if ci[0] is not None:
        sub.append(f"95% CI {E.pts(ci[0])} to {E.pts(ci[1])}")
    sub.append(E.p_text(c.get("p_value")))
    badge = T.badge("significant" if sig else "not significant", "learn" if sig else "")
    with common.card(key):
        st.html(
            f'<div class="ca-stat"><div class="ca-card-head"><div><p class="ca-eyebrow">'
            f'{T.esc(E.metric_label(c["matchup"]))}</p></div><div>{badge}</div></div>'
            f'<div class="ca-stat-v {tone}">{E.pts(d)}<small> pts</small></div>'
            f'<div class="ca-stat-sub">{" · ".join(sub)}</div></div>'
        )
        T.chart(charts.contrast_dots_figure({**c, "_sig": sig}, height=92, span=span), key=f"dots_{key}")
        T.caption(T.esc(c.get("verdict") or E.verdict(c)))


def did_it_work(rows: list[dict]) -> None:
    if not rows:
        return
    word = {"yes": "Yes", "no": "No", "mixed": "Mixed", "unclear": "Unclear", "suggestive": "Suggestive",
            "suggestive-negative": "Possible cost"}
    html = []
    for r in rows:
        html.append(f'<div class="ca-dw-row"><span class="ca-dw-ans {r["answer"]}">{word[r["answer"]]}</span><div>'
                    f'<p class="q">{T.esc(r["question"])}</p><p class="h">{T.esc(r["headline"])}</p>'
                    f'<p class="d">{T.esc(r["detail"])}</p></div></div>')  # fmt: skip
    with common.card("didit"):
        T.card_header("Did it work?", "Separate questions with separate answers: whether the detectors learned, and "
                      "whether each agent did better because of it. Suggestive = positive in most seeds but not "
                      "significant (p ≥ 0.05).")  # fmt: skip
        st.html('<div class="ca-dw">' + "".join(html) + "</div>")


def methods(agg: dict | None, extra: str = "") -> None:
    defs = E.definitions(agg or {})
    with st.expander("Methods: seeds, measures, intervals and known limitations", key="ev_methods"):
        st.markdown(
            "**What a seed is.** One complete training run (2,000 games by default) with its own random number "
            "stream: network layout draws, which dataset rows feed the sensors, exploration and opponent moves. "
            "Seeds are independent, so the spread across seeds shows how much a result depends on luck.\n\n"
            "**What is measured.** At every checkpoint each learned agent plays fresh evaluation games against a "
            "scripted opponent with exploration off; the number is the learned side's win rate. Learned blue is "
            "also tested against the scripted red *disguised* at fixed levels (0.4 and 0.7: its activity blended "
            "that far toward normal traffic), which isolates whether retrained detectors beat disguise. "
            "*Second half of training* averages each seed's checkpoints after half the games, then averages seeds.\n\n"
            "**Intervals and p-values.** The 95% interval is a t-interval across seeds "
            "(mean ± t₀.₉₇₅,ₙ₋₁ · sd/√n). A comparison pairs the two conditions *within each seed* and runs a "
            "paired t-test on the per-seed differences (two-sided). It is called significant when p < 0.05 and the "
            "interval excludes zero. With 3–5 seeds only fairly large effects can be detected; 'no significant "
            "difference' means 'not shown', not 'shown equal'.\n\n"
            "**Disjoint data partitions.** Each detector's held-out rows are split three ways: *metric* rows (used "
            "only for the recall/AUC numbers, never seen in training), *arena_train* rows (the training games' "
            "sensors; revealed labels from these feed retraining) and *arena_eval* rows (the disguised-red "
            "evaluation; never used for retraining). So a retrained detector cannot win the evaluation by "
            "memorising its rows.\n\n"
            "**Known limitations.** Everything is simulated: the network is small (12–16 hosts), 'disguise' is "
            "interpolation between dataset rows in feature space, the scripted opponents are simple rule sets, and "
            "five seeds give limited power. Detector recall improving does not imply the blue agent benefits: the "
            "agent learns its policy against shifting detector scores, and the experiments here test that "
            "separately." + extra
        )
        if defs:
            st.html('<div class="ca-notes">' + "".join(
                f"<p><b>{T.esc(k)}</b>: {T.esc(v)}</p>" for k, v in defs.items()) + "</div>")  # fmt: skip


def run_status_card(e: dict) -> None:
    man = E.load_manifest(e["dir"]) or {}
    prog = E.run_progress(man, E.runs_root(e["dir"]))
    with common.card("ev-running"):
        done = int((prog["status"] == "done").sum()) if not prog.empty else 0
        T.card_header(f"{e['name']}: {man.get('status', 'running')}",
                      f"{done} of {len(prog)} runs finished. Results appear when every run has finished and "
                      "<code>aggregate.json</code> is written.")  # fmt: skip
        if not prog.empty:
            st.dataframe(prog, hide_index=True, column_config={
                "seed": st.column_config.NumberColumn("Seed", format="%d"), "condition": "Condition",
                "status": "Status", "progress": st.column_config.ProgressColumn("Progress", min_value=0, max_value=1),
                "game": st.column_config.NumberColumn("Game", format="%d"),
                "games": st.column_config.NumberColumn("Of", format="%d"), "run_id": "Run", "error": "Error"})  # fmt: skip


# ================================================================================================ factorial view

if choice in fam_by:
    fam = fam_by[choice]
    T.caption(
        "Reads <code>aggregate.json</code> of " + ", ".join(f"<code>{T.esc(c['exp']['name'])}</code>" for c in fam["cells"])
        + ". Each experiment is one cell of the design; main effects and the interaction are computed here from the "
        "cells' per-seed values, paired by seed (same seed = same network and data draws)."
    )
    cf_all = []
    labels = {}
    for c in fam["cells"]:
        agg_c = aggregate(c["exp"]) or {}
        cf = E.curve_frame(agg_c)
        name = E.cell_name(c["levels"])
        cf["condition"] = name
        labels[name] = name
        cf_all.append(cf)
    cf = pd.concat(cf_all, ignore_index=True) if cf_all else E.curve_frame({})
    metrics = E.sort_metrics(cf["matchup"].unique())
    if not metrics:
        T.empty_state("No results in these cells yet", "Their aggregates list no matchups.")
        st.stop()
    with common.toolbar("fam"):
        mu = st.segmented_control("Matchup", metrics, key="fam_mu", default=metrics[0], required=True,
                                  format_func=E.metric_short)  # fmt: skip
    effects = E.factorial_effects(fam, mu, stat)
    (f1, l1), *rest = fam["factors"]
    with common.card("fam-grid"):
        T.card_header(f"{E.metric_label(mu)} · {STATS[stat].lower()}",
                      "Mean win rate across seeds in each cell, with its 95% interval.")  # fmt: skip
        if len(fam["factors"]) == 2:
            f2, l2 = rest[0]
            head = "".join(f"<th>{T.esc(E.level_word(f2, v))}</th>" for v in l2)
            body = []
            for a in l1:
                tds = []
                for b in l2:
                    cell = next((c for c in fam["cells"] if c["levels"].get(f1) == a and c["levels"].get(f2) == b), None)
                    s = E.cell_summary(cell, mu, stat) if cell else None
                    if not s:
                        tds.append("<td>—</td>")
                        continue
                    ci = s.get("ci95") or [None, None]
                    tds.append(f'<td class="num"><b>{E.pct(s["mean"])}</b><br><span class="ca-caption">'
                               f"{E.pct(max(0, ci[0]))}–{E.pct(min(1, ci[1]))}</span></td>" if ci[0] is not None
                               else f'<td class="num"><b>{E.pct(s["mean"])}</b></td>')  # fmt: skip
                body.append(f"<tr><td><b>{T.esc(E.level_word(f1, a))}</b></td>{''.join(tds)}</tr>")
            st.html(f'<div class="ca-table-wrap" style="--ca-table-h:400px;--ca-table-head:{T.tok("surface")}">'
                    f'<table class="ca-table ca-grid"><thead><tr><th></th>{head}</tr></thead>'
                    f'<tbody>{"".join(body)}</tbody></table></div>')  # fmt: skip
        else:
            rows = []
            for c in fam["cells"]:
                s = E.cell_summary(c, mu, stat) or {}
                rows.append({"cell": T.esc(E.cell_name(c["levels"])), "mean": E.pct(s.get("mean")),
                             "exp": f'<span class="ca-id">{T.esc(c["exp"]["name"])}</span>'})  # fmt: skip
            st.html(T.table([("cell", "Cell"), ("mean:num", "Mean win rate"), ("exp", "Experiment")], rows))
        main = next((e for e in effects if e["kind"] == "main effect" and e.get("flag") == "--detectors"), None)
        if main is not None:
            T.takeaway(T.esc(main["verdict"]), learning=True)
    cols = st.columns(min(3, max(1, len(effects))), gap="medium")
    for i, e in enumerate(effects):
        with cols[i % len(cols)]:
            sig = E.significant(e)
            d = e.get("diff_mean") or 0.0
            tone = ("pos" if d > 0 else "neg") if sig else "ns"
            title = (f"{e['a']} − {e['b']}" if e["kind"] == "main effect"
                     else f"Interaction: {e['inner'][0]} gap, {e['a']} vs {e['b']}")  # fmt: skip
            with common.card(f"eff{i}"):
                st.html(f'<div class="ca-stat"><div class="ca-card-head"><div><p class="ca-eyebrow">'
                        f'{T.esc(e["factor"])}</p></div><div>'
                        f'{T.badge("significant" if sig else "not significant", "learn" if sig else "")}</div></div>'
                        f'<div class="ca-stat-v {tone}">{E.pts(d)}<small> pts</small></div>'
                        f'<div class="ca-stat-sub">{T.esc(title)} · {E.p_text(e.get("p_value"))}</div></div>')  # fmt: skip
                T.chart(charts.contrast_dots_figure({**e, "_sig": sig}, height=92), key=f"fdots{i}_{mu}")
                T.caption(T.esc(e["verdict"]))
    with common.card("fam-curves"):
        T.card_header(f"Every cell over training · {E.metric_label(mu)}",
                      "Line: mean across seeds; band: 95% interval; thin lines: individual seeds.")  # fmt: skip
        conds = list(labels)
        T.chart(charts.multiseed_figure(cf, mu, conds, height=340, labels=labels), key=f"fam_curve_{mu}")
        T.takeaway(T.esc(E.curve_takeaway(cf, mu, conds)), learning=True)
    methods(aggregate(fam["cells"][0]["exp"]),
            "\n\n**This factorial.** Cells differ only in diagnostic train flags (" + ", ".join(
                f"`{f}`" for f, _ in fam["factors"]) + "). A main effect averages the difference over the other "
            "factor's levels, per seed, then tests the per-seed values; the interaction is the difference of "
            "differences. These are computed by the dashboard from the cells' aggregates.")  # fmt: skip
    st.stop()


# ================================================================================================ one experiment

e = by_name[choice]
agg = aggregate(e)
T.caption(f"Reads <code>runs/experiments/{T.esc(e['name'])}/manifest.json</code> and <code>aggregate.json</code>.")
if agg is None:
    if e["status"] == "running":
        st.fragment(run_every=3.0)(run_status_card)(e)
    else:
        run_status_card(e)
    methods(None)
    st.stop()

man = E.load_manifest(e["dir"]) or {}
conds = list(agg.get("conditions") or man.get("conditions") or [])
names = E.experiment_condition_names(man, conds)
used = agg.get("runs_used") or {}
failed = agg.get("runs_failed") or []
first_run = next((E.run_dir_of(r["run_dir"], common.runs_dir()) for r in man.get("runs") or [] if r.get("run_dir")),
                 None)  # fmt: skip
kind = L.agent_kind(common.info(first_run)) if first_run is not None and first_run.exists() else ""
bits = [T.badge(f"{agg.get('n_seeds', len(e['seeds']))} seeds"),
        T.badge(f"{int(agg.get('episodes') or e.get('episodes') or 0):,} games per run")]  # fmt: skip
bits += [T.badge(names[c], "learn" if names[c] == "adaptive detectors" else "",
                 T.VIOLET if names[c] == "adaptive detectors" else None) for c in conds]  # fmt: skip
if kind:
    bits.append(T.badge(kind))
rules = E.experiment_rules(e, man)
if rules:
    bits.append(T.badge(E.RULES_LABEL[rules], "" if rules == "current" else "red"))
if man.get("extra_args"):
    bits.append(T.badge("train flags: " + " ".join(man["extra_args"])))
bits.append(T.badge(f"{sum(len(v) for v in used.values())} runs used" + (f" · {len(failed)} failed" if failed else "")))
T.badges(bits)

pairs = E.contrast_pairs(agg)
pair = None
if len(pairs) > 1:
    pair = st.selectbox("Comparison", pairs, key="ev_pair", width=420,
                        format_func=lambda p: f"{E.condition_label(p[0])} − {E.condition_label(p[1])}")  # fmt: skip
elif pairs:
    pair = pairs[0]
cs = E.contrasts(agg, stat, pair)

if stat == "late_mean" and (pair is None or pair == ("adaptive", "frozen")):
    did_it_work(E.did_it_work(agg))

if cs:
    a, b = E.condition_label(cs[0]["a"]), E.condition_label(cs[0]["b"])
    st.html(f'<p class="ca-section">Headline comparisons · {T.esc(a)} minus {T.esc(b)}, '
            f"{T.esc(STATS[stat].lower())}</p>")  # fmt: skip
    span = max([abs(100 * x) for c in cs for x in (c.get("per_seed_diff") or []) + list(c.get("ci95") or [0, 0])
                if x is not None] + [10]) * 1.12  # fmt: skip
    for i in range(0, len(cs), 2):
        cols = st.columns(2, gap="medium")
        for col, c in zip(cols, cs[i : i + 2], strict=False):
            with col:
                stat_card(c, f"ct{i}{c['matchup']}".replace(".", "_").replace("@", "_"), span)
else:
    with common.card("ev-single"):
        T.card_header("One condition, no comparison",
                      f"This experiment ran one condition ({T.esc(', '.join(names[c] for c in conds))}), so there is nothing "
                      "to compare inside it. Its curves and arms race are below; a factorial family combines such "
                      "experiments.")  # fmt: skip
        block = (agg.get(stat) or {}).get(conds[0] if conds else "", {})
        tl = [T.tile(E.metric_label(m), E.pct(s.get("mean")),
                     f"95% CI {E.pct(max(0, (s.get('ci95') or [0])[0]))}–{E.pct(min(1, (s.get('ci95') or [0, 1])[1]))}"
                     f" · {s.get('n', '?')} seeds", swatch=T.TEAM[E.metric_side(m)])
              for m, s in sorted(block.items(), key=lambda kv: E.sort_metrics(block).index(kv[0]))]  # fmt: skip
        T.tiles(tl)

# ------------------------------------------------------------------------------------------------ curves
cf = E.curve_frame(agg)
metrics = E.sort_metrics(cf["matchup"].unique()) if not cf.empty else []
if metrics:
    with common.card("ev-curves"):
        T.card_header("Win rate over training, every seed",
                      "Line: mean across seeds; band: 95% interval; thin lines: each seed on its own. The grey rule is "
                      "a coin flip.")  # fmt: skip
        with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
            mu = st.segmented_control("Matchup", metrics, key="ev_mu", default=metrics[0], required=True,
                                      format_func=E.metric_short)  # fmt: skip
            seeds_on = st.toggle("Show each seed", value=True, key="ev_seeds")
        labels = names
        T.chart(charts.multiseed_figure(cf, mu, conds, height=340, show_seeds=seeds_on, labels=labels),
                key=f"ev_curve_{mu}")  # fmt: skip
        T.takeaway(T.esc(E.curve_takeaway(cf, mu, conds, names)), learning=True)

# ------------------------------------------------------------------------------------------------ arms race
ar = E.arms_race_rows(agg)
if ar:
    with common.card("ev-arms"):
        T.card_header("Arms race across seeds",
                      "Detector recall on held-out malicious rows disguised at 0.7, before any retraining (open "
                      "circle) and after the last retrain (filled). Thin lines are individual seeds.")  # fmt: skip
        left, right = st.columns([1.6, 1], gap="medium")
        with left:
            T.chart(charts.recall_dumbbell_figure(ar, height=60 + 62 * len(ar)), key="ev_dumbbell")
        with right:
            rows = [{"m": T.esc(r["model"]), "cyc": f"{r['cycles']:.1f}" if r["cycles"] is not None else "—",
                     "rng": (f"{min(x for x in r['per_cycles'] if x is not None)}–{max(x for x in r['per_cycles'] if x is not None)}"
                             if any(x is not None for x in r["per_cycles"]) else "—"),
                     "up": f"{r['updates']:.0f}" if r["updates"] is not None else "—"} for r in ar]  # fmt: skip
            st.html(T.table([("m", "Detector"), ("cyc:num", "Cycles / run"), ("rng:num", "Range"),
                             ("up:num", "Retrains")], rows, height=240))  # fmt: skip
            T.caption("A cycle: red's disguise rises at least 0.2 from a low, then falls at least 0.2 from that peak "
                      "after the detector catches up.")  # fmt: skip
        gains = [(r["model"], (r["last"] or 0) - (r["first"] or 0)) for r in ar]
        best = max(gains, key=lambda g: g[1])
        allup = all(g > 0 for r in ar for g in r["gains"])
        T.takeaway(f"Retraining raised recall on disguised red for every detector ({', '.join(f'{m} {E.pts(g)} pts' for m, g in gains)}), "
                   f"most for {T.esc(best[0])}" + ("; it rose in every seed." if allup else "; not in every seed.")
                   if all(g > 0 for _, g in gains) else
                   "Retraining did not raise recall on disguised red for every detector: "
                   + ", ".join(f"{m} {E.pts(g)} pts" for m, g in gains) + ".", learning=True)  # fmt: skip

# ------------------------------------------------------------------------------------------------ cross-evaluation
xe = E.load_crosseval(common.runs_dir())
exp_runs = {L.path_name(r["run_dir"]) for r in man.get("runs") or [] if r.get("run_dir")}
if not xe.empty and not (set(xe["blue_run"]) & exp_runs):
    xe = pd.DataFrame()  # the cross-evaluation belongs to another experiment's runs
if not xe.empty:
    with common.card("ev-xeval"):
        T.card_header("Cross-evaluation: same blue agent, different detectors",
                      "Each trained blue agent (from the adaptive and the frozen runs) replayed against the scripted red "
                      "with pretrained detectors (v0), its run's final detectors, or detector scores shuffled across "
                      "hosts (scores carry no information). Mean blue win rate across seeds.")  # fmt: skip
        g = xe.groupby(["trained_with", "detectors", "evasion"])["blue_win_rate"].mean().unstack("evasion")
        lv = list(g.columns)
        rows = []
        for (tw, det), r in g.iterrows():
            rows.append({"tw": f"blue trained with {T.esc(tw)} detectors", "det": T.esc({"v0": "pretrained v0", "final": "final (retrained)",
                         "shuffled": "shuffled scores"}.get(det, det)), **{f"e{i}": E.pct(r[x]) for i, x in enumerate(lv)}})  # fmt: skip
        st.html(T.table([("tw", "Agent"), ("det", "Detectors"), *[(f"e{i}:num", f"red disguise {x:.1f}") for i, x in enumerate(lv)]],
                        rows, height=320))  # fmt: skip
        m_ = xe.groupby(["trained_with", "detectors"])["blue_win_rate"].mean()
        try:
            da = m_[("adaptive", "final")] - m_[("adaptive", "shuffled")]
            dfz = m_[("frozen", "final")] - m_[("frozen", "shuffled")]
            T.takeaway(f"Shuffling the detector scores costs the adaptive-trained blue {E.pts(da)} pts but the "
                       f"frozen-trained blue {E.pts(dfz)} pts: "
                       + ("the frozen-trained agent barely uses the scores at all." if abs(dfz) < 0.05 < abs(da)
                          else "both rely on them." if min(abs(da), abs(dfz)) >= 0.05 else "neither relies on them much.")
                       + f" {xe['seed'].nunique()} seeds, {int(xe['n'].iloc[0])} games per cell.")  # fmt: skip
        except KeyError:
            pass

# ------------------------------------------------------------------------------------------------ DQN vs tabular
runs = L.list_runs(common.runs_dir())
infos = {r: common.info(r) for r in runs}
pair_runs = E.agent_comparison_runs(runs, infos)
if pair_runs:
    dq, tb = pair_runs
    with common.card("ev-agents"):
        T.card_header("DQN vs tabular agents, same seed",
                      f"Two single runs with seed {infos[dq].get('seed')}, identical except for the learner: a "
                      "TensorFlow Q-network that scores every concrete move, and the older Q-table that picks an action "
                      "type and lets a fixed rule pick the host. One seed each, so read it as an illustration.")  # fmt: skip
        cd, ct = L.eval_curve(common.summary(dq)[1]), L.eval_curve(common.summary(tb)[1])
        T.chart(charts.compare_figure([
            {"name": f"DQN · {infos[dq].get('label') or dq.name}", "curve": cd, "slot": 0},
            {"name": f"tabular · {infos[tb].get('label') or tb.name}", "curve": ct, "slot": 1}], height=320),
            key="ev_dqn_tab")  # fmt: skip
        fd, ft = E.final_rates(cd), E.final_rates(ct)
        parts = []
        for side, opp in (("red", "blue"), ("blue", "red")):
            if fd.get(f"{side}_late") is not None and ft.get(f"{side}_late") is not None:
                parts.append(f"learned {side} beats scripted {opp} {E.pct(fd[f'{side}_late'])} of the time with DQN "
                             f"vs {E.pct(ft[f'{side}_late'])} tabular")  # fmt: skip
        if parts:
            line = "; ".join(parts)
            T.takeaway(line[:1].upper() + line[1:] + " (second half of training, single seed).")

# ------------------------------------------------------------------------------------------------ lessons
def lessons() -> None:
    """The previous-rules result and its 2×2 diagnosis, shown with the current experiment as a documented finding."""
    old = by_name.get("main-cheap-isolation")
    old_agg = aggregate(old) if old else None
    fam = fam_by.get(FAMILY + "diag")
    if old_agg is None and fam is None:
        return
    st.html('<p class="ca-section">What we learned along the way</p>')
    with common.card("ev-lessons"):
        T.card_header("Under the previous rules, adaptive detectors made blue worse",
                      "Before the current rules, isolating a clean host was nearly free (0.1 + 0.01 per turn; now 0.3 + "
                      "0.03). The same experiment then gave a clear negative result, and a 2×2 factorial found why. "
                      "Kept here as a diagnosed finding.")  # fmt: skip
        items = []
        if old_agg is not None:
            row = next((r for r in E.did_it_work(old_agg) if r["key"] == "blue"), None)
            if row:
                items.append(("Previous rules, same experiment", f"{row['headline']}. {row['detail']}"))
        if fam is not None:
            for eff in E.factorial_effects(fam, "blue_learned_vs_red_baseline", "late_mean"):
                if eff["kind"] == "main effect":
                    items.append((f"2×2 diagnosis · {eff['factor']}", eff["verdict"]))
        xs = E.load_crosseval(common.runs_dir())
        if old is not None and not xs.empty and "isolate_clean_share" in xs:
            old_runs = {L.path_name(r["run_dir"]) for r in (E.load_manifest(old["dir"]) or {}).get("runs") or []
                        if r.get("run_dir")}  # fmt: skip
            xs = xs[xs["blue_run"].isin(old_runs)]
            g = xs.groupby("detectors")["isolate_clean_share"].mean()
            if {"v0", "final"} <= set(g.index):
                items.append(("Cross-evaluation", (f"Replayed with the pretrained detectors, {E.pct(g['v0'])} of blue's "
                              f"isolations hit clean hosts; with the retrained detectors {E.pct(g['final'])}. Fewer false "
                              "alarms meant fewer of the near-free isolations that had been helping blue.")))  # fmt: skip
        items.append(("What changed", ("Isolating a clean host now costs 0.3 plus 0.03 per turn, so isolation has to be "
                      "earned by evidence. The current-rules experiment above is the result under those rules; "
                      "<code>--rules cheap-isolation</code> reproduces the previous ones.")))  # fmt: skip
        st.html('<div class="ca-lessons">' + "".join(
            f'<div class="ca-lesson"><p class="k">{T.esc(k)}</p><p class="v">{v if k == "What changed" else T.esc(v)}</p></div>'
            for k, v in items) + "</div>")  # fmt: skip
        with st.container(horizontal=True, gap="small"):
            if old is not None and st.button("Open the previous-rules experiment", key="ev_open_old",
                                              icon=":material/history:"):  # fmt: skip
                ss["_open_experiment"] = old["name"]
                st.rerun()
            if fam is not None and st.button("Open the 2×2 diagnosis", key="ev_open_fam", icon=":material/grid_view:"):
                ss["_open_experiment"] = FAMILY + "diag"
                st.rerun()


if e["name"] == "main":
    lessons()

methods(agg)
