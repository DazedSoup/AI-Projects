"""Learning: how the detectors and the agents learned during training, read from ``learning.jsonl``.

Every chart on this page is drawn from logged rows; nothing is simulated or smoothed. Runs that predate
adaptive learning show an empty state for the detector panels and Q-table growth from saved checkpoints.
"""

from __future__ import annotations

import streamlit as st

from cyberarena.dashboard import charts, common
from cyberarena.dashboard import learning as LL
from cyberarena.dashboard import theme as T

run = common.current_run()
lr = common.learning(run)
T.page_header(
    "Learning",
    "How blue's TensorFlow detectors were retrained during play, how red adapted its evasion in response, and how "
    "both agents' decisions changed. Everything here comes from <code>learning.jsonl</code>.",
)

if lr is None or lr.empty:
    with common.card("predates"):
        T.empty_state(
            "This run predates adaptive learning",
            "It has no <code>learning.jsonl</code>: its detectors were the fixed pretrained models and red never "
            "adapted its evasion, so there is no arms race, learning landscape, detector history or probe data to "
            "show. Launch a new run from the Simulation Lab (adaptive learning is on by default) to fill this page.",
        )
    q = common.q_states(run)
    with common.card("agents-v1"):
        T.card_header("Q-table growth", "Distinct situations each learned agent has values for, at every saved "
                      "checkpoint (from <code>agents/checkpoints/</code>).")  # fmt: skip
        if q.empty:
            T.empty_state("No saved checkpoints", "This run did not save agent checkpoints.")
        else:
            T.chart(charts.agent_figure(q.iloc[0:0].rename(columns={}), qstates=q, height=280), key="lr_q_v1")
            last = q.sort_values("after_episode").groupby("side").tail(1).set_index("side")["n_states"]
            first = q.sort_values("after_episode").groupby("side").head(1).set_index("side")["n_states"]
            T.takeaway(" ".join(
                f"Learned {s} grew from {int(first[s]):,} to {int(last[s]):,} states." for s in ("red", "blue") if s in last
            ) + " TD error, exploration and action mix need adaptive-learning telemetry.")  # fmt: skip
    st.stop()

models = lr.models
metric = LL.detector_metric(lr)
word = "recall" if metric == "recall" else "AUC"
NAMES = {
    "network": "Network traffic detector",
    "malware": "Malware detector",
    "phishing": "Phishing detector",
}
info = common.info(run)
adaptive_run = bool(models)
partners = common.comparison_partners(run)

with common.toolbar("learning"):
    if adaptive_run:
        model = st.selectbox("Detector", models, key="lr_model", format_func=lambda m: NAMES.get(m, m), width=240,
                             help="Which of blue's three detectors the landscape and the version history show.")  # fmt: skip
    side = st.segmented_control("Agent", ["red", "blue"], key="lr_side", default="red", required=True,
                                format_func=lambda s: f"Learned {s}",
                                help="Whose action mix and probe situations to show.")  # fmt: skip
    if adaptive_run:
        ls_view = st.segmented_control("Landscape", ["3D", "2D"], key="lr_view", default="3D", required=True)
    partner = None
    if partners:
        partner = st.selectbox(
            "Compare with", partners, key="lr_partner", width=340,
            format_func=lambda p: common.L.run_label(common.info(p)),
            help="A run trained the same way but with the other detector setting (adaptive vs frozen).",
        )  # fmt: skip
T.caption(
    f"Reads <code>runs/{T.esc(run.name)}/learning.jsonl</code>: {len(lr.updates)} detector updates, "
    f"{lr.evasion['episode'].nunique()} red evasion readings, {lr.stats['episode'].nunique()} agent snapshots and "
    f"{lr.probes['after_episode'].nunique()} probe checkpoints"
    + (
        f"; the comparison also reads <code>runs/{T.esc(partner.name)}/summary.jsonl</code>."
        if partner
        else "."
    )
)


def comparison_card() -> None:
    if partner is None:
        return
    a_run, f_run = (run, partner) if adaptive_run else (partner, run)
    a_curve = common.L.eval_curve(common.summary(a_run)[1])
    f_curve = common.L.eval_curve(common.summary(f_run)[1])
    kind = common.L.agent_kind(common.info(a_run)) or "agents of an earlier version"
    with common.card("versus"):
        T.card_header(
            "Adaptive vs frozen detectors · one seed",
            "Learned blue against the scripted red, in two runs that differ only in whether blue's detectors keep "
            f"learning. Single seed, {T.esc(kind)}; bands are 95% intervals for each checkpoint. The multi-seed answer "
            "is on the Evidence page.",
            right=T.badge(f"single seed · {kind}"),
        )
        T.chart(charts.adaptive_vs_frozen_figure(a_curve, f_curve, height=320), key="lr_versus")
        T.takeaway(common.L.adaptive_vs_frozen_takeaway(a_curve, f_curve), learning=True)


if not adaptive_run:
    with common.card("frozen"):
        T.empty_state(
            "Detectors were frozen in this run",
            "It was trained with adaptive learning off (<code>--no-adaptive</code>): blue kept the pretrained "
            "detectors and red never disguised its activity, so there is no arms race, landscape or version history. "
            "Agent learning and probe situations below still come from its <code>learning.jsonl</code>.",
        )
    comparison_card()

if adaptive_run:
    # ------------------------------------------------------------------------------------------------ arms race
    with common.card("arms"):
        T.card_header(
            "Arms race",
            f"Each column is one sensor. Bottom: red's evasion level, how far it blends its activity toward normal "
            f"traffic (it rises when red keeps getting caught). Top: the detector's {word} against traffic at red's "
            "current level. Diamonds mark detector updates, where blue retrains on newly confirmed examples.",
        )
        T.chart(charts.arms_race_figure(lr, models, height=460), key="lr_arms")
        stories = [LL.sensor_story(lr, m) for m in models]
        head = next((x for x in stories if x["model"] == LL.headline_model(lr)), stories[0])
        T.takeaway(head["text"], learning=True)
        rest = [x for x in stories if x is not head]
        if rest:
            st.html('<div class="ca-notes">' + "".join(f"<p>{T.esc(x['text'])}</p>" for x in rest) + "</div>")

    comparison_card()

    # ------------------------------------------------------------------------------------------------ landscape + versions
    with common.card("landscape"):
        T.card_header(
            f"Learning landscape · {NAMES.get(model, model)}",
            "Detector AUC (1.0 = perfect, 0.5 = guessing) for every version (training time, left to right) at every "
            "evasion level (back to front). The red line traces the level red was using at each update.",
        )
        ls = LL.landscape(lr, model, "auc")
        if not ls["episodes"]:
            T.empty_state("No updates for this detector yet", "It is still the pretrained model.")
        else:
            path = []
            for ep, v in zip(ls["episodes"][1:], ls["versions"][1:], strict=True):
                upd = lr.updates[(lr.updates["model"] == model) & (lr.updates["version"] == v)]
                s = (
                    LL.update_level(lr, upd.iloc[-1])
                    if not upd.empty
                    else LL.evasion_at(lr, model, ep, strict=True)
                )
                k = min(range(len(ls["levels"])), key=lambda i: abs(ls["levels"][i] - s))
                col = ls["versions"].index(v)
                if ls["z"][k][col] is not None:
                    path.append((ep, ls["levels"][k], ls["z"][k][col]))
            T.chart(charts.landscape_figure(ls, path, three_d=ls_view == "3D", height=480, revision=f"{run.name}{model}"),
                    key=f"lr_ls_{ls_view}", three_d=ls_view == "3D")  # fmt: skip
            T.takeaway(LL.landscape_takeaway(lr, model), learning=True)

    with common.card("versions"):
        T.card_header(
            f"Detector versions · {NAMES.get(model, model)}",
            "Each update fine-tunes the detector on newly confirmed examples mixed with replayed original training "
            "data. The clean-traffic check shows whether it forgot how to spot ordinary attacks.",
        )
        vt = LL.version_table(lr, model)
        if vt.empty:
            T.empty_state("No updates for this detector yet", "It is still the pretrained model (v0).")
        else:
            rows = []
            for r in vt.iloc[::-1].itertuples(index=False):
                d = r.clean_vs_pretrained
                flag = (
                    ""
                    if d is None
                    else f'<span class="ca-flag {"warn" if d < -0.02 else ""}">'
                    f"{d:+.3f}{' · forgetting' if d < -0.02 else ' · ok'}</span>"
                )
                gain = (
                    None
                    if r.evasive_before is None or r.evasive_after is None
                    else r.evasive_after - r.evasive_before
                )
                rows.append({
                    "v": f"v{r.version}", "ep": f"{r.episode:,}",
                    "n": f"{int(r.n_new):,} + {int(r.n_replay):,}" if r.n_new is not None and r.n_replay is not None else "—",
                    "loss": f"{r.loss_before:.3f} → {r.loss_after:.3f}" if r.loss_before is not None else "—",
                    "clean": f"{r.clean_after:.3f}" if r.clean_after is not None else "—", "forget": flag,
                    "lvl": f"{r.red_level:.1f}",
                    "eva": (f"{r.evasive_before:.3f} → {r.evasive_after:.3f}" if gain is not None else "—"),
                    "gain": f'<span class="ca-flag {"good" if (gain or 0) >= 0 else "warn"}">{gain:+.3f}</span>'
                    if gain is not None else "",
                })  # fmt: skip
            st.html(T.table([("v:mono", "Version"), ("ep:num", "After game"), ("n:num", "Samples (new + replay)"),
                             ("loss:num", "Loss before → after"), ("clean:num", "Clean AUC"),
                             ("forget", "vs pretrained"), ("lvl:num", "Red level"),
                             ("eva:num", "AUC at red's level"), ("gain:num", "Gain")], rows, height=320))  # fmt: skip
            T.takeaway(LL.versions_takeaway(lr, model), learning=True)


# ------------------------------------------------------------------------------------------------ agents
with common.card("agents"):
    if lr.dqn:
        T.card_header(
            "Agent learning · Q-networks",
            "Each agent is a TensorFlow network that scores every legal concrete move. Loss is its training error on "
            "batches replayed from memory; TD error is how far its value estimates still move per step; mean Q is the "
            "value it expects from the moves it picks; replay memory and gradient steps show how much it has trained; "
            "exploration is the share of moves picked at random.",
        )
        T.chart(charts.dqn_agent_figure(lr.stats, height=440), key="lr_agents")
    else:
        T.card_header(
            "Agent learning · Q-tables",
            "Q-table size counts the distinct situations an agent has values for. TD error is how much its value "
            "estimates still change per step. Exploration is the share of moves picked at random to try something new.",
        )
        T.chart(charts.agent_figure(lr.stats, height=270), key="lr_agents")
    T.takeaway(LL.agent_takeaway(lr), learning=True)
    actions, eps_, z = LL.mix_matrix(lr, side)
    st.html(f'<p class="ca-card-title" style="margin-top:6px">Action mix · learned {side}</p>'
            '<p class="ca-card-sub">Share of moves spent on each action over training (stronger colour = more often).</p>')  # fmt: skip
    if actions:
        T.chart(
            charts.mix_figure(actions, eps_, z, side, height=56 + 30 * len(actions)), key=f"lr_mix_{side}"
        )
        T.takeaway(LL.mix_takeaway(lr, side), learning=True)
    else:
        T.empty_state("No action mix logged", "agent_stats rows carry no action_mix.")

# ------------------------------------------------------------------------------------------------ probes
with common.card("probes"):
    T.card_header(
        f"Probe situations · learned {side}",
        "Hand-picked situations put to the agent at every checkpoint. Each tile shows the value it gives every "
        + ("candidate move (its top moves by value; stronger colour = higher value) " if lr.dqn else
           "action (stronger colour = higher value) ")
        + "over training; the dot marks the move it would pick.",
    )
    ids = LL.probe_ids(lr, side)
    if not ids:
        T.empty_state("No probe situations logged", "This run has no probe rows for this side.")
    else:
        cols = st.columns(3, gap="medium")
        for i, pid in enumerate(ids):
            pm = LL.probe_matrix(lr, side, pid)
            ch = LL.probe_change(pm)
            with cols[i % 3]:
                if ch["first"] != ch["last"]:
                    change = f"{T.esc(LL.move_label(ch['first']))} → <b>{T.esc(LL.move_label(ch['last']))}</b> · settled by game {ch['settled_at']:,}"
                elif ch["changed"]:
                    n = ch["switches"]
                    change = f"<b>{T.esc(LL.move_label(ch['last']))}</b> at the end · wavered {n} time{'s' if n != 1 else ''} on the way"
                else:
                    change = f"always <b>{T.esc(LL.move_label(ch['last']))}</b>"
                st.html(f'<p class="ca-card-title" style="font-size:13.5px">{T.esc(pm["description"])}</p>'
                        f'<p class="ca-card-sub"><span class="ca-mono">{T.esc(pid)}</span> · {change}</p>')  # fmt: skip
                T.chart(
                    charts.probe_figure(pm, side, height=60 + 26 * len(pm["actions"])),
                    key=f"lr_probe_{side}_{pid}",
                )
        T.takeaway(LL.probes_takeaway(lr, side), learning=True)
