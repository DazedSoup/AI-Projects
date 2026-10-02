"""Simulation Lab page: pick variables, launch a simulation, watch it train, compare and replay runs.

The form is generated entirely from ``python -m cyberarena.arena.train --describe-params`` (see
docs/contracts.md, "Simulation Lab"). Launching starts ``lab_runner`` detached (train, then offline enrich).
This page imports no arena, ml or explain code and never passes ``--online``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from cyberarena import config
from cyberarena.dashboard import charts, lab
from cyberarena.dashboard import loaders as L

RUNS_DIR = Path(config.RUNS_DIR)
ROOT = Path(config.ROOT)
PY = sys.executable
POLL_S = 1.5
REPLAY_PAGE = "replay_page.py"
FRESH, CONTINUE = "Start fresh", "Continue from run…"

ss = st.session_state


@st.cache_data(show_spinner="Reading the parameter list from the arena (train --describe-params)…")
def cached_spec(python: str, cwd: str) -> tuple[dict | None, str | None]:
    """Called once per session lifetime (errors are cached too; the Retry button clears the cache)."""
    try:
        return lab.load_param_spec(python, Path(cwd)), None
    except lab.SpecError as e:
        return None, str(e)


def wkey(k: str) -> str:
    return f"lab_w:{k}"


def rkey(k: str) -> str:
    return f"lab_r:{k}"


def fmt_val(v) -> str:
    if v is None:
        return "random"
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def pct(v) -> str:
    return "-" if v is None or pd.isna(v) else f"{float(v):.0%}"


# ================================================================================================ form state


def _put_widget_state(p: dict, value) -> None:
    """Mirror one value into its widget keys (allowed in callbacks and before the widget is created)."""
    w = lab.widget_for(p)
    if w["widget"] == "unsupported":
        return
    k = p["key"]
    if w["nullable"]:
        ss[rkey(k)] = value is None
    if w["widget"] in ("slider", "number"):
        ss[wkey(k)] = (
            w["fallback"] if value is None else lab.snap(p, value) if w["widget"] == "slider" else value
        )
    elif w["widget"] == "toggle":
        ss[wkey(k)] = bool(value)
    elif w["widget"] == "selectbox":
        ss[wkey(k)] = value if value in w["options"] else w["options"][0]


def _set_values(spec: dict, values: dict) -> None:
    idx = lab.param_index(spec)
    for k, v in values.items():
        if k in idx:
            ss.lab_vals[k] = lab.coerce(idx[k], v)
            _put_widget_state(idx[k], ss.lab_vals[k])


def _apply_preset(name: str, spec: dict) -> None:
    _set_values(spec, lab.preset_values(name, spec, ss.lab_vals))
    ss.lab_preset = name


def _reset_group(group: dict, spec: dict) -> None:
    _set_values(spec, {p["key"]: p.get("default") for p in group["params"]})


def init_state(spec: dict) -> None:
    if "lab_vals" not in ss:
        ss.lab_vals = lab.defaults(spec)
    for _, p in lab.iter_params(spec):
        k = p["key"]
        if k not in ss.lab_vals:  # a param added to the arena since this session started
            ss.lab_vals[k] = lab.coerce(p, p.get("default"))
        # widget keys are dropped while another page is shown; restore them from lab_vals
        if wkey(k) not in ss or (p.get("nullable") and rkey(k) not in ss):
            _put_widget_state(p, ss.lab_vals[k])
        elif lab.widget_for(p)["widget"] != "unsupported":
            # widget state already holds this rerun's new value; read it now so group headers are current
            ss.lab_vals[k] = None if (p.get("nullable") and ss[rkey(k)]) else lab.coerce(p, ss[wkey(k)])


def render_param(p: dict, disabled: bool = False) -> None:
    w = lab.widget_for(p)
    k = p["key"]
    if w["widget"] == "unsupported":
        st.caption(f"{w['label']}: not editable here ({w['reason']}).")
        return
    random = False
    if w["nullable"]:
        random = st.checkbox(
            f"{w['label']}: random", key=rkey(k), disabled=disabled,
            help=(w["help"] or "") + "\n\nTicked: leave it unset so the arena picks it.",
        )  # fmt: skip
    common = {"key": wkey(k), "help": w["help"], "disabled": disabled or random}
    if w["widget"] == "slider":
        st.slider(w["label"], min_value=w["min"], max_value=w["max"], step=w["step"], **common)
    elif w["widget"] == "number":
        st.number_input(w["label"], min_value=w["min"], max_value=w["max"], step=w["step"], **common)
    elif w["widget"] == "toggle":
        st.toggle(w["label"], **common)
    elif w["widget"] == "selectbox":
        st.selectbox(w["label"], w["options"], **common)
    ss.lab_vals[k] = None if random else lab.coerce(p, ss[wkey(k)])
    d = lab.coerce(p, p.get("default"))
    if not lab.same(ss.lab_vals[k], d):
        st.caption(f":orange[●] changed (default {fmt_val(d)})")


def render_group(group: dict, spec: dict, disabled: bool, skip: tuple = ()) -> None:
    params = [p for p in group["params"] if p["key"] not in skip]
    basic = [p for p in params if not p.get("advanced")]
    adv = [p for p in params if p.get("advanced")]
    for p in basic:
        render_param(p, disabled)
    if adv:
        n_adv = len([k for k in lab.group_changed({"params": adv}, ss.lab_vals)])
        with st.expander(f"Advanced ({len(adv)})" + (f" · ● {n_adv} changed" if n_adv else ""),
                         key=f"lab_adv:{group['id']}"):  # fmt: skip
            for p in adv:
                render_param(p, disabled)


# ================================================================================================ live panel


def _stop(job: dict, force: bool = False) -> None:
    lab.stop_job(job, force=force)
    ss.lab_token = job.get("token")


def live_panel(token: str) -> None:
    job = lab.find_job(RUNS_DIR, token)
    stage = lab.effective_stage(job)
    if stage not in lab.ACTIVE_STAGES:
        ss.lab_token = token
        st.rerun()  # finished: re-render the whole page (enables the form, shows the summary)
    run_dir = Path(job["run_dir"]) if job.get("run_dir") else None
    prog = lab.read_progress(run_dir) if run_dir else None
    started = lab.parse_time(job.get("started"))
    elapsed = (lab.local_now() - started).total_seconds() if started else None
    title = job.get("label") or job.get("run_id") or "new run"
    st.markdown(f"#### Running: {title}")
    c = st.columns(5)
    phase = (prog or {}).get("phase")
    c[0].metric("Stage", stage + (f" · {phase}" if phase and stage == "training" else ""),
                help="Lab stage (training / enriching) and, while training, train's phase (setup / train / eval)")  # fmt: skip
    ep, n = (prog or {}).get("episode"), (prog or {}).get("episodes")
    c[1].metric("Episode", f"{ep} / {n}" if n else "-")
    c[2].metric("Elapsed", lab.format_elapsed(elapsed))
    le = (prog or {}).get("last_eval") or {}
    c[3].metric("Learned red vs baseline", pct(le.get("red_win_rate")),
                help=f"Latest eval (after episode {le.get('after_episode', '-')})")  # fmt: skip
    c[4].metric("Learned blue vs baseline", pct(le.get("blue_win_rate")),
                help=f"Latest eval (after episode {le.get('after_episode', '-')})")  # fmt: skip
    if stage == "enriching":
        st.progress(
            1.0, text="Training done · enriching episodes offline (SHAP, MITRE, template rationales)…"
        )
    elif stage == "starting" or prog is None:
        st.progress(0.0, text="Starting: creating the run and loading classifiers…")
    else:
        frac = lab.progress_fraction(prog)
        st.progress(frac, text=f"Training: {frac:.0%} ({ep} / {n} episodes)")
    st.caption(job.get("message") or "")
    asked = lab.stop_requested_at(job)
    if asked is None:
        st.button("Stop", key="lab_stop", on_click=_stop, args=(job,), icon=":material/stop_circle:",
                  help="Cancel the run: training gets a cancel signal, records 'cancelled' in progress.json and "
                  "exits. The partial run stays on disk.")  # fmt: skip
    else:
        waited = (lab.local_now() - asked).total_seconds()
        st.warning(f"Stopping… (requested {waited:.0f}s ago; waiting for training to save and exit)")
        if waited > lab.STOP_GRACE_S:
            st.button("Force stop", key="lab_force_stop", on_click=_stop, args=(job, True),
                      icon=":material/dangerous:", help="Kill the runner and its child processes now.")  # fmt: skip
    if run_dir and (run_dir / L.SUMMARY_FILE).exists():
        try:
            _, evals = L.load_summary(run_dir)
        except L.RunFormatError:
            evals = pd.DataFrame()
        curve = L.eval_curve(evals) if not evals.empty else pd.DataFrame()
        if not curve.empty:
            st.plotly_chart(charts.win_rate_figure(curve, pd.DataFrame(), 0), key="lab_live_curve",
                            config={"displayModeBar": False})  # fmt: skip


def finished_panel(job: dict) -> None:
    stage = lab.effective_stage(job)
    title = job.get("label") or job.get("run_id") or job.get("token")
    started, finished = lab.parse_time(job.get("started")), lab.parse_time(job.get("finished"))
    took = lab.format_elapsed((finished - started).total_seconds()) if started and finished else "-"
    if stage == "done":
        st.success(f"**{title}** finished in {took}. {job.get('message') or ''}")
        run_dir = Path(job["run_dir"])
        try:
            _, evals = L.load_summary(run_dir)
        except (L.RunFormatError, OSError):
            evals = pd.DataFrame(columns=["after_episode", "matchup", "n", "red_win_rate", "blue_win_rate"])
        fe = lab.final_eval(evals)
        c = st.columns(4)
        c[0].metric("Run", job.get("run_id") or "-")
        c[1].metric(
            "Learned red vs baseline", pct(fe["red"]), help=f"final eval, after ep {fe['after_episode']}"
        )
        c[2].metric(
            "Learned blue vs baseline", pct(fe["blue"]), help=f"final eval, after ep {fe['after_episode']}"
        )
        c[3].metric("Changed params", len(job.get("changed") or {}))
        if job.get("changed"):
            st.caption("Changed from defaults: " + lab.format_changed(job["changed"]))
    else:
        msg = job.get("message") or lab.stale_message(job)
        if lab.effective_stage(job) == "error" and job.get("stage") in lab.ACTIVE_STAGES:
            msg = lab.stale_message(job)
        (st.warning if job.get("stopped") else st.error)(f"**{title}**: {msg}")
        logs = [job.get("train_log"), job.get("enrich_log")]
        logs = [lg for lg in logs if lg]
        if logs:
            st.caption("Logs: " + " · ".join(f"`{lg}`" for lg in logs))
    b1, b2 = st.columns([1, 4])
    can_open = bool(job.get("run_dir")) and (Path(job["run_dir"]) / L.SUMMARY_FILE).exists()
    if can_open and b1.button("Open in Replay", type="primary", key="lab_open_new", icon=":material/replay:"):
        ss["_open_run"] = Path(job["run_dir"]).name
        st.switch_page(REPLAY_PAGE)
    if b2.button("Dismiss", key="lab_dismiss"):
        ss.pop("lab_token", None)
        st.rerun()


# ================================================================================================ page

st.title("Simulation Lab")
st.caption(
    "Pick the variables, launch a new simulated training run, watch it train, then replay it. Simulation only: "
    "training runs locally and enrichment is **offline** (template rationales, no API, no cost)."
)

active = lab.active_job(RUNS_DIR)
if active is not None:
    ss.lab_token = active["token"]
    st.fragment(run_every=POLL_S)(live_panel)(active["token"])
elif ss.get("lab_token"):
    job = lab.find_job(RUNS_DIR, ss.lab_token)
    if job:
        finished_panel(job)

spec, spec_err = cached_spec(PY, str(ROOT))
busy = active is not None

# ------------------------------------------------------------------------------------------------ form
st.header("New simulation")
if spec is None:
    cmd = lab.format_cmd(lab.describe_params_cmd(PY))
    st.error(f"Could not read the parameter list from the arena: {spec_err}")
    st.markdown(
        "**How to fix:** run this from the repo root and check that it prints a JSON object:\n\n"
        f"```\n{cmd}\n```\n"
        "If it reports `unrecognized arguments: --describe-params`, the arena does not implement the Simulation "
        "Lab contract yet (docs/contracts.md, *Simulation Lab*). If it fails on an import, fix the venv "
        "(`pip install -r requirements.txt`). Then press **Retry**. Replay and the run history below still work."
    )
    if st.button("Retry", key="lab_retry"):
        cached_spec.clear()
        st.rerun()
else:
    init_state(spec)
    if busy:
        st.info("A simulation is running, so launching is disabled until it finishes or is stopped. "
                "You can prepare the next run meanwhile.")  # fmt: skip

    st.markdown("**Presets**")
    pc = st.columns(len(lab.PRESETS))
    for col, name in zip(pc, lab.PRESETS, strict=True):
        col.button(name, key=f"lab_preset:{name}", on_click=_apply_preset, args=(name, spec),
                   help=lab.PRESET_HELP[name], width="stretch")  # fmt: skip

    # memory
    groups = spec["groups"]
    mem = next((g for g in groups if g["id"] == "memory"), None)
    st.markdown("**Memory**")
    mode = st.radio("Agents start", [FRESH, CONTINUE], key="lab_mem", horizontal=True,
                    label_visibility="collapsed")  # fmt: skip
    st.caption(
        "*Start fresh*: both agents begin with empty Q-tables. *Continue from run…*: the chosen side(s) start "
        "from that run's learned Q-tables (`agents/red.json`, `agents/blue.json`) and keep learning, so you can "
        "change a variable and see how already-trained agents adapt. Works across graph sizes."
    )
    parent = None
    side_keys = ("init_side",)
    if mode == CONTINUE:
        parents = [p for p in lab.list_run_dirs(RUNS_DIR) if lab.has_agents(p) and lab.run_is_done(p)]
        if not parents:
            st.warning("No finished run with saved agents yet. Start fresh first.")
        else:
            parent = st.selectbox(
                "Continue from", parents, format_func=lab.run_display_name, key="lab_parent"
            )
            if mem:
                for p in mem["params"]:
                    if p["key"] in side_keys:
                        render_param(p)
    if mem:
        render_group(mem, spec, False, skip=side_keys)

    st.text_input("Run label", key="lab_label", max_chars=80, placeholder="e.g. phish 0.5, blue noise up",
                  help="Free text stored in config.json; shown in the run pickers and history.")  # fmt: skip

    st.markdown("**Parameters**")
    for g in groups:
        if g["id"] == "memory" or not g["params"]:
            continue
        n_changed = len(lab.group_changed(g, ss.lab_vals))
        with st.expander(
            g["label"] + (f"  · ● {n_changed} changed" if n_changed else ""), key=f"lab_g:{g['id']}"
        ):
            if n_changed:
                st.button("Reset group", key=f"lab_reset:{g['id']}", on_click=_reset_group, args=(g, spec),
                          icon=":material/restart_alt:", help="Put every parameter in this group back to its default")  # fmt: skip
            render_group(g, spec, False)

    changed = lab.changed_params(spec, ss.lab_vals)
    if mode == FRESH:
        changed = {k: v for k, v in changed.items() if k not in side_keys}
    init_side = ss.lab_vals.get("init_side", "both")
    train_args = lab.build_train_args(
        spec, ss.lab_vals, init_from=parent, label=ss.get("lab_label") or "", runs_dir=RUNS_DIR
    )
    st.markdown(
        "**Changed from defaults:** "
        + (lab.format_changed(changed) if changed else "nothing (all defaults)")
        + (f" · continuing from `{Path(parent).name}` ({init_side})" if parent else "")
    )
    with st.expander("Command line", key="lab_cmd"):
        st.code(lab.format_cmd(lab.train_cmd(train_args, PY)), language="text", wrap_lines=True)
        st.code(lab.format_cmd(lab.enrich_cmd("<run_dir>", PY)), language="text", wrap_lines=True)
        st.caption(
            "Run by a detached runner, so it finishes even if you close this tab. Enrichment is offline."
        )

    if st.button(
        "Run simulation", type="primary", key="lab_run", disabled=busy, icon=":material/play_arrow:"
    ):
        try:
            job = lab.launch_job(
                RUNS_DIR, train_args, label=ss.get("lab_label") or "", changed=changed,
                init_from={"run_id": Path(parent).name, "side": init_side} if parent else None,
                python=PY, cwd=ROOT,
            )  # fmt: skip
        except lab.LabBusyError as e:
            st.warning(str(e))
        else:
            ss.lab_token = job["token"]
            st.rerun()

# ------------------------------------------------------------------------------------------------ history
st.header("Run history")
run_dirs = lab.list_run_dirs(RUNS_DIR)
if not run_dirs:
    st.info(f"No runs under `{RUNS_DIR}` yet.")
    st.stop()
summaries = {}
for p in run_dirs:
    try:
        summaries[str(p)] = L.cached_summary(str(p), L.file_stamp(p / L.SUMMARY_FILE))[1]
    except L.RunFormatError:
        pass
hist = lab.history_table(RUNS_DIR, spec, summaries)
slots = {rid: i for i, rid in enumerate(sorted(hist["run_id"]))}  # colour follows the run, oldest = slot 0
with_evals = [i for i, r in hist.iterrows() if r["red_eval"] is not None or r["blue_eval"] is not None]
event = st.dataframe(
    hist.drop(columns=["run_dir"]),
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    selection_default={"selection": {"rows": with_evals[:2]}},
    key="lab_hist",
    column_config={
        "label": st.column_config.TextColumn("Label"),
        "run_id": st.column_config.TextColumn("Run id"),
        "started": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm"),
        "parent": st.column_config.TextColumn("Continued from", help="init_from run (side)"),
        "changed": st.column_config.TextColumn("Changed from defaults", width="large"),
        "red_eval": st.column_config.NumberColumn("Red vs baseline", format="percent",
                                                  help="Final eval: learned red's win rate vs the baseline blue"),
        "blue_eval": st.column_config.NumberColumn("Blue vs baseline", format="percent",
                                                   help="Final eval: learned blue's win rate vs the baseline red"),
        "eval_at": st.column_config.NumberColumn("Eval @ ep"),
        "episodes": st.column_config.NumberColumn("Episodes"),
        "status": st.column_config.TextColumn("Status"),
    },
)  # fmt: skip
rows = list(getattr(getattr(event, "selection", None), "rows", None) or [])
st.caption(
    "Select rows to overlay their eval win-rate curves below and compare the effect of a variable change."
)

sel = hist.iloc[rows] if rows else hist.iloc[0:0]
if not sel.empty:
    o1, _ = st.columns([1, 4])
    if o1.button(f"Open {sel.iloc[0]['run_id']} in Replay", key="lab_open_sel", icon=":material/replay:"):
        ss["_open_run"] = sel.iloc[0]["run_id"]
        st.switch_page(REPLAY_PAGE)
    to_plot = []
    for r in sel.itertuples(index=False):
        evals = summaries.get(r.run_dir)
        curve = L.eval_curve(evals) if evals is not None and not evals.empty else pd.DataFrame()
        if curve.empty:
            continue
        to_plot.append({"name": (f"{r.label} · " if r.label else "") + r.run_id, "curve": curve,
                        "slot": slots[r.run_id]})  # fmt: skip
    if to_plot:
        st.subheader("Compare eval win rates")
        st.plotly_chart(charts.compare_figure(to_plot), key="lab_compare", config={"displayModeBar": False})
        st.caption(
            "Each point: the learned agent's win rate over n greedy eval episodes against the heuristic baseline, "
            "with a 95% Wilson interval. Left panel: learned red; right panel: learned blue. Colour and marker "
            "identify the run. Where two runs' bars overlap, the difference is within noise."
        )
    else:
        st.info("The selected runs have no eval rows yet.")
