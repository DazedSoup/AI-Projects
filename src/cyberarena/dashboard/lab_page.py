"""Simulation Lab page: pick variables, launch a simulation, watch it train, compare and replay runs.

The form is generated entirely from ``python -m cyberarena.arena.train --describe-params`` (see
docs/contracts.md, "Simulation Lab"), so a new parameter group such as ``adaptation`` shows up as a new tab
automatically. Launching starts ``lab_runner`` detached (train, then offline enrich). This page imports no
arena, ml or explain code and never passes ``--online``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from cyberarena import config
from cyberarena.dashboard import charts, common, lab
from cyberarena.dashboard import evidence as E
from cyberarena.dashboard import loaders as L
from cyberarena.dashboard import showcase as SC
from cyberarena.dashboard import theme as T

RUNS_DIR = Path(config.RUNS_DIR)
ROOT = Path(config.ROOT)
PY = sys.executable
POLL_S = 1.5
FRESH, CONTINUE = "Start fresh", "Continue from a run"
SINGLE_RUN, EXPERIMENT = "Single run", "Experiment"
CHOICE_LABELS = {
    "agent": {"dqn": "Q-network (TensorFlow DQN)", "tabular": "Q-table (tabular, v1–v3)"},
    "eval_evasion_levels": {"0.4,0.7": "disguise 0.4 and 0.7", "0.7": "disguise 0.7 only",
                            "0.2,0.4,0.7": "disguise 0.2, 0.4 and 0.7", "0.1,0.3,0.5,0.7": "disguise 0.1, 0.3, 0.5 and 0.7",
                            "": "none (skip the disguised-attacker test)"},
    "dqn_hidden": {"32": "1 layer × 32", "64,64": "2 layers × 64", "128,128": "2 layers × 128", "64,64,64": "3 layers × 64"},
    "baseline": {"heuristic": "rule-based (heuristic)", "random": "random moves"},
    "init_side": {"both": "both sides", "red": "red only", "blue": "blue only"},
    "detectors": {"auto": "follow 'Adaptive detectors'", "adaptive": "always retrain", "frozen": "never retrain"},
    "red_evasion": {"auto": "follow 'Adaptive detectors'", "bandit": "red adapts its disguise", "off": "red never disguises"},
    "blue_score_view": {"raw": "raw detector scores", "clean_quantile": "scores as quantiles of clean traffic"},
    "rules": {"default": "current: isolating a clean host is costly", "cheap-isolation": "previous: cheap isolation (v4)"},
}  # fmt: skip
TABULAR_ONLY = {"alpha", "alpha_end", "gamma"}
DQN_ONLY = {"dqn_lr", "dqn_hidden", "dqn_gamma", "n_step", "replay_size", "batch_size", "target_sync", "train_every"}


def choice_label(p: dict, v) -> str:
    return CHOICE_LABELS.get(p["key"], {}).get(v, "none" if v == "" else str(v))

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


def pct(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{float(v):.0%}"


# ================================================================================================ formatting

_PERCENT_HINTS = ("p_", "noise", "threshold", "leak", "frac", "cost", "prob", "rate")
_UNITS = {"episodes": "games", "eval_every": "games", "eval_n": "games", "eval_log_n": "games",
          "n_nodes": "hosts", "max_rounds": "rounds", "detector_update_every": "games",
          "detector_min_samples": "samples", "stats_every": "games", "pool_size": "rows"}  # fmt: skip


def is_percent(p: dict) -> bool:
    if p.get("type") != "float" or p.get("min") is None or p.get("max") is None:
        return False
    if float(p["min"]) < 0 or float(p["max"]) > 1:
        return False
    k = p["key"].lower()
    return any(h in k for h in _PERCENT_HINTS) and "lr" not in k.split(".")[-1].split("_")


def slider_format(p: dict) -> str | None:
    if p.get("unit"):
        return f"%d {p['unit']}" if p.get("type") == "int" else f"%.2f {p['unit']}"
    if p.get("type") == "int" and p["key"] in _UNITS:
        return f"%d {_UNITS[p['key']]}"
    if is_percent(p):
        return "percent"
    step = p.get("step") or 0
    if p.get("type") == "float" and step and float(step) < 0.001:
        return "%.1e"
    return None


def fmt_val(p: dict | None, v) -> str:
    if v is None:
        return "random"
    if p is not None and p.get("type") == "choice":
        return choice_label(p, v)
    if isinstance(v, bool):
        return "on" if v else "off"
    if p is not None and is_percent(p) and isinstance(v, (int, float)):
        return f"{float(v):.0%}"
    if isinstance(v, float):
        return f"{v:g}"
    if p is not None and p["key"] in _UNITS:
        return f"{v:,} {_UNITS[p['key']]}"
    return str(v)


def changed_text(spec: dict | None, changed: dict) -> str:
    idx = lab.param_index(spec)
    return ", ".join(
        f"{(idx.get(k) or {}).get('label', k)} {fmt_val(idx.get(k), v)}" for k, v in changed.items()
    )


def readable_changed(spec: dict | None, text: str) -> str:
    """``"p_phish=0.5, episodes=300"`` (lab.format_changed) -> ``"Phish success 50%, Training games 300 games"``."""
    idx = lab.param_index(spec)
    out = []
    for part in (text or "").split(", "):
        k, sep, v = part.partition("=")
        p = idx.get(k)
        if not sep or p is None:
            out.append(part)
            continue
        try:
            val = lab.coerce(p, float(v)) if p.get("type") in ("int", "float") else v
        except ValueError:
            val = v
        out.append(f"{p.get('label', k)} {fmt_val(p, val)}")
    return ", ".join(o for o in out if o)


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


def _apply_preset(spec: dict) -> None:
    name = ss.get("lab_preset_sel")
    if name:
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
            # widget state already holds this rerun's new value; read it now so tab headers are current
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
            f"{w['label']}: let the arena pick", key=rkey(k), disabled=disabled,
            help=(w["help"] or "") + "\n\nTicked: leave it unset so the arena picks it.",
        )  # fmt: skip
    common_kw = {"key": wkey(k), "help": w["help"], "disabled": disabled or random}
    if w["widget"] == "slider":
        st.slider(w["label"], min_value=w["min"], max_value=w["max"], step=w["step"], format=slider_format(p),
                  **common_kw)  # fmt: skip
    elif w["widget"] == "number":
        st.number_input(w["label"], min_value=w["min"], max_value=w["max"], step=w["step"], **common_kw)
    elif w["widget"] == "toggle":
        st.toggle(w["label"], **common_kw)
    elif w["widget"] == "selectbox":
        st.selectbox(w["label"], w["options"], format_func=lambda o, p=p: choice_label(p, o), **common_kw)
    ss.lab_vals[k] = None if random else lab.coerce(p, ss[wkey(k)])
    d = lab.coerce(p, p.get("default"))
    learner = ss.lab_vals.get("agent")
    # always one caption line, so a change marker never shifts the grid
    if (k in TABULAR_ONLY and learner == "dqn") or (k in DQN_ONLY and learner == "tabular"):
        st.caption(f"only used by the {'tabular' if k in TABULAR_ONLY else 'Q-network'} learner · default "
                   f"{fmt_val(p, d)}")  # fmt: skip
    elif not lab.same(ss.lab_vals[k], d):
        st.caption(f":violet[●] changed (default {fmt_val(p, d)})")
    else:
        st.caption(f"default {fmt_val(p, d)}")


def render_grid(params: list[dict], disabled: bool = False) -> None:
    for i in range(0, len(params), 2):
        cols = st.columns(2, gap="large")
        for col, p in zip(cols, params[i : i + 2], strict=False):
            with col:
                render_param(p, disabled)


def render_group(group: dict, spec: dict, disabled: bool, skip: tuple = ()) -> None:
    params = [p for p in group["params"] if p["key"] not in skip]
    basic = [p for p in params if not p.get("advanced")]
    adv = [p for p in params if p.get("advanced")]
    render_grid(basic, disabled)
    if adv:
        n_adv = len(lab.group_changed({"params": adv}, ss.lab_vals))
        with st.expander(f"Advanced · {len(adv)} setting{'s' if len(adv) != 1 else ''}"
                         + (f" · ● {n_adv} changed" if n_adv else ""), key=f"lab_adv:{group['id']}"):  # fmt: skip
            render_grid(adv, disabled)


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
    with common.card("live"):
        T.card_header(f"Running: {title}", "Training runs in a detached process; you can leave this page.",
                      right=T.badge(stage, "learn", T.VIOLET))  # fmt: skip
        c = st.columns(5)
        phase = (prog or {}).get("phase")
        c[0].metric("Stage", stage + (f" · {phase}" if phase and stage == "training" else ""),
                    help="Lab stage (training / enriching) and, while training, train's phase (setup / train / eval)")  # fmt: skip
        ep, n = (prog or {}).get("episode"), (prog or {}).get("episodes")
        c[1].metric("Game", f"{ep:,} / {n:,}" if n else "—")
        c[2].metric("Elapsed", lab.format_elapsed(elapsed))
        le = (prog or {}).get("last_eval") or {}
        c[3].metric("Learned red vs scripted blue", pct(le.get("red_win_rate")),
                    help=f"Latest checkpoint (after game {le.get('after_episode', '—')})")  # fmt: skip
        c[4].metric("Learned blue vs scripted red", pct(le.get("blue_win_rate")),
                    help=f"Latest checkpoint (after game {le.get('after_episode', '—')})")  # fmt: skip
        if stage == "enriching":
            st.progress(
                1.0, text="Training done · narrating games offline (SHAP, MITRE, template rationales)…"
            )
        elif stage == "starting" or prog is None:
            st.progress(0.0, text="Starting: creating the run and loading detectors…")
        else:
            frac = lab.progress_fraction(prog)
            st.progress(frac, text=f"Training: {frac:.0%} ({ep:,} / {n:,} games)")
        if job.get("message"):
            st.caption(job.get("message"))
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
                T.chart(charts.win_rate_figure(curve, pd.DataFrame(), 0, height=280), key="lab_live_curve")


def finished_panel(job: dict, spec: dict | None) -> None:
    stage = lab.effective_stage(job)
    title = job.get("label") or job.get("run_id") or job.get("token")
    started, finished = lab.parse_time(job.get("started")), lab.parse_time(job.get("finished"))
    took = lab.format_elapsed((finished - started).total_seconds()) if started and finished else "—"
    with common.card("finished"):
        if stage == "done":
            st.success(f"**{title}** finished in {took}. {job.get('message') or ''}")
            run_dir = Path(job["run_dir"])
            try:
                _, evals = L.load_summary(run_dir)
            except (L.RunFormatError, OSError):
                evals = pd.DataFrame(
                    columns=["after_episode", "matchup", "n", "red_win_rate", "blue_win_rate"]
                )
            fe = lab.final_eval(evals)
            c = st.columns(4)
            c[0].metric("Run", job.get("run_id") or "—")
            c[1].metric("Learned red vs scripted blue", pct(fe["red"]),
                        help=f"final checkpoint, after game {fe['after_episode']}")  # fmt: skip
            c[2].metric("Learned blue vs scripted red", pct(fe["blue"]),
                        help=f"final checkpoint, after game {fe['after_episode']}")  # fmt: skip
            c[3].metric("Changed settings", len(job.get("changed") or {}))
            if job.get("changed"):
                st.caption("Changed from defaults: " + changed_text(spec, job["changed"]))
        else:
            msg = job.get("message") or lab.stale_message(job)
            if lab.effective_stage(job) == "error" and job.get("stage") in lab.ACTIVE_STAGES:
                msg = lab.stale_message(job)
            (st.warning if job.get("stopped") else st.error)(f"**{title}**: {msg}")
            logs = [lg for lg in (job.get("train_log"), job.get("enrich_log")) if lg]
            if logs:
                st.caption("Logs: " + " · ".join(f"`{lg}`" for lg in logs))
        can_open = bool(job.get("run_dir")) and (Path(job["run_dir"]) / L.SUMMARY_FILE).exists()
        with st.container(horizontal=True, gap="small"):
            if can_open and st.button(
                "Open in Replay", type="primary", key="lab_open_new", icon=":material/play_circle:"
            ):
                ss["_open_run"] = Path(job["run_dir"]).name
                st.switch_page(common.PAGES["replay"])
            if st.button("Dismiss", key="lab_dismiss"):
                ss.pop("lab_token", None)
                st.rerun()


# ================================================================================================ experiment panels


def experiment_live_panel(token: str) -> None:
    job = lab.find_job(RUNS_DIR, token)
    stage = lab.effective_stage(job)
    if stage not in lab.ACTIVE_STAGES:
        ss.lab_token = token
        st.rerun()
    man = E.load_manifest(Path(job["exp_dir"])) if job.get("exp_dir") else None
    prog = E.run_progress(man or {})
    started = lab.parse_time(job.get("started"))
    elapsed = (lab.local_now() - started).total_seconds() if started else None
    with common.card("live"):
        T.card_header(f"Running experiment: {job.get('exp_name')}",
                      "Every (seed, condition) runs as its own training process, several in parallel. You can leave "
                      "this page; results land on the Evidence page when all runs finish.",
                      right=T.badge(stage, "learn", T.VIOLET))  # fmt: skip
        n_done = int((prog["status"] == "done").sum()) if not prog.empty else 0
        n_run = int(prog["status"].isin(["running", "finishing"]).sum()) if not prog.empty else 0
        c = st.columns(4)
        c[0].metric("Runs finished", f"{n_done} / {len(prog)}" if len(prog) else "—")
        c[1].metric("Running now", f"{n_run}")
        c[2].metric("Elapsed", lab.format_elapsed(elapsed))
        frac = float(prog["progress"].mean()) if not prog.empty else 0.0
        c[3].metric("Overall", f"{frac:.0%}")
        st.progress(frac, text=f"Experiment: {frac:.0%} of all training games")
        if not prog.empty:
            st.dataframe(prog.drop(columns=["games"]), hide_index=True, key="lab_exp_prog", column_config={
                "seed": st.column_config.NumberColumn("Seed", format="%d"), "condition": "Condition", "status": "Status",
                "progress": st.column_config.ProgressColumn("Progress", min_value=0, max_value=1),
                "game": st.column_config.NumberColumn("Game", format="%d"), "run_id": "Run", "error": "Error"})  # fmt: skip
        asked = lab.stop_requested_at(job)
        if asked is None:
            st.button("Stop experiment", key="lab_stop", on_click=_stop, args=(job,), icon=":material/stop_circle:",
                      help="Sends the experiment a cancel signal (CTRL_BREAK on Windows); it cancels every training "
                      "run, which records 'cancelled' and exits. Partial runs stay on disk.")  # fmt: skip
        else:
            waited = (lab.local_now() - asked).total_seconds()
            st.warning(f"Stopping… (requested {waited:.0f}s ago; waiting for the runs to save and exit)")
            if waited > lab.STOP_GRACE_S:
                st.button("Force stop", key="lab_force_stop", on_click=_stop, args=(job, True),
                          icon=":material/dangerous:", help="Kill the runner, the experiment and its runs now.")  # fmt: skip


def experiment_finished_panel(job: dict) -> None:
    stage = lab.effective_stage(job)
    name = job.get("exp_name")
    with common.card("finished"):
        if stage == "done":
            st.success(f"Experiment **{name}** finished. {job.get('message') or ''}")
        else:
            (st.warning if job.get("stopped") else st.error)(f"Experiment **{name}**: {job.get('message') or lab.stale_message(job)}")
            if job.get("train_log"):
                st.caption(f"Log: `{job['train_log']}`")
        with st.container(horizontal=True, gap="small"):
            if stage == "done" and st.button("Open in Evidence", type="primary", key="lab_open_ev",
                                             icon=":material/fact_check:"):  # fmt: skip
                ss["_open_experiment"] = name
                st.switch_page(common.PAGES["evidence"])
            if st.button("Dismiss", key="lab_dismiss"):
                ss.pop("lab_token", None)
                st.rerun()


# ================================================================================================ showcase


def _on_publish_run(run_id: str, key: str) -> None:
    SC.set_published(RUNS_DIR, "runs", run_id, bool(ss[key]))


def _on_default_run(key: str) -> None:
    if ss.get(key):
        SC.set_default_run(RUNS_DIR, ss[key])


@st.cache_data(show_spinner="Estimating the export size (indexes each published run's game log once)…",
               max_entries=8)  # fmt: skip
def _estimate(runs_dir: str, runs: tuple, exps: tuple, default: str | None, stamp) -> dict:
    plan = SC.make_plan(Path(runs_dir), list(runs), list(exps), default)
    return {"bytes": plan.bytes, "files": len(plan.items) + 1, "warnings": plan.warnings,
            "default": plan.default_run}  # fmt: skip


def showcase_panel() -> None:
    """What the public dashboard will show (``runs/.showcase.json``), its estimated size, and the export."""
    pub = SC.read_selection(RUNS_DIR)
    runs = [r for r in pub["runs"]]
    exps = [e for e in pub["experiments"]]
    with common.card("showcase"):
        T.card_header("Showcase",
                      "What the public, read-only dashboard shows. Select a row in the run history and switch on "
                      "<b>Publish</b>; publish experiments from the Evidence page. Export writes the "
                      f"<code>{SC.DEFAULT_OUT}/</code> folder that hosting serves "
                      "(<code>CYBERARENA_PUBLIC=1</code>, <code>CYBERARENA_RUNS_DIR</code>).",
                      right=T.badge(f"{len(runs)} runs · {len(exps)} experiments"))  # fmt: skip
        if not runs and not exps:
            T.caption("Nothing published yet.")
        else:
            rows = []
            for r in runs:
                d = RUNS_DIR / r
                label = L.run_info(d)["label"] if d.is_dir() else "(missing from runs/)"
                rows.append({"k": "run", "n": T.esc(label or "—"), "id": f'<span class="ca-id">{T.esc(r)}</span>'
                             + (" · default" if r == pub["default_run"] else "")})  # fmt: skip
            for e in exps:
                ok = (RUNS_DIR / SC.EXPERIMENTS / e / "manifest.json").exists()
                rows.append({"k": "experiment", "n": T.esc(e), "id": "" if ok else "(missing)"})
            st.html(T.table([("k", "Kind"), ("n", "Label / name"), ("id", "Run id")], rows, height=260))
            if runs:
                key = "lab_showcase_default"
                ss[key] = pub["default_run"] if pub["default_run"] in runs else runs[0]
                st.selectbox("Public dashboard opens on", runs, key=key, on_change=_on_default_run, args=(key,),
                             width=420, format_func=lambda r: lab.run_display_name(RUNS_DIR / r))  # fmt: skip
            stamp = tuple(L.file_stamp(RUNS_DIR / r / f) for r in runs for f in (L.EPISODES_FILE, L.ENRICHED_FILE))
            est = _estimate(str(RUNS_DIR), tuple(runs), tuple(exps), pub["default_run"], stamp)
            over = est["bytes"] > SC.DEFAULT_MAX_MB * SC.MB
            T.caption(f"Estimated export: <b>{est['bytes'] / SC.MB:.1f} MB</b> in {est['files']} files (limit "
                      f"{SC.DEFAULT_MAX_MB:.0f} MB){' · <b>over the limit</b>' if over else ''}. Copies logs and "
                      "summaries only: game logs are trimmed to the narrated and probe games; agent weights, detectors "
                      "and checkpoints stay here.")  # fmt: skip
            for w in est["warnings"]:
                T.caption(T.esc(w))
        cmd = lab.format_cmd(lab.showcase_export_cmd(PY))
        if st.button("Export showcase", key="lab_showcase_export", icon=":material/publish:",
                     disabled=not (runs or exps), help=f"Runs {cmd} in the repo folder and waits for it."):  # fmt: skip
            with st.spinner("Exporting the showcase…"):
                ss["lab_showcase_result"] = lab.run_showcase_export(PY, ROOT)
        res = ss.get("lab_showcase_result")
        if res:
            (st.success if res["returncode"] == 0 else st.error)(
                "Exported the showcase." if res["returncode"] == 0 else f"Export failed (exit code {res['returncode']}).")
            st.code(res["output"] or "(no output)", language="text", wrap_lines=True)


# ================================================================================================ page

if common.public():  # never reached through navigation (the page isn't registered); defence in depth
    T.empty_state("Not available in the public showcase", "The Simulation Lab only runs on the author's machine.")
    st.stop()

T.page_header(
    "Simulation Lab",
    "Set the variables, launch a simulated training run or a multi-seed experiment, watch it train, then compare "
    "it with earlier runs. Simulation only: training runs locally and narration is offline (template rationales, no "
    "API, no cost).",
)

spec, spec_err = cached_spec(PY, str(ROOT))
active = lab.active_job(RUNS_DIR)
if active is not None:
    ss.lab_token = active["token"]
    panel = experiment_live_panel if active.get("kind") == "experiment" else live_panel
    st.fragment(run_every=POLL_S)(panel)(active["token"])
elif ss.get("lab_token"):
    job = lab.find_job(RUNS_DIR, ss.lab_token)
    if job:
        (experiment_finished_panel(job) if job.get("kind") == "experiment" else finished_panel(job, spec))
busy = active is not None

# ------------------------------------------------------------------------------------------------ form
with common.card("form"):
    T.card_header("New simulation", "Every setting below comes from the arena's own parameter list, so new "
                  "settings appear here automatically.")  # fmt: skip
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
        if st.button("Retry", key="lab_retry", icon=":material/refresh:"):
            cached_spec.clear()
            st.rerun()
    else:
        init_state(spec)
        if busy:
            st.info("A simulation is running, so launching is disabled until it finishes or is stopped. "
                    "You can prepare the next one meanwhile.", icon=":material/hourglass_top:")  # fmt: skip
        groups = spec["groups"]
        mem = next((g for g in groups if g["id"] == "memory"), None)
        side_keys = ("init_side",)
        parent = None
        with common.toolbar("lab"):
            kind = st.segmented_control(
                "What to run", [SINGLE_RUN, EXPERIMENT], key="lab_kind", default=SINGLE_RUN, required=True,
                help="Single run: one training run you can replay. Experiment: the same setup trained with several "
                "seeds, per condition (adaptive / frozen detectors), for the Evidence page.",
            )  # fmt: skip
            st.segmented_control(
                "Preset", list(lab.PRESETS), key="lab_preset_sel", on_change=_apply_preset, args=(spec,),
                help="\n\n".join(f"**{k}**: {v}" for k, v in lab.PRESET_HELP.items()),
            )  # fmt: skip
            mode = FRESH
            if kind == SINGLE_RUN:
                mode = st.segmented_control("Agents start", [FRESH, CONTINUE], key="lab_mem", default=FRESH,
                                            required=True)  # fmt: skip
                if mode == CONTINUE:
                    parents = [p for p in lab.list_run_dirs(RUNS_DIR) if lab.has_agents(p) and lab.run_is_done(p)]
                    if parents:
                        parent = st.selectbox("Continue from", parents, key="lab_parent", width=340,
                                              format_func=lambda p: L.run_label(L.run_info(p)))  # fmt: skip
        if kind == SINGLE_RUN:
            T.caption(
                "<b>Start fresh</b>: both agents begin untrained. <b>Continue from a run</b>: the chosen side(s) start "
                "from that run's saved agents (Q-networks or Q-tables) and keep learning, so you can change one "
                "variable and watch already-trained agents adapt. Presets change a handful of settings at once."
            )
            if mode == CONTINUE and parent is None:
                st.warning("No finished run with saved agents yet. Start fresh first.")
            if mode == CONTINUE and mem:
                render_grid([p for p in mem["params"] if p["key"] in side_keys])
        else:
            ss.setdefault("lab_exp_name", f"lab-{lab.local_now():%Y%m%d-%H%M}")
            with st.container(key="toolbar-exp", horizontal=True, vertical_alignment="bottom", gap="medium"):
                st.text_input("Experiment name", key="lab_exp_name", max_chars=60, width=240,
                              help="Folder under runs/experiments/; must be new.")  # fmt: skip
                seeds = st.slider("Seeds", 1, 10, value=(1, 3), key="lab_exp_seeds",
                                  help="One run per seed and condition; more seeds = tighter intervals.")  # fmt: skip
                conds = st.multiselect("Conditions", list(lab.EXPERIMENT_CONDITIONS), default=list(lab.EXPERIMENT_CONDITIONS),
                                       key="lab_exp_conds", format_func=E.condition_label, width=320,
                                       help="Frozen adds --no-adaptive. Contrasts need both.")  # fmt: skip
                games = st.number_input("Games per run", 100, 20000, value=600, step=100, key="lab_exp_games")
            n_runs = (seeds[1] - seeds[0] + 1) * max(1, len(conds))
            T.caption(f"{n_runs} training runs of {int(games):,} games, several in parallel (the experiment uses "
                      "all but two CPU cores). Settings changed in the tabs below apply to every run; seed, game count, "
                      "label and detector setting are set per run by the experiment.")  # fmt: skip
        if mem and kind == SINGLE_RUN:
            render_group(mem, spec, False, skip=side_keys)

        tab_groups = [g for g in groups if g["id"] != "memory" and g["params"]]
        changed_by = {g["id"]: len(lab.group_changed(g, ss.lab_vals)) for g in tab_groups}
        if any(changed_by.values()):
            T.caption("Changed: " + " · ".join(f"<b>{T.esc(g['label'])}</b> {changed_by[g['id']]}"
                                                for g in tab_groups if changed_by[g["id"]]))  # fmt: skip
        tabs = st.tabs([g["label"] for g in tab_groups], key="lab_tabs")
        for tab, g in zip(tabs, tab_groups, strict=True):
            with tab:
                n_changed = changed_by[g["id"]]
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.html(f'<p class="ca-card-sub" style="margin:0">{len(g["params"])} settings'
                            + (f" · <b>{n_changed} changed</b>" if n_changed else " · all at defaults") + "</p>")  # fmt: skip
                    st.button("Reset to defaults", key=f"lab_reset:{g['id']}", on_click=_reset_group, args=(g, spec),
                              icon=":material/restart_alt:", disabled=not n_changed, type="tertiary",
                              help="Put every setting in this group back to its default")  # fmt: skip
                render_group(g, spec, False)

        changed = lab.changed_params(spec, ss.lab_vals)
        if mode == FRESH:
            changed = {k: v for k, v in changed.items() if k not in side_keys}
        init_side = ss.lab_vals.get("init_side", "both")
        train_args = lab.build_train_args(
            spec, ss.lab_vals, init_from=parent, label=ss.get("lab_label") or "", runs_dir=RUNS_DIR
        )
        if kind == EXPERIMENT:
            name = (ss.get("lab_exp_name") or "").strip()
            problem = lab.valid_experiment_name(name, RUNS_DIR) or (None if conds else "Pick at least one condition.")
            extra = lab.experiment_extra_args(train_args)
            exp_cmd = lab.experiment_cmd(name or "<name>", lab.seeds_arg(*seeds), conds or ["adaptive"], int(games),
                                         extra=extra, runs_dir=RUNS_DIR, python=PY)  # fmt: skip
            with st.container(horizontal=True, vertical_alignment="center", gap="medium", key="toolbar-launch"):
                launch = st.button("Run experiment", type="primary", key="lab_run_exp", disabled=busy or bool(problem),
                                   icon=":material/play_arrow:")  # fmt: skip
                if problem:
                    T.caption(T.esc(problem))
            T.caption("<b>Applied to every run:</b> " + (T.esc(changed_text(spec, {k: v for k, v in changed.items()
                      if k not in ("episodes", "adaptive", "detectors")})) or "nothing (all defaults)"))  # fmt: skip
            with st.expander("Command line", key="lab_cmd"):
                st.code(lab.format_cmd(exp_cmd), language="text", wrap_lines=True)
                st.caption("Run by a detached runner, so it finishes even if you close this tab. Stop sends the "
                           "experiment CTRL_BREAK, which it forwards to every run.")  # fmt: skip
            if launch and not problem:
                try:
                    job = lab.launch_experiment(RUNS_DIR, name, exp_cmd, python=PY, cwd=ROOT)
                except lab.LabBusyError as e:
                    st.warning(str(e))
                else:
                    ss.lab_token = job["token"]
                    st.rerun()
        else:
            with st.container(horizontal=True, vertical_alignment="bottom", gap="medium", key="toolbar-launch"):
                st.text_input("Run label", key="lab_label", max_chars=80, placeholder="e.g. phish 50%, louder red",
                              help="Free text stored in config.json; shown in the run picker and history.")  # fmt: skip
                launch = st.button("Run simulation", type="primary", key="lab_run", disabled=busy,
                                   icon=":material/play_arrow:")  # fmt: skip
            T.caption(
                "<b>Changed from defaults:</b> "
                + (T.esc(changed_text(spec, changed)) if changed else "nothing (all defaults)")
                + (f" · continuing from <code>{T.esc(Path(parent).name)}</code> ({T.esc(init_side)})" if parent else "")
            )
            with st.expander("Command line", key="lab_cmd"):
                st.code(lab.format_cmd(lab.train_cmd(train_args, PY)), language="text", wrap_lines=True)
                st.code(lab.format_cmd(lab.enrich_cmd("<run_dir>", PY)), language="text", wrap_lines=True)
                st.caption("Run by a detached runner, so it finishes even if you close this tab. Narration is offline.")
            if launch:
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
run_dirs = lab.list_run_dirs(RUNS_DIR)
with common.card("history"):
    T.card_header(
        "Run history", "Select rows to overlay their win-rate curves below and see what a change did."
    )
    if not run_dirs:
        T.empty_state("No runs yet", f"Nothing under <code>{T.esc(RUNS_DIR)}</code>. Launch one above.")
        st.stop()
    summaries = {}
    for p in run_dirs:
        try:
            summaries[str(p)] = L.cached_summary(str(p), L.file_stamp(p / L.SUMMARY_FILE))[1]
        except L.RunFormatError:
            pass
    hist = lab.history_table(RUNS_DIR, spec, summaries)
    hist["changed"] = [readable_changed(spec, c) for c in hist["changed"]]
    pub = SC.read_selection(RUNS_DIR)
    hist.insert(1, "showcase", ["default" if r == pub["default_run"] else "published" if r in pub["runs"] else ""
                                for r in hist["run_id"]])  # fmt: skip
    is_exp = hist["label"].fillna("").str.contains(" · seed ", regex=False)
    if is_exp.any():
        show_exp = st.toggle(f"Include the {int(is_exp.sum())} per-seed runs of experiments", value=False,
                             key="lab_hist_exp", help="Experiment runs are summarised on the Evidence page.")  # fmt: skip
        if not show_exp:
            hist = hist[~is_exp].reset_index(drop=True)
    slots = {
        rid: i for i, rid in enumerate(sorted(hist["run_id"]))
    }  # colour follows the run, oldest = slot 0
    with_evals = [i for i, r in hist.iterrows() if r["red_eval"] is not None or r["blue_eval"] is not None]
    shown = hist.drop(columns=["run_dir"])
    for col in ("red_eval", "blue_eval"):  # whole percentages, aligned
        shown[col] = pd.to_numeric(shown[col], errors="coerce") * 100
    event = st.dataframe(
        shown,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        selection_default={"selection": {"rows": with_evals[:2]}},
        key="lab_hist",
        column_config={
            "label": st.column_config.TextColumn("Label"),
            "showcase": st.column_config.TextColumn("Showcase", help="Published to the public showcase "
                                                    "(select the row, then use Publish below)"),
            "run_id": st.column_config.TextColumn("Run id"),
            "started": st.column_config.DatetimeColumn("Started", format="MMM D, HH:mm"),
            "parent": st.column_config.TextColumn("Continued from", help="Warm-started from this run (side)"),
            "changed": st.column_config.TextColumn("Changed from defaults", width="large"),
            "red_eval": st.column_config.NumberColumn("Red vs scripted", format="%.0f%%",
                                                      help="Final checkpoint: learned red's win rate vs scripted blue"),
            "blue_eval": st.column_config.NumberColumn("Blue vs scripted", format="%.0f%%",
                                                       help="Final checkpoint: learned blue's win rate vs scripted red"),
            "eval_at": st.column_config.NumberColumn("Checkpoint", format="%d"),
            "episodes": st.column_config.NumberColumn("Games", format="%d"),
            "status": st.column_config.TextColumn("Status"),
        },
    )  # fmt: skip
    rows = list(getattr(getattr(event, "selection", None), "rows", None) or [])
    sel = hist.iloc[rows] if rows else hist.iloc[0:0]
    if not sel.empty:
        with st.container(horizontal=True, gap="medium", vertical_alignment="center"):
            if st.button(f"Open {sel.iloc[0]['run_id']} in Replay", key="lab_open_sel", icon=":material/play_circle:"):
                ss["_open_run"] = sel.iloc[0]["run_id"]
                st.switch_page(common.PAGES["replay"])
            for r in sel.itertuples(index=False):
                key = f"lab_publish:{r.run_id}"
                ss[key] = r.run_id in pub["runs"]  # mirror the file every rerun (Evidence may have changed it)
                st.toggle(f"Publish {r.label or r.run_id}", key=key, on_change=_on_publish_run, args=(r.run_id, key),
                          help=f"Include run {r.run_id} in the public showcase (runs/.showcase.json).")  # fmt: skip

if not sel.empty:
    to_plot = []
    for r in sel.itertuples(index=False):
        evals = summaries.get(r.run_dir)
        curve = L.eval_curve(evals) if evals is not None and not evals.empty else pd.DataFrame()
        if curve.empty:
            continue
        to_plot.append({"name": (f"{r.label} · " if r.label else "") + r.run_id, "curve": curve,
                        "slot": slots[r.run_id]})  # fmt: skip
    with common.card("compare"):
        T.card_header(
            "Compare win rates",
            "Each point is the learned agent's win rate against its scripted opponent at a checkpoint; bands are 95% "
            "confidence intervals. Colour and marker identify the run. Where two runs' bands overlap, the "
            "difference is within noise.",
        )
        if to_plot:
            T.chart(charts.compare_figure(to_plot), key="lab_compare")
            finals = []
            for r in to_plot:
                c = r["curve"]
                red = c[c["side"] == "red"].sort_values("after_episode")
                if not red.empty:
                    finals.append((float(red["win_rate"].iloc[-1]), r["name"]))
            if len(finals) >= 2:
                finals.sort(reverse=True)
                T.takeaway(f"At the final checkpoint learned red wins {finals[0][0]:.0%} in "
                           f"<b>{T.esc(finals[0][1])}</b> against {finals[-1][0]:.0%} in "
                           f"<b>{T.esc(finals[-1][1])}</b>.")  # fmt: skip
            else:
                T.takeaway("Select a second run to compare.")
        else:
            T.empty_state("No checkpoints yet", "The selected runs have no evaluation rows yet.")

showcase_panel()
