"""
Simudo GUI — Simulation panel.

Sections:
  • Simulation mode  (4-way Select: Equilibrium / Dark J-V / Illuminated / Illuminated J-V)
  • Intensity ramp   (selfconsistent_optics, step_size — shown when mode includes ramp)
  • Voltage sweep    (values list / linspace helper, selfconsistent_optics, step_size)
  • Checkpoints      (at_end flags, directory)
  • Resume-from path
  • Spatially resolved output (xdmf_mesh, spatial_profiles, output folder)
  • Execution        (backend type + settings; collapsed card)
  • Run button

On Run:  saves project, builds the appropriate RunnerBackend from the execution
         profile, launches the runner, and streams stdout into the log panel.
"""

from __future__ import annotations
import shutil, subprocess, os, shlex
from typing import TYPE_CHECKING

import panel as pn

from simudo.gui.batch import enumerate_sweep_params
from simudo.gui.panels.shared import (
    INPUT_SS, SELECT_SS, BTN_LIGHT_SS, RADIO_SS, ADD_BTN_SS, CHECKBOX_SS,
    sec_header, label, scan_checkpoint_dir,
    read_checkpoint_metadata, already_computed_values, value_already_computed,
)

if TYPE_CHECKING:
    from simudo.gui.app import SimudoApp

# Stylesheet for the big Run button
_RUN_SS = ["""
.bk-btn {
    background: #1a6e38 !important; color: #c8f0d8 !important;
    border: 1px solid #27ae60 !important; border-radius: 5px !important;
    font-size: 14px !important; font-weight: 700 !important;
    padding: 6px 24px !important; cursor: pointer !important;
}
.bk-btn:hover { background: #218838 !important; border-color: #2ecc71 !important; }
.bk-btn:disabled { background: #1a2e22 !important; color: #4a6a55 !important;
    border-color: #2d4a3e !important; cursor: default !important; }
"""]

_SECTION_STYLE = (
    "font-size:11px;font-weight:700;letter-spacing:0.08em;color:#8a9bb5;"
    "border-left:2px solid #2d4a6e;padding-left:6px;margin-top:4px;"
)

# ── Simulation mode ────────────────────────────────────────────────────────────
# Maps to intensity_ramp.enabled / voltage_sweep.enabled flags.

_MODES = [
    "Equilibrium only",   # ir=off, vs=off
    "Dark J-V",           # ir=off, vs=on
    "Illuminated (V=0)",  # ir=on,  vs=off
    "Illuminated J-V",    # ir=on,  vs=on  (default)
]

def _mode_from_flags(ir_enabled: bool, vs_enabled: bool) -> str:
    if not ir_enabled and not vs_enabled: return "Equilibrium only"
    if not ir_enabled and vs_enabled:     return "Dark J-V"
    if ir_enabled     and not vs_enabled: return "Illuminated (V=0)"
    return "Illuminated J-V"

def _flags_from_mode(mode: str):
    """Return (ir_enabled, vs_enabled)."""
    if mode == "Equilibrium only":  return False, False
    if mode == "Dark J-V":          return False, True
    if mode == "Illuminated (V=0)": return True,  False
    return True, True   # "Illuminated J-V"


def _sub_header(title: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f'<div style="{_SECTION_STYLE}">{title}</div>',
        sizing_mode="stretch_width", height=20, margin=(6, 0, 2, 0),
    )

_CB_LABEL_STYLE = "font-size:12px;color:#c8d6e5;cursor:pointer;"
_CB_DIM_STYLE   = "font-size:11px;color:#8a9bb5;"


def _cb_row(cb: pn.widgets.Checkbox, label_html: str) -> pn.Row:
    """Checkbox + explicit HTML label side-by-side (avoids Panel name= rendering issues)."""
    return pn.Row(
        cb,
        pn.pane.HTML(label_html, margin=0),
        margin=0, styles={"gap": "6px", "align-items": "center"},
    )


class SimulationPanel:
    def __init__(self, app: "SimudoApp"):
        self.app = app
        self._log_lines: list = []
        self._out_folder_in = None
        self._folder_display_updating = False
        self._run_handle = None   # active RunHandle; set during run, cleared after
        self._last_run_abs_dir: str | None = None
        # Persistent widgets — survive _rebuild_view() — same pattern as _log_ta.
        # Use TextInput (read-only via CSS) so .value updates via pn.state.execute
        # use the same proven mechanism as _log_ta.
        self._actual_folder_input = pn.widgets.TextInput(
            value="Last run: —",
            disabled=False,
            height=24,
            sizing_mode="stretch_width",
            margin=(0, 0, 2, 0),
            stylesheets=["""
                input {
                    background: transparent !important;
                    border: none !important;
                    color: #4a9eff !important;
                    font-size: 10px !important;
                    font-family: monospace !important;
                    padding: 1px 0 1px 50px !important;
                    caret-color: transparent !important;
                    pointer-events: none !important;
                }
            """],
        )
        self._stop_btn = pn.widgets.Button(
            name="■  Interrupt", height=38,
            sizing_mode="stretch_width", margin=(4, 0, 0, 0),
            stylesheets=["""
                .bk-btn { background: #3d1515 !important; color: #e05555 !important;
                    border: 1px solid #7a2020 !important; border-radius: 4px !important;
                    font-size: 14px !important; font-weight: 600 !important;
                    cursor: pointer !important; }
                .bk-btn:hover { background: #5a1c1c !important; color: #ff7070 !important;
                    border-color: #e05555 !important; }
            """],
            visible=False,
        )
        self._stop_btn.on_click(lambda e: self._interrupt_run())
        self._view_log_btn = pn.widgets.Button(
            name="📄 View info.log",
            height=26, margin=0, stylesheets=BTN_LIGHT_SS,
        )
        self._view_log_btn.on_click(lambda e: self._open_info_log())
        self._info_log_pane = pn.pane.HTML("", height=0, margin=0)
        self._log_ta = pn.widgets.TextAreaInput(
            value="Run output will appear here.",
            disabled=False,
            height=660,
            sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
            stylesheets=["""
                textarea {
                    background: #0d1520 !important; color: #ffffff !important;
                    border: 1px solid #2d3748 !important; border-radius: 3px !important;
                    font-family: monospace !important; font-size: 12px !important;
                    padding: 8px !important; resize: vertical !important;
                    box-sizing: border-box !important;
                    overflow-y: scroll !important; overscroll-behavior: contain !important;
                    caret-color: transparent !important;
                }
            """],
        )
        self._view = pn.Column(
            sizing_mode="stretch_width",
            styles={"padding": "12px", "gap": "4px"},
        )

        # ── Parameter sweep persistent state ──────────────────────────────────
        self._sweep_variations: list = []   # list of batch.BatchVariation
        self._sweep_mode: str = "cartesian"
        self._sweep_max_parallel: int = 2
        self._sweep_log_lines: list = []
        self._sweep_loaded: bool = False
        self._sweep_last_filepath: str = ""
        self._sweep_launcher = None
        self._sweep_status_ta = pn.widgets.TextAreaInput(
            value="Sweep output will appear here.",
            disabled=False, height=280, sizing_mode="stretch_width",
            margin=(8, 0, 0, 0),
            stylesheets=["""
                textarea {
                    background: #0d1520 !important; color: #ffffff !important;
                    border: 1px solid #2d3748 !important; border-radius: 3px !important;
                    font-family: monospace !important; font-size: 12px !important;
                    padding: 8px !important; resize: vertical !important;
                    caret-color: transparent !important;
                }
            """],
        )
        self._sweep_stop_btn = pn.widgets.Button(
            name="■  Stop Sweep", height=32, sizing_mode="stretch_width",
            margin=(4, 0, 0, 0), visible=False,
            stylesheets=["""
                .bk-btn { background: #3d1515 !important; color: #e05555 !important;
                    border: 1px solid #7a2020 !important; border-radius: 4px !important;
                    font-size: 13px !important; font-weight: 600 !important; }
                .bk-btn:hover { background: #5a1c1c !important; }
            """],
        )
        self._sweep_stop_btn.on_click(lambda e: self._stop_sweep())
        self._sweep_rows_col = pn.Column(
            sizing_mode="stretch_width", styles={"gap": "0px"},
        )
        self._sweep_n_runs_label = pn.pane.HTML(
            "", sizing_mode="stretch_width", margin=(4, 0, 0, 0),
        )
        self._sweep_zip_warning = pn.pane.HTML(
            "", sizing_mode="stretch_width", margin=(2, 0, 0, 0),
        )
        # Per-run status table (populated when a sweep is launched)
        self._sweep_table_col = pn.Column(
            sizing_mode="stretch_width",
            styles={"gap": "2px"},
            margin=(4, 0, 4, 0),
        )
        self._sweep_row_htmls: list = []   # list of pn.pane.HTML, one per run
        self._sweep_row_dirs: list = []    # list of abs output dir strings
        self._sweep_row_params: list = []  # list of compact param strings

    def view(self) -> pn.Column:
        self._try_load_sweep_sidecar()
        self._rebuild_view()
        return self._view

    def _rebuild_view(self):
        sim = self.app.project.simulation
        ir  = sim.intensity_ramp
        vs  = sim.voltage_sweep
        out = sim.output
        ck  = sim.checkpoints
        has_optical = bool(self.app.project.optical_fields
                           or self.app.project.optical_field_sets)

        proj_dir = (os.path.dirname(os.path.abspath(self.app.project.filepath))
                    if self.app.project.filepath else os.getcwd())

        # ── Resume-state banner ─────────────────────────────────────────────────
        # Represented generically as (parameter name, parameter value, stage
        # label) rather than anything voltage-specific -- see gui/TODO.md
        # "Checkpoint / Resume UX" (the "generic resume-state model" item) --
        # so this keeps working once the optical ramp gets a configurable
        # endpoint beyond I=1. resume_ir_note goes in the Intensity Ramp
        # section, resume_vs_note + the live skip-note pane in Voltage
        # Sweep, depending on which stage the checkpoint belongs to.
        resume_ir_note = None
        resume_vs_note = None
        already_v_for_note = None   # set below only for a V-type checkpoint
        if sim.resume_from:
            run_folder = os.path.join(proj_dir, self.app.project.output_folder or "out/a")
            ckpt_abs = os.path.join(run_folder, sim.resume_from)
            meta = read_checkpoint_metadata(ckpt_abs)
            if meta is None:
                warn = pn.pane.HTML(
                    f'<div style="font-size:11px;color:#e07b39;background:rgba(224,123,57,0.08);'
                    f'border:1px solid rgba(224,123,57,0.25);border-radius:4px;'
                    f'padding:6px 8px;margin:2px 0 6px;">'
                    f'⚠ Could not read checkpoint metadata at '
                    f'<code style="color:#f0a060;">{ckpt_abs}</code>.</div>',
                    sizing_mode="stretch_width", margin=0,
                )
                resume_ir_note = resume_vs_note = warn
            else:
                param_name, param_value = meta
                stage_label = {"V": "voltage sweep", "I": "optical ramp"}.get(
                    param_name, param_name)
                banner = pn.pane.HTML(
                    f'<div style="font-size:11px;color:#7ec8e3;background:rgba(74,158,255,0.08);'
                    f'border:1px solid rgba(74,158,255,0.25);border-radius:4px;'
                    f'padding:6px 8px;margin:2px 0 6px;">'
                    f'↻ Resuming from <strong>{param_name} = {param_value:g}</strong> '
                    f'({stage_label}) — continues from there in whichever '
                    f'direction(s) new values need; already-computed points are '
                    f'skipped automatically.</div>',
                    sizing_mode="stretch_width", margin=0,
                )
                if param_name == "I":
                    resume_ir_note = banner
                else:
                    resume_vs_note = banner
                    already_v_for_note = already_computed_values(run_folder, "V")

        # Persistent pane (present whenever a V-type checkpoint is active),
        # updated live so it doesn't go stale as the user edits the values
        # without navigating away and back.
        resume_vs_skip_pane = pn.pane.HTML(
            "", sizing_mode="stretch_width", margin=0, visible=False)

        def _update_vs_skip_note():
            if already_v_for_note is None:
                return
            skip_vals = sorted({v for v in vs.values
                                if value_already_computed(v, already_v_for_note)})
            if skip_vals:
                skip_str = ", ".join(f"{v:g}" for v in skip_vals)
                resume_vs_skip_pane.object = (
                    f'<div style="font-size:10px;color:#8a9bb5;padding:1px 0 4px;">'
                    f'Already computed, will be skipped: {skip_str}</div>')
                resume_vs_skip_pane.visible = True
            else:
                resume_vs_skip_pane.object = ""
                resume_vs_skip_pane.visible = False
        _update_vs_skip_note()

        # ── Mode selector ─────────────────────────────────────────────────────
        current_mode = _mode_from_flags(ir.enabled, vs.enabled)

        mode_sel = pn.widgets.Select(
            value=current_mode, options=_MODES,
            height=26, width=180, margin=0, stylesheets=SELECT_SS,
        )

        # ── Intensity ramp ────────────────────────────────────────────────────
        ir_sco_cb = pn.widgets.Checkbox(
            name="", value=ir.selfconsistent_optics,
            height=20, width=18, margin=0, stylesheets=CHECKBOX_SS,
        )
        ir_step_in = pn.widgets.TextInput(
            value="" if ir.step_size is None else str(ir.step_size),
            placeholder="auto", height=26, width=120, margin=0, stylesheets=INPUT_SS,
        )

        def _on_ir_sco(e):  ir.selfconsistent_optics = e.new; self.app.autosave()
        def _on_ir_step(e):
            s = e.new.strip()
            try:    ir.step_size = float(s) if s else None
            except ValueError: pass
            self.app.autosave()
        ir_sco_cb.param.watch(_on_ir_sco, "value")
        ir_step_in.param.watch(_on_ir_step, "value")

        ir_section = pn.Column(
            _sub_header("Intensity Ramp"),
            *([resume_ir_note] if resume_ir_note is not None else []),
            _cb_row(ir_sco_cb, f'<span style="{_CB_LABEL_STYLE}">Self-consistent optics</span>'),
            pn.Row(label("Step size:", 64), ir_step_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"}),
            sizing_mode="stretch_width",
            visible=ir.enabled,
        )

        # ── Voltage sweep ─────────────────────────────────────────────────────
        vs_sco_cb = pn.widgets.Checkbox(
            name="", value=vs.selfconsistent_optics,
            height=20, width=18, margin=0, stylesheets=CHECKBOX_SS,
        )
        vs_step_in = pn.widgets.TextInput(
            value="" if vs.step_size is None else str(vs.step_size),
            placeholder="auto", height=26, width=120, margin=0, stylesheets=INPUT_SS,
        )

        # Voltage values: textarea (one value per line)
        vs_vals_str = "\n".join(str(v) for v in vs.values)
        vs_vals_ta = pn.widgets.TextAreaInput(
            value=vs_vals_str,
            placeholder="One voltage per line, e.g.:\n0.0\n0.2\n0.4\n…",
            height=120, margin=0, stylesheets=INPUT_SS,
            sizing_mode="stretch_width",
        )

        # Linspace helper — start defaults to 0 (solver begins at equilibrium)
        # but can be overridden when resuming from a checkpoint at a non-zero voltage.
        ls_start = pn.widgets.NumberInput(
            value=0.0, name="", height=26, width=60, margin=0, stylesheets=INPUT_SS)
        ls_stop  = pn.widgets.NumberInput(
            value=1.0, name="", height=26, width=60, margin=0, stylesheets=INPUT_SS)
        ls_n     = pn.widgets.NumberInput(
            value=11, name="", step=1, start=2,
            height=26, width=50, margin=0, stylesheets=INPUT_SS)
        ls_apply = pn.widgets.Button(
            name="Apply", button_type="light", height=26, margin=0,
            stylesheets=BTN_LIGHT_SS)

        def _apply_ls(e):
            import numpy as np
            vals = list(np.linspace(float(ls_start.value or 0),
                                    float(ls_stop.value  or 1),
                                    int(ls_n.value        or 11)))
            vs.values = [round(v, 8) for v in vals]
            vs_vals_ta.value = "\n".join(str(v) for v in vs.values)
            self.app.autosave()
            _update_vs_skip_note()
        ls_apply.on_click(_apply_ls)

        def _on_vs_sco(e):  vs.selfconsistent_optics = e.new; self.app.autosave()
        def _on_vs_step(e):
            s = e.new.strip()
            try:    vs.step_size = float(s) if s else None
            except ValueError: pass
            self.app.autosave()
        def _on_vs_vals(e):
            vals = []
            for line in e.new.splitlines():
                line = line.strip()
                if line:
                    try: vals.append(float(line))
                    except ValueError: pass
            vs.values = vals
            self.app.autosave()
            _update_vs_skip_note()
        vs_sco_cb.param.watch(_on_vs_sco, "value")
        vs_step_in.param.watch(_on_vs_step, "value")
        vs_vals_ta.param.watch(_on_vs_vals, "value")
        # Also save on every keystroke (not just on blur/Enter) -- a
        # TextAreaInput's "value" param only updates when the widget loses
        # focus, so an edit that's typed but never blurred (e.g. switching
        # to a terminal to run the CLI directly, which is a supported
        # workflow here) would never reach disk. "value_input" mirrors
        # "value" but fires live, closing that gap.
        vs_vals_ta.param.watch(_on_vs_vals, "value_input")

        vs_section = pn.Column(
            _sub_header("Voltage Sweep"),
            *([resume_vs_note] if resume_vs_note is not None else []),
            _cb_row(vs_sco_cb, f'<span style="{_CB_LABEL_STYLE}">Self-consistent optics</span>'),
            pn.Row(label("Step size:", 64), vs_step_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"}),
            pn.Column(
                pn.pane.HTML('<div style="font-size:11px;color:#8a9bb5;margin-bottom:2px;">'
                             'Voltage values (V):</div>', height=18, margin=0),
                vs_vals_ta,
                resume_vs_skip_pane,
                pn.Row(
                    label("linspace:", 56),
                    ls_start,
                    pn.pane.HTML('<span style="font-size:12px;color:#8a9bb5;">→</span>',
                                 width=14, height=26, margin=0,
                                 styles={"display":"flex","align-items":"center"}),
                    ls_stop,
                    label("V,  n=", 38), ls_n, ls_apply,
                    margin=0, styles={"gap": "4px", "align-items": "center"},
                ),
                sizing_mode="stretch_width", styles={"gap": "4px"},
            ),
            sizing_mode="stretch_width",
            visible=vs.enabled,
        )

        # ── Mode callback (defined after ir_section / vs_section) ─────────────
        no_optical_note = pn.pane.HTML(
            '<div style="font-size:11px;color:#e07b39;padding:2px 0;">'
            '⚠ No optical fields defined — illuminated modes will be treated as dark.</div>',
            sizing_mode="stretch_width", margin=0,
            visible=(not has_optical and current_mode.startswith("Illuminated")),
        )

        def _on_mode(e):
            new_ir, new_vs = _flags_from_mode(e.new)
            ir.enabled = new_ir
            vs.enabled = new_vs
            ir_section.visible = new_ir
            vs_section.visible = new_vs
            no_optical_note.visible = (not has_optical and e.new.startswith("Illuminated"))
            self.app.autosave()
        mode_sel.param.watch(_on_mode, "value")

        # ── Checkpoints ───────────────────────────────────────────────────────
        ck_ir_end_cb = pn.widgets.Checkbox(
            name="", value=ck.intensity_ramp.checkpoint_at_end,
            height=20, width=18, margin=0, stylesheets=CHECKBOX_SS,
        )
        ck_ir_at_in = pn.widgets.TextInput(
            value=", ".join(str(v) for v in ck.intensity_ramp.at_values),
            placeholder="also checkpoint at these intensities, e.g. 0.5, 0.8",
            height=24, margin=0, stylesheets=INPUT_SS,
            sizing_mode="stretch_width",
        )
        ck_vs_end_cb = pn.widgets.Checkbox(
            name="", value=ck.voltage_sweep.checkpoint_at_end,
            height=20, width=18, margin=0, stylesheets=CHECKBOX_SS,
        )
        ck_vs_at_in = pn.widgets.TextInput(
            value=", ".join(str(v) for v in ck.voltage_sweep.at_values),
            placeholder="also checkpoint at these voltages, e.g. 0.2, 0.4",
            height=24, margin=0, stylesheets=INPUT_SS,
            sizing_mode="stretch_width",
        )
        ck_note = pn.pane.HTML(
            '<div style="font-size:11px;color:#556070;padding:2px 0 4px 0;">'
            'Checkpoints save the full solver state so a run can be resumed '
            'later — e.g. extend a voltage sweep further without redoing it, '
            'or branch a fresh sweep (like negative voltages) off an '
            'already-computed optical ramp. They are large files written to '
            'the directory below.</div>',
            sizing_mode="stretch_width", margin=0,
        )
        ck_dir_in = pn.widgets.TextInput(
            value=ck.directory, placeholder="checkpoints",
            height=26, margin=0, stylesheets=INPUT_SS,
            sizing_mode="stretch_width",
        )

        def _parse_floats(s: str) -> list[float]:
            vals = []
            for tok in s.replace(",", " ").split():
                try: vals.append(float(tok))
                except ValueError: pass
            return vals

        def _on_ck_ir_end(e): ck.intensity_ramp.checkpoint_at_end = e.new; self.app.autosave()
        def _on_ck_vs_end(e): ck.voltage_sweep.checkpoint_at_end  = e.new; self.app.autosave()
        def _on_ck_ir_at(e):  ck.intensity_ramp.at_values = _parse_floats(e.new); self.app.autosave()
        def _on_ck_vs_at(e):  ck.voltage_sweep.at_values  = _parse_floats(e.new); self.app.autosave()
        def _on_ck_dir(e):
            ck.directory = e.new.strip() or "checkpoints"
            self.app.autosave()
            _rescan_checkpoints()
        ck_ir_end_cb.param.watch(_on_ck_ir_end, "value")
        ck_vs_end_cb.param.watch(_on_ck_vs_end, "value")
        ck_ir_at_in.param.watch(_on_ck_ir_at, "value")
        ck_vs_at_in.param.watch(_on_ck_vs_at, "value")
        ck_dir_in.param.watch(_on_ck_dir, "value")

        # ── Resume from checkpoint ───────────────────────────────────────────────
        # A dropdown scans the checkpoints directory (from the most recent run
        # this session, or the configured output folder otherwise) and lists
        # what's actually there in human-readable form — picking an entry
        # fills the path box below, which remains the editable source of
        # truth (same pattern as the voltage-sweep linspace helper above).
        _NONE_LABEL = "— pick a checkpoint —"

        resume_sel = pn.widgets.Select(
            options={_NONE_LABEL: ""}, value="",
            height=26, margin=0, stylesheets=SELECT_SS,
            sizing_mode="stretch_width",
        )
        resume_rescan_btn = pn.widgets.Button(
            name="↻ Rescan", height=26, margin=0, stylesheets=BTN_LIGHT_SS,
        )
        resume_in = pn.widgets.TextInput(
            value=sim.resume_from or "",
            placeholder="(no resume — start fresh)",
            height=26, margin=0, stylesheets=INPUT_SS,
            sizing_mode="stretch_width",
        )
        resume_scan_note = pn.pane.HTML(
            "", sizing_mode="stretch_width", margin=(2, 0, 0, 0),
        )

        def _checkpoint_scan_dir() -> str:
            base = self.app._last_run_output_dir or os.path.join(
                proj_dir, self.app.project.output_folder or "out/a")
            return os.path.join(base, ck.directory or "checkpoints")

        def _rescan_checkpoints():
            scan_dir = _checkpoint_scan_dir()
            found = scan_checkpoint_dir(scan_dir)
            rel_ckdir = ck.directory or "checkpoints"
            options = {_NONE_LABEL: ""}
            for label_str, fname in found:
                options[label_str] = f"{rel_ckdir}/{fname}"
            resume_sel.options = options
            resume_sel.value = ""
            note = f'Scanned: <code style="color:#7a8caa;">{scan_dir}</code>'
            if not found:
                note += ' — no checkpoints found yet.'
            resume_scan_note.object = (
                f'<div style="font-size:10px;color:#556070;">{note}</div>')

        def _on_resume_sel(e):
            if e.new:
                resume_in.value = e.new   # fills the path box; its own watcher persists it
        def _on_resume(e):
            s = e.new.strip()
            new_val = s or None
            changed = new_val != sim.resume_from
            sim.resume_from = new_val
            self.app.autosave()
            if changed:
                # Picking (or clearing) a checkpoint changes the resume
                # banner and skip-note shown above -- both are computed once
                # per _rebuild_view() call, so without this they'd stay
                # stale (or absent) until the user navigates away and back.
                self._rebuild_view()
        resume_sel.param.watch(_on_resume_sel, "value")
        resume_rescan_btn.on_click(lambda e: _rescan_checkpoints())
        resume_in.param.watch(_on_resume, "value")

        _rescan_checkpoints()

        # ── Mesh settings ─────────────────────────────────────────────────────
        mesh_max_in = pn.widgets.FloatInput(
            value=self.app.project.mesh_max_edge_length or 0.02,
            name="", width=100, height=26, margin=0, stylesheets=INPUT_SS,
        )
        def _on_mesh_max(e):
            self.app.project.mesh_max_edge_length = e.new or 0.02
            self.app.autosave()
        mesh_max_in.param.watch(_on_mesh_max, "value")

        # ── Output settings ───────────────────────────────────────────────────
        sp_cb = pn.widgets.Checkbox(
            name="", value=out.spatial_profiles,
            height=20, width=18, margin=0, stylesheets=CHECKBOX_SS,
        )
        xdmf_cb = pn.widgets.Checkbox(
            name="", value=out.xdmf_mesh,
            height=20, width=18, margin=0, stylesheets=CHECKBOX_SS,
        )
        out_folder_in = pn.widgets.TextInput(
            value=self.app.project.output_folder or "out/a",
            placeholder="out/a",
            height=26, margin=0, stylesheets=INPUT_SS,
            sizing_mode="stretch_width",
        )
        self._out_folder_in = out_folder_in

        folder_context = pn.pane.HTML(
            f'<div style="font-size:10px;color:#556070;padding:1px 0 4px 0;">'
            f'Relative to: <code style="color:#8a9bb5;">{proj_dir}</code></div>',
            sizing_mode="stretch_width", margin=0,
        )

        # ── Stale-date warning for output folder ──────────────────────────────
        import re as _re, datetime as _dt
        _date_match = _re.search(
            r'(\d{4}[A-Za-z]{3}\d{2})',
            self.app.project.output_folder or "",
        )
        _today_str = _dt.date.today().strftime("%Y%b%d")
        if _date_match and _date_match.group(1) != _today_str:
            _stale_date = _date_match.group(1)
            _update_date_btn = pn.widgets.Button(
                name=f"Update to {_today_str}", height=24, margin=0,
                stylesheets=BTN_LIGHT_SS,
            )
            def _on_update_date(e, _old=_stale_date, _new=_today_str):
                new_folder = (self.app.project.output_folder or "").replace(_old, _new, 1)
                self.app.project.output_folder = new_folder
                self._folder_display_updating = True
                out_folder_in.value = new_folder
                self._folder_display_updating = False
                self.app.autosave()
                self._rebuild_view()
            _update_date_btn.on_click(_on_update_date)
            _date_warn = pn.Row(
                pn.pane.HTML(
                    f'<span style="font-size:11px;color:#e07b39;">'
                    f'⚠ Folder date <code style="color:#f0a060;">{_stale_date}</code>'
                    f' is not today</span>',
                    margin=0,
                    styles={"display": "flex", "align-items": "center"},
                ),
                _update_date_btn,
                margin=(2, 0, 2, 0),
                styles={"gap": "8px", "align-items": "center"},
            )
        else:
            _date_warn = None

        def _on_sp(e):  out.spatial_profiles = e.new; self.app.autosave()
        def _on_xdmf(e): out.xdmf_mesh = e.new; self.app.autosave()
        def _on_out_folder(e):
            if self._folder_display_updating:
                return
            self.app.project.output_folder = e.new.strip() or "out/a"
            self.app.autosave()
        sp_cb.param.watch(_on_sp, "value")
        xdmf_cb.param.watch(_on_xdmf, "value")
        out_folder_in.param.watch(_on_out_folder, "value")

        # ── Run button ────────────────────────────────────────────────────────
        run_btn = pn.widgets.Button(
            name="▶  Run Simulation", stylesheets=_RUN_SS,
            height=38, sizing_mode="stretch_width", margin=(8, 0, 0, 0),
        )
        run_btn.on_click(lambda e: self._run_simulation())
        self._run_btn = run_btn

        # ── Assemble ─────────────────────────────────────────────────────────
        param_opts = enumerate_sweep_params(self.app.project)

        self._view.objects = [
            sec_header("SIMULATION"),

            _sub_header("Mode"),
            pn.Row(label("Mode:", 38), mode_sel,
                   pn.Spacer(sizing_mode="stretch_width", margin=0),
                   margin=0, styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            no_optical_note,

            ir_section,
            vs_section,

            _sub_header("Checkpoints"),
            ck_note,
            _cb_row(ck_ir_end_cb, f'<span style="{_CB_LABEL_STYLE}">Save checkpoint after intensity ramp</span>'),
            pn.Row(label("", 24), ck_ir_at_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            _cb_row(ck_vs_end_cb, f'<span style="{_CB_LABEL_STYLE}">Save checkpoint after voltage sweep</span>'),
            pn.Row(label("", 24), ck_vs_at_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.Row(label("Directory:", 64), ck_dir_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),

            _sub_header("Resume from checkpoint"),
            pn.pane.HTML(
                '<div style="font-size:11px;color:#556070;padding:0 0 4px 0;">'
                'Pick a saved checkpoint to continue from — the run will skip '
                'stages that checkpoint already covers.</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            pn.Row(resume_sel, resume_rescan_btn, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            resume_scan_note,
            pn.Row(label("Path:", 36), resume_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),

            _sub_header("Mesh"),
            pn.Row(
                label("Max spacing:", 80),
                mesh_max_in,
                pn.pane.HTML('<span style="font-size:11px;color:#556070;margin-left:4px;">µm</span>', margin=0),
                margin=0, styles={"gap": "4px", "align-items": "center"},
            ),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#556070;padding:1px 0 4px 0;">'
                'Global maximum mesh spacing. Per-layer mesh options are set in the Layers panel.</div>',
                margin=0, sizing_mode="stretch_width",
            ),

            _sub_header("Spatially resolved output"),
            pn.pane.HTML(
                f'<div style="{_CB_DIM_STYLE}; margin: 0 0 4px 2px;">'
                f'Choose format for spatially resolved outputs of bands, carrier '
                f'densities, currents, etc. at each sweep point.</div>',
                sizing_mode="stretch_width", margin=0),
            _cb_row(xdmf_cb, (
                f'<span style="{_CB_LABEL_STYLE}">XDMF (binary) output</span>'
                f'<span style="{_CB_DIM_STYLE}"> — required for spatially resolved plots in this GUI. Also readable by Paraview</span>'
            )),
            _cb_row(sp_cb, (
                f'<span style="{_CB_LABEL_STYLE}">CSV format output</span>'
                f'<span style="{_CB_DIM_STYLE}"> — all quantities projected onto mesh points</span>'
            )),
            pn.Row(label("Folder:", 46), out_folder_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            *([_date_warn] if _date_warn is not None else []),
            self._actual_folder_input,
            folder_context,

            _sub_header("Execution"),
            self._build_execution_card(),

            self._build_sweep_card(param_opts),

            run_btn,
            self._stop_btn,
            pn.Row(
                pn.Spacer(sizing_mode="stretch_width", margin=0),
                self._view_log_btn,
                margin=(2, 0, 0, 0),
                sizing_mode="stretch_width",
            ),
            self._log_ta,
            self._info_log_pane,
        ]

    # ── Execution settings card ───────────────────────────────────────────────

    def _build_execution_card(self) -> pn.Card:
        """Collapsible card showing the active execution profile."""
        from simudo.gui.model import ExecutionProfile
        profile = self.app.get_effective_execution_profile()

        _TYPE_OPTS = ["docker", "local", "ssh"]
        _TYPE_LABELS = {
            "docker": "Docker (local container)",
            "local":  "Local subprocess",
            "ssh":    "Remote SSH",
        }
        type_sel = pn.widgets.Select(
            options={_TYPE_LABELS[t]: t for t in _TYPE_OPTS},
            value=profile.type,
            height=26, margin=0, stylesheets=SELECT_SS, sizing_mode="stretch_width",
        )

        # ── Docker fields ─────────────────────────────────────────────────────
        container_in = pn.widgets.TextInput(
            value=profile.docker_container,
            placeholder="name of the running dolfin container",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        host_root_in = pn.widgets.TextInput(
            value=profile.docker_host_root,
            placeholder="bind-mounted host path  (e.g. /Users/you/Simudo  or  C:/Users/you/Simudo)",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        container_root_in = pn.widgets.TextInput(
            value=profile.docker_container_root,
            placeholder="mount point inside container  (e.g. /home/user/simudo)",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        docker_simudo_in = pn.widgets.TextInput(
            value=getattr(profile, 'docker_simudo_path', ''),
            placeholder="(optional — only needed if Simudo is pip-installed inside the container)",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        docker_section = pn.Column(
            pn.Row(label("Container:", 80), container_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.Row(label("Host root:", 80), host_root_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.Row(label("Container root:", 80), container_root_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#556070;padding:1px 0;">'
                'Host root must be bind-mounted to container root before launching the container '
                '(e.g. <code>docker run -v /host/path:/container/path …</code>). '
                'Windows: use forward slashes (<code>C:/Users/you/…</code>) or Cygwin paths.</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            pn.Row(label("Simudo path:", 80), docker_simudo_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#556070;padding:1px 0;">'
                '<b style="color:#8a9bb5;">Simudo path</b> — inside the container, run: '
                '<code style="color:#7ec8e3;">docker exec '
                '&lt;container&gt; python3 -c \'import simudo; print(simudo.__file__)\'</code>'
                ' and paste the output above. '
                'Leave blank if the Simudo source tree is mounted under Host root.</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            sizing_mode="stretch_width", styles={"gap": "4px"},
            visible=(profile.type == "docker"),
        )

        # ── SSH fields ────────────────────────────────────────────────────────
        ssh_host_in = pn.widgets.TextInput(
            value=profile.ssh_host,
            placeholder="~/.ssh/config alias  or  hostname  or  user@hostname",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        ssh_user_in = pn.widgets.TextInput(
            value=profile.ssh_user,
            placeholder="username  (leave blank if host already includes it or uses ~/.ssh/config)",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        ssh_key_in = pn.widgets.TextInput(
            value=profile.ssh_identity_file,
            placeholder="~/.ssh/id_ed25519  (blank = ssh-agent / ~/.ssh/config default)",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        ssh_work_in = pn.widgets.TextInput(
            value=profile.ssh_remote_work_dir, placeholder="~/simudo_runs",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        ssh_simudo_in = pn.widgets.TextInput(
            value=profile.ssh_simudo_path,
            placeholder="/home/you/simudo/code/simudo/__init__.py",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width",
        )
        ssh_delete_cb = pn.widgets.Checkbox(
            name="", value=profile.ssh_delete_remote_after_sync,
            height=20, width=18, margin=0, stylesheets=CHECKBOX_SS,
        )
        ssh_section = pn.Column(
            pn.Row(label("Host:", 80), ssh_host_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.Row(label("User:", 80), ssh_user_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.Row(label("Identity file:", 80), ssh_key_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.Row(label("Remote work dir:", 80), ssh_work_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.Row(label("Simudo path:", 80), ssh_simudo_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#556070;padding:2px 0 1px 0;">'
                '<b style="color:#8a9bb5;">Simudo path</b> — on the remote machine, run: '
                '<code style="color:#7ec8e3;">python3 -c \'import simudo; print(simudo.__file__)\'</code>'
                ' and paste the output above.</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            _cb_row(ssh_delete_cb, (
                f'<span style="{_CB_LABEL_STYLE}">Delete remote run directory after sync</span>'
                f'<span style="{_CB_DIM_STYLE}"> — output files remain locally</span>'
            )),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#556070;padding:1px 0;">'
                'Password auth is not supported — set up key-based auth or configure '
                'the host in <code>~/.ssh/config</code>.</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            sizing_mode="stretch_width", styles={"gap": "4px"},
            visible=(profile.type == "ssh"),
        )

        python_in = pn.widgets.TextInput(
            value=profile.python_cmd, placeholder="python3",
            height=26, margin=0, stylesheets=INPUT_SS, width=120,
        )
        test_btn = pn.widgets.Button(
            name="Test connection", height=26, margin=0, stylesheets=BTN_LIGHT_SS,
        )
        default_btn = pn.widgets.Button(
            name="Set as default", height=26, margin=0, stylesheets=BTN_LIGHT_SS,
        )
        restore_btn = pn.widgets.Button(
            name="Restore default", height=26, margin=0, stylesheets=BTN_LIGHT_SS,
        )
        test_result = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)

        def _save_profile():
            """Save current fields to the per-project execution override."""
            ep = ExecutionProfile(
                type=type_sel.value,
                docker_container=container_in.value.strip(),
                docker_host_root=host_root_in.value.strip(),
                docker_container_root=container_root_in.value.strip(),
                docker_simudo_path=docker_simudo_in.value.strip(),
                ssh_host=ssh_host_in.value.strip(),
                ssh_user=ssh_user_in.value.strip(),
                ssh_identity_file=ssh_key_in.value.strip(),
                ssh_remote_work_dir=ssh_work_in.value.strip() or "~/simudo_runs",
                ssh_simudo_path=ssh_simudo_in.value.strip(),
                ssh_delete_remote_after_sync=ssh_delete_cb.value,
                python_cmd=python_in.value.strip() or "python3",
            )
            self.app.project.execution = ep
            self.app.autosave()
            return ep

        def _current_ep_for_type(t):
            return ExecutionProfile(
                type=t,
                docker_container=container_in.value.strip(),
                docker_host_root=host_root_in.value.strip(),
                docker_container_root=container_root_in.value.strip(),
                docker_simudo_path=docker_simudo_in.value.strip(),
                ssh_host=ssh_host_in.value.strip(),
                ssh_user=ssh_user_in.value.strip(),
                ssh_identity_file=ssh_key_in.value.strip(),
                ssh_remote_work_dir=ssh_work_in.value.strip() or "~/simudo_runs",
                ssh_simudo_path=ssh_simudo_in.value.strip(),
                ssh_delete_remote_after_sync=ssh_delete_cb.value,
                python_cmd=python_in.value.strip() or "python3",
            )

        def _apply_ep_to_widgets(ep):
            container_in.value = ep.docker_container
            host_root_in.value = ep.docker_host_root
            container_root_in.value = ep.docker_container_root
            docker_simudo_in.value = getattr(ep, 'docker_simudo_path', '')
            ssh_host_in.value = ep.ssh_host
            ssh_user_in.value = ep.ssh_user
            ssh_key_in.value = ep.ssh_identity_file
            ssh_work_in.value = ep.ssh_remote_work_dir
            ssh_simudo_in.value = ep.ssh_simudo_path
            ssh_delete_cb.value = ep.ssh_delete_remote_after_sync
            python_in.value = ep.python_cmd

        def _on_type(e):
            # Save the current widget values for the type being left.
            self.app.save_execution_profile_for_type(
                e.old, _current_ep_for_type(e.old)
            )
            # Restore saved values for the new type if we have any.
            saved = self.app.get_execution_profile_for_type(e.new)
            if saved:
                _apply_ep_to_widgets(saved)
            t = e.new
            docker_section.visible = (t == "docker")
            ssh_section.visible    = (t == "ssh")
            _save_profile()

        def _on_test(e):
            import subprocess as sp
            import html as _html
            ep = _save_profile()
            test_result.object = '<span style="color:#8a9bb5;font-size:11px;">Testing…</span>'
            try:
                if ep.type == "docker":
                    # Step 1: check the container is reachable
                    result = sp.run(
                        ["docker", "exec", ep.docker_container, "echo", "ok"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if result.returncode != 0 or "ok" not in result.stdout:
                        test_result.object = (
                            f'<span style="color:#e74c3c;font-size:11px;">'
                            f'✗ {_html.escape((result.stderr or result.stdout).strip())}</span>'
                        )
                        return

                    # Step 2: try to resolve the runner path inside the container
                    # and verify that it exists.
                    try:
                        from simudo.gui.runner_backends import DockerBackend
                        _gui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        _runner_host = os.path.join(_gui_dir, "simudo_1d_runner.py")
                        _backend = DockerBackend(
                            container=ep.docker_container,
                            host_root=ep.docker_host_root,
                            container_root=ep.docker_container_root,
                            python_cmd=ep.python_cmd,
                            container_simudo_path=getattr(ep, 'docker_simudo_path', ''),
                        )
                        _runner_c = _backend._derive_container_runner_path(_runner_host)
                    except ValueError as _exc:
                        test_result.object = (
                            f'<span style="color:#e74c3c;font-size:11px;">'
                            f'✓ Container reachable, but runner path error: '
                            f'{_html.escape(str(_exc))}</span>'
                        )
                        return
                    except Exception as _exc:
                        test_result.object = (
                            f'<span style="color:#e74c3c;font-size:11px;">'
                            f'✓ Container reachable, but could not resolve runner path: '
                            f'{_html.escape(str(_exc))}</span>'
                        )
                        return

                    runner_check = sp.run(
                        ["docker", "exec", ep.docker_container, "test", "-f", _runner_c],
                        capture_output=True, text=True, timeout=10,
                    )
                    if runner_check.returncode == 0:
                        test_result.object = (
                            f'<span style="color:#27ae60;font-size:11px;">'
                            f'✓ Docker OK — runner found at '
                            f'<code>{_html.escape(_runner_c)}</code></span>'
                        )
                    else:
                        test_result.object = (
                            f'<span style="color:#e07b39;font-size:11px;">'
                            f'✓ Container reachable, but runner not found at '
                            f'<code>{_html.escape(_runner_c)}</code>. '
                            f'Check Host root / Container root, or set the '
                            f'Simudo path field.</span>'
                        )
                elif ep.type == "local":
                    import sys as _sys
                    result = sp.run(
                        [_sys.executable, "--version"], capture_output=True, text=True, timeout=5,
                    )
                    test_result.object = (
                        f'<span style="color:#27ae60;font-size:11px;">'
                        f'✓ Local Python: {result.stdout.strip() or result.stderr.strip()}</span>'
                    )
                elif ep.type == "ssh":
                    if not ep.ssh_host:
                        test_result.object = (
                            '<span style="color:#e74c3c;font-size:11px;">✗ No SSH host configured</span>'
                        )
                        return
                    host_spec = f"{ep.ssh_user}@{ep.ssh_host}" if ep.ssh_user else ep.ssh_host
                    ssh_cmd = ["ssh"]
                    if ep.ssh_identity_file:
                        ssh_cmd += ["-i", os.path.expanduser(ep.ssh_identity_file)]
                    ssh_cmd += [
                        "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=10",
                        host_spec, "echo ok",
                    ]
                    result = sp.run(ssh_cmd, capture_output=True, text=True, timeout=15)
                    if result.returncode == 0 and "ok" in result.stdout:
                        test_result.object = (
                            '<span style="color:#27ae60;font-size:11px;">✓ SSH connection OK</span>'
                        )
                    else:
                        err = (result.stderr or result.stdout).strip() or "Connection failed"
                        test_result.object = (
                            f'<span style="color:#e74c3c;font-size:11px;">✗ {err}</span>'
                        )
                else:
                    test_result.object = (
                        '<span style="color:#8a6020;font-size:11px;">'
                        '⚠ Test not yet implemented for this backend</span>'
                    )
            except Exception as exc:
                test_result.object = (
                    f'<span style="color:#e74c3c;font-size:11px;">✗ {exc}</span>'
                )

        def _on_set_default(e):
            ep = _save_profile()
            self.app.set_default_execution_profile(ep)
            test_result.object = (
                '<span style="color:#27ae60;font-size:11px;">✓ Saved as default</span>'
            )

        def _on_restore_default(e):
            d = self.app._default_execution
            type_sel.value = d.type   # triggers _on_type (saves old, loads cached new)
            _apply_ep_to_widgets(d)   # overwrite with actual default values
            # Clear per-project override so the default is used going forward.
            self.app.project.execution = None
            self.app.autosave()
            test_result.object = (
                '<span style="color:#27ae60;font-size:11px;">✓ Restored default</span>'
            )

        type_sel.param.watch(_on_type, "value")
        for w in (container_in, host_root_in, container_root_in, docker_simudo_in,
                  ssh_host_in, ssh_user_in, ssh_key_in, ssh_work_in,
                  ssh_simudo_in, python_in, ssh_delete_cb):
            w.param.watch(lambda e: _save_profile(), "value")
        test_btn.on_click(_on_test)
        default_btn.on_click(_on_set_default)
        restore_btn.on_click(_on_restore_default)

        return pn.Card(
            pn.Row(label("Type:", 46), type_sel, margin=0,
                   styles={"gap": "4px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            docker_section,
            ssh_section,
            pn.Row(label("Python:", 46), python_in, margin=0,
                   styles={"gap": "4px", "align-items": "center"}),
            pn.Row(test_btn, default_btn, restore_btn, test_result, margin=(4, 0, 0, 0),
                   styles={"gap": "8px", "align-items": "center"},
                   sizing_mode="stretch_width"),
            title="Execution",
            collapsed=False,
            sizing_mode="stretch_width",
            styles={"background": "#1a2435", "border": "1px solid #2d3748",
                    "border-radius": "4px"},
            margin=(0, 0, 4, 0),
        )

    def _run_simulation(self):
        """Save project then launch the runner via the configured backend."""
        try:
            self._run_simulation_inner()
        except Exception as e:
            import traceback
            self._set_log(f"⚠ Unexpected error:\n{traceback.format_exc()}", error=True)

    def _run_simulation_inner(self):
        import sys, threading
        from simudo.gui.runner_backends import backend_from_profile

        proj = self.app.project
        if not proj.filepath:
            self._set_log("⚠ Please save the project before running.", error=True)
            return
        try:
            self.app.save_yaml()
        except Exception as e:
            self._set_log(f"⚠ Could not save project: {e}", error=True)
            return

        runner_host = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "simudo_1d_runner.py")
        if not os.path.exists(runner_host):
            self._set_log(f"⚠ Runner not found: {runner_host}", error=True)
            return

        profile = self.app.get_effective_execution_profile()
        try:
            backend = backend_from_profile(profile)
        except Exception as e:
            self._set_log(f"⚠ Could not build runner backend: {e}", error=True)
            return

        extra_args = ["--resume"] if proj.simulation.resume_from else []
        self._log_lines = []

        try:
            handle = backend.launch(
                runner_host_path=runner_host,
                project_yaml_host_path=proj.filepath,
                lib_dirs_host=self.app.get_library_dirs(),
                extra_args=extra_args,
            )
        except Exception as e:
            self._set_log(f"⚠ Failed to launch: {e}", error=True)
            return

        # Synchronous update in the main thread — shows command immediately.
        self._log_lines = [f"$ {handle.cmd_str}\n"]
        self._log_ta.value = "".join(self._log_lines)
        self._run_handle = handle
        self._stop_btn.visible = True

        def _stream():
            # Start periodic sync thread if the backend supports it (e.g. SSH, 120 s).
            import threading as _threading
            _stop_sync = _threading.Event()
            _sync_interval = getattr(handle, "periodic_sync_interval", 0)

            def _periodic_sync():
                while not _stop_sync.wait(timeout=_sync_interval):
                    for msg in handle.do_intermediate_sync():
                        self._append_log(msg)

            if _sync_interval > 0:
                _threading.Thread(target=_periodic_sync, daemon=True).start()

            try:
                for msg in handle.initial_msgs():
                    self._append_log(msg)
                    # SSH backend announces local output dir here (before runner starts).
                    for _pfx in ("[GUI] Local output dir:", "Local output dir:"):
                        if msg.startswith(_pfx):
                            self._last_run_abs_dir = msg[len(_pfx):].strip()
                            break

                for line in handle.stdout_lines():
                    # Detect output-directory announcement and translate to host path.
                    notify_dir = None
                    if line.startswith("Remote output dir:"):
                        raw_path = line.split(":", 1)[1].strip()
                        try:
                            notify_dir = handle.translate_output_path(raw_path)
                        except Exception:
                            notify_dir = raw_path

                    def _update(text=line, nd=notify_dir):
                        self._log_lines.append(text)
                        if len(self._log_lines) > 5000:
                            self._log_lines = self._log_lines[-5000:]
                        if nd is not None:
                            self._last_run_abs_dir = nd
                            self._log_lines.append(f"[GUI] Local output dir: {nd}\n")
                        self._log_ta.value = "".join(self._log_lines)
                        if nd is not None:
                            proj = self.app.project
                            try:
                                if proj.filepath:
                                    proj_dir = os.path.dirname(
                                        os.path.abspath(proj.filepath))
                                    rel = os.path.relpath(nd, proj_dir)
                                else:
                                    rel = nd
                            except Exception:
                                rel = nd
                            self._actual_folder_input.value = f"▶ Last run: {rel}"
                            try:
                                self.app.notify_run_output_dir(nd)
                            except Exception:
                                pass

                    try:
                        pn.state.execute(_update)
                    except Exception:
                        _update()

                rc = handle.returncode
                status = "OK" if rc == 0 else "FAILED"
                self._append_log(f"\n[{status}] Exit code: {rc}\n")

                # Post-run finalization (e.g. SSH rsync of output files back to local).
                for msg in handle.finalize_msgs():
                    self._append_log(msg)

            except Exception as exc:
                self._append_log(f"\n[ERROR] {exc}\n")
            finally:
                _stop_sync.set()   # stop periodic sync thread if running
                self._run_handle = None
                # Fallback: scan collected log lines for output dir in case
                # the in-flight detection was missed.
                fallback_dir = None
                for logged_line in self._log_lines:
                    if logged_line.startswith("Remote output dir:"):
                        raw = logged_line.split(":", 1)[1].strip()
                        try:
                            fallback_dir = handle.translate_output_path(raw)
                        except Exception:
                            fallback_dir = raw
                        break

                def _done(fd=fallback_dir):
                    self._stop_btn.visible = False
                    if fd is not None and self._actual_folder_input.value.startswith("Last run: —"):
                        proj = self.app.project
                        try:
                            if proj.filepath:
                                proj_dir = os.path.dirname(os.path.abspath(proj.filepath))
                                rel = os.path.relpath(fd, proj_dir)
                            else:
                                rel = fd
                        except Exception:
                            rel = fd
                        self._actual_folder_input.value = f"▶ Last run: {rel}"
                        try:
                            self.app.notify_run_output_dir(fd)
                        except Exception:
                            pass
                    self.app.refresh_output_panel()
                try:
                    pn.state.execute(_done)
                except Exception:
                    _done()

        threading.Thread(target=_stream, daemon=True).start()

    def _append_log(self, text: str):
        """Append plain text to the log; safe to call from a background thread."""
        def _update():
            self._log_lines.append(text)
            if len(self._log_lines) > 5000:
                self._log_lines = self._log_lines[-5000:]
            self._log_ta.value = "".join(self._log_lines)
        try:
            pn.state.execute(_update)
        except Exception:
            _update()

    def _set_log(self, msg: str, error: bool = False):
        """Show a plain-text status message. Called from main thread."""
        self._log_lines = [msg, "\n"]
        self._log_ta.value = "".join(self._log_lines)

    def _interrupt_run(self):
        handle = self._run_handle
        if handle is not None:
            handle.stop()
            self._append_log("\n[INTERRUPTED by user]\n")

    def _open_info_log(self):
        import html as _html, time

        # Resolve output dir: prefer tracked value, then scan log lines, then
        # fall back to the displayed folder path.
        abs_dir = self._last_run_abs_dir

        if not abs_dir:
            # Scan collected log lines for any output-dir announcement.
            for logged in reversed(self._log_lines):
                for prefix in ("[GUI] Local output dir: ", "Local output dir: ",
                               "Remote output dir: "):
                    if logged.startswith(prefix):
                        candidate = logged[len(prefix):].strip()
                        if os.path.isabs(candidate):
                            abs_dir = candidate
                        break
                if abs_dir:
                    break

        if not abs_dir:
            raw = self._actual_folder_input.value
            # strip "▶ Last run: " or "Last run: —" prefixes
            for prefix in ("▶ Last run: ", "Last run: "):
                if raw.startswith(prefix):
                    raw = raw[len(prefix):]
                    break
            if raw and raw != "—":
                if os.path.isabs(raw):
                    abs_dir = raw
                elif self.app.project.filepath:
                    proj_dir = os.path.dirname(os.path.abspath(self.app.project.filepath))
                    abs_dir = os.path.join(proj_dir, raw)

        log_path = os.path.join(abs_dir, "info.log") if abs_dir else None
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path, errors="replace") as f:
                    content = f.read()
            except Exception as exc:
                content = f"Error reading {log_path}: {exc}"
            header = _html.escape(log_path)
        elif abs_dir:
            content = f"info.log not found in:\n{abs_dir}"
            header = _html.escape(abs_dir)
        else:
            content = "No output directory known yet.\nRun a simulation first."
            header = "info.log"

        escaped = _html.escape(content)
        # Nonce ensures the string always differs so Panel pushes a browser update
        # even when content is the same (e.g. overlay was JS-hidden, not removed).
        nonce = int(time.monotonic() * 1000)
        self._info_log_pane.object = f"""<!-- {nonce} -->
<div class="simudo-log-overlay"
     style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:9999;
            background:rgba(0,0,0,0.72);display:flex;align-items:center;
            justify-content:center;pointer-events:auto;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
     onclick="if(event.target===this)this.style.display='none'">
  <div style="background:#0d1520;border:1px solid #2d4a6e;border-radius:8px;
              width:82vw;height:82vh;display:flex;flex-direction:column;overflow:hidden;
              box-shadow:0 8px 32px rgba(0,0,0,0.6);pointer-events:auto;">
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:8px 14px;border-bottom:1px solid #2d3748;flex-shrink:0;">
      <span style="font-size:12px;font-weight:600;color:#c8d6e5;">
        info.log &nbsp;<code style="font-size:10px;color:#7a8caa;font-weight:normal;">{header}</code>
      </span>
      <button onclick="this.closest('.simudo-log-overlay').style.display='none'"
              style="background:transparent;border:none;color:#7a8caa;cursor:pointer;
                     font-size:20px;line-height:1;padding:0 2px;margin:0;pointer-events:auto;">×</button>
    </div>
    <pre style="flex:1;overflow:auto;color:#c8d6e5;font-size:11px;margin:0;
                padding:12px;font-family:monospace;white-space:pre-wrap;
                word-break:break-word;user-select:text;-webkit-user-select:text;">{escaped}</pre>
  </div>
</div>
"""

    # ── Parameter sweep ───────────────────────────────────────────────────────

    def _try_load_sweep_sidecar(self):
        """Load the sidecar .batch.yaml when the project filepath changes."""
        from simudo.gui.batch import load_batch_spec

        fp = self.app.project.filepath or ""
        if fp == self._sweep_last_filepath:
            return
        self._sweep_last_filepath = fp
        if not fp:
            return
        spec = load_batch_spec(fp)
        if spec is not None:
            self._sweep_variations = spec.variations
            self._sweep_mode = spec.mode
            self._sweep_max_parallel = spec.max_parallel
        self._sweep_rebuild_rows()

    def _save_sweep_sidecar(self):
        from simudo.gui.batch import BatchSpec, save_batch_spec
        fp = self.app.project.filepath or ""
        if not fp:
            return
        spec = BatchSpec(
            mode=self._sweep_mode,
            max_parallel=self._sweep_max_parallel,
            variations=list(self._sweep_variations),
        )
        try:
            save_batch_spec(fp, spec)
        except Exception:
            pass
        self._update_sweep_n_runs()

    def _update_sweep_n_runs(self):
        """Refresh the 'N sub-runs will be generated' label and zip-warning."""
        from simudo.gui.batch import BatchSpec
        spec = BatchSpec(
            mode=self._sweep_mode,
            max_parallel=self._sweep_max_parallel,
            variations=list(self._sweep_variations),
        )
        n = spec.run_count()
        if n == 0:
            self._sweep_n_runs_label.object = ""
        else:
            self._sweep_n_runs_label.object = (
                f'<div style="font-size:11px;color:#8a9bb5;">'
                f'→ {n} sub-run{"s" if n != 1 else ""} will be generated.</div>'
            )
        # Zip-length mismatch warning
        if spec.mode == "zip" and len(spec.variations) > 1:
            lengths = [len(v.values) for v in spec.variations]
            if len(set(lengths)) > 1:
                pairs = ", ".join(
                    f'"{v.label}": {len(v.values)}' for v in spec.variations
                )
                self._sweep_zip_warning.object = (
                    f'<div style="font-size:11px;color:#e07b39;padding:3px 0;">'
                    f'⚠ Paired mode requires equal-length value lists. '
                    f'Current lengths — {pairs}.</div>'
                )
            else:
                self._sweep_zip_warning.object = ""
        else:
            self._sweep_zip_warning.object = ""

    def _add_empty_variation(self, param_opts):
        from simudo.gui.batch import BatchVariation
        if not param_opts:
            return
        lbl, tgt = param_opts[0]
        self._sweep_variations.append(BatchVariation(label=lbl, target=tgt, values=[]))
        self._sweep_rebuild_rows()
        self._save_sweep_sidecar()

    def _sweep_rebuild_rows(self):
        """Rebuild the inline-editable variation rows in _sweep_rows_col."""
        from simudo.gui.batch import enumerate_sweep_params, get_current_param_value
        from simudo.gui.panels.shared import INPUT_SS, SELECT_SS

        param_opts = enumerate_sweep_params(self.app.project)
        param_labels = [lbl for lbl, _ in param_opts]
        param_map   = {lbl: tgt for lbl, tgt in param_opts}

        _DEL_SS = ["""
            .bk-btn { background: #2a1515 !important; color: #e05555 !important;
                border: 1px solid #5a2020 !important; border-radius: 3px !important;
                font-size: 13px !important; padding: 0 !important; line-height:1; }
            .bk-btn:hover { background: #5a1c1c !important; }
        """]

        rows = []
        for i, var in enumerate(self._sweep_variations):
            # Keep existing label if still valid; otherwise fall back to first option.
            cur_label = var.label if var.label in param_map else (
                param_labels[0] if param_labels else ""
            )

            cur_sel = cur_label or (param_labels[0] if param_labels else "(no parameters defined)")
            param_sel = pn.widgets.Select(
                options=param_labels or ["(no parameters defined)"],
                value=cur_sel,
                height=26, margin=0, stylesheets=SELECT_SS,
                sizing_mode="stretch_width",
                disabled=not param_labels,
            )

            # Current-value hint — updates when dropdown changes
            def _hint_html(lbl):
                tgt = param_map.get(lbl)
                if tgt is None:
                    return ""
                cur = get_current_param_value(self.app.project, tgt)
                if not cur:
                    return ""
                return (
                    f'<span style="font-size:10px;color:#556070;">'
                    f'current: <code style="color:#7a9fc0;">{cur}</code></span>'
                )

            hint_pane = pn.pane.HTML(
                _hint_html(cur_sel), margin=0,
                styles={"display": "flex", "align-items": "center"},
            )

            vals_str = "\n".join(str(v) for v in var.values)
            vals_ta = pn.widgets.TextAreaInput(
                value=vals_str,
                placeholder="One value per line:\n0.5\n1.0\n1.5",
                height=72, margin=0, stylesheets=INPUT_SS,
                sizing_mode="stretch_width",
            )

            del_btn = pn.widgets.Button(
                name="×", width=26, height=26, margin=0,
                stylesheets=_DEL_SS,
            )

            idx = i

            def _on_param(e, _idx=idx, _hp=hint_pane):
                if e.new in param_map:
                    self._sweep_variations[_idx].label  = e.new
                    self._sweep_variations[_idx].target = param_map[e.new]
                    _hp.object = _hint_html(e.new)
                    self._save_sweep_sidecar()

            def _on_vals(e, _idx=idx):
                values = []
                for line in e.new.splitlines():
                    s = line.strip()
                    if s:
                        try:
                            values.append(float(s))
                        except ValueError:
                            pass
                self._sweep_variations[_idx].values = values
                self._save_sweep_sidecar()

            def _on_del(e, _idx=idx):
                self._sweep_variations.pop(_idx)
                self._sweep_rebuild_rows()
                self._save_sweep_sidecar()

            param_sel.param.watch(_on_param, "value")
            vals_ta.param.watch(_on_vals, "value")
            del_btn.on_click(_on_del)

            row_col = pn.Column(
                pn.Row(
                    param_sel, del_btn,
                    margin=0,
                    styles={"gap": "4px", "align-items": "center"},
                    sizing_mode="stretch_width",
                ),
                hint_pane,
                vals_ta,
                sizing_mode="stretch_width",
                styles={"gap": "3px", "background": "#111d2e",
                        "border": "1px solid #2d3748", "border-radius": "4px",
                        "padding": "6px 6px 6px 6px"},
                margin=(0, 0, 6, 0),
            )
            rows.append(row_col)

        self._sweep_rows_col.objects = rows
        self._update_sweep_n_runs()

    def _build_sweep_card(self, param_opts) -> pn.Card:
        """Build the collapsible Parameter Sweep card."""
        from simudo.gui.panels.shared import INPUT_SS, BTN_LIGHT_SS

        param_map = {lbl: tgt for lbl, tgt in param_opts}

        note = pn.pane.HTML(
            '<div style="font-size:11px;color:#556070;padding:2px 0 6px 0;">'
            'Add parameters to vary. Each combination becomes a numbered sub-directory '
            'under the output folder with its own project YAML. '
            'Settings are saved to <code>.batch.yaml</code> alongside the project.<br>'
            '<span style="color:#445060;">'
            'Only parameters set manually in the Layers panel are listed. '
            'Parameters supplied by the material library cannot be swept here.</span></div>',
            sizing_mode="stretch_width", margin=0,
        )

        # ── "+" button ───────────────────────────────────────────────────────
        add_row_btn = pn.widgets.Button(
            name="＋  Add swept parameter", height=28,
            sizing_mode="stretch_width", margin=(0, 0, 6, 0),
            stylesheets=BTN_LIGHT_SS,
        )
        add_row_btn.on_click(lambda e: self._add_empty_variation(param_opts))

        # ── Mode and max-parallel ─────────────────────────────────────────────
        _MODE_SS = ["""
            .bk-btn-group button {
                background: #1a2435 !important; color: #8a9bb5 !important;
                border: 1px solid #2d3748 !important; font-size: 11px !important;
                padding: 3px 8px !important; border-radius: 3px !important; }
            .bk-btn-group button.bk-active {
                background: #1a4a8a !important; color: #c8d6e5 !important;
                border-color: #2d6abf !important; }
        """]
        # Cartesian first, as default
        mode_sel = pn.widgets.RadioButtonGroup(
            options={
                "All combinations (Cartesian product)": "cartesian",
                "Pair up variations (zip — all must be same length)": "zip",
            },
            value=self._sweep_mode,
            height=26, margin=0, stylesheets=_MODE_SS,
        )
        parallel_in = pn.widgets.IntInput(
            value=self._sweep_max_parallel,
            start=1, end=64,
            width=56, height=26, margin=0, stylesheets=INPUT_SS,
        )

        def _on_mode(e):
            self._sweep_mode = e.new
            self._save_sweep_sidecar()   # also updates zip warning
        def _on_parallel(e):
            self._sweep_max_parallel = int(e.new or 1)
            self._save_sweep_sidecar()
        mode_sel.param.watch(_on_mode, "value")
        parallel_in.param.watch(_on_parallel, "value")

        # ── Run Sweep button ──────────────────────────────────────────────────
        _RUN_SWEEP_SS = ["""
            .bk-btn {
                background: #1a3a6e !important; color: #c8d6e5 !important;
                border: 1px solid #2d6abf !important; border-radius: 5px !important;
                font-size: 13px !important; font-weight: 700 !important;
                padding: 5px 20px !important; cursor: pointer !important;
            }
            .bk-btn:hover { background: #1e4a8a !important; }
            .bk-btn:disabled { background: #101f35 !important; color: #3a4a5a !important;
                border-color: #1e2d3d !important; cursor: default !important; }
        """]
        run_sweep_btn = pn.widgets.Button(
            name="▶▶  Run Sweep", height=36,
            sizing_mode="stretch_width", margin=(6, 0, 0, 0),
            stylesheets=_RUN_SWEEP_SS,
        )
        run_sweep_btn.on_click(lambda e: self._run_sweep(param_map))

        # Rebuild rows from persisted state and update n-runs / warning labels
        self._sweep_rebuild_rows()

        return pn.Card(
            note,
            add_row_btn,
            self._sweep_rows_col,
            self._sweep_n_runs_label,
            pn.pane.HTML(
                '<div style="font-size:11px;color:#8a9bb5;margin-top:6px;">Combination mode:</div>',
                margin=0,
            ),
            mode_sel,
            self._sweep_zip_warning,
            pn.Row(
                label("Max number of cores to use:", 168), parallel_in,
                margin=(4, 0, 0, 0),
                styles={"gap": "4px", "align-items": "center"},
            ),
            run_sweep_btn,
            self._sweep_stop_btn,
            self._sweep_table_col,
            self._sweep_status_ta,
            title="Parameter Sweep",
            header_color="#c8d6e5",
            header_background="#1a2435",
            collapsed=not bool(self._sweep_variations),
            sizing_mode="stretch_width",
            styles={"background": "#1a2435", "border": "1px solid #2d3748",
                    "border-radius": "4px"},
            margin=(0, 0, 4, 0),
        )

    def _run_sweep(self, param_map: dict):
        """Validate, generate sub-YAMLs, and launch the sweep."""
        from simudo.gui.batch import BatchSpec, BatchLauncher, generate_sub_yamls
        from simudo.gui.runner_backends import backend_from_profile
        import threading

        proj = self.app.project
        if not proj.filepath:
            self._append_sweep_log("⚠ Save the project before running a sweep.\n")
            return

        spec = BatchSpec(
            mode=self._sweep_mode,
            max_parallel=self._sweep_max_parallel,
            variations=list(self._sweep_variations),
        )

        if not spec.variations:
            self._append_sweep_log("⚠ Add at least one variation before running.\n")
            return

        n = spec.run_count()
        if n == 0:
            self._append_sweep_log("⚠ All value lists are empty — nothing to run.\n")
            return

        # Zip mode: require equal-length value arrays
        if spec.mode == "zip":
            lengths = [len(v.values) for v in spec.variations]
            if len(set(lengths)) > 1:
                self._append_sweep_log(
                    f"⚠ Zip mode requires equal-length value lists. "
                    f"Got lengths: {lengths}\n"
                )
                return

        try:
            self.app.save_yaml()
        except Exception as exc:
            self._append_sweep_log(f"⚠ Could not save project: {exc}\n")
            return

        runner_host = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "simudo_1d_runner.py"
        )
        if not os.path.exists(runner_host):
            self._append_sweep_log(f"⚠ Runner not found: {runner_host}\n")
            return

        profile = self.app.get_effective_execution_profile()
        try:
            backend = backend_from_profile(profile)
        except Exception as exc:
            self._append_sweep_log(f"⚠ Could not build backend: {exc}\n")
            return

        # Generate sub-YAMLs
        try:
            sub_yaml_paths = generate_sub_yamls(proj.filepath, spec)
        except Exception as exc:
            self._append_sweep_log(f"⚠ Failed to generate sub-YAMLs: {exc}\n")
            return

        self._sweep_log_lines = []
        self._append_sweep_log(
            f"[Sweep] Generated {n} sub-run(s) in "
            f"{os.path.dirname(sub_yaml_paths[0])}/…\n"
        )

        # Build the per-run status table
        self._init_sweep_table(spec, sub_yaml_paths)

        self._sweep_launcher = BatchLauncher()
        self._sweep_stop_btn.visible = True

        lib_dirs = self.app.get_library_dirs()

        def _log_cb(line: str):
            self._append_sweep_log(line)

        def _status_cb(idx: int, status: str):
            self._update_run_status(idx, status)

        # launch_all is non-blocking — starts an internal supervisor thread.
        # The supervisor logs "[Sweep] All runs complete." when done,
        # and _append_sweep_log auto-hides the stop button on that message.
        self._sweep_launcher.launch_all(
            sub_yaml_paths=sub_yaml_paths,
            spec=spec,
            backend=backend,
            runner_host=runner_host,
            lib_dirs=lib_dirs,
            log_cb=_log_cb,
            status_cb=_status_cb,
        )

    def _stop_sweep(self):
        if self._sweep_launcher is not None:
            self._sweep_launcher.stop()
        self._sweep_stop_btn.visible = False
        self._append_sweep_log("[Sweep] Stop requested.\n")

    # ── Per-run status table ───────────────────────────────────────────────────

    _STATUS_STYLES = {
        "queued":  ("⏳ Queued",  "#556070", "#1a2435"),
        "running": ("▶ Running", "#4a9eff", "#0d1a2e"),
        "done":    ("✓ Done",    "#2ecc71", "#0d1e14"),
        "failed":  ("✗ Failed",  "#e05555", "#2a0d0d"),
        "skipped": ("— Skipped", "#556070", "#1a2435"),
    }

    def _row_html(self, idx: int, status: str) -> str:
        label, color, bg = self._STATUS_STYLES.get(
            status, ("? Unknown", "#8a9bb5", "#1a2435"))
        param_str = self._sweep_row_params[idx] if idx < len(self._sweep_row_params) else ""
        subdir = self._sweep_row_dirs[idx] if idx < len(self._sweep_row_dirs) else ""
        # Shorten the path to just the last 2 components for display
        parts = subdir.replace("\\", "/").rstrip("/").split("/")
        short_dir = "/".join(parts[-2:]) if len(parts) >= 2 else subdir
        return (
            f'<div style="display:flex;align-items:center;gap:6px;'
            f'padding:3px 6px;background:{bg};border-radius:3px;'
            f'border:1px solid #2d3748;font-size:12px;">'
            f'<span style="color:#8a9bb5;min-width:28px;text-align:right;">#{idx}</span>'
            f'<span style="color:#c8d6e5;flex:1;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap;" title="{param_str}">{param_str}</span>'
            f'<span style="color:{color};font-weight:600;min-width:80px;'
            f'text-align:center;">{label}</span>'
            f'<span style="color:#556070;font-size:11px;min-width:80px;'
            f'text-align:right;" title="{subdir}">{short_dir}</span>'
            f'</div>'
        )

    def _init_sweep_table(self, spec, sub_yaml_paths: list):
        """Build the status table rows (called after sub-YAMLs are generated)."""
        from simudo.gui.batch import build_combinations, compact_param_label
        combos = build_combinations(spec)

        self._sweep_row_htmls = []
        self._sweep_row_dirs = []
        self._sweep_row_params = []

        rows = []
        for i, (combo, yp) in enumerate(zip(combos, sub_yaml_paths)):
            sub_dir = os.path.dirname(yp)
            param_str = ", ".join(compact_param_label(tgt, val) for tgt, val in combo)
            self._sweep_row_dirs.append(sub_dir)
            self._sweep_row_params.append(param_str)

            html_pane = pn.pane.HTML(
                self._row_html(i, "queued"),
                sizing_mode="stretch_width", margin=0,
            )
            self._sweep_row_htmls.append(html_pane)

            # Navigation button — opens Output panel pointed at this subdir
            nav_btn = pn.widgets.Button(
                name="▶ Open", width=60, height=24, margin=(0, 0, 0, 4),
                stylesheets=["""
                    .bk-btn { background: #1a2e50 !important; color: #4a9eff !important;
                        border: 1px solid #2d4a6e !important; border-radius: 3px !important;
                        font-size: 11px !important; padding: 0 4px !important; }
                    .bk-btn:hover { background: #2d4a6e !important; }
                """],
            )
            _sub_dir = sub_dir  # capture for closure
            def _on_nav(e, d=_sub_dir):
                op = self.app._panels.get("output")
                if op is not None:
                    op.set_folder(d)
                self.app._show_section("output")
            nav_btn.on_click(_on_nav)

            rows.append(pn.Row(html_pane, nav_btn,
                               sizing_mode="stretch_width", margin=(1, 0, 1, 0)))

        self._sweep_table_col.objects = rows

    def _update_run_status(self, idx: int, status: str):
        """Update the status badge for run idx; safe to call from background threads."""
        def _update():
            if idx < len(self._sweep_row_htmls):
                self._sweep_row_htmls[idx].object = self._row_html(idx, status)
        try:
            pn.state.execute(_update)
        except Exception:
            _update()

    def _append_sweep_log(self, text: str):
        """Append text to the sweep log; safe to call from background threads."""
        def _update():
            self._sweep_log_lines.append(text)
            if len(self._sweep_log_lines) > 2000:
                self._sweep_log_lines = self._sweep_log_lines[-2000:]
            self._sweep_status_ta.value = "".join(self._sweep_log_lines)
            # Auto-hide stop button when sweep finishes
            if "All runs complete" in text or "interrupted" in text.lower():
                self._sweep_stop_btn.visible = False
        try:
            pn.state.execute(_update)
        except Exception:
            _update()

    def set_actual_output_dir(self, abs_path: str):
        """Show the directory the runner actually used below the configured Folder field."""
        proj = self.app.project
        try:
            if proj.filepath:
                proj_dir = os.path.dirname(os.path.abspath(proj.filepath))
                rel = os.path.relpath(abs_path, proj_dir)
            else:
                rel = abs_path
        except ValueError:
            rel = abs_path
        self._actual_folder_input.value = f"▶ Last run: {rel}"


