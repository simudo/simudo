"""
Simudo GUI — Output panel.

Sections::

  1. Folder selector  (editable path + open-in-finder + refresh buttons)
  2. J-V plot         (Bokeh interactive; linear / log|J| toggle)
  3. Band diagram     (spatial data via spatial_extractor.py -> dolfin in
                       container)
       - layer schematic linked to band-diagram x-axis (Bokeh shared x_range)
       - band plot  (Ephi_* solid black, qfl_* dashed coloured)
  4. Add-plot dashboards  (user-defined lines from any spatial quantity, each
                           with its own schematic linked via shared x_range)

Design notes::

  - Each Bokeh figure is wrapped in a resizable pn.Column card
    (CSS resize:vertical).  The Bokeh figure inside uses
    sizing_mode="stretch_both" so it fills the card.
  - Voltage-dropdown changes update the band diagram in-place (pane.object
    swap) rather than clearing/appending to a Column, to avoid page-scroll
    resets.
  - Schematic figures share x_range with their companion data figure so that
    pan/zoom on the data figure automatically updates the schematic.
"""

from __future__ import annotations

import glob
import html as _html
import logging
import os
import subprocess
import threading
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import panel as pn

from simudo.gui import runner_backends

from simudo.gui.panels.shared import sec_header, label, INPUT_SS

if TYPE_CHECKING:
    from simudo.gui.app import SimudoApp

import platform
import threading

# ── Shared widget styles ──────────────────────────────────────────────────────

_BROWSE_BTN_SS = ["""
    .bk-btn { background:#1e2a3a !important; color:#7a9cc5 !important;
        border:1px solid #2d3d50 !important; border-radius:3px !important;
        font-size:15px !important; line-height:1 !important;
        cursor:pointer !important; padding:0 4px !important; }
    .bk-btn:hover { background:#2d4a6e !important; color:#c8d6e5 !important; }
"""]

_BTN_SS = ["""
.bk-btn { background: #1a2435 !important; color: #8a9bb5 !important;
    border: 1px solid #2d3748 !important; border-radius: 3px !important;
    font-size: 12px !important; padding: 2px 10px !important; cursor: pointer !important; }
.bk-btn:hover { color: #c8d6e5 !important; border-color: #4a9eff !important; }
"""]

_PATH_SS = ["""
.bk-input {
    background: #0d1520 !important; color: #c8d6e5 !important;
    border: 1px solid #2d3748 !important; border-radius: 3px !important;
    font-size: 11px !important; font-family: monospace !important;
    padding: 2px 6px !important; height: 26px !important;
}
.bk-input:focus { border-color: #4a9eff !important; outline: none !important; }
"""]

_SEL_SS = ["""
.bk-input {
    background: #1a2435 !important; color: #c8d6e5 !important;
    border: 1px solid #2d3748 !important; border-radius: 3px !important;
    font-size: 12px !important;
}
"""]

_PALETTE = ["#4a9eff", "#e07b39", "#27ae60", "#9b59b6", "#e74c3c",
            "#1abc9c", "#f39c12", "#2980b9", "#8e44ad", "#16a085"]

# 10 preset swatches: evenly-spaced hues, similar saturation/brightness,
# chosen to look good on the dark theme and be clearly distinct from each other.
_SWATCH_COLORS = [
    "#52a8ec",  # sky blue
    "#e85d52",  # coral red
    "#4dbb78",  # emerald green
    "#f0922b",  # orange
    "#b56de3",  # violet
    "#2ec9c9",  # teal
    "#e06aaa",  # rose
    "#c5e04a",  # lime
    "#f0cc40",  # golden yellow
    "#9090a8",  # slate
]
_N_SWATCHES = len(_SWATCH_COLORS)
_SWATCH_W = 18   # px per swatch button
_SWATCH_ROW_W = _N_SWATCHES * (_SWATCH_W + 1) - 1  # tight to content

_SWATCH_BTN_CSS = """
.bk-btn {{
    background: {color} !important;
    border: {border} !important;
    border-radius: 3px !important;
    padding: 0 !important;
    min-width: {w}px !important;
    width: {w}px !important;
    height: {w}px !important;
    cursor: pointer !important;
    line-height: 1 !important;
}}
.bk-btn:hover {{ border: 2px solid #ffffff !important; opacity: 0.9; }}
"""


def _fill_swatches(swatch_row: pn.Row, selected: str, on_color):
    """Rebuild the swatch row, highlighting the currently selected colour."""
    swatch_row.clear()
    for hex_c in _SWATCH_COLORS:
        border = "2px solid #ffffff" if hex_c == selected else "1px solid #333333"
        btn = pn.widgets.Button(
            name="", width=_SWATCH_W, height=_SWATCH_W, margin=(0, 1, 0, 0),
            stylesheets=[_SWATCH_BTN_CSS.format(color=hex_c, border=border, w=_SWATCH_W)],
        )
        btn.on_click(lambda e, c=hex_c: on_color(c))
        swatch_row.append(btn)


def _row_lbl(text: str) -> pn.pane.HTML:
    """Label for line-config rows — vertically centered, sized tight to text."""
    # ~7 px per character at 11 px font, plus small padding
    w = len(text) * 7 + 6
    return pn.pane.HTML(
        f'<div style="font-size:11px;color:#8a9bb5;white-space:nowrap;'
        f'height:26px;display:flex;align-items:center;">{text}</div>',
        width=w, height=26, margin=0,
    )


def _native_dir_dialog(initial_dir: str = ".") -> str | None:
    """Native OS directory picker — same architecture as app._native_dir_dialog."""
    initial_dir = os.path.abspath(os.path.expanduser(initial_dir or "."))
    if not os.path.isdir(initial_dir):
        initial_dir = os.path.expanduser("~")

    if platform.system() == "Darwin":
        def _esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            'try\n'
            f'    set f to choose folder with prompt "Select output folder"'
            f' default location POSIX file "{_esc(initial_dir)}"\n'
            '    return POSIX path of f\n'
            'on error\n'
            '    return ""\n'
            'end try'
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            path = result.stdout.strip().rstrip("/")
            return path if path else None
        except Exception as e:
            print(f"osascript dir dialog error: {e}")
            return None

    # Windows / Linux
    result = [None]
    def _run():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", 1)
            path = filedialog.askdirectory(initialdir=initial_dir,
                                           title="Select output folder")
            root.destroy()
            result[0] = path or None
        except Exception as e:
            print(f"tkinter dir dialog error: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=120)
    return result[0]


_log = logging.getLogger('gui.output')


def _bd_label_html(text: str) -> str:
    return (f'<div style="font-size:11px;color:#8a9bb5;white-space:nowrap;'
            f'height:26px;display:flex;align-items:center;">{text}</div>')


def _key_to_display(key: tuple, has_v: bool, has_i: bool) -> str:
    """Format a (v, i) cache key as a display string for selectors."""
    v, i = key
    if has_v and has_i:
        return f"({v:.4g}, {i:.4g})"
    elif has_i:
        return f"{i:.4g}"
    else:
        return f"{v:.4g}"


def _display_to_key(s: str, has_v: bool, has_i: bool) -> tuple:
    """Parse a display string back to a (v, i) tuple."""
    if has_v and has_i:
        s = s.strip("() ")
        parts = s.split(",")
        return (float(parts[0]), float(parts[1]))
    elif has_i:
        return (0.0, float(s))
    else:
        return (float(s), 0.0)


def _no_data_html(msg: str = "No output data available yet.") -> str:
    return (
        f'<div style="background:#1a2435;border:1px solid #2d3748;border-radius:4px;'
        f'padding:24px;text-align:center;color:#556070;font-size:13px;">{msg}</div>'
    )


def _empty_bk_fig():
    """Minimal placeholder Bokeh figure for pre-created panes."""
    import bokeh.plotting as bp
    fig = bp.figure(width=1, height=1, toolbar_location=None)
    fig.toolbar.logo = None
    return fig


def _resizable_card(*contents, height: int = 360) -> pn.Column:
    """Wrap plot content in a fixed-height card.

    NOTE (future work): true user-draggable resize is not yet implemented.
    CSS `resize: vertical` is fought by Bokeh's layout engine, which
    recalculates and re-pins the height on every render cycle.  A working
    approach would require either (a) a ResizeObserver JS shim that notifies
    Bokeh of the new size, or (b) a Panel IntSlider that drives `plot_box.height`
    through Panel/Bokeh's own layout path.  See OUTPUT_PANEL_TODO.md.
    """
    return pn.Column(
        *contents,
        sizing_mode="stretch_width",
        height=height,
    )


# ══════════════════════════════════════════════════════════════════════════════
# _PlotDashboard — one user-added spatial plot
# ══════════════════════════════════════════════════════════════════════════════

class _PlotDashboard:
    """One entry in the 'Add Plot' section."""

    def __init__(self, output_panel: "OutputPanel", idx: int,
                 available_cols: List[str]):
        self._op = output_panel
        self._idx = idx
        self._lines: List[dict] = []
        self._available_cols = available_cols

        self._title_input = pn.widgets.TextInput(
            value=f"Plot {idx + 1}", placeholder="Plot title",
            stylesheets=_PATH_SS, height=26, margin=0, sizing_mode="stretch_width",
        )
        self._title_input.param.watch(lambda e: self._rebuild(), "value")

        self._scale_btn = pn.widgets.RadioButtonGroup(
            options=["Linear", "Log", "Symlog"], value="Linear",
            button_type="default", height=26, margin=0,
        )
        self._scale_btn.param.watch(lambda e: self._rebuild(), "value")

        self._add_line_btn = pn.widgets.Button(
            name="＋ Add Line", stylesheets=_BTN_SS, height=26, margin=0,
        )
        self._add_line_btn.on_click(lambda e: self._add_line())

        self._remove_btn = pn.widgets.Button(
            name="✕ Remove Plot", stylesheets=_BTN_SS, height=26, margin=0,
        )
        self._remove_btn.on_click(lambda e: self._op._remove_dashboard(self))

        # Persistent panes — updated in-place to avoid DOM disruption
        self._fig_msg = pn.pane.HTML(
            _no_data_html("No lines added yet. Click '＋ Add Line'."),
            sizing_mode="stretch_width",
        )
        self._fig_bk = pn.pane.Bokeh(
            _empty_bk_fig(), sizing_mode="stretch_both", visible=False,
        )
        self._fig_card = _resizable_card(self._fig_msg, self._fig_bk, height=340)

        self._lines_col = pn.Column(sizing_mode="stretch_width", margin=0)

        self._view = pn.Column(
            pn.Row(
                self._title_input,
                self._scale_btn,
                self._add_line_btn,
                self._remove_btn,
                margin=0,
                styles={"gap": "6px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            self._lines_col,
            self._fig_card,
            sizing_mode="stretch_width",
            styles={
                "background": "#131e2e", "border": "1px solid #2d3748",
                "border-radius": "4px", "padding": "8px", "gap": "6px",
            },
        )

    def view(self) -> pn.Column:
        return self._view

    def _voltage_options(self) -> List[str]:
        if not self._op._spatial_cache:
            return ["All"]
        has_v = self._op._sweep_has_v
        has_i = self._op._sweep_has_i
        return ["All"] + [_key_to_display(k, has_v, has_i)
                          for k in sorted(self._op._spatial_cache.keys())]

    def _point_label(self) -> str:
        has_v = self._op._sweep_has_v
        has_i = self._op._sweep_has_i
        if has_v and has_i:
            return "(V,I)"
        elif has_i:
            return "I"
        return "V"

    def _update_voltage_options(self, v_opts: List[str]):
        """Refresh point dropdowns on all existing lines (called when cache loads)."""
        for cfg in self._lines:
            sel = cfg.get("_v_sel")
            if sel is None:
                continue
            old = sel.value
            sel.options = v_opts
            if old not in v_opts:
                sel.value = "All"

    def _add_line(self):
        idx = len(self._lines)
        color = _SWATCH_COLORS[idx % len(_SWATCH_COLORS)]
        cfg = {
            "quantity": self._available_cols[0] if self._available_cols else "",
            "color": color,
            "voltage": "All",
            "linestyle": "solid",
            "width": 1.5,
        }
        self._lines.append(cfg)

        qty_sel = pn.widgets.Select(
            options=self._available_cols, value=cfg["quantity"],
            stylesheets=_SEL_SS, height=26, margin=0, width=180,
        )
        v_sel = pn.widgets.Select(
            options=self._voltage_options(), value="All",
            stylesheets=_SEL_SS, height=26, margin=0, width=62,
        )
        cfg["_v_sel"] = v_sel

        # Colour swatches — rebuilt in-place when selection changes
        swatch_row = pn.Row(margin=0, width=_SWATCH_ROW_W)

        def _set_color(new_color, c=cfg, sr=swatch_row):
            c["color"] = new_color
            _fill_swatches(sr, new_color, _set_color)
            self._rebuild()

        _fill_swatches(swatch_row, color, _set_color)

        ls_sel = pn.widgets.Select(
            options=["solid", "dashed", "dotted", "dotdash"],
            value="solid", stylesheets=_SEL_SS, height=26, margin=0, width=76,
        )
        w_sel = pn.widgets.Select(
            options=["0.5", "1.0", "1.5", "2.0", "3.0"],
            value="1.5", stylesheets=_SEL_SS, height=26, margin=0, width=52,
        )
        rm_btn = pn.widgets.Button(name="✕", stylesheets=_BTN_SS,
                                   height=26, width=28, margin=0)

        line_row = pn.Row(
            _row_lbl("Qty"), qty_sel,
            _row_lbl(self._point_label()), v_sel,
            swatch_row,
            _row_lbl("Style"), ls_sel,
            _row_lbl("Linewidth"), w_sel,
            pn.HSpacer(),
            rm_btn,
            margin=0, styles={"gap": "4px", "align-items": "center"},
        )

        def _update(e=None, c=cfg):
            c["quantity"] = qty_sel.value
            c["voltage"] = v_sel.value
            c["linestyle"] = ls_sel.value
            c["width"] = float(w_sel.value)
            self._rebuild()

        qty_sel.param.watch(_update, "value")
        v_sel.param.watch(_update, "value")
        ls_sel.param.watch(_update, "value")
        w_sel.param.watch(_update, "value")

        def _remove(e, row=line_row, c=cfg):
            self._lines.remove(c)
            self._lines_col.remove(row)
            self._rebuild()

        rm_btn.on_click(_remove)
        self._lines_col.append(line_row)
        self._rebuild()

    def _show_msg(self, html: str):
        self._fig_msg.object = html
        self._fig_msg.visible = True
        self._fig_bk.visible = False

    def _show_fig(self, layout):
        self._fig_bk.object = layout
        self._fig_bk.visible = True
        self._fig_msg.visible = False

    def _rebuild(self):
        if not self._lines:
            self._show_msg(_no_data_html("No lines added yet. Click '＋ Add Line'."))
            return

        cache = self._op._spatial_cache
        if not cache:
            self._show_msg(_no_data_html("Spatial data not loaded. Use 'Load Spatial Data'."))
            return

        first_v = next(iter(cache))
        coord_x = cache[first_v].get("coord_x")
        if coord_x is None:
            return

        try:
            layout = _build_custom_with_schematic(
                lines=self._lines,
                cache=cache,
                app=self._op.app,
                scale_mode=self._scale_btn.value,
                title=self._title_input.value or f"Plot {self._idx + 1}",
                output_dir=self._op._spatial_data_dir,
            )
            self._show_fig(layout)
        except Exception as exc:
            self._show_msg(_no_data_html(f"Error: {exc}"))


# ══════════════════════════════════════════════════════════════════════════════
# OutputPanel
# ══════════════════════════════════════════════════════════════════════════════

class OutputPanel:
    def __init__(self, app: "SimudoApp"):
        self.app = app
        # Cache keys are (v, i) tuples. V-only sweeps use i=0; I-only use v=0.
        self._spatial_cache: Dict[tuple, Dict[str, np.ndarray]] = {}
        self._spatial_data_dir: str = ""   # directory that produced the current cache
        self._spatial_loading = False
        self._dashboards: List[_PlotDashboard] = []
        self._sweep_has_v: bool = False
        self._sweep_has_i: bool = False

        # ── Folder row ────────────────────────────────────────────────────────
        self._folder_input = pn.widgets.TextInput(
            placeholder="Path to output folder",
            value="", stylesheets=_PATH_SS,
            sizing_mode="stretch_width", height=28, margin=0,
        )
        self._folder_input.param.watch(lambda e: self._on_folder_changed(), "value")

        self._browse_btn = pn.widgets.Button(
            name="📁  Browse…", width=110, height=28, margin=0,
            stylesheets=_BROWSE_BTN_SS,
        )
        self._browse_btn.on_click(lambda e: self._browse_folder())

        self._refresh_btn = pn.widgets.Button(
            name="↻ Refresh", stylesheets=_BTN_SS, height=28, margin=0,
        )
        self._refresh_btn.on_click(lambda e: self._do_refresh())

        # ── J-V / J-I section ─────────────────────────────────────────────────
        self._jv_scale = pn.widgets.RadioButtonGroup(
            options=["Linear", "Log|J|"], value="Linear",
            button_type="default", height=26, margin=0,
        )
        self._jv_scale.param.watch(lambda e: self._refresh_plot(), "value")

        # Shown only when the directory contains both V= and I= files
        self._ji_mode_btn = pn.widgets.RadioButtonGroup(
            options=["J(V)", "J(I)"], value="J(V)",
            button_type="default", height=26, margin=0, visible=False,
        )
        self._ji_mode_btn.param.watch(lambda e: self._refresh_plot(), "value")

        # Persistent panes — swapped in-place
        self._jv_msg = pn.pane.HTML(_no_data_html(), sizing_mode="stretch_width")
        self._jv_bk  = pn.pane.Bokeh(_empty_bk_fig(), sizing_mode="stretch_both",
                                      visible=False)
        self._jv_card = _resizable_card(self._jv_msg, self._jv_bk, height=360)

        # ── Band diagram section ──────────────────────────────────────────────
        self._bd_label_pane = pn.pane.HTML(
            _bd_label_html("Voltage:"), width=58, height=26, margin=0,
        )
        self._bd_voltage = pn.widgets.Select(
            options=[], value=None, name="",
            stylesheets=_SEL_SS, height=26, margin=0, width=120,
        )
        self._bd_voltage.param.watch(lambda e: self._rebuild_band_diagram(), "value")

        self._load_btn = pn.widgets.Button(
            name="⟳ Load Spatial Data", stylesheets=_BTN_SS, height=26, margin=0,
        )
        self._load_btn.on_click(lambda e: self._start_spatial_load())

        self._load_status = pn.pane.HTML("", sizing_mode="stretch_width")

        # _band_inner is cleared and a fresh pn.pane.Bokeh appended each time
        # spatial data loads, avoiding first-time pane-visibility failures.
        self._band_inner = pn.Column(
            pn.pane.HTML(
                _no_data_html("Load spatial data to view band diagram."),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_both", margin=0,
        )
        self._band_card = _resizable_card(self._band_inner, height=440)

        self._band_col = pn.Column(
            sec_header("BAND DIAGRAM"),
            pn.Row(
                self._bd_label_pane, self._bd_voltage,
                self._load_btn,
                pn.pane.HTML(
                    '<span style="font-size:10px;color:#556070;">'
                    '(requires dolfin — uses Execution settings from Simulation panel)</span>',
                    margin=0,
                ),
                margin=0, styles={"gap": "8px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            self._load_status,
            self._band_card,
            sizing_mode="stretch_width", margin=0, styles={"gap": "6px"},
        )

        # ── Add-plot section ──────────────────────────────────────────────────
        self._add_plot_btn = pn.widgets.Button(
            name="＋ Add Plot", stylesheets=_BTN_SS, height=28, margin=0,
        )
        self._add_plot_btn.on_click(lambda e: self._add_dashboard())
        self._dashboards_col = pn.Column(sizing_mode="stretch_width", margin=0)

        # ── Main layout ───────────────────────────────────────────────────────
        self._view = pn.Column(
            sec_header("OUTPUT"),
            pn.Row(
                self._folder_input, self._browse_btn, self._refresh_btn,
                margin=0,
                styles={"gap": "6px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            pn.Column(
                pn.Row(
                    label("J scale:"), self._jv_scale,
                    self._ji_mode_btn,
                    margin=0, styles={"gap": "6px", "align-items": "center"},
                ),
                self._jv_card,
                sizing_mode="stretch_width", margin=0, styles={"gap": "6px"},
            ),
            self._band_col,
            pn.Column(
                pn.Row(
                    sec_header("SPATIAL PLOTS"),
                    self._add_plot_btn,
                    margin=0, styles={"gap": "10px", "align-items": "center"},
                ),
                self._dashboards_col,
                sizing_mode="stretch_width", margin=0, styles={"gap": "8px"},
            ),
            sizing_mode="stretch_width",
            styles={"padding": "12px", "gap": "12px"},
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def view(self) -> pn.Column:
        self._seed_folder()
        self._refresh_plot()
        return self._view

    def _show_error_card(self, title: str, body: str):
        """Insert a dismissible error card at the top of the output panel."""
        close_btn = pn.widgets.Button(
            name="✕ Close",
            stylesheets=["""
                .bk-btn { background:#2d1a1a; color:#e74c3c; border:1px solid #6b2020;
                          border-radius:4px; padding:3px 14px; cursor:pointer;
                          font-size:12px; }
                .bk-btn:hover { background:#4a2020; }
            """],
        )
        card = pn.Column(
            pn.pane.HTML(
                f'<span style="font-size:13px;font-weight:700;color:#e74c3c;">'
                f'{title}</span>'
            ),
            pn.pane.HTML(
                f'<pre style="background:#0d0808;color:#ff9999;font-family:monospace;'
                f'font-size:11px;white-space:pre-wrap;word-break:break-all;'
                f'overflow-y:auto;max-height:280px;padding:6px;margin:0;'
                f'user-select:text;-webkit-user-select:text;cursor:text;border:none;">'
                f'{_html.escape(body)}</pre>',
                sizing_mode="stretch_width",
            ),
            pn.Row(pn.Spacer(), close_btn, margin=0),
            sizing_mode="stretch_width",
            margin=(4, 0),
            stylesheets=["""
                :host { background:#1a0808; border:1px solid #6b2020;
                        border-radius:6px; padding:10px; }
            """],
        )
        # Insert after _load_status (index 2) in the band diagram column,
        # so the error appears where the band diagram would otherwise be.
        self._band_col.insert(3, card)

        def _close(e, c=card):
            try:
                self._band_col.remove(c)
            except Exception:
                pass

        close_btn.on_click(_close)

    def set_folder(self, path: str):
        self._folder_input.value = path
        self._refresh_plot()
        self._spatial_cache = {}
        self._update_band_diagram_state()

    def refresh(self):
        last = getattr(self.app, "_last_run_output_dir", None)
        if last:
            self.set_folder(last)
        else:
            self._refresh_plot()

    # ── Folder helpers ─────────────────────────────────────────────────────────

    def _seed_folder(self):
        if self._folder_input.value:
            return
        last = getattr(self.app, "_last_run_output_dir", None)
        if last:
            self._folder_input.value = last
            return
        d = self._default_output_dir()
        if d:
            self._folder_input.value = d

    def _default_output_dir(self) -> Optional[str]:
        proj = self.app.project
        if not proj.filepath:
            return None
        return os.path.join(os.path.dirname(proj.filepath),
                            proj.output_folder or "out")

    def _current_search_dir(self) -> Optional[str]:
        # Prefer value_input (live keystroke) so Refresh sees the typed path
        # even before the user presses Enter.
        live = getattr(self._folder_input, 'value_input', None)
        v = ((live if live else self._folder_input.value) or "").strip()
        return v if v else self._default_output_dir()

    def _browse_folder(self):
        """Open a native directory picker (synchronous, same pattern as Save As)."""
        initial = self._current_search_dir() or os.path.expanduser("~")
        folder = _native_dir_dialog(initial_dir=initial)
        if folder:
            self._folder_input.value = folder
            # param.watch fires _on_folder_changed; call _refresh_plot too in
            # case value was already equal (no watch fires on equal assignment).
            self._refresh_plot()

    def _do_refresh(self):
        """Commit any live-typed folder path then refresh the plot."""
        live = getattr(self._folder_input, 'value_input', None)
        committed = (self._folder_input.value or "").strip()
        if live and live.strip() and live.strip() != committed:
            # Setting value fires the param watch → _on_folder_changed → _refresh_plot
            self._folder_input.value = live.strip()
        else:
            self._refresh_plot()

    def _on_folder_changed(self):
        self._spatial_cache = {}
        self._update_band_diagram_state()
        self._refresh_plot()

    # ── J-V / J-I plot ────────────────────────────────────────────────────────

    def _set_jv_msg(self, html: str):
        self._jv_msg.object = html
        self._jv_msg.visible = True
        self._jv_bk.visible = False

    def _set_jv_fig(self, fig):
        self._jv_bk.object = fig
        self._jv_bk.visible = True
        self._jv_msg.visible = False

    def _scan_sweep_types(self, folder: Optional[str]) -> tuple:
        """Return (has_v, has_i) based on xdmf files present."""
        if not folder or not os.path.isdir(folder):
            return False, False
        has_v = bool(glob.glob(os.path.join(folder, "sim_V=*_full.xdmf")))
        has_i = bool(glob.glob(os.path.join(folder, "sim_I=*_full.xdmf")))
        return has_v, has_i

    def _find_sweep_csv(self):
        """Find sim_V.csv in search dir or immediate subdirs."""
        d = self._current_search_dir()
        if not d or not os.path.isdir(d):
            return None, None
        candidates = []
        search_dirs = [d] + [
            os.path.join(d, e) for e in sorted(os.listdir(d))
            if os.path.isdir(os.path.join(d, e))
        ]
        for sd in search_dirs:
            csv = os.path.join(sd, "sim_V.csv")
            if os.path.isfile(csv):
                candidates.append((os.path.getmtime(csv), csv, sd))
        if not candidates:
            return None, None
        _, csv_path, exp_dir = max(candidates)
        return csv_path, exp_dir

    def _refresh_plot(self):
        d = self._current_search_dir()
        has_v, has_i = self._scan_sweep_types(d)
        self._sweep_has_v = has_v
        self._sweep_has_i = has_i

        # Show V/I mode toggle only when both types exist
        self._ji_mode_btn.visible = has_v and has_i

        show_ji = has_i and (not has_v or self._ji_mode_btn.value == "J(I)")

        if show_ji:
            self._refresh_ji_plot(d)
        else:
            self._refresh_jv_plot(d)

        exp_dir = d  # for xdmf point scanning
        # If we found a V csv, use its directory for xdmf scanning
        if has_v:
            csv_path, csv_dir = self._find_sweep_csv()
            if csv_dir:
                exp_dir = csv_dir
        self._refresh_point_dropdown(exp_dir)

    def _refresh_jv_plot(self, d: Optional[str]):
        csv_path, exp_dir = self._find_sweep_csv()
        if csv_path is None:
            if d and os.path.isdir(d):
                msg = (f'Folder <code>{d}</code> found but no '
                       f'<code>sim_V.csv</code> located.')
            elif d:
                msg = f'Folder <code>{d}</code> does not exist.'
            else:
                msg = "No output data yet. Run a simulation first."
            self._set_jv_msg(_no_data_html(msg))
            return
        try:
            voltages, currents, units = _parse_sweep_csv(csv_path)
        except Exception as e:
            self._set_jv_msg(_no_data_html(
                f'Error reading <code>{os.path.basename(csv_path)}</code>: {e}'))
            return
        if len(voltages) < 2:
            self._set_jv_msg(_no_data_html(
                f'<code>{os.path.basename(csv_path)}</code>: fewer than 2 data points.'))
            return
        try:
            fig = _build_jv_figure(
                voltages, currents, units,
                log_mode=(self._jv_scale.value == "Log|J|"),
                title=f"J-V — {exp_dir}",
            )
            self._set_jv_fig(fig)
        except Exception as exc:
            self._set_jv_msg(_no_data_html(f"Plot error: {exc}"))

    def _refresh_ji_plot(self, d: Optional[str]):
        if not self._spatial_cache:
            self._set_jv_msg(_no_data_html(
                "Load spatial data to view J(I) plot."))
            return
        try:
            intensities, currents = _extract_ji_from_cache(self._spatial_cache)
        except Exception as exc:
            self._set_jv_msg(_no_data_html(f"J(I) error: {exc}"))
            return
        if len(intensities) < 1:
            self._set_jv_msg(_no_data_html("No j_tot data found in spatial cache."))
            return
        try:
            fig = _build_ji_figure(intensities, currents, title=f"J(I) — {d}")
            self._set_jv_fig(fig)
        except Exception as exc:
            self._set_jv_msg(_no_data_html(f"Plot error: {exc}"))

    # ── Point (V or I) dropdown ───────────────────────────────────────────────

    def _find_xdmf_points(self, folder: Optional[str]) -> List[tuple]:
        """Return sorted list of (v, i) tuples from xdmf filenames."""
        if not folder or not os.path.isdir(folder):
            return []
        # NOTE: this must label points exactly as gui/spatial_extractor.py does,
        # or a selection made here will not match a key in the extracted cache.
        # The voltage sweep runs after the intensity ramp, at the intensity the
        # ramp finished on -- so sweep points are not dark unless the run was.
        def _vals(pattern, prefix):
            out = []
            for m in glob.glob(os.path.join(folder, pattern)):
                val_str = os.path.basename(m)[len(prefix):-len("_full.xdmf")]
                try:
                    out.append(float(val_str))
                except ValueError:
                    pass
            return out

        ramp = _vals("sim_I=*_full.xdmf", "sim_I=")
        sweep_intensity = max(ramp) if ramp else 0.0

        points = [(0.0, i) for i in ramp]
        for v in _vals("sim_V=*_full.xdmf", "sim_V="):
            pt = (v, sweep_intensity)
            if pt not in points:      # end of ramp == start of sweep
                points.append(pt)
        return sorted(points)

    def _refresh_point_dropdown(self, exp_dir: Optional[str]):
        points = self._find_xdmf_points(exp_dir)
        has_v = any(v != 0.0 for v, i in points)
        has_i = any(i != 0.0 for v, i in points)

        if has_v and has_i:
            label_text = "(V, I):"
        elif has_i:
            label_text = "Intensity:"
        else:
            label_text = "Voltage:"
        self._bd_label_pane.object = _bd_label_html(label_text)

        if not points:
            self._bd_voltage.options = []
            self._bd_voltage.value = None
            return

        options = [_key_to_display(p, has_v, has_i) for p in points]
        # Default: first point near zero voltage (or just first)
        default = options[0]
        if has_v and not has_i:
            for (v, i_val), opt in zip(points, options):
                if abs(v) < 1e-9:
                    default = opt
                    break
        self._bd_voltage.options = options
        if self._bd_voltage.value not in options:
            self._bd_voltage.value = default

    # ── Band diagram state ─────────────────────────────────────────────────────

    def _update_band_diagram_state(self):
        if self._spatial_cache:
            self._rebuild_band_diagram()
        else:
            self._band_inner.clear()
            self._band_inner.append(pn.pane.HTML(
                _no_data_html("Load spatial data to view band diagram."),
                sizing_mode="stretch_width",
            ))
            self._load_status.object = ""

    # ── Spatial data loading ───────────────────────────────────────────────────

    def _start_spatial_load(self):
        if self._spatial_loading:
            return
        folder = self._current_search_dir()
        if not folder or not os.path.isdir(folder):
            self._load_status.object = (
                '<span style="color:#e74c3c;">No valid output folder selected.</span>')
            return
        self._spatial_loading = True
        self._load_status.object = (
            '<span style="color:#f39c12;">'
            '⟳ Extracting spatial data — this may take a moment…</span>')
        self._load_btn.disabled = True
        threading.Thread(
            target=self._run_extractor_thread, args=(folder,), daemon=True,
        ).start()

    def _run_extractor_thread(self, output_dir: str):
        try:
            cache, error = _extract_spatial_data(output_dir, self.app)
        except Exception as exc:
            cache, error = {}, str(exc)

        def _done(c=cache, err=error, od=output_dir):
            self._spatial_loading = False
            self._load_btn.disabled = False
            if err:
                first_line = err.split("\n")[0]
                self._load_status.object = (
                    f'<span style="color:#e74c3c;">'
                    f'Extraction error: {first_line}'
                    f'{"  (details ↓)" if chr(10) in err else ""}'
                    f'</span>'
                )
                if "\n" in err:
                    self._show_error_card("Extraction error", err)
            else:
                self._spatial_cache = c
                self._spatial_data_dir = od
                n = len(c)
                cols = len(next(iter(c.values()))) - 1 if c else 0
                self._load_status.object = (
                    f'<span style="color:#27ae60;">✓ Loaded {n} points, '
                    f'{cols} fields.</span>')
                try:
                    self._rebuild_band_diagram()
                except Exception:
                    pass
                self._refresh_dashboards()
                # Determine sweep type from the loaded cache keys directly,
                # rather than relying on _sweep_has_i/_sweep_has_v which may
                # not have been set yet if the folder wasn't committed via Enter.
                cache_has_i = any(k[1] != 0.0 for k in c)
                cache_has_v = any(k[0] != 0.0 for k in c)
                if cache_has_i and (not cache_has_v or self._ji_mode_btn.value == "J(I)"):
                    self._refresh_ji_plot(od)

        pn.state.execute(_done)

    # ── Band diagram ───────────────────────────────────────────────────────────

    def _rebuild_band_diagram(self):
        sel_str = self._bd_voltage.value
        if not sel_str or not self._spatial_cache:
            return

        has_v = self._sweep_has_v
        has_i = self._sweep_has_i
        try:
            key = _display_to_key(sel_str, has_v, has_i)
        except (ValueError, IndexError):
            return

        # Direct lookup, falling back to the nearest key only when it is close
        # enough to be float-formatting noise.  A loose fallback silently drew a
        # completely different bias point when the selector and the extracted
        # data disagreed about a point's label.
        if key not in self._spatial_cache:
            nearest = min(self._spatial_cache.keys(),
                          key=lambda k: (k[0] - key[0]) ** 2 + (k[1] - key[1]) ** 2)
            if (nearest[0] - key[0]) ** 2 + (nearest[1] - key[1]) ** 2 > 1e-6:
                _log.warning(
                    "No spatial data for point %s; nearest is %s. Not plotting.",
                    key, nearest)
                return
            key = nearest
        data = self._spatial_cache[key]
        coord_x = data.get("coord_x")
        if coord_x is None:
            return

        v_label = key[0]
        try:
            layout = _build_band_and_schematic(
                data, coord_x, v_label, self.app, self._spatial_data_dir)
            bk_pane = pn.pane.Bokeh(layout, sizing_mode="stretch_both")
            self._band_inner.clear()
            self._band_inner.append(bk_pane)
        except Exception as exc:
            self._band_inner.clear()
            self._band_inner.append(pn.pane.HTML(
                _no_data_html(f"Band diagram error: {exc}"),
                sizing_mode="stretch_width",
            ))

    # ── Add-plot dashboards ────────────────────────────────────────────────────

    def _add_dashboard(self):
        cols = _non_coord_columns(self._spatial_cache)
        dash = _PlotDashboard(self, len(self._dashboards), cols)
        self._dashboards.append(dash)
        self._dashboards_col.append(dash.view())
        dash._add_line()  # start with one line already present

    def _remove_dashboard(self, dash: "_PlotDashboard"):
        if dash in self._dashboards:
            self._dashboards.remove(dash)
            self._dashboards_col.remove(dash.view())

    def _refresh_dashboards(self):
        cols = _non_coord_columns(self._spatial_cache)
        has_v = self._sweep_has_v
        has_i = self._sweep_has_i
        v_opts = (["All"] + [_key_to_display(k, has_v, has_i)
                              for k in sorted(self._spatial_cache.keys())]
                  if self._spatial_cache else ["All"])
        for dash in self._dashboards:
            dash._available_cols = cols
            dash._update_voltage_options(v_opts)
            dash._rebuild()


# ══════════════════════════════════════════════════════════════════════════════
# Bokeh figure builders
# ══════════════════════════════════════════════════════════════════════════════

def _style_figure(fig):
    fig.background_fill_color = "#0d1520"
    fig.border_fill_color = "#131e2e"
    fig.grid.grid_line_color = "#1e2d3d"
    fig.grid.grid_line_dash = "dashed"
    fig.axis.axis_line_color = "#2d3748"
    fig.axis.major_tick_line_color = "#2d3748"
    fig.axis.minor_tick_line_color = "#2d3748"
    fig.axis.major_label_text_color = "#8a9bb5"
    fig.axis.axis_label_text_color = "#8a9bb5"
    fig.title.text_color = "#c8d6e5"
    fig.title.text_font_size = "12px"
    fig.toolbar.logo = None


def _ls_to_dash(ls: str):
    return {"solid": "solid", "dashed": "dashed",
            "dotted": "dotted", "dotdash": "dotdash"}.get(ls, "solid")


def _build_jv_figure(voltages, currents, units, log_mode=False, title="J-V"):
    import bokeh.plotting as bp
    from bokeh.models import ColumnDataSource

    vs = np.array(voltages, dtype=float)
    js = np.array(currents, dtype=float)
    ylabel = f"|J| ({units})" if log_mode else f"J ({units})"

    if log_mode:
        ys = np.where(js != 0, np.abs(js), np.nan)
        fig = bp.figure(
            title=title, x_axis_label="Voltage (V)", y_axis_label=ylabel,
            y_axis_type="log",
            tools="pan,box_zoom,wheel_zoom,reset,save", toolbar_location="right",
            sizing_mode="stretch_both",
        )
    else:
        ys = js
        fig = bp.figure(
            title=title, x_axis_label="Voltage (V)", y_axis_label=ylabel,
            tools="pan,box_zoom,wheel_zoom,reset,save", toolbar_location="right",
            sizing_mode="stretch_both",
        )

    _style_figure(fig)
    src = ColumnDataSource({"v": vs, "j": ys})
    fig.line("v", "j", source=src, color="#4a9eff", line_width=2)
    fig.scatter("v", "j", source=src, color="#4a9eff", size=5)
    return fig


def _build_schematic_figure(app: "SimudoApp", coord_x: np.ndarray,
                             x_range=None, output_dir: str = ""):
    """Thin coloured-block layer schematic.

    If *x_range* is supplied (a Bokeh Range1d from another figure), the
    schematic shares that range so pan/zoom stays in sync.

    If *output_dir* contains a project YAML (e.g. a batch sub-run directory),
    that YAML's layer thicknesses are used instead of app.project.layers so
    the schematic reflects the actual geometry of the displayed run.
    """
    import glob
    import bokeh.plotting as bp
    from bokeh.models import ColumnDataSource, LabelSet

    layers = app.project.layers
    if output_dir and os.path.isdir(output_dir):
        yamls = glob.glob(os.path.join(output_dir, "*.yaml"))
        if yamls:
            try:
                from simudo.gui.yaml_io import load_project as _lp
                sub_proj = _lp(yamls[0])
                layers = sub_proj.layers
            except Exception:
                pass  # fall back to app.project.layers
    x_min = float(coord_x.min())
    x_max = float(coord_x.max())

    thicknesses = []
    for lyr in layers:
        try:
            t = float(lyr.thickness)
        except Exception:
            t = 0.0
        thicknesses.append(t)

    total = sum(thicknesses) or 1.0
    scale = (x_max - x_min) / total

    lefts, rights, colors, names = [], [], [], []
    x = x_min
    for lyr, t in zip(layers, thicknesses):
        st = t * scale
        lefts.append(x)
        rights.append(x + st)
        colors.append(lyr.color)
        names.append(lyr.name)
        x += st

    kwargs = dict(
        x_range=x_range if x_range is not None else (x_min, x_max),
        y_range=(-0.5, 0.5),
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location=None,
        sizing_mode="stretch_width",
        height=55,
    )
    fig = bp.figure(**kwargs)
    _style_figure(fig)
    fig.yaxis.visible = False
    fig.xaxis.visible = False
    fig.grid.visible = False
    fig.outline_line_color = None
    fig.min_border_top = 2
    fig.min_border_bottom = 2
    fig.min_border_left = 4
    fig.min_border_right = 4

    src = ColumnDataSource({
        "left": lefts, "right": rights, "color": colors, "name": names,
        "top": [0.4] * len(lefts), "bottom": [-0.4] * len(lefts),
    })
    fig.quad(left="left", right="right", top="top", bottom="bottom",
             fill_color="color", line_color="#0d1520", line_width=1,
             source=src, alpha=0.85)
    labels = LabelSet(x="left", y=0, text="name",
                      text_font_size="10px", text_color="#ffffff",
                      x_offset=4, y_offset=-5, source=src)
    fig.add_layout(labels)
    return fig


def _build_band_figure(data: dict, coord_x: np.ndarray, voltage: float):
    import bokeh.plotting as bp

    fig = bp.figure(
        title=f"Band diagram  V = {voltage:.4g} V",
        x_axis_label="x (µm)", y_axis_label="Energy (eV)",
        tools="pan,box_zoom,wheel_zoom,reset,save", toolbar_location="right",
        sizing_mode="stretch_both",
    )
    _style_figure(fig)

    ephi_keys = sorted(k for k in data if k.startswith("Ephi_"))
    qfl_keys  = sorted(k for k in data if k.startswith("qfl_"))

    for key in ephi_keys:
        fig.line(coord_x, data[key], color="#c8d6e5", line_width=1.8,
                 legend_label=key)
    for i, key in enumerate(qfl_keys):
        fig.line(coord_x, data[key], color=_PALETTE[i % len(_PALETTE)],
                 line_width=1.5, line_dash="dashed", legend_label=key)

    if ephi_keys or qfl_keys:
        fig.legend.background_fill_color = "#131e2e"
        fig.legend.label_text_color = "#c8d6e5"
        fig.legend.border_line_color = "#2d3748"
        fig.legend.click_policy = "hide"
        fig.legend.location = "top_right"

    return fig


def _build_band_and_schematic(data: dict, coord_x: np.ndarray,
                               voltage: float, app: "SimudoApp",
                               output_dir: str = ""):
    """Return a single Bokeh column layout: schematic (top) + band diagram (bottom).

    The schematic shares x_range with the band diagram so that pan/zoom on
    the band diagram automatically updates the schematic.
    """
    from bokeh.layouts import column as bk_column

    band_fig = _build_band_figure(data, coord_x, voltage)
    schematic_fig = _build_schematic_figure(app, coord_x,
                                             x_range=band_fig.x_range,
                                             output_dir=output_dir)
    return bk_column([schematic_fig, band_fig], sizing_mode="stretch_both")


def _symlog(y: np.ndarray, linthresh: float) -> np.ndarray:
    """Symmetric-log transform: sign(y) * log10(1 + |y| / linthresh)."""
    return np.sign(y) * np.log10(1.0 + np.abs(y) / linthresh)


def _symlog_linthresh(y_all: np.ndarray) -> float:
    """Auto-detect linear threshold from the 10th percentile of |nonzero| values."""
    flat = y_all[np.isfinite(y_all)]
    nonzero = np.abs(flat[flat != 0])
    if len(nonzero) == 0:
        return 1.0
    return float(np.percentile(nonzero, 10))


def _setup_symlog_yaxis(fig, y_all: np.ndarray, linthresh: float):
    """Add FixedTicker + JS formatter to show original values on a symlog y-axis."""
    from bokeh.models import FixedTicker, CustomJSTickFormatter

    finite = y_all[np.isfinite(y_all)]
    y_max = float(np.max(np.abs(finite))) if len(finite) else 1.0

    # Collect candidate tick values in original space
    mag_hi = int(np.ceil(np.log10(max(y_max, linthresh)))) + 1
    mag_lo = int(np.floor(np.log10(linthresh))) - 1
    orig_vals = [0.0]
    for exp in range(mag_lo, mag_hi + 1):
        for sign in (1, -1):
            v = sign * (10.0 ** exp)
            if np.abs(v) <= y_max * 1.5:
                orig_vals.append(v)
    orig_vals = sorted(set(orig_vals))

    tick_positions = [float(_symlog(np.array([v]), linthresh)[0]) for v in orig_vals]

    fig.yaxis.ticker = FixedTicker(ticks=tick_positions)
    fig.yaxis.formatter = CustomJSTickFormatter(
        args={"linthresh": linthresh},
        code="""
        // tick is in transformed (symlog) space; recover original value
        var absT = Math.abs(tick);
        var orig = Math.sign(tick) * linthresh * (Math.pow(10, absT) - 1);
        if (Math.abs(tick) < 1e-9) return "0";
        var absO = Math.abs(orig);
        if (absO === 0) return "0";
        var exp = Math.floor(Math.log10(absO));
        var m = orig / Math.pow(10, exp);
        var mStr = (Math.abs(m - Math.round(m)) < 0.05)
                   ? String(Math.round(m))
                   : m.toFixed(1);
        if (exp === 0) return mStr;
        if (exp === 1) return String(Math.round(orig));
        return mStr + "e" + exp;
        """,
    )


def _resolve_line_voltages(cfg: dict, cache: dict) -> List[tuple]:
    """Return [(cache_key, color), ...] for one line config.

    When voltage=="All", each point gets an auto-colour from _SWATCH_COLORS.
    When a specific point is chosen, one entry with the user's colour is returned.
    Cache keys are (v, i) tuples.
    """
    v_target = cfg.get("voltage", "All")
    if v_target == "All":
        return [
            (key, _SWATCH_COLORS[i % len(_SWATCH_COLORS)])
            for i, key in enumerate(sorted(cache.keys()))
        ]
    # Detect sweep type from cache keys
    has_v = any(k[0] != 0.0 for k in cache)
    has_i = any(k[1] != 0.0 for k in cache)
    try:
        target_key = _display_to_key(v_target, has_v, has_i)
    except (ValueError, IndexError):
        return []
    # Direct lookup; fall back to nearest
    if target_key in cache:
        return [(target_key, cfg["color"])]
    nearest = min(cache.keys(),
                  key=lambda k: (k[0] - target_key[0]) ** 2 + (k[1] - target_key[1]) ** 2)
    return [(nearest, cfg["color"])]


def _build_custom_with_schematic(lines: list, cache: dict,
                                  app: "SimudoApp",
                                  scale_mode: str, title: str,
                                  output_dir: str = ""):
    """Return a Bokeh column: schematic (shared x_range) + custom data figure.

    scale_mode: "Linear" | "Log" | "Symlog".
    Each line cfg has a "voltage" field: "All" (auto-coloured per voltage) or a
    specific voltage string (use cfg["color"]).
    """
    import bokeh.plotting as bp
    from bokeh.layouts import column as bk_column

    first_v = next(iter(cache))
    coord_x = cache[first_v].get("coord_x")
    log_mode = (scale_mode == "Log")
    symlog_mode = (scale_mode == "Symlog")

    if log_mode:
        data_fig = bp.figure(
            title=title,
            x_axis_label="x (µm)", y_axis_label="value",
            y_axis_type="log",
            tools="pan,box_zoom,wheel_zoom,reset,save", toolbar_location="right",
            sizing_mode="stretch_both",
        )
    else:
        data_fig = bp.figure(
            title=title,
            x_axis_label="x (µm)", y_axis_label="value",
            tools="pan,box_zoom,wheel_zoom,reset,save", toolbar_location="right",
            sizing_mode="stretch_both",
        )
    _style_figure(data_fig)

    # For symlog: collect all y values first to auto-detect linthresh
    all_ys_symlog: list = []
    if symlog_mode:
        for cfg in lines:
            col = cfg["quantity"]
            for v_key, _ in _resolve_line_voltages(cfg, cache):
                d = cache.get(v_key, {})
                if col in d:
                    all_ys_symlog.append(np.asarray(d[col], dtype=float))
        linthresh = _symlog_linthresh(np.concatenate(all_ys_symlog)) if all_ys_symlog else 1.0
    else:
        linthresh = 1.0

    drawn = False
    for cfg in lines:
        col = cfg["quantity"]
        ls = _ls_to_dash(cfg["linestyle"])
        lw = cfg["width"]

        for v_key, color in _resolve_line_voltages(cfg, cache):
            d = cache.get(v_key, {})
            xs = d.get("coord_x")
            if xs is None or col not in d:
                continue
            ys = np.asarray(d[col], dtype=float)

            if log_mode:
                ys_plot = np.where(ys > 0, ys, np.nan)
                if not np.any(np.isfinite(ys_plot)):
                    continue
            elif symlog_mode:
                ys_plot = _symlog(ys, linthresh)
            else:
                ys_plot = ys

            v_key, i_key = v_key
            if i_key == 0.0:
                pt_label = f"V={v_key:.2g}"
            elif v_key == 0.0:
                pt_label = f"I={i_key:.2g}"
            else:
                pt_label = f"V={v_key:.2g},I={i_key:.2g}"
            data_fig.line(xs, ys_plot, color=color, line_width=lw, line_dash=ls,
                          legend_label=f"{col} {pt_label}")
            drawn = True

    if symlog_mode and all_ys_symlog:
        _setup_symlog_yaxis(data_fig, np.concatenate(all_ys_symlog), linthresh)

    if drawn and len(data_fig.legend) > 0:
        data_fig.legend.background_fill_color = "#131e2e"
        data_fig.legend.label_text_color = "#c8d6e5"
        data_fig.legend.border_line_color = "#2d3748"
        data_fig.legend.click_policy = "hide"

    schematic_fig = _build_schematic_figure(app, coord_x,
                                             x_range=data_fig.x_range,
                                             output_dir=output_dir)
    return bk_column([schematic_fig, data_fig], sizing_mode="stretch_both")


# ══════════════════════════════════════════════════════════════════════════════
# Spatial data extraction (background thread)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_spatial_data(output_dir: str, app: "SimudoApp"):
    """Launch spatial_extractor.py and parse the CSV stream.

    Returns (cache_dict, error_str).  error_str is "" on success.
    On failure, error_str includes the extractor's stderr so the caller can
    show a useful traceback to the user.
    """
    extractor_path = os.path.join(_GUI_DIR, "spatial_extractor.py")
    if not os.path.exists(extractor_path):
        return {}, f"spatial_extractor.py not found at {extractor_path}"

    local_proj_dir = None
    if app.project and app.project.filepath:
        local_proj_dir = os.path.dirname(os.path.abspath(app.project.filepath))

    try:
        profile = app.get_effective_execution_profile()
        backend = runner_backends.backend_from_profile(profile)
        handle = backend.launch_extractor(
            extractor_path, output_dir,
            local_proj_dir=local_proj_dir,
        )
    except Exception as exc:
        return {}, f"Could not launch extractor: {exc}"

    cache: Dict[tuple, Dict[str, np.ndarray]] = {}
    error = ""

    try:
        current_key: Optional[tuple] = None  # (v, i)
        headers: List[str] = []
        rows: List[List[float]] = []

        for raw_line in handle.stdout_lines():
            line = raw_line.rstrip("\n")
            if line.startswith("# BEGIN_POINT"):
                # Format: # BEGIN_POINT V=<v> I=<i>
                parts = line.split()
                try:
                    v_part = next(p for p in parts if p.startswith("V="))
                    i_part = next(p for p in parts if p.startswith("I="))
                    current_key = (float(v_part[2:]), float(i_part[2:]))
                except (StopIteration, ValueError):
                    current_key = None
                headers = []
                rows = []
            elif line.startswith("# END_POINT"):
                if current_key is not None and headers and rows:
                    arr = np.array(rows, dtype=float)
                    cache[current_key] = {col: arr[:, i]
                                          for i, col in enumerate(headers)}
                current_key = None
                headers = []
                rows = []
            elif current_key is not None:
                if not headers:
                    headers = line.split(",")
                else:
                    try:
                        rows.append([float(x) for x in line.split(",")])
                    except ValueError:
                        pass

        if handle.returncode not in (None, 0):
            stderr = handle.stderr_text().strip()
            error = f"Extractor exited with code {handle.returncode}"
            if stderr:
                error += f"\n\n{stderr}"
    except Exception as exc:
        error = str(exc)

    return cache, error


# ══════════════════════════════════════════════════════════════════════════════
# J-V CSV parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_sweep_csv(csv_path: str):
    with open(csv_path, newline="") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("Empty file")

    headers = [h.strip() for h in lines[0].split(",")]
    units_row: list = []
    data_start = 1
    if len(lines) > 1 and lines[1].lstrip().startswith("#"):
        units_row = [u.strip().lstrip("#").strip() for u in lines[1].split(",")]
        data_start = 2

    def _unit(i):
        return units_row[i].strip() if i < len(units_row) else ""

    v_idx = next((i for i, h in enumerate(headers) if h == "sweep_parameter:V"), None)
    if v_idx is None:
        raise ValueError("No 'sweep_parameter:V' column")

    left_idxs = [i for i, h in enumerate(headers) if h.endswith(":left_contact")
                 and h.startswith("avg:current_")]
    right_idxs = [i for i, h in enumerate(headers) if h.endswith(":right_contact")
                  and h.startswith("avg:current_")]
    contact_idxs = left_idxs or right_idxs
    if not contact_idxs:
        raise ValueError("No avg:current_* contact columns")

    current_unit = _unit(contact_idxs[0]) or "mA/cm²"
    voltages, currents = [], []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        try:
            v = float(parts[v_idx])
            j = sum(float(parts[i]) for i in contact_idxs if i < len(parts))
            voltages.append(v)
            currents.append(j)
        except (ValueError, IndexError):
            continue

    return voltages, currents, current_unit


def _extract_ji_from_cache(cache: dict):
    """Extract (intensities, currents) from spatial cache for J(I) plot.

    Cache keys are (v, i) tuples. j_tot is taken at the last spatial point
    (right contact). Returns lists sorted by intensity.
    """
    points = []
    for (v, i_val), data in cache.items():
        j_arr = data.get("j_tot")
        if j_arr is not None and len(j_arr) > 0:
            points.append((i_val, float(j_arr[-1])))
    points.sort(key=lambda x: x[0])
    intensities = [p[0] for p in points]
    currents = [p[1] for p in points]
    return intensities, currents


def _build_ji_figure(intensities, currents, title="J(I)"):
    import bokeh.plotting as bp
    from bokeh.models import ColumnDataSource

    xs = np.array(intensities, dtype=float)
    ys = np.abs(np.array(currents, dtype=float))

    # Filter zeros/negatives for log axes
    mask = (xs > 0) & (ys > 0)
    xs = xs[mask]
    ys = ys[mask]

    fig = bp.figure(
        title=title,
        x_axis_label="Intensity (suns)",
        y_axis_label="|J| (mA/cm²)",
        x_axis_type="log", y_axis_type="log",
        tools="pan,box_zoom,wheel_zoom,reset,save", toolbar_location="right",
        sizing_mode="stretch_both",
    )
    _style_figure(fig)
    src = ColumnDataSource({"x": xs, "y": ys})
    fig.line("x", "y", source=src, color="#4a9eff", line_width=2)
    fig.scatter("x", "y", source=src, color="#4a9eff", size=5)
    return fig


def _non_coord_columns(cache: dict) -> List[str]:
    if not cache:
        return []
    return [k for k in next(iter(cache.values())) if not k.startswith("coord_")]
