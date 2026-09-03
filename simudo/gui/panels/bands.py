"""
Simudo GUI — Bands panel.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List  # noqa: F401

import panel as pn

from simudo.gui.model import Band, BAND_TYPE_GUI_TO_YAML, BAND_TYPE_YAML_TO_GUI
from simudo.gui.panels.shared import (
    INPUT_SS, SELECT_SS, BTN_LIGHT_SS, ADD_BTN_SS, MULTICHOICE_SS,
    sec_header, label,
)

if TYPE_CHECKING:
    from simudo.gui.app import SimudoApp

_BAND_TYPE_OPTIONS = list(BAND_TYPE_GUI_TO_YAML.keys())
_BC_OPTIONS = ["ohmic", "zero_current"]

# Left-border accent colours by sign: electrons=blue, holes=orange
_SIGN_COLOR = {-1: "#4a9eff", 1: "#e07b39"}

_SIGN_SS = ["""
.bk-btn-group { display:flex; gap:0; }
.bk-btn-group .bk-btn {
    background: #1a2435 !important; color: #8a9bb5 !important;
    border: 1px solid #2d3748 !important;
    font-size: 12px !important; padding: 2px 10px !important;
    min-width: 70px !important; text-align: center !important;
}
.bk-btn-group .bk-btn.bk-active {
    background: #2d4a6e !important; color: #4a9eff !important;
    border-color: #4a9eff !important;
}
"""]



class BandsPanel:
    def __init__(self, app: "SimudoApp"):
        self.app = app

        self._list_col = pn.Column(sizing_mode="stretch_width", styles={"gap": "8px"})
        self._add_btn = pn.widgets.Button(
            name="+ Add band", stylesheets=ADD_BTN_SS,
            sizing_mode="stretch_width", height=30, margin=(4, 0, 0, 0),
        )
        self._add_btn.on_click(lambda e: self._add_band())

        _note = pn.pane.HTML(
            '<div style="font-size:11px;color:#7a8caa;padding:2px 0 6px;">'
            'Band parameters (energy levels, effective density of states, etc.) '
            'are set in the '
            '<strong style="color:#8a9bb5;">Layers</strong> or '
            '<strong style="color:#8a9bb5;">Materials</strong> panels.</div>',
            sizing_mode="stretch_width", margin=0,
        )
        self._view = pn.Column(
            sec_header("BANDS"),
            _note,
            self._list_col,
            self._add_btn,
            sizing_mode="stretch_width",
            styles={"padding": "12px", "gap": "4px"},
        )

    def view(self) -> pn.Column:
        self._refresh()
        return self._view

    def _refresh(self):
        self._list_col.objects = [
            self._build_band_row(i, b)
            for i, b in enumerate(self.app.project.bands)
        ]

    def _add_band(self):
        existing = {b.name for b in self.app.project.bands}
        for candidate in ["CB", "VB", "IB", "CB2", "VB2", "IB2", "band"]:
            if candidate not in existing:
                name = candidate
                break
        else:
            name = f"band{len(self.app.project.bands) + 1}"
        # Auto-assign carrier sign: holes (+1) for VB-like names, electrons (-1) otherwise
        sign = +1 if name.startswith("VB") else -1
        self.app.project.bands.append(Band(name=name, sign=sign))
        self.app.autosave()
        self.app.refresh_layers_missing_params()
        self._refresh()

    def _delete_band(self, idx: int):
        bands = self.app.project.bands
        if len(bands) <= 1:
            return
        bands.pop(idx)
        self.app.autosave()
        self.app.refresh_layers_missing_params()
        self._refresh()

    def _extent_options(self) -> List[str]:
        """Layer names + non-domain overlay region names, in order."""
        layer_names  = [l.name for l in self.app.project.layers]
        region_names = [ov.name for ov in self.app.project.overlay_regions
                        if not ov.is_domain]
        return layer_names + region_names

    def _build_band_row(self, idx: int, band: Band) -> pn.Column:
        bands       = self.app.project.bands
        accent = _SIGN_COLOR.get(band.sign, "#4a9eff")
        extent_opts = self._extent_options()
        extent_is_all = (band.extent == "all")
        current_sel   = band.extent if isinstance(band.extent, list) else []

        # ── Row 1: name + delete ──────────────────────────────────────────────
        name_in = pn.widgets.TextInput(
            value=band.name, placeholder="name",
            height=26, margin=0, stylesheets=INPUT_SS,
            sizing_mode="stretch_width",
        )
        del_btn = pn.widgets.Button(
            name="✕", width=24, height=26, margin=0, stylesheets=BTN_LIGHT_SS,
        )
        del_btn.on_click(lambda e, i=idx: self._delete_band(i))

        row1 = pn.Row(
            name_in, del_btn,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )

        # ── Row 2: type + sign ────────────────────────────────────────────────
        type_sel = pn.widgets.Select(
            value=BAND_TYPE_YAML_TO_GUI.get(band.type, "Boltzmann"),
            options=_BAND_TYPE_OPTIONS,
            height=26, margin=0, stylesheets=SELECT_SS,
            sizing_mode="stretch_width",
        )
        sign_rbg = pn.widgets.RadioButtonGroup(
            options=["Electrons", "Holes"],
            value="Electrons" if band.sign == -1 else "Holes",
            height=26, width=150, margin=0, stylesheets=_SIGN_SS,
        )
        row2 = pn.Row(
            label("Type:", 34), type_sel,
            label("Carriers:", 52), sign_rbg,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )

        # ── Extent selector ───────────────────────────────────────────────────
        extent_sel = pn.widgets.Select(
            value="Domain (all layers)" if extent_is_all else "Select regions…",
            options=["Domain (all layers)", "Select regions…"],
            height=26, width=170, margin=0, stylesheets=SELECT_SS,
        )
        extent_row = pn.Row(
            label("Extent:", 42), extent_sel,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )

        # ── MultiChoice for region selection ──────────────────────────────────
        mc = pn.widgets.MultiChoice(
            value=[r for r in current_sel if r in extent_opts],
            options=extent_opts,
            placeholder="Select layers / regions…",
            height=60, margin=0, stylesheets=MULTICHOICE_SS,
            sizing_mode="stretch_width",
            visible=not extent_is_all,
        )

        # ── BC sub-card (only for finite-extent bands) ────────────────────────
        bc_note = pn.pane.HTML(
            '<div style="font-size:11px;color:#7a8caa;padding:2px 0 4px;">'
            'By default, finite-extent bands have zero-current boundary '
            'conditions at their edges.</div>',
            sizing_mode="stretch_width", margin=0,
        )
        bc_left_sel = pn.widgets.Select(
            value=band.extent_boundary_bc_left, options=_BC_OPTIONS,
            height=26, width=110, margin=0, stylesheets=SELECT_SS,
        )
        bc_right_sel = pn.widgets.Select(
            value=band.extent_boundary_bc_right, options=_BC_OPTIONS,
            height=26, width=110, margin=0, stylesheets=SELECT_SS,
        )
        bc_row = pn.Row(
            label("BC left:", 48), bc_left_sel,
            label("right:", 34), bc_right_sel,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )
        bc_card = pn.Column(
            bc_note, bc_row,
            sizing_mode="stretch_width",
            visible=not extent_is_all,
            styles={
                "background": "#111827",
                "border": "1px solid #2d3748",
                "border-left": f"3px solid {accent}",
                "border-radius": "3px",
                "padding": "6px 8px",
                "gap": "2px",
            },
        )

        # ── Callbacks ─────────────────────────────────────────────────────────
        def _on_name(e, i=idx):
            bands[i].name = e.new.strip() or bands[i].name
            self.app.autosave()
        name_in.param.watch(_on_name, "value")

        def _on_type(e, i=idx):
            bands[i].type = BAND_TYPE_GUI_TO_YAML[e.new]
            self.app.autosave()
        type_sel.param.watch(_on_type, "value")

        def _on_sign(e, i=idx):
            bands[i].sign = -1 if e.new == "Electrons" else +1
            self.app.autosave()
        sign_rbg.param.watch(_on_sign, "value")

        def _on_extent_sel(e, i=idx):
            is_all = (e.new == "Domain (all layers)")
            bands[i].extent = "all" if is_all else (mc.value or [])
            mc.visible      = not is_all
            bc_card.visible = not is_all
            self.app.autosave()
        extent_sel.param.watch(_on_extent_sel, "value")

        def _on_mc(e, i=idx):
            if bands[i].extent != "all":
                bands[i].extent = e.new
                self.app.autosave()
        mc.param.watch(_on_mc, "value")

        def _on_bc_left(e, i=idx):
            bands[i].extent_boundary_bc_left = e.new
            self.app.autosave()
        def _on_bc_right(e, i=idx):
            bands[i].extent_boundary_bc_right = e.new
            self.app.autosave()
        bc_left_sel.param.watch(_on_bc_left, "value")
        bc_right_sel.param.watch(_on_bc_right, "value")

        # ── Outer card ────────────────────────────────────────────────────────
        return pn.Column(
            row1, row2, extent_row, mc, bc_card,
            sizing_mode="stretch_width",
            styles={
                "background": "#141c2b",
                "border":      f"1px solid {accent}50",
                "border-left": f"4px solid {accent}",
                "border-radius": "5px",
                "padding": "10px",
                "gap": "5px",
            },
        )
