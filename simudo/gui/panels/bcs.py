"""
Simudo GUI — Boundary Conditions panel.

Shows left and right contact cards.  Each card has:
  • role   (Select: swept / reference)
  • per-band BC  (RadioButtonGroup: ohmic / zero_current / SRV)
  • when SRV selected: value + unit inputs for surface recombination velocity
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import panel as pn

from simudo.gui.panels.shared import (
    INPUT_SS, SELECT_SS, RADIO_SS,
    sec_header, label,
)

if TYPE_CHECKING:
    from simudo.gui.app import SimudoApp

_ROLES = ["swept", "reference"]
_BC_OPTS = ["ohmic", "zero_current", "SRV"]

# Map display label ↔ YAML type string
_DISPLAY_TO_TYPE = {
    "ohmic":         "ohmic",
    "zero_current":  "zero_current",
    "SRV":           "surface_recombination",
}
_TYPE_TO_DISPLAY = {v: k for k, v in _DISPLAY_TO_TYPE.items()}

_SRV_UNITS = ["cm/s", "m/s"]


def _bc_display(bc_spec) -> str:
    """Return the RadioButtonGroup label for a bc_spec (string or dict)."""
    if isinstance(bc_spec, dict):
        t = bc_spec.get("type", "ohmic")
    else:
        t = bc_spec or "ohmic"
    return _TYPE_TO_DISPLAY.get(t, "ohmic")


def _srv_params(bc_spec):
    """Return (value_str, unit) from a bc_spec, with defaults."""
    if isinstance(bc_spec, dict):
        srv = bc_spec.get("srv", {})
        if isinstance(srv, dict):
            return str(srv.get("value", 1000.0)), str(srv.get("unit", "cm/s"))
    return "1000.0", "cm/s"


class BCsPanel:
    def __init__(self, app: "SimudoApp"):
        self.app = app

        self._left_col = pn.Column(sizing_mode="stretch_width", styles={"gap": "4px"})
        self._right_col = pn.Column(sizing_mode="stretch_width", styles={"gap": "4px"})

        self._view = pn.Column(
            sec_header("BOUNDARY CONDITIONS"),
            pn.Row(
                pn.Column(
                    pn.pane.HTML(
                        '<div style="font-size:11px;font-weight:700;color:#8a9bb5;'
                        'padding:4px 0 6px;">LEFT CONTACT</div>',
                        height=24, margin=0,
                    ),
                    self._left_col,
                    sizing_mode="stretch_width",
                    styles={
                        "background": "#1a2435", "border": "1px solid #2d3748",
                        "border-radius": "4px", "padding": "8px", "gap": "4px",
                    },
                ),
                pn.Column(
                    pn.pane.HTML(
                        '<div style="font-size:11px;font-weight:700;color:#8a9bb5;'
                        'padding:4px 0 6px;">RIGHT CONTACT</div>',
                        height=24, margin=0,
                    ),
                    self._right_col,
                    sizing_mode="stretch_width",
                    styles={
                        "background": "#1a2435", "border": "1px solid #2d3748",
                        "border-radius": "4px", "padding": "8px", "gap": "4px",
                    },
                ),
                sizing_mode="stretch_width",
                styles={"gap": "8px"},
            ),
            sizing_mode="stretch_width",
            styles={"padding": "12px", "gap": "8px"},
        )

    def view(self) -> pn.Column:
        self._refresh()
        return self._view

    def _refresh(self):
        self._left_col.objects  = self._build_contact_widgets("left")
        self._right_col.objects = self._build_contact_widgets("right")

    def _build_contact_widgets(self, side: str) -> list:
        contacts = self.app.project.contacts
        contact = contacts.get(side)
        if contact is None:
            return [pn.pane.HTML('<span style="color:#556070;font-size:12px;">No contact defined.</span>')]

        bands = self.app.project.bands

        # ── Role ─────────────────────────────────────────────────────────────
        role_sel = pn.widgets.Select(
            value=contact.role if contact.role in _ROLES else _ROLES[0],
            options=_ROLES, height=26, width=110, margin=0, stylesheets=SELECT_SS,
        )
        def _on_role(e, s=side):
            contacts[s].role = e.new
            self.app.autosave()
        role_sel.param.watch(_on_role, "value")

        role_row = pn.Row(
            label("Role:", 36), role_sel,
            margin=0, styles={"gap": "4px", "align-items": "center"},
        )

        # ── Per-band BCs ──────────────────────────────────────────────────────
        band_rows = []
        for band in bands:
            current_bc = contact.band_bcs.get(band.name, "ohmic")
            display_val = _bc_display(current_bc)
            srv_val_str, srv_unit = _srv_params(current_bc)

            bc_rbg = pn.widgets.RadioButtonGroup(
                options=_BC_OPTS,
                value=display_val if display_val in _BC_OPTS else "ohmic",
                height=26, margin=0, stylesheets=RADIO_SS,
            )

            # SRV sub-row: value TextInput + unit Select
            srv_val_in = pn.widgets.TextInput(
                value=srv_val_str,
                placeholder="1000.0",
                width=80, height=26, margin=0,
                stylesheets=INPUT_SS,
            )
            srv_unit_sel = pn.widgets.Select(
                value=srv_unit if srv_unit in _SRV_UNITS else "cm/s",
                options=_SRV_UNITS,
                width=70, height=26, margin=0,
                stylesheets=SELECT_SS,
            )
            srv_row = pn.Row(
                label("SRV:", 36),
                srv_val_in,
                srv_unit_sel,
                margin=0,
                styles={"gap": "4px", "align-items": "center"},
                visible=(display_val == "SRV"),
            )

            def _on_bc(e, s=side, bn=band.name, srow=srv_row,
                       sval=srv_val_in, sunit=srv_unit_sel):
                srow.visible = (e.new == "SRV")
                if e.new == "SRV":
                    try:
                        val = float(sval.value)
                    except (ValueError, TypeError):
                        val = 1000.0
                    contacts[s].band_bcs[bn] = {
                        "type": "surface_recombination",
                        "srv": {"value": val, "unit": sunit.value},
                    }
                else:
                    contacts[s].band_bcs[bn] = _DISPLAY_TO_TYPE[e.new]
                self.app.autosave()
            bc_rbg.param.watch(_on_bc, "value")

            def _on_srv_val(e, s=side, bn=band.name, sunit=srv_unit_sel):
                try:
                    val = float(e.new)
                except (ValueError, TypeError):
                    return
                contacts[s].band_bcs[bn] = {
                    "type": "surface_recombination",
                    "srv": {"value": val, "unit": sunit.value},
                }
                self.app.autosave()
            srv_val_in.param.watch(_on_srv_val, "value")

            def _on_srv_unit(e, s=side, bn=band.name, sval=srv_val_in):
                try:
                    val = float(sval.value)
                except (ValueError, TypeError):
                    val = 1000.0
                contacts[s].band_bcs[bn] = {
                    "type": "surface_recombination",
                    "srv": {"value": val, "unit": e.new},
                }
                self.app.autosave()
            srv_unit_sel.param.watch(_on_srv_unit, "value")

            band_col = pn.Column(
                pn.Row(
                    label(f"{band.name}:", 36), bc_rbg,
                    margin=0, styles={"gap": "4px", "align-items": "center"},
                    sizing_mode="stretch_width",
                ),
                srv_row,
                margin=0,
                styles={"gap": "2px"},
            )
            band_rows.append(band_col)

        return [role_row] + band_rows
