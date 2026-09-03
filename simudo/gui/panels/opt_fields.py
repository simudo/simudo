"""
Simudo GUI — Optical Fields panel.

Lists optical fields with add / delete.  Each field has:
  • name            (TextInput)
  • direction       (+x / −x radio)
  • photon_energy   (value + unit)
  • intensity mode  (Explicit | Blackbody)
      Explicit  → value + unit
      Blackbody → T_source, concentration, E_min, E_max (∞ checkbox)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import panel as pn

from simudo.gui.model import (OpticalField, OpticalFieldSet, ExplicitIntensity,
                   BlackbodyIntensity)
from simudo.gui.panels.shared import (
    INPUT_SS, SELECT_SS, BTN_LIGHT_SS, RADIO_SS, ADD_BTN_SS, CHECKBOX_SS,
    sec_header, label,
)

if TYPE_CHECKING:
    from simudo.gui.app import SimudoApp

_ENERGY_UNITS    = ["eV", "J", "cm^-1", "nm", "um"]
_FLUX_UNITS      = ["cm^-2 s^-1", "m^-2 s^-1"]
_POWER_UNITS     = ["mW/cm^2", "W/m^2", "suns", "W/cm^2", "kW/m^2"]
_TEMP_UNITS      = ["K", "°C"]
_SPECTRA         = ["am15g", "am15d", "blackbody"]
_BIN_EDGE_MODES  = ["equal_flux", "uniform_energy"]

# Physical constants for live flux preview (no pint dependency in GUI)
_EV_TO_J = 1.60218e-19   # J per eV
_H_JS    = 6.62607e-34   # J·s
_C_MS    = 2.99792e8     # m/s

# Power-density → W/cm² conversion factors for supported units
_TO_W_CM2 = {
    "mw/cm^2": 1e-3,
    "w/cm^2":  1.0,
    "w/m^2":   1e-4,
    "kw/m^2":  0.1,
    "suns":    0.1,   # 1 sun ≈ 100 mW/cm²
    "sun":     0.1,
}


def _power_to_flux_str(power_val: float, power_unit: str,
                       E_val: float, E_unit: str) -> str:
    """Return a human-readable photon-flux string, or empty string on failure."""
    try:
        factor = _TO_W_CM2.get(power_unit.lower().replace(" ", ""))
        if factor is None:
            return ""
        I_wcm2 = float(power_val) * factor          # W/cm²
        # Photon energy in Joules
        eu = E_unit.lower()
        if eu == "ev":
            E_J = float(E_val) * _EV_TO_J
        elif eu == "j":
            E_J = float(E_val)
        elif eu == "nm":
            E_J = _H_JS * _C_MS / (float(E_val) * 1e-9)
        elif eu == "um":
            E_J = _H_JS * _C_MS / (float(E_val) * 1e-6)
        elif eu == "cm^-1":
            E_J = _H_JS * _C_MS * float(E_val) * 100.0
        else:
            return ""
        if E_J <= 0:
            return ""
        flux = I_wcm2 / E_J                         # photons/cm²/s
        return f"{flux:.3e} cm⁻²s⁻¹"
    except Exception:
        return ""


class OptFieldsPanel:
    def __init__(self, app: "SimudoApp"):
        self.app = app

        self._list_col = pn.Column(sizing_mode="stretch_width", styles={"gap": "2px"})
        self._add_btn = pn.widgets.Button(
            name="+ Add monochromatic field", stylesheets=ADD_BTN_SS,
            sizing_mode="stretch_width", height=30, margin=(4, 0, 0, 0),
        )
        self._add_btn.on_click(lambda e: self._add_field())

        # Spectrum-expanded field sets.
        self._sets_col = pn.Column(sizing_mode="stretch_width", styles={"gap": "2px"})
        self._add_set_btn = pn.widgets.Button(
            name="+ Add field set (spectrum)", stylesheets=ADD_BTN_SS,
            sizing_mode="stretch_width", height=30, margin=(4, 0, 0, 0),
        )
        self._add_set_btn.on_click(lambda e: self._add_field_set())

        self._view = pn.Column(
            sec_header("OPTICAL FIELDS"),
            self._list_col,
            self._add_btn,
            sec_header("OPTICAL FIELD SETS"),
            self._sets_col,
            self._add_set_btn,
            sizing_mode="stretch_width",
            styles={"padding": "12px", "gap": "4px"},
        )

    def view(self) -> pn.Column:
        self._refresh()
        return self._view

    def _refresh(self):
        fields = self.app.project.optical_fields
        self._list_col.objects = [self._build_row(i, f) for i, f in enumerate(fields)]
        sets = self.app.project.optical_field_sets
        self._sets_col.objects = [self._build_set_row(i, s) for i, s in enumerate(sets)]

    def _add_field(self):
        fields = self.app.project.optical_fields
        n = len(fields) + 1
        fields.append(OpticalField(name=f"field{n}"))
        self.app.autosave()
        self._refresh()

    def _delete_field(self, idx: int):
        self.app.project.optical_fields.pop(idx)
        self.app.autosave()
        self._refresh()

    def _add_field_set(self):
        sets = self.app.project.optical_field_sets
        existing = {s.name_prefix for s in sets}
        base = "sun"
        name = base
        n = 2
        while name in existing:
            name = f"{base}{n}"
            n += 1
        sets.append(OpticalFieldSet(name_prefix=name))
        self.app.autosave()
        self._refresh()

    def _delete_field_set(self, idx: int):
        self.app.project.optical_field_sets.pop(idx)
        self.app.autosave()
        self._refresh()

    def _build_set_row(self, idx: int, fs: OpticalFieldSet) -> pn.Column:
        sets = self.app.project.optical_field_sets

        name_in = pn.widgets.TextInput(
            value=fs.name_prefix, placeholder="name prefix",
            height=26, margin=0, stylesheets=INPUT_SS, sizing_mode="stretch_width")
        def _on_name(e, i=idx):
            sets[i].name_prefix = e.new.strip() or sets[i].name_prefix
            self.app.autosave()
        name_in.param.watch(_on_name, "value")

        dir_rbg = pn.widgets.RadioButtonGroup(
            options=["+x", "−x"], value=fs.direction if fs.direction == "+x" else "−x",
            height=26, margin=0, stylesheets=RADIO_SS)
        def _on_dir(e, i=idx):
            sets[i].direction = "+x" if e.new == "+x" else "-x"; self.app.autosave()
        dir_rbg.param.watch(_on_dir, "value")

        del_btn = pn.widgets.Button(name="✕", width=24, height=26, margin=0,
                                    stylesheets=BTN_LIGHT_SS)
        del_btn.on_click(lambda e, i=idx: self._delete_field_set(i))

        header = pn.Row(
            name_in, dir_rbg, del_btn,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width")

        # Spectrum selector + T_source (only for blackbody).
        spec_sel = pn.widgets.Select(
            value=fs.spectrum if fs.spectrum in _SPECTRA else "am15g",
            options=_SPECTRA, height=26, width=110, margin=0, stylesheets=SELECT_SS)
        T_in = pn.widgets.NumberInput(
            value=fs.T_source, step=100.0, start=0, height=26, width=80, margin=0,
            stylesheets=INPUT_SS, visible=(fs.spectrum == "blackbody"))
        T_unit = pn.widgets.Select(
            value=fs.T_source_unit if fs.T_source_unit in _TEMP_UNITS else "K",
            options=_TEMP_UNITS, height=26, width=46, margin=0, stylesheets=SELECT_SS,
            visible=(fs.spectrum == "blackbody"))
        spec_row = pn.Row(
            label("Spectrum:", 60), spec_sel, label("T:", 14), T_in, T_unit,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width")

        # N bins, concentration, bin-edge mode.
        N_in = pn.widgets.IntInput(
            value=fs.N_bins, start=1, step=1, height=26, width=60, margin=0,
            stylesheets=INPUT_SS)
        conc_in = pn.widgets.NumberInput(
            value=fs.concentration, step=1.0, start=0, height=26, width=70, margin=0,
            stylesheets=INPUT_SS)
        bin_sel = pn.widgets.Select(
            value=fs.bin_edges if fs.bin_edges in _BIN_EDGE_MODES else "equal_flux",
            options=_BIN_EDGE_MODES, height=26, width=120, margin=0, stylesheets=SELECT_SS)
        nbc_row = pn.Row(
            label("N bins:", 44), N_in, label("Conc.:", 38), conc_in,
            label("Edges:", 40), bin_sel,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width")

        # Energy range.
        Emin_in = pn.widgets.NumberInput(
            value=fs.E_min, step=0.01, start=0, height=26, width=70, margin=0,
            stylesheets=INPUT_SS)
        Emin_unit = pn.widgets.Select(
            value=fs.E_min_unit if fs.E_min_unit in _ENERGY_UNITS else "eV",
            options=_ENERGY_UNITS, height=26, width=60, margin=0, stylesheets=SELECT_SS)
        Emax_in = pn.widgets.NumberInput(
            value=fs.E_max, step=0.01, start=0, height=26, width=70, margin=0,
            stylesheets=INPUT_SS)
        Emax_unit = pn.widgets.Select(
            value=fs.E_max_unit if fs.E_max_unit in _ENERGY_UNITS else "eV",
            options=_ENERGY_UNITS, height=26, width=60, margin=0, stylesheets=SELECT_SS)
        e_row = pn.Row(
            label("E min:", 40), Emin_in, Emin_unit,
            label("E max:", 38), Emax_in, Emax_unit,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width")

        help_pane = pn.pane.HTML(
            '<div style="font-size:11px;color:#8ab4d0;background:#0d1a2a;'
            'border-left:2px solid #2d4a6e;padding:5px 8px;'
            'border-radius:0 3px 3px 0;line-height:1.5;">'
            f'Expands at run time into N monochromatic fields named '
            f'<code>{fs.name_prefix}_000</code> … binned across the energy '
            'range. <code>equal_flux</code> bins carry equal photon flux; '
            '<code>uniform_energy</code> bins are evenly spaced in energy. '
            'Concentration multiplies the spectrum flux.'
            '</div>',
            sizing_mode="stretch_width", margin=0)

        def _commit(_=None, i=idx):
            s = sets[i]
            s.spectrum     = spec_sel.value
            s.N_bins       = int(N_in.value or 1)
            s.concentration = float(conc_in.value or 0.0)
            s.bin_edges    = bin_sel.value
            s.E_min        = float(Emin_in.value or 0.0)
            s.E_min_unit   = Emin_unit.value
            s.E_max        = float(Emax_in.value or 0.0)
            s.E_max_unit   = Emax_unit.value
            s.T_source     = float(T_in.value or 0.0)
            s.T_source_unit = T_unit.value
            self.app.autosave()

        def _on_spec(e, i=idx):
            is_bb = (e.new == "blackbody")
            T_in.visible = is_bb
            T_unit.visible = is_bb
            _commit(i=i)
        spec_sel.param.watch(_on_spec, "value")
        for w in (N_in, conc_in, bin_sel, Emin_in, Emin_unit, Emax_in, Emax_unit,
                  T_in, T_unit):
            w.param.watch(lambda e, i=idx: _commit(i=i), "value")

        return pn.Column(
            header, spec_row, nbc_row, e_row, help_pane,
            sizing_mode="stretch_width",
            styles={
                "background": "#1a2435",
                "border": "1px solid #2d3748",
                "border-radius": "4px",
                "padding": "8px",
                "gap": "4px",
            },
        )

    def _build_row(self, idx: int, field: OpticalField) -> pn.Column:
        fields = self.app.project.optical_fields

        # ── Name ─────────────────────────────────────────────────────────────
        name_in = pn.widgets.TextInput(
            value=field.name, placeholder="name",
            height=26, margin=0, stylesheets=INPUT_SS,
            sizing_mode="stretch_width",
        )
        def _on_name(e, i=idx): fields[i].name = e.new.strip() or fields[i].name; self.app.autosave()
        name_in.param.watch(_on_name, "value")

        # ── Direction ─────────────────────────────────────────────────────────
        dir_rbg = pn.widgets.RadioButtonGroup(
            options=["+x", "−x"], value=field.direction if field.direction == "+x" else "−x",
            height=26, margin=0, stylesheets=RADIO_SS,
        )
        def _on_dir(e, i=idx):
            fields[i].direction = "+x" if e.new == "+x" else "-x"
            self.app.autosave()
        dir_rbg.param.watch(_on_dir, "value")

        # ── Delete ────────────────────────────────────────────────────────────
        del_btn = pn.widgets.Button(name="✕", width=24, height=26, margin=0,
                                     stylesheets=BTN_LIGHT_SS)
        del_btn.on_click(lambda e, i=idx: self._delete_field(i))

        header = pn.Row(
            name_in, dir_rbg, del_btn,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )

        # ── Photon energy ─────────────────────────────────────────────────────
        pe_in = pn.widgets.NumberInput(
            value=field.photon_energy, step=0.001, start=0,
            height=26, width=90, margin=0, stylesheets=INPUT_SS,
        )
        pe_unit = pn.widgets.Select(
            value=field.photon_energy_unit if field.photon_energy_unit in _ENERGY_UNITS else "eV",
            options=_ENERGY_UNITS, height=26, width=60, margin=0, stylesheets=SELECT_SS,
        )
        def _on_pe(e, i=idx): fields[i].photon_energy = float(e.new or 0); self.app.autosave()
        def _on_pe_unit(e, i=idx): fields[i].photon_energy_unit = e.new; self.app.autosave()
        pe_in.param.watch(_on_pe, "value")
        pe_unit.param.watch(_on_pe_unit, "value")

        pe_row = pn.Row(
            label("Photon E:", 56), pe_in, pe_unit,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )

        # ── Intensity mode ────────────────────────────────────────────────────
        is_bb = isinstance(field.intensity, BlackbodyIntensity)
        mode_sel = pn.widgets.Select(
            value="Blackbody" if is_bb else "Explicit",
            options=["Explicit", "Blackbody"],
            height=26, width=110, margin=0, stylesheets=SELECT_SS,
        )

        # ---- Explicit intensity sub-form ─────────────────────────────────────
        inten = field.intensity
        is_exp  = isinstance(inten, ExplicitIntensity)
        is_pd   = is_exp and inten.input_type == "power_density"

        exp_val  = inten.value if is_exp else 100.0
        exp_unit = inten.unit  if is_exp else "mW/cm^2"

        # Sub-mode toggle: photon flux vs power density
        exp_type_rbg = pn.widgets.RadioButtonGroup(
            value="Power density" if is_pd else "Photon flux",
            options=["Photon flux", "Power density"],
            height=26, margin=0, stylesheets=RADIO_SS,
        )

        # Value input (shared)
        exp_val_in = pn.widgets.NumberInput(
            value=exp_val, step=1.0, start=0,
            height=26, width=90, margin=0, stylesheets=INPUT_SS,
        )

        # Unit selector — options differ by sub-mode
        _init_units = _POWER_UNITS if is_pd else _FLUX_UNITS
        _init_unit  = (exp_unit if exp_unit in _init_units
                       else (_POWER_UNITS[0] if is_pd else _FLUX_UNITS[0]))
        exp_unit_sel = pn.widgets.Select(
            value=_init_unit, options=_init_units,
            height=26, width=100, margin=0, stylesheets=SELECT_SS,
        )

        # Hint shown when power-density sub-mode is active
        def _flux_hint() -> str:
            fs = _power_to_flux_str(
                exp_val_in.value or 0.0, exp_unit_sel.value,
                field.photon_energy, field.photon_energy_unit,
            )
            body = (f"≈ {fs}" if fs else "(enter value above to see conversion)")
            return (
                '<div style="font-size:11px;color:#8ab4d0;background:#0d1a2a;'
                'border-left:2px solid #2d4a6e;padding:5px 8px;'
                'border-radius:0 3px 3px 0;line-height:1.5;">'
                'Simudo needs photon flux (photons · cm⁻² · s⁻¹). '
                'The value you enter will be divided by the photon energy '
                'to convert it automatically.<br>'
                f'<span style="color:#c8d6e5;font-family:monospace;">{body}</span>'
                '</div>'
            )

        exp_hint_pane = pn.pane.HTML(
            _flux_hint() if is_pd else "",
            sizing_mode="stretch_width", margin=(0, 0, 0, 0),
            visible=(not is_bb and is_pd),
        )

        exp_row = pn.Column(
            pn.Row(
                exp_type_rbg,
                margin=0, styles={"gap": "4px", "align-items": "center"},
            ),
            pn.Row(
                label("Value:", 42), exp_val_in, exp_unit_sel,
                margin=0, styles={"gap": "4px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            exp_hint_pane,
            margin=0, styles={"gap": "4px"},
            sizing_mode="stretch_width",
            visible=not is_bb,
        )

        # ---- Blackbody sub-form ──────────────────────────────────────────────
        bb = inten if isinstance(inten, BlackbodyIntensity) else BlackbodyIntensity()

        T_in = pn.widgets.NumberInput(
            value=bb.T_source, step=100.0, start=0,
            height=26, width=80, margin=0, stylesheets=INPUT_SS,
        )
        T_unit = pn.widgets.Select(
            value=bb.T_source_unit if bb.T_source_unit in _TEMP_UNITS else "K",
            options=_TEMP_UNITS, height=26, width=46, margin=0, stylesheets=SELECT_SS,
        )
        conc_in = pn.widgets.NumberInput(
            value=bb.concentration, step=1.0, start=0,
            height=26, width=70, margin=0, stylesheets=INPUT_SS,
        )
        Emin_in = pn.widgets.NumberInput(
            value=bb.E_min, step=0.01, start=0,
            height=26, width=70, margin=0, stylesheets=INPUT_SS,
        )
        Emin_unit = pn.widgets.Select(
            value=bb.E_min_unit if bb.E_min_unit in _ENERGY_UNITS else "eV",
            options=_ENERGY_UNITS, height=26, width=60, margin=0, stylesheets=SELECT_SS,
        )
        Emax_val = bb.E_max if bb.E_max is not None else 0.0
        Emax_in = pn.widgets.NumberInput(
            value=Emax_val, step=0.01, start=0,
            height=26, width=70, margin=0, stylesheets=INPUT_SS,
            visible=(bb.E_max is not None),
        )
        Emax_unit = pn.widgets.Select(
            value=bb.E_max_unit if bb.E_max_unit in _ENERGY_UNITS else "eV",
            options=_ENERGY_UNITS, height=26, width=60, margin=0, stylesheets=SELECT_SS,
            visible=(bb.E_max is not None),
        )
        Emax_inf_cb = pn.widgets.Checkbox(
            name="∞", value=(bb.E_max is None),
            height=26, margin=0, stylesheets=CHECKBOX_SS,
        )

        # ── Blackbody help text ───────────────────────────────────────────────
        bb_help_pane = pn.pane.HTML(
            '<div style="font-size:11px;color:#8ab4d0;background:#0d1a2a;'
            'border-left:2px solid #2d4a6e;padding:5px 8px;'
            'border-radius:0 3px 3px 0;line-height:1.5;">'
            'Specify the temperature of the blackbody source, the concentration '
            '(relative to the solar solid-angle fraction '
            '<em>f</em><sub>s</sub> = 1/26200), '
            'and the spectral range to be integrated into this field.'
            '</div>',
            sizing_mode="stretch_width", margin=(0, 0, 2, 0),
            visible=is_bb,
        )

        # ── Concentration warning (shown when > 46 200×) ──────────────────────
        _CONC_THRESH = 46200
        _init_conc = bb.concentration if isinstance(inten, BlackbodyIntensity) else 0.0
        conc_warn_pane = pn.pane.HTML(
            '<div style="font-size:11px;color:#f0c040;'
            'background:rgba(240,192,64,0.10);'
            'border:1px solid rgba(240,192,64,0.35);'
            'border-radius:4px;padding:5px 8px;">'
            f'⚠ Concentration exceeds {_CONC_THRESH:,}×, which is the '
            'theoretical maximum solar concentration. Verify this is intentional.'
            '</div>',
            sizing_mode="stretch_width", margin=0,
            visible=(is_bb and _init_conc > _CONC_THRESH),
        )

        # ── E_max ∞ checkbox hint ─────────────────────────────────────────────
        emax_hint_pane = pn.pane.HTML(
            '<div style="font-size:11px;color:#556070;padding:1px 2px;">'
            'Check ∞ to use no upper energy cutoff '
            '(integrate the blackbody spectrum to infinity).'
            '</div>',
            sizing_mode="stretch_width", margin=0,
            visible=is_bb,
        )

        bb_T_row = pn.Row(
            label("T source:", 56), T_in, T_unit,
            label("Conc.:", 38), conc_in,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
            visible=is_bb,
        )
        bb_E_row = pn.Row(
            label("E min:", 40), Emin_in, Emin_unit,
            label("E max:", 38), Emax_in, Emax_unit, Emax_inf_cb,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
            visible=is_bb,
        )

        # ── Callbacks ─────────────────────────────────────────────────────────
        def _on_exp_type(event, i=idx):
            """Switch explicit sub-mode between Photon flux and Power density."""
            pd = (event.new == "Power density")
            new_units = _POWER_UNITS if pd else _FLUX_UNITS
            exp_unit_sel.options = new_units
            exp_unit_sel.value   = new_units[0]
            exp_hint_pane.visible = pd
            exp_hint_pane.object  = _flux_hint() if pd else ""
            if isinstance(fields[i].intensity, ExplicitIntensity):
                fields[i].intensity.input_type = "power_density" if pd else "flux"
                fields[i].intensity.unit = new_units[0]
                self.app.autosave()
        exp_type_rbg.param.watch(_on_exp_type, "value")

        def _on_mode(event, i=idx):
            if event.new == "Explicit":
                pd = (exp_type_rbg.value == "Power density")
                fields[i].intensity = ExplicitIntensity(
                    value=exp_val_in.value or 100.0,
                    unit=exp_unit_sel.value,
                    input_type="power_density" if pd else "flux",
                )
                exp_row.visible = True
                exp_hint_pane.visible = pd
                bb_help_pane.visible = False
                bb_T_row.visible = False
                conc_warn_pane.visible = False
                bb_E_row.visible = False
                emax_hint_pane.visible = False
            else:
                fields[i].intensity = BlackbodyIntensity(
                    T_source=T_in.value or 6000.0,
                    T_source_unit=T_unit.value,
                    concentration=conc_in.value or 1.0,
                    E_min=Emin_in.value or 0.0,
                    E_min_unit=Emin_unit.value,
                    E_max=None if Emax_inf_cb.value else (Emax_in.value or 0.0),
                    E_max_unit=Emax_unit.value,
                )
                exp_row.visible = False
                exp_hint_pane.visible = False
                bb_help_pane.visible = True
                bb_T_row.visible = True
                conc_warn_pane.visible = (conc_in.value or 0) > _CONC_THRESH
                bb_E_row.visible = True
                emax_hint_pane.visible = True
            self.app.autosave()
        mode_sel.param.watch(_on_mode, "value")

        def _sync_explicit(i=idx):
            if isinstance(fields[i].intensity, ExplicitIntensity):
                fields[i].intensity.value = exp_val_in.value or 0.0
                fields[i].intensity.unit = exp_unit_sel.value
                # Refresh conversion hint
                if exp_hint_pane.visible:
                    exp_hint_pane.object = _flux_hint()
                self.app.autosave()
        exp_val_in.param.watch(lambda e, i=idx: _sync_explicit(i), "value")
        exp_unit_sel.param.watch(lambda e, i=idx: _sync_explicit(i), "value")

        def _sync_bb(i=idx):
            if isinstance(fields[i].intensity, BlackbodyIntensity):
                b = fields[i].intensity
                b.T_source = T_in.value or 0.0
                b.T_source_unit = T_unit.value
                b.concentration = conc_in.value or 0.0
                b.E_min = Emin_in.value or 0.0
                b.E_min_unit = Emin_unit.value
                b.E_max = None if Emax_inf_cb.value else (Emax_in.value or 0.0)
                b.E_max_unit = Emax_unit.value
                conc_warn_pane.visible = (conc_in.value or 0) > _CONC_THRESH
                self.app.autosave()
        for w in (T_in, T_unit, conc_in, Emin_in, Emin_unit, Emax_in, Emax_unit):
            w.param.watch(lambda e, i=idx: _sync_bb(i), "value")

        def _on_emax_inf(event, i=idx):
            Emax_in.visible = not event.new
            Emax_unit.visible = not event.new
            _sync_bb(i)
        Emax_inf_cb.param.watch(_on_emax_inf, "value")

        mode_row = pn.Row(
            label("Intensity mode:", 96), mode_sel,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )

        return pn.Column(
            header, pe_row, mode_row, bb_help_pane,
            exp_row, bb_T_row, conc_warn_pane, bb_E_row, emax_hint_pane,
            sizing_mode="stretch_width",
            styles={
                "background": "#1a2435",
                "border": "1px solid #2d3748",
                "border-radius": "4px",
                "padding": "8px",
                "gap": "4px",
            },
        )
