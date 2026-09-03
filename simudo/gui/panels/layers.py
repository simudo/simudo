"""
Simudo GUI — Layers panel.

Uses the "update in place" pattern: persistent pn.Column/.pane objects whose
.objects / .object attributes are mutated directly, giving immediate DOM updates
without relying on Panel's reactive parameter machinery.
"""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, Dict, List

import panel as pn

from simudo.gui.model import (Interface, InterfaceBCSpec, Layer, MeshParams,
                    OverlayRegion, default_layer_color, default_bands)

if TYPE_CHECKING:
    from simudo.gui.app import SimudoApp

# Imported lazily at first use to avoid a circular import at module load time.
# Access via _get_eop_required_params() / _get_eop_required_callables().
_EOP_REQUIRED_PARAMS_CACHE: Dict[str, Dict[str, str]] | None = None
_EOP_REQUIRED_CALLABLES_CACHE: Dict[str, Dict[str, Dict[str, str]]] | None = None

def _get_eop_required_params() -> Dict[str, Dict[str, str]]:
    global _EOP_REQUIRED_PARAMS_CACHE
    if _EOP_REQUIRED_PARAMS_CACHE is None:
        from simudo.gui.panels.processes import _EOP_REQUIRED_PARAMS
        _EOP_REQUIRED_PARAMS_CACHE = _EOP_REQUIRED_PARAMS
    return _EOP_REQUIRED_PARAMS_CACHE

def _get_eop_required_callables() -> Dict[str, Dict[str, Dict[str, str]]]:
    global _EOP_REQUIRED_CALLABLES_CACHE
    if _EOP_REQUIRED_CALLABLES_CACHE is None:
        from simudo.gui.panels.processes import _EOP_REQUIRED_CALLABLES
        _EOP_REQUIRED_CALLABLES_CACHE = _EOP_REQUIRED_CALLABLES
    return _EOP_REQUIRED_CALLABLES_CACHE

_EOP_OPTICAL_CLASSES_CACHE = None
_EOP_RAD_DEFAULTS_CACHE: Dict[str, bool] | None = None

def _get_eop_optical_classes():
    global _EOP_OPTICAL_CLASSES_CACHE
    if _EOP_OPTICAL_CLASSES_CACHE is None:
        from simudo.gui.panels.processes import _OPTICAL_CLASSES
        _EOP_OPTICAL_CLASSES_CACHE = _OPTICAL_CLASSES
    return _EOP_OPTICAL_CLASSES_CACHE

def _get_eop_rad_defaults() -> Dict[str, bool]:
    global _EOP_RAD_DEFAULTS_CACHE
    if _EOP_RAD_DEFAULTS_CACHE is None:
        from simudo.gui.panels.processes import _EOP_RAD_DEFAULTS
        _EOP_RAD_DEFAULTS_CACHE = _EOP_RAD_DEFAULTS
    return _EOP_RAD_DEFAULTS_CACHE

# Global parameters always required (independent of bands/processes)
_GLOBAL_REQUIRED: Dict[str, str] = {
    "temperature": "K",
}

# Per-band-type required suffixes → unit (hardcoded; band types are stable)
_BAND_REQUIRED: Dict[str, Dict[str, str]] = {
    "nondegenerate": {"energy_level": "eV", "effective_density_of_states": "cm^-3"},
    "degenerate":    {"energy_level": "eV", "effective_density_of_states": "cm^-3"},
    "intermediate":  {"energy_level": "eV", "number_of_states": "cm^-3"},
    "mixed_qfl_nondegenerate": {"energy_level": "eV", "effective_density_of_states": "cm^-3"},
}


# ── Helpers ────────────────────────────────────────────────────────────────────

import math as _math

# Physical constants for v_th formula
_KB = 1.380649e-23   # J/K  (Boltzmann)
_ME = 9.10938e-31    # kg   (electron mass)
# Prefactor: 8 k_B / (π m_e) in SI, result in m²/s²  → take sqrt → m/s → ×100 → cm/s
_VTH_PREFACTOR = 8.0 * _KB / (_math.pi * _ME)   # m²/s² per K per m*


def _project_temperature(project) -> float:
    """Read temperature in K from the domain overlay's spatial properties, fallback 300 K.

    Uses Pint to convert whatever unit the user set (K, degC, etc.) to Kelvin.
    Returns 300.0 if temperature is unset or cannot be parsed.
    """
    try:
        for ov in project.overlay_regions:
            if ov.is_domain:
                entry = ov.properties.get("temperature")
                if entry is not None:
                    if isinstance(entry, dict):
                        val  = float(entry.get("value", 300.0))
                        unit = entry.get("unit", "K") or "K"
                    else:
                        val, unit = float(entry), "K"
                    import pint as _pint
                    _ureg = _pint.UnitRegistry()
                    T_K = float(_ureg.Quantity(val, unit).to("K").magnitude)
                    return T_K if T_K > 0 else 300.0
    except Exception:
        pass
    return 300.0


def _vth_hint_html(T_K: float) -> str:
    """Return amber hint HTML for v_th, computed at temperature T_K with m*=1."""
    vth_cms = _math.sqrt(_VTH_PREFACTOR * T_K) * 100    # cm/s, m*=1
    vth_str = f"{vth_cms:.3e}"
    return (
        '<div style="background:rgba(240,160,40,0.10);border:1px solid rgba(240,160,40,0.30);'
        'border-radius:4px;padding:7px 10px;margin-top:4px;font-size:11px;line-height:1.55;">'
        '<div style="color:#f0c040;font-weight:700;margin-bottom:4px;">v_th formula helper</div>'
        '<div style="color:#c8a060;font-family:monospace;margin-bottom:5px;">'
        'v_th = √( 8 k<sub>B</sub> T / π m* m<sub>e</sub> )'
        '</div>'
        f'<div style="color:#c8d6e5;">At project temperature <b>T = {T_K:.0f} K</b>, <b>m* = 1</b>: '
        f'<span style="font-family:monospace;color:#f0c040;">{vth_str} cm/s</span></div>'
        '<div style="color:#8a9bb5;margin-top:4px;">'
        'Scales as 1/√m* — e.g. m*=0.063 (GaAs CB) → ×3.98.'
        '</div>'
        '<div style="color:#556070;margin-top:5px;font-size:10px;">'
        'If a material file is loaded, v_th is set automatically. '
        'Set project temperature via <em>Physics → Temperature</em> (currently '
        f'{T_K:.0f} K).'
        '</div>'
        '</div>'
    )


_TEMP_WARNING_HTML = (
    '<div style="background:rgba(231,76,60,0.12);border:1px solid rgba(231,76,60,0.40);'
    'border-radius:4px;padding:7px 10px;margin-top:4px;font-size:11px;line-height:1.55;">'
    '<div style="color:#e74c3c;font-weight:700;margin-bottom:4px;">⚠ Temperature warning</div>'
    '<div style="color:#c8d6e5;">Temperature should normally be set in the '
    '<code>domain</code> region so it applies everywhere.</div>'
    '<div style="color:#8a9bb5;margin-top:4px;">'
    'If you set it here but not in every other region, it will be <b>0 K</b> '
    'in any region where it is unset — which will almost certainly cause solver errors.'
    '</div>'
    '</div>'
)


def _fmt_sci(v: float) -> str:
    """Format a concentration as scientific notation, e.g. '1.000e+18'. Zero → '0'."""
    if v == 0.0:
        return "0"
    return f"{v:.3e}"


def _sanitize_ident(s: str) -> str:
    """Return a valid Python identifier derived from s.

    Spaces and hyphens → underscore; non-alphanumeric/underscore chars removed;
    leading digit → prefixed with underscore; empty result → '_'.
    """
    s = s.strip().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "_"


def _to_um(thickness: float, unit: str) -> float:
    """Convert a thickness to micrometres."""
    if unit == "nm":  return thickness * 1e-3
    if unit == "mm":  return thickness * 1e3
    return thickness   # already um


# ── Overlay colour palette ─────────────────────────────────────────────────────

_OV_COLORS = ["#4a9eff", "#ff6b4a", "#4aff9e", "#ff9e4a", "#c44aff", "#ff4ac8"]

# ── Schematic flex constants ────────────────────────────────────────────────────
# MUST be kept in sync between _build_schematic_objects and _build_overlay_strip_html.
#
# Schematic buttons and overlay strip divs both use:
#   flex: {fv} 1 0px   (flex-basis = 0 → purely proportional by flex-grow)
#
# Using flex-basis:0 (not 26px) is critical: it means position% = flex_cum/total_flex
# exactly, which is what _um_to_frac returns for coordinate overlays.  Any non-zero
# flex-basis adds a width-dependent offset that can't be compensated in pure Python.
#
# _SCHEMATIC_MIN_FLEX sets the minimum flex-grow for any layer so that zero-thickness
# or very thin layers still get a visible block in the schematic.
_SCHEMATIC_MIN_FLEX = 5.0

# Width (px) of the clickable interface-divider buttons between layer blocks.
# MUST be kept in sync with _build_overlay_strip_html which inserts identically-
# sized transparent placeholder divs so the overlay segments stay aligned.
_IFACE_DIV_WIDTH = 12


# ── Compact widget stylesheets ─────────────────────────────────────────────────

_INPUT_SS = ["""
:host { font-size: 13px !important; }
.bk-input { background: #1a2435 !important; color: #c8d6e5 !important;
    border: 1px solid #2d3748 !important; border-radius: 3px !important;
    padding: 2px 6px !important; font-size: 13px !important; }
.bk-input:focus { border-color: #4a9eff !important; outline: none !important; }
"""]

_SELECT_SS = ["""
:host { font-size: 12px !important; }
select { background: #1a2435 !important; color: #c8d6e5 !important;
    border: 1px solid #2d3748 !important; border-radius: 3px !important;
    padding: 1px 2px !important; font-size: 12px !important; }
"""]

_BTN_LIGHT_SS = ["""
.bk-btn { background: #1a2435 !important; color: #8a9bb5 !important;
    border: 1px solid #2d3748 !important; border-radius: 3px !important;
    font-size: 12px !important; padding: 0 !important; cursor: pointer !important; }
.bk-btn:hover { color: #e74c3c !important; border-color: #e74c3c !important; }
"""]

_BTN_MOVE_SS = ["""
.bk-btn { background: transparent !important; color: #556070 !important;
    border: none !important; font-size: 11px !important;
    padding: 0 !important; cursor: pointer !important; line-height: 1 !important; }
.bk-btn:hover { color: #c8d6e5 !important; }
"""]

_RADIO_SS = ["""
.bk-btn-group .bk-btn {
    background: #1a2435 !important; color: #8a9bb5 !important;
    border: 1px solid #2d3748 !important;
    font-size: 12px !important; padding: 2px 8px !important;
}
.bk-btn-group .bk-btn.bk-active {
    background: #2d4a6e !important; color: #4a9eff !important;
    border-color: #4a9eff !important;
}
"""]

_ADD_BTN_SS = ["""
.bk-btn { background: transparent !important; color: #7a8caa !important;
    border: 1px dashed #2d3748 !important; border-radius: 4px !important;
    font-size: 13px !important; cursor: pointer !important;
    padding: 4px 12px !important; }
.bk-btn:hover { color: #4a9eff !important; border-color: #4a9eff !important; }
"""]

_MULTICHOICE_SS = ["""
:host { font-size: 12px !important; }
.choices { background: #1a2435 !important; border: 1px solid #2d3748 !important;
    border-radius: 3px !important; }
.choices__inner { background: transparent !important; border: none !important;
    padding: 2px 4px !important; min-height: 28px !important; }
.choices__input { background: transparent !important; color: #c8d6e5 !important;
    font-size: 12px !important; }
.choices__item--selectable { color: #c8d6e5 !important;
    background: #2d4a6e !important; font-size: 11px !important; }
.choices__list--dropdown { background: #1a2435 !important;
    border: 1px solid #4a9eff !important; z-index: 200 !important; }
.choices__list--dropdown .choices__item--selectable {
    background: transparent !important; color: #c8d6e5 !important; }
.choices__list--dropdown .choices__item--selectable.is-highlighted {
    background: #2d4a6e !important; }
"""]

_CONTACT_HTML = (
    '<div style="width:20px;min-width:20px;height:100%;display:flex;'
    'align-items:center;justify-content:center;background:#1a2435;'
    'border:1px solid #2d3748;border-radius:3px;cursor:pointer;'
    'font-size:8px;color:#556070;writing-mode:vertical-rl;'
    'padding:3px 1px;box-sizing:border-box;user-select:none;"'
    ' title="Click to configure external boundary conditions">BC</div>'
)


class LayersPanel:
    """Layers sidebar panel — builds layout once, updates components in place."""

    def __init__(self, app: "SimudoApp"):
        self.app = app
        self._selected: int = -1           # selected layer index (-1 = none)
        self._selected_overlay: int = -1   # selected overlay index (-1 = none)
        self._selected_interface: int = -1 # left-layer index of selected divider (-1 = none)
        self._iface_warning_dismissed: bool = False  # "default bands" warning dismissed
        self._same_mat_warning_dismissed: bool = False  # same-material HJBC warning dismissed

        # ── Persistent panes ──────────────────────────────────────────────────
        # Schematic: a Column containing [layer_row, overlay_strip]
        self._schematic = pn.Column(
            sizing_mode="stretch_width", margin=0,
            styles={"margin-bottom": "8px", "gap": "3px"},
        )
        self._overlay_strip = pn.pane.HTML(
            "", sizing_mode="stretch_width", height=10, margin=0,
        )
        self._table = pn.Column(sizing_mode="stretch_width", styles={"gap": "1px"})
        self._detail = pn.Column(sizing_mode="stretch_width", styles={"gap": "4px"})
        self._overlays_inner = pn.Column(styles={"gap": "4px"})

        # Missing-params section — updated in rebuild()
        self._missing_params_pane = pn.pane.HTML(
            "", sizing_mode="stretch_width", margin=0,
        )

        _SEC = ('font-size:11px;font-weight:700;letter-spacing:0.10em;'
                'color:#8a9bb5;border-left:2px solid #4a9eff;padding-left:6px;')
        _sec = lambda title, extra="": pn.pane.HTML(
            f'<div style="{_SEC}{extra}">{title}</div>',
            sizing_mode="stretch_width", height=20, margin=(8, 0, 4, 0),
        )
        _divider = lambda: pn.pane.HTML(
            '<hr style="border:none;border-top:1px solid #1e2d3d;margin:0;">',
            sizing_mode="stretch_width", height=1, margin=(10, 0, 0, 0),
        )

        # Left column: layer stack list + overlays + missing params
        _left_col = pn.Column(
            _sec("LAYER STACK"),
            self._schematic,
            self._overlay_strip,
            self._table,
            _divider(),
            _sec("OVERLAY REGIONS"),
            self._overlays_inner,
            _divider(),
            self._missing_params_pane,
            sizing_mode="stretch_width",
            styles={"gap": "4px", "min-width": "300px"},
        )

        # Right column: properties / detail panel
        _right_col = pn.Column(
            _sec("PROPERTIES"),
            self._detail,
            width=380,
            styles={"gap": "4px", "flex-shrink": "0"},
        )

        self._layout = pn.Column(
            pn.Row(
                _left_col,
                _right_col,
                sizing_mode="stretch_width",
                styles={"gap": "16px", "align-items": "flex-start"},
            ),
            sizing_mode="stretch_width",
            styles={"padding": "12px", "overflow-y": "auto", "gap": "4px"},
        )

        self.rebuild()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def project(self):
        return self.app.project

    def rebuild(self):
        """Refresh every dynamic component in place."""
        self._schematic.objects = self._build_schematic_objects()
        self._overlay_strip.object = self._build_overlay_strip_html()
        self._table.objects = self._build_table_rows()
        self._detail.objects = [self._build_detail()]
        self._overlays_inner.objects = self._build_overlays()
        self._missing_params_pane.object = self._build_missing_params_html()

    # ── Suggested / missing parameter helpers ────────────────────────────────

    def _compute_suggested_params(self) -> Dict[str, str]:
        """Return {full_key: unit} for all params implied by current bands + processes.

        Callable requirements (alpha_function, sigma_function, ...) are emitted
        with a display 'unit' string of the form ``callable: <arg>→<ret>`` so
        the missing-params table distinguishes them from scalar rules.
        """
        suggested: Dict[str, str] = dict(_GLOBAL_REQUIRED)
        for band in self.project.bands:
            for suffix, unit in _BAND_REQUIRED.get(band.type, {}).items():
                suggested[f"{band.name}/{suffix}"] = unit
        eop_rp = _get_eop_required_params()
        for proc in self.project.processes:
            rsp = eop_rp.get(proc.cls, {})
            for suffix, unit in rsp.items():
                key = (suffix
                       .replace("{dst_band}", proc.dst_band)
                       .replace("{src_band}", proc.src_band))
                suggested[f"{proc.name}/{key}"] = unit
        eop_rc = _get_eop_required_callables()
        for proc in self.project.processes:
            rsc = eop_rc.get(proc.cls, {})
            for suffix, meta in rsc.items():
                key = (suffix
                       .replace("{dst_band}", proc.dst_band)
                       .replace("{src_band}", proc.src_band))
                arg = meta.get("arg_units", "?")
                ret = meta.get("return_units", "?")
                suggested[f"{proc.name}/{key}"] = f"callable: {arg}→{ret}"
        # refractive_index is required by the Shockley–van Roosbroeck radiative
        # recombination integral, which runs for any optical process that has
        # radiative recombination enabled.  The per-class default comes from the
        # AST-extracted enable_radiative_recombination attribute; a process-level
        # radiative_recombination value of False/None overrides it.
        if self._any_radiative_recombination():
            suggested["refractive_index"] = ""
        return suggested

    def _any_radiative_recombination(self) -> bool:
        """True if any optical process will run SVR (and thus needs refractive_index)."""
        optical  = _get_eop_optical_classes()
        defaults = _get_eop_rad_defaults()
        for proc in self.project.processes:
            if proc.cls not in optical:
                continue
            rr = proc.radiative_recombination
            # None → use the class default; False → off; anything else → on.
            enabled = defaults.get(proc.cls, False) if rr is None else bool(rr)
            if enabled:
                return True
        return False

    def _compute_missing_params(self) -> List[str]:
        """Return suggested params not set in any region or applied material."""
        suggested = self._compute_suggested_params()
        if not suggested:
            return []
        # Keys set in any layer or overlay region
        set_keys: set = set()
        for layer in self.project.layers:
            set_keys.update(layer.properties.keys())
        for ov in self.project.overlay_regions:
            set_keys.update(ov.properties.keys())
        # Keys provided by materials that are applied to at least one layer
        applied_mats = {l.material for l in self.project.layers if l.material}
        lib_dirs = self.app.get_library_dirs()
        from simudo.gui.panels.mat_ast import get_property_keys_for_material
        for mat in self.project.materials:
            if mat.name in applied_mats:
                set_keys.update(get_property_keys_for_material(mat, lib_dirs))
        # Callables satisfied by an inline top-hat spec on the process itself
        # (the GUI process-card editor) count as set.
        for proc in self.project.processes:
            for shorthand, rows in (getattr(proc, "callable_top_hats", {}) or {}).items():
                if rows:
                    set_keys.add(f"{proc.name}/{shorthand}")
        return sorted(k for k in suggested if k not in set_keys)

    def _build_missing_params_html(self) -> str:
        """HTML for the global 'missing parameters' section at the bottom of the left panel."""
        missing = self._compute_missing_params()
        if not missing:
            return ""
        suggested = self._compute_suggested_params()
        rows = "".join(
            f'<tr>'
            f'<td style="padding:2px 6px 2px 0;color:#c8d6e5;font-size:11px;'
            f'font-family:monospace;">{k}</td>'
            f'<td style="padding:2px 0;color:#7a8caa;font-size:11px;">'
            f'{suggested.get(k, "")}</td>'
            f'</tr>'
            for k in missing
        )
        return (
            '<div style="margin-top:18px;border-top:1px solid #1e2d3d;padding-top:10px;">'
            '<div style="font-size:11px;font-weight:700;letter-spacing:0.10em;'
            'color:#f0c040;border-left:2px solid #f0c040;padding-left:6px;'
            'margin-bottom:6px;">PARAMETERS NOT SET IN ANY REGION</div>'
            '<div style="font-size:11px;color:#7a8caa;margin-bottom:6px;">'
            'Any unset parameter defaults to zero. If you set a parameter in only one region, '
            'it will be zero in all other regions. If you want it to be set globally, set the '
            'parameter in the <code>domain</code> region.</div>'
            f'<table style="border-collapse:collapse;width:100%;">'
            f'<thead><tr>'
            f'<th style="text-align:left;font-size:10px;color:#556070;'
            f'font-weight:600;padding-bottom:3px;">Key</th>'
            f'<th style="text-align:left;font-size:10px;color:#556070;'
            f'font-weight:600;padding-bottom:3px;">Unit</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _select(self, idx: int):
        """Select a layer; clear overlay and interface selection."""
        self._selected = idx
        self._selected_overlay = -1
        self._selected_interface = -1
        self._schematic.objects = self._build_schematic_objects()
        self._table.objects = self._build_table_rows()
        self._detail.objects = [self._build_detail()]

    def _select_overlay(self, idx: int):
        """Select an overlay; clear layer and interface selection."""
        self._selected_overlay = idx
        self._selected = -1
        self._selected_interface = -1
        self._overlay_strip.object = self._build_overlay_strip_html()
        self._overlays_inner.objects = self._build_overlays()
        self._table.objects = self._build_table_rows()   # clear layer highlight
        self._detail.objects = [self._build_detail()]

    def _select_interface(self, left_idx: int):
        """Select an interface divider; clear layer and overlay selection."""
        self._selected_interface = left_idx
        self._selected = -1
        self._selected_overlay = -1
        self._iface_warning_dismissed = False
        self._same_mat_warning_dismissed = False
        self._schematic.objects = self._build_schematic_objects()
        self._table.objects = self._build_table_rows()
        self._detail.objects = [self._build_detail()]

    def _move(self, idx: int, direction: int):
        layers = self.project.layers
        new_idx = idx + direction
        if 0 <= new_idx < len(layers):
            layers[idx], layers[new_idx] = layers[new_idx], layers[idx]
            self._selected = new_idx
            self.rebuild()

    def _delete(self, idx: int):
        del self.project.layers[idx]
        self._selected = min(self._selected, len(self.project.layers) - 1)
        self.rebuild()

    # ── Schematic ─────────────────────────────────────────────────────────────

    def _build_schematic_objects(self) -> list:
        """Layer-block row (each block is a real Button so it receives clicks).

        No separator panes are used between blocks — the container uses
        gap:2px so the flex-fraction math stays clean and the overlay strip
        can mirror positions exactly.  Interface edges are marked by a dark
        right border on each non-last button (future: make them clickable for
        interface-condition editing).
        """
        layers = self.project.layers
        if not layers:
            return [pn.pane.HTML(
                '<div style="color:#556070;font-size:13px;padding:8px;">'
                'No layers yet.</div>'
            )]

        total_t = sum(l.thickness for l in layers) or 1.0

        inner_items: list = []
        for i, layer in enumerate(layers):
            # flex-basis MUST be 0px so position% = flex_cum/total_flex exactly.
            # See _SCHEMATIC_MIN_FLEX comment above.
            flex_val = max(_SCHEMATIC_MIN_FLEX, (layer.thickness / total_t) * 80)
            sel = i == self._selected
            sel_outline = ("outline:2px solid #fff !important;"
                           "outline-offset:-2px !important;" if sel else
                           "outline:none !important;")
            btn = pn.widgets.Button(
                name=layer.name, margin=0,
                stylesheets=[f"""
                    :host {{ flex:{flex_val:.2f} 1 0px !important;
                             min-width:0 !important; overflow:hidden !important;
                             align-self:stretch !important; }}
                    .bk-btn {{ width:100% !important; height:100% !important;
                        background:{layer.color} !important;
                        border:none !important;
                        border-radius:3px !important; {sel_outline}
                        color:#fff !important; font-size:11px !important;
                        font-weight:600 !important; cursor:pointer !important;
                        text-shadow:0 1px 2px rgba(0,0,0,0.6) !important;
                        overflow:hidden !important; text-overflow:ellipsis !important;
                        white-space:nowrap !important; padding:2px !important;
                        box-sizing:border-box !important; }}
                    .bk-btn:hover {{ filter:brightness(1.15) !important; }}
                """],
            )
            btn.on_click(lambda e, idx=i: self._select(idx))
            inner_items.append(btn)

            # Clickable divider between this layer and the next.
            # Uses explicit width/height Panel params (not just CSS flex) so the
            # button renders at exactly _IFACE_DIV_WIDTH × 52 px regardless of
            # how the browser resolves :host flex rules.
            if i < len(layers) - 1:
                iface_sel = (self._selected_interface == i)
                if iface_sel:
                    div_bg     = "#4a9eff"
                    div_border = "border:1px solid #4a9eff !important;"
                else:
                    div_bg     = "rgba(74,158,255,0.18)"
                    div_border = "border:1px solid rgba(74,158,255,0.38) !important;"
                div_btn = pn.widgets.Button(
                    name="", margin=0,
                    width=_IFACE_DIV_WIDTH,
                    stylesheets=[f"""
                        :host {{
                            flex-shrink: 0 !important;
                            align-self: stretch !important;
                            height: auto !important;
                            display: flex !important;
                            flex-direction: column !important;
                        }}
                        .bk-btn-group {{
                            flex: 1 1 0 !important;
                            display: flex !important;
                            flex-direction: column !important;
                            min-height: 0 !important;
                        }}
                        .bk-btn {{
                            flex: 1 1 0 !important;
                            width: 100% !important;
                            min-height: 0 !important;
                            background: {div_bg} !important; {div_border}
                            border-radius: 2px !important;
                            cursor: pointer !important;
                            padding: 0 !important;
                        }}
                        .bk-btn:hover {{
                            background: rgba(74,158,255,0.52) !important;
                            border-color: #4a9eff !important;
                        }}
                    """],
                )
                div_btn.on_click(lambda e, idx=i: self._select_interface(idx))
                inner_items.append(div_btn)

        # gap:2px — must match the overlay strip's flex container gap
        inner = pn.Row(
            *inner_items, margin=0, sizing_mode="stretch_width",
            styles={"align-items": "stretch", "overflow": "hidden",
                    "flex": "1", "gap": "2px"},
        )
        # Create separate pane objects — reusing one pane in two layout slots
        # causes Panel to render only one of them.
        contact_l = pn.pane.HTML(_CONTACT_HTML, width=22, height=52, margin=0)
        contact_r = pn.pane.HTML(_CONTACT_HTML, width=22, height=52, margin=0)
        layer_row = pn.Row(contact_l, inner, contact_r,
                           height=52, margin=0,
                           styles={"align-items": "stretch"})
        return [layer_row]

    def _build_overlay_strip_html(self) -> str:
        """Coloured bar below the schematic showing where each overlay sits.

        For **layer-based** overlays the strip uses the same CSS flex layout as
        the schematic (same flex values, same gap:2px), so layer boundaries
        align pixel-perfectly regardless of window width or min-flex inflation.
        Disjoint selections (e.g. p + n without I) are shown as two separate
        coloured segments.

        For **coordinate-based** overlays a position:absolute bar is overlaid
        on top; its left/right percentages use the flex-fraction mapping so
        sub-layer precision is preserved (a tiny ~1 % error from the 2 px
        gaps is visually undetectable).

        The whole strip is inset 22 px each side to align with the inner layer
        area of the schematic row above it.
        """
        layers = self.project.layers
        all_ovs = self.project.overlay_regions
        custom_ovs = [(idx, ov) for idx, ov in enumerate(all_ovs) if not ov.is_domain]
        if not layers or not custom_ovs:
            return ""

        # ── Same flex values as _build_schematic_objects ──────────────────────
        total_t   = sum(l.thickness for l in layers) or 1.0
        flex_vals = [max(_SCHEMATIC_MIN_FLEX, (l.thickness / total_t) * 80)
                     for l in layers]
        total_flex = sum(flex_vals)

        flex_cum = [0.0]
        for fv in flex_vals:
            flex_cum.append(flex_cum[-1] + fv)

        layer_idx_map: dict[str, int] = {l.name: i for i, l in enumerate(layers)}

        # ── Per-layer colour assignment (last overlay wins; selected overrides) ─
        # layer_paint[i] = (color, alpha, is_selected) or None
        layer_paint: list = [None] * len(layers)

        nd = 0
        for full_idx, ov in enumerate(all_ovs):
            if ov.is_domain:
                continue
            color = _OV_COLORS[nd % len(_OV_COLORS)]
            sel   = (self._selected_overlay == full_idx)
            alpha = 0.75 if sel else 0.42
            nd   += 1
            if ov.layer_names:
                for n in ov.layer_names:
                    if n in layer_idx_map:
                        i = layer_idx_map[n]
                        # Selected overlay always wins; otherwise last wins.
                        if layer_paint[i] is None or sel or not layer_paint[i][2]:
                            layer_paint[i] = (color, alpha, sel)

        # ── Build flex-mirror layer segments ──────────────────────────────────
        # After each layer segment (except the last) insert a transparent
        # placeholder whose width exactly mirrors the interface-divider button
        # in the schematic row.  This keeps every layer column aligned even
        # though the dividers are fixed-width elements that sit outside the
        # layer flex values.  (Coordinate-based absolute overlays will have a
        # small ≲5 % error from these fixed-pixel dividers, which is acceptable.)
        segs: list[str] = []
        for i, fv in enumerate(flex_vals):
            paint = layer_paint[i]
            if paint:
                c, a, s = paint
                ol = (f"outline:2px solid {c};outline-offset:-1px;" if s else "")
                style = f"background:{c};opacity:{a:.2f};{ol}"
            else:
                style = "background:transparent;"
            segs.append(
                f'<div style="flex:{fv:.2f} 1 0;min-width:0;height:100%;'
                f'border-radius:2px;box-sizing:border-box;{style}"></div>'
            )
            if i < len(flex_vals) - 1:
                # Transparent placeholder mirroring the _IFACE_DIV_WIDTH divider
                segs.append(
                    f'<div style="flex:0 0 {_IFACE_DIV_WIDTH}px;'
                    f'min-width:{_IFACE_DIV_WIDTH}px;height:100%;'
                    f'background:transparent;"></div>'
                )

        # ── Name labels for layer-based overlays (position:absolute) ─────────
        # One label per contiguous run of selected layers so disjoint overlays
        # each get a label on every separate segment.
        label_divs: list[str] = []
        nd = 0
        for full_idx, ov in enumerate(all_ovs):
            if ov.is_domain:
                continue
            color = _OV_COLORS[nd % len(_OV_COLORS)]
            nd += 1
            if not ov.layer_names:
                continue
            indices = sorted(layer_idx_map[n] for n in ov.layer_names
                             if n in layer_idx_map)
            if not indices:
                continue
            # Group into contiguous runs
            runs, start, prev = [], indices[0], indices[0]
            for idx in indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    runs.append((start, prev))
                    start = prev = idx
            runs.append((start, prev))
            for (g_s, g_e) in runs:
                x0p = flex_cum[g_s]     / total_flex * 100
                x1p = flex_cum[g_e + 1] / total_flex * 100
                label_divs.append(
                    f'<div style="position:absolute;left:{x0p:.3f}%;'
                    f'width:{x1p - x0p:.3f}%;height:100%;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'pointer-events:none;overflow:hidden;">'
                    f'<span style="font-size:9px;color:#fff;font-weight:600;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
                    f'padding:0 2px;">{ov.name}</span></div>'
                )

        # ── Coordinate overlays as position:absolute on top ───────────────────
        um_cum = [0.0]
        for l in layers:
            um_cum.append(um_cum[-1] + _to_um(l.thickness, l.thickness_unit))
        total_um = um_cum[-1] or 1.0

        def _um_to_frac(pos: float) -> float:
            """Map a µm position to its fractional position in the flex space."""
            pos = max(0.0, min(total_um, pos))
            for i in range(len(layers)):
                lo, hi = um_cum[i], um_cum[i + 1]
                if lo <= pos <= hi:
                    span = hi - lo
                    if span == 0:
                        return flex_cum[i] / total_flex
                    return (flex_cum[i] + (pos - lo) / span * flex_vals[i]) / total_flex
            return 1.0

        abs_divs: list[str] = []
        nd = 0
        for full_idx, ov in enumerate(all_ovs):
            if ov.is_domain:
                continue
            color = _OV_COLORS[nd % len(_OV_COLORS)]
            sel   = (self._selected_overlay == full_idx)
            alpha = 0.75 if sel else 0.42
            nd   += 1
            if ov.layer_names:
                continue  # handled by flex segments above
            if ov.start is None or ov.end is None:
                continue
            x0 = _um_to_frac(_to_um(ov.start, ov.extent_unit or "um"))
            x1 = _um_to_frac(_to_um(ov.end,   ov.extent_unit or "um"))
            x0p = max(0.0, min(100.0, x0 * 100))
            x1p = max(0.0, min(100.0, x1 * 100))
            if x1p <= x0p:
                continue
            ol = (f"outline:2px solid {color};outline-offset:-1px;" if sel else "")
            abs_divs.append(
                f'<div title="{ov.name}" style="position:absolute;left:{x0p:.3f}%;'
                f'width:{x1p - x0p:.3f}%;height:100%;background:{color};'
                f'opacity:{alpha:.2f};border-radius:2px;overflow:hidden;{ol}'
                f'display:flex;align-items:center;justify-content:center;pointer-events:none;">'
                f'<span style="font-size:9px;color:#fff;font-weight:600;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
                f'padding:0 2px;">{ov.name}</span></div>'
            )

        # Suppress empty strips (all layers uncoloured, no coordinate overlays)
        if all(p is None for p in layer_paint) and not abs_divs and not label_divs:
            return ""

        inner = (
            # gap:2px must match the schematic inner Row gap
            '<div style="position:relative;display:flex;gap:2px;width:100%;height:10px;'
            'border-radius:2px;overflow:visible;background:rgba(26,36,53,0.3);">'
            + "".join(segs)
            + "".join(label_divs)   # layer-based overlay names
            + "".join(abs_divs)     # coordinate overlay bars (drawn on top)
            + "</div>"
        )
        # 22 px inset aligns the strip with the inner layer area (excluding BC markers)
        return (
            '<div style="padding-left:22px;padding-right:22px;'
            'box-sizing:border-box;width:100%;">'
            + inner + "</div>"
        )

    # ── Layer table ───────────────────────────────────────────────────────────
    #
    # Layout (all items margin=0, outer Row gap=6px, padding="2px 4px"):
    #   [20px move-col] [18px dot] [82px name] [112px thickness] [stretch mat] [22px del]

    def _build_table_rows(self) -> list:
        layers = self.project.layers
        rows: list = []

        _H = 'style="font-size:11px;color:#556070;font-weight:600;letter-spacing:0.04em;"'
        _mat_tip = "(Optional) Set a material — use Materials panel to set up options"
        rows.append(pn.Row(
            pn.Spacer(width=20, height=1, margin=0),
            pn.Spacer(width=18, height=1, margin=0),
            pn.pane.HTML(f'<span {_H}>Layer Name</span>',
                         width=82, height=16, margin=0),
            pn.pane.HTML(f'<span {_H}>Thickness</span>',
                         width=112, height=16, margin=0),
            pn.pane.HTML(
                f'<span {_H} title="{_mat_tip}">Material'
                f' <span style="color:#4a9eff;font-size:10px;cursor:help;">ℹ</span></span>',
                sizing_mode="stretch_width", height=16, margin=0),
            pn.Spacer(width=22, height=1, margin=0),
            margin=0,
            styles={"padding": "2px 4px", "align-items": "center", "gap": "6px"},
            sizing_mode="stretch_width",
        ))

        for i, layer in enumerate(layers):
            sel = (i == self._selected)
            bg = "rgba(74,158,255,0.07)" if sel else "transparent"

            _dot_border = "2px solid #fff" if sel else "2px solid rgba(255,255,255,0.15)"
            dot_btn = pn.widgets.Button(name="", width=18, height=18, margin=0,
                stylesheets=[f"""
                    :host {{ width:18px !important; height:18px !important;
                             display:flex !important; align-items:center !important;
                             justify-content:center !important; flex-shrink:0; }}
                    .bk-btn {{ width:100% !important; height:100% !important;
                        background:{layer.color} !important; border-radius:50% !important;
                        border:{_dot_border} !important; padding:0 !important;
                        cursor:pointer !important; min-width:0 !important;
                        box-sizing:border-box !important; }}
                    .bk-btn:hover {{ filter:brightness(1.35) !important; }}
                """])
            dot_btn.on_click(lambda e, idx=i: self._select(idx))

            up_btn = pn.widgets.Button(name="▲", width=18, height=14, margin=0,
                                       stylesheets=_BTN_MOVE_SS)
            dn_btn = pn.widgets.Button(name="▼", width=18, height=14, margin=0,
                                       stylesheets=_BTN_MOVE_SS)
            up_btn.on_click(lambda e, idx=i: self._move(idx, -1))
            dn_btn.on_click(lambda e, idx=i: self._move(idx, +1))

            name_in = pn.widgets.TextInput(
                value=layer.name, width=82, height=26, margin=0,
                stylesheets=_INPUT_SS,
            )
            def _name_cb(event, idx=i):
                clean = _sanitize_ident(event.new)
                old_name = self.project.layers[idx].name
                self.project.layers[idx].name = clean
                if clean != event.new:          # show corrected value in-place
                    name_in.value = clean
                # Auto-update any interface entries referencing the old name
                if old_name != clean:
                    for iface in self.project.interfaces:
                        if iface.left  == old_name: iface.left  = clean
                        if iface.right == old_name: iface.right = clean
                    self.app.autosave()
                self._schematic.objects = self._build_schematic_objects()
                self._overlay_strip.object = self._build_overlay_strip_html()
            name_in.param.watch(_name_cb, "value")

            thick_in = pn.widgets.FloatInput(
                value=layer.thickness, width=62, height=26, margin=0,
                stylesheets=_INPUT_SS,
            )
            unit_sel = pn.widgets.Select(
                value=layer.thickness_unit, options=["nm", "um", "mm"],
                width=46, height=26, margin=0, stylesheets=_SELECT_SS,
            )
            def _thick_cb(event, idx=i):
                self.project.layers[idx].thickness = event.new
                self._schematic.objects = self._build_schematic_objects()
                self._overlay_strip.object = self._build_overlay_strip_html()
            def _unit_cb(event, idx=i):
                self.project.layers[idx].thickness_unit = event.new
                self._overlay_strip.object = self._build_overlay_strip_html()
            thick_in.param.watch(_thick_cb, "value")
            unit_sel.param.watch(_unit_cb, "value")

            _NO_MAT = "(Optional) — use Materials panel to add options"
            mat_opts = [_NO_MAT] + [m.name for m in self.project.materials]
            mat_val  = layer.material if layer.material in mat_opts else _NO_MAT
            mat_in = pn.widgets.Select(
                options=mat_opts, value=mat_val,
                height=26, width=280, margin=0,
                stylesheets=_SELECT_SS,
            )
            def _mat_cb(event, idx=i, _no=_NO_MAT):
                self.project.layers[idx].material = "" if event.new == _no else event.new
                self.app.autosave()
                self.app.refresh_layers_missing_params()
                # If the user is viewing an interface detail, rebuild it so the
                # same-material warning appears/disappears to reflect the new material.
                if self._selected_interface >= 0:
                    self._same_mat_warning_dismissed = False
                    self._detail.objects = [self._build_detail()]
            mat_in.param.watch(_mat_cb, "value")

            del_btn = pn.widgets.Button(name="✕", width=22, height=22, margin=0,
                                        stylesheets=_BTN_LIGHT_SS)
            del_btn.on_click(lambda e, idx=i: self._delete(idx))

            row = pn.Row(
                pn.Column(up_btn, dn_btn, width=20, margin=0, styles={"gap": "0px"}),
                dot_btn,
                name_in,
                pn.Row(thick_in, unit_sel, width=112, margin=0,
                       styles={"gap": "2px", "align-items": "center"}),
                mat_in,
                pn.Spacer(sizing_mode="stretch_width"),
                del_btn,
                margin=0,
                styles={
                    "background": bg,
                    "border-radius": "3px",
                    "padding": "2px 4px",
                    "align-items": "center",
                    "gap": "6px",
                },
                sizing_mode="stretch_width",
            )
            rows.append(row)

        add_btn = pn.widgets.Button(
            name="+ Add layer", stylesheets=_ADD_BTN_SS,
            sizing_mode="stretch_width", height=32,
            styles={"margin-top": "6px"},
        )
        def _on_add(e):
            idx = len(self.project.layers)
            self.project.layers.append(Layer(
                name=f"layer_{idx + 1}",
                thickness=1.0,
                thickness_unit="um",
                color=default_layer_color(idx),
            ))
            self._selected = idx
            self.rebuild()
        add_btn.on_click(_on_add)
        rows.append(add_btn)

        return rows

    # ── Detail panel — shared dispatcher ──────────────────────────────────────

    def _build_detail(self) -> pn.viewable.Viewable:
        if self._selected_interface >= 0:
            return self._build_interface_detail(self._selected_interface)
        if self._selected_overlay >= 0:
            return self._build_overlay_detail(self._selected_overlay)
        if self._selected >= 0:
            return self._build_layer_detail(self._selected)
        return pn.pane.HTML(
            '<div style="color:#556070;font-size:13px;padding:8px;">'
            'Click the ● dot on a layer row, or the ● dot on an overlay, '
            'to edit its properties here.<br><br>'
            'Click the narrow <span style="display:inline-block;width:10px;'
            'height:12px;background:rgba(74,158,255,0.35);border:1px solid '
            'rgba(74,158,255,0.55);border-radius:2px;vertical-align:middle;'
            'margin:0 2px;"></span> blue bar between layers in the schematic '
            'to configure heterojunction boundary conditions at that '
            'interface.</div>'
        )

    # ── Interface detail panel ────────────────────────────────────────────────

    def _find_interface(self, left: str, right: str):
        """Return the Interface model object for (left, right), or None."""
        for iface in self.project.interfaces:
            if iface.left == left and iface.right == right:
                return iface
        return None

    def _get_or_create_interface(self, left: str, right: str):
        iface = self._find_interface(left, right)
        if iface is None:
            iface = Interface(left=left, right=right, bcs=[])
            self.project.interfaces.append(iface)
        return iface

    def _get_thermionic_spec(self, iface):
        """Return the ThermionicHeterojunction BCSpec from iface.bcs, creating if absent."""
        for bc in iface.bcs:
            if bc.type == "ThermionicHeterojunction":
                return bc
        bc = InterfaceBCSpec(type="ThermionicHeterojunction", bands=[], HJBC_enhancement=1.0)
        iface.bcs.append(bc)
        return bc

    def _prune_empty_interfaces(self):
        """Remove interface entries that have no active bands in any BC spec."""
        self.project.interfaces = [
            iface for iface in self.project.interfaces
            if any(bc.bands for bc in iface.bcs)
        ]

    def _build_interface_detail(self, left_idx: int) -> pn.viewable.Viewable:
        """Build the interface configuration panel for the divider at left_idx."""
        INPUT_SS = _INPUT_SS
        layers = self.project.layers
        if left_idx < 0 or left_idx >= len(layers) - 1:
            return pn.pane.HTML(
                '<div style="color:#e74c3c;font-size:12px;padding:8px;">Invalid interface.</div>'
            )

        left_layer  = layers[left_idx]
        right_layer = layers[left_idx + 1]
        left_name   = left_layer.name
        right_name  = right_layer.name

        items: list = []

        # ── Header ────────────────────────────────────────────────────────────
        l_mat = left_layer.material  or "(no material)"
        r_mat = right_layer.material or "(no material)"
        items.append(pn.pane.HTML(
            f'<div style="font-size:13px;font-weight:700;color:#c8d6e5;">'
            f'{left_name}|{right_name} interface</div>'
            f'<div style="font-size:11px;color:#556070;margin-top:2px;">'
            f'{l_mat} / {r_mat}</div>',
            sizing_mode="stretch_width", margin=(0, 0, 6, 0),
        ))

        # ── Bands check ───────────────────────────────────────────────────────
        bands = self.project.bands
        if not bands:
            items.append(pn.pane.HTML(
                '<div style="font-size:12px;color:#e74c3c;background:rgba(231,76,60,0.1);'
                'border:1px solid rgba(231,76,60,0.4);border-radius:4px;padding:8px;">'
                '⚠ No bands defined. Visit the Bands panel to add bands before '
                'configuring interface BCs.</div>',
                sizing_mode="stretch_width", margin=(0, 0, 6, 0),
            ))
            return pn.Column(*items, sizing_mode="stretch_width")

        # ── Default-bands soft warning (dismissable) ──────────────────────────
        if not self._iface_warning_dismissed and bands == default_bands():
            warn_col = pn.Column(sizing_mode="stretch_width")
            dismiss_btn = pn.widgets.Button(
                name="✕", width=20, height=20, margin=0,
                stylesheets=["""
                    .bk-btn { background:transparent !important; border:none !important;
                        color:#c8a060 !important; font-size:12px !important;
                        cursor:pointer !important; padding:0 !important; }
                    .bk-btn:hover { color:#fff !important; }
                """],
            )
            def _dismiss(e):
                self._iface_warning_dismissed = True
                self._detail.objects = [self._build_detail()]
            dismiss_btn.on_click(_dismiss)
            warn_col.objects = [pn.Row(
                pn.pane.HTML(
                    '<div style="font-size:11px;color:#c8a060;flex:1;">'
                    '⚠ Default bands (CB + VB) are present. Visit the Bands panel '
                    'to add or change bands before applying HJBCs.</div>',
                    sizing_mode="stretch_width", margin=0,
                ),
                dismiss_btn,
                margin=0,
                styles={"align-items": "flex-start", "gap": "4px",
                        "background": "rgba(200,160,60,0.10)",
                        "border": "1px solid rgba(200,160,60,0.4)",
                        "border-radius": "4px", "padding": "6px 8px"},
                sizing_mode="stretch_width",
            )]
            items.append(warn_col)

        # ── Per-band HJBC toggles + per-band enhancement ─────────────────────
        iface    = self._find_interface(left_name, right_name)
        bc_spec  = None
        if iface:
            for bc in iface.bcs:
                if bc.type == "ThermionicHeterojunction":
                    bc_spec = bc
                    break
        enabled_bands = set(bc_spec.bands) if bc_spec else set()
        # enh_dict: band_name → float, missing keys default to 1.0
        enh_dict = dict(bc_spec.HJBC_enhancement) if bc_spec else {}

        items.append(pn.pane.HTML(
            '<div style="font-size:11px;font-weight:700;letter-spacing:0.08em;'
            'color:#8a9bb5;margin-top:4px;margin-bottom:2px;">'
            'THERMIONIC HETEROJUNCTION BC — PER BAND</div>'
            '<div style="font-size:11px;color:#556070;margin-bottom:4px;">'
            'Enhancement penalizes deviations from the BC more strongly. '
            'Suggested values: 1, 1e3, 1e6.</div>',
            sizing_mode="stretch_width", margin=0,
        ))

        def _save_state(new_enabled_bands, new_enh_dict):
            """Persist current per-band toggle + enhancement to project.interfaces."""
            if not new_enabled_bands:
                self.project.interfaces = [
                    ifc for ifc in self.project.interfaces
                    if not (ifc.left == left_name and ifc.right == right_name)
                ]
            else:
                ifc = self._get_or_create_interface(left_name, right_name)
                spec = self._get_thermionic_spec(ifc)
                spec.bands = sorted(new_enabled_bands)
                spec.HJBC_enhancement = {
                    b: new_enh_dict.get(b, 1.0) for b in new_enabled_bands
                }
            self.app.autosave()

        # Shared mutable state across all band-row closures
        state = {
            "enabled": set(enabled_bands),
            "enh": {band.name: enh_dict.get(band.name, 1.0) for band in bands},
        }

        # ── Same-material warning (reactive: shown only when ≥1 HJBC enabled) ─
        # Build the warning row up-front so _on_toggle can flip its visibility.
        _same_mat = (bool(left_layer.material) and bool(right_layer.material)
                     and left_layer.material == right_layer.material)
        if _same_mat:
            _sm_dismiss_ss = ["""
                .bk-btn { background:transparent !important; border:none !important;
                    color:#c8a060 !important; font-size:12px !important;
                    cursor:pointer !important; padding:0 !important; }
                .bk-btn:hover { color:#fff !important; }
            """]
            _sm_dismiss_btn = pn.widgets.Button(
                name="✕", width=20, height=20, margin=0,
                stylesheets=_sm_dismiss_ss,
            )
            same_mat_warn = pn.Row(
                pn.pane.HTML(
                    f'<div style="font-size:11px;color:#f0c040;flex:1;">'
                    f'⚠ Both layers use material '
                    f'<code>{left_layer.material}</code>. '
                    f'A ThermionicHeterojunction BC between identical materials '
                    f'is unusual — verify this is intentional.</div>',
                    sizing_mode="stretch_width", margin=0,
                ),
                _sm_dismiss_btn,
                margin=(4, 0, 0, 0),
                styles={"align-items": "flex-start", "gap": "4px",
                        "background": "rgba(240,192,64,0.10)",
                        "border": "1px solid rgba(240,192,64,0.35)",
                        "border-radius": "4px", "padding": "6px 8px"},
                sizing_mode="stretch_width",
                visible=False,  # shown only when ≥1 band is enabled
            )
            def _refresh_same_mat_warn():
                same_mat_warn.visible = (
                    not self._same_mat_warning_dismissed
                    and bool(state["enabled"])
                )
            def _dismiss_same_mat(e):
                self._same_mat_warning_dismissed = True
                same_mat_warn.visible = False
            _sm_dismiss_btn.on_click(_dismiss_same_mat)
        else:
            same_mat_warn = None
            def _refresh_same_mat_warn(): pass

        toggle_rows = []
        for band in bands:
            bname  = band.name
            is_on  = bname in enabled_bands
            enh_v  = enh_dict.get(bname, 1.0)

            cb = pn.widgets.Checkbox(value=is_on, margin=0, width=18, height=18)

            enh_in  = pn.widgets.TextInput(
                value=str(enh_v), placeholder="1",
                width=80, height=24, margin=0, stylesheets=INPUT_SS,
            )
            enh_err = pn.pane.HTML("", visible=False, margin=0)

            def _on_toggle(e, _bname=bname):
                if e.new:
                    state["enabled"].add(_bname)
                else:
                    state["enabled"].discard(_bname)
                _save_state(state["enabled"], state["enh"])
                _refresh_same_mat_warn()

            def _on_enh(e, _bname=bname, _err=enh_err):
                raw = e.new.strip()
                try:
                    v = float(raw)
                    if v <= 0:
                        raise ValueError
                except (ValueError, TypeError):
                    _err.object = (
                        '<span style="font-size:10px;color:#e74c3c;">'
                        '⚠ must be a positive number</span>')
                    _err.visible = True
                    return
                _err.visible = False
                state["enh"][_bname] = v
                _save_state(state["enabled"], state["enh"])

            cb.param.watch(_on_toggle, "value")
            enh_in.param.watch(_on_enh, "value")

            toggle_rows.append(pn.Column(
                pn.Row(
                    cb,
                    pn.pane.HTML(
                        f'<span style="font-size:12px;color:#c8d6e5;'
                        f'font-weight:600;">{bname}</span>',
                        width=60, height=24, margin=0,
                    ),
                    pn.pane.HTML(
                        '<span style="font-size:11px;color:#8a9bb5;'
                        'white-space:nowrap;">Enhancement (optional):</span>',
                        margin=0,
                    ),
                    enh_in,
                    margin=0,
                    styles={"align-items": "center", "gap": "6px",
                            "padding": "3px 6px",
                            "background": "rgba(74,158,255,0.05)",
                            "border-radius": "3px"},
                    sizing_mode="stretch_width",
                ),
                enh_err,
                margin=(0, 0, 2, 0),
                styles={"gap": "2px"},
                sizing_mode="stretch_width",
            ))

        items.extend(toggle_rows)

        # Add same-material warning after toggles; set initial visibility
        if same_mat_warn is not None:
            _refresh_same_mat_warn()
            items.append(same_mat_warn)

        return pn.Column(*items, sizing_mode="stretch_width")

    # ── Property editor (shared by layer + overlay detail) ────────────────────

    def _build_property_editor(
        self,
        props: dict,
        on_change,
        is_domain: bool = False,
    ) -> pn.Column:
        """Build an editable properties block for any region.

        `props` is the region's .properties dict (mutated in place).
        `on_change` is called with no args after any edit so callers can
        refresh the missing-params section and badge.
        """
        SKIP = {"poisson/static_rho"}   # doping shown separately above

        # ── Existing property rows ────────────────────────────────────────────
        prop_rows_col = pn.Column(sizing_mode="stretch_width", styles={"gap": "3px"})

        def _refresh_prop_rows():
            rows = []
            for key in [k for k in props if k not in SKIP]:
                val_d = props[key]
                cur_val = str(val_d.get("value", "")) if isinstance(val_d, dict) else str(val_d)
                cur_unit = val_d.get("unit", "") if isinstance(val_d, dict) else ""

                val_w = pn.widgets.TextInput(
                    value=cur_val, placeholder="value",
                    height=24, margin=0, stylesheets=_INPUT_SS,
                )
                unit_w = pn.widgets.TextInput(
                    value=cur_unit, placeholder="unit",
                    width=80, height=24, margin=0, stylesheets=_INPUT_SS,
                )
                del_btn = pn.widgets.Button(
                    name="✕", width=22, height=22, margin=0,
                    stylesheets=_BTN_LIGHT_SS,
                )

                def _on_val(event, k=key):
                    try:
                        props[k]["value"] = float(event.new)
                    except (ValueError, TypeError, KeyError):
                        if isinstance(props.get(k), dict):
                            props[k]["value"] = event.new
                    on_change()

                def _on_unit(event, k=key):
                    if isinstance(props.get(k), dict):
                        props[k]["unit"] = event.new
                    on_change()

                def _on_del(e, k=key):
                    props.pop(k, None)
                    _refresh_prop_rows()
                    on_change()

                val_w.param.watch(_on_val, "value")
                unit_w.param.watch(_on_unit, "value")
                del_btn.on_click(_on_del)

                key_box = pn.pane.HTML(
                    f'<div style="background:#0d1a2a;color:#8ab4d0;'
                    f'border:1px solid #1e2d3d;border-radius:3px;'
                    f'padding:2px 6px;font-family:monospace;font-size:11px;'
                    f'height:22px;line-height:22px;overflow:hidden;'
                    f'text-overflow:ellipsis;white-space:nowrap;'
                    f'box-sizing:border-box;">{key}</div>',
                    width=210, height=26, margin=0,
                )
                rows.append(pn.Column(
                    pn.Row(
                        key_box, del_btn,
                        pn.Spacer(sizing_mode="stretch_width", margin=0),
                        margin=0,
                        styles={"align-items": "center", "gap": "4px"},
                        sizing_mode="stretch_width",
                    ),
                    pn.Row(
                        val_w, unit_w,
                        margin=0,
                        styles={"align-items": "center", "gap": "4px"},
                    ),
                    margin=0,
                    styles={
                        "gap": "3px",
                        "border-left": "2px solid #1e2d3d",
                        "padding-left": "6px",
                        "margin-bottom": "4px",
                    },
                    sizing_mode="stretch_width",
                ))
            prop_rows_col.objects = rows

        _refresh_prop_rows()

        # ── Add-property inline form ──────────────────────────────────────────
        suggested = self._compute_suggested_params()
        available_sugg = {k: v for k, v in suggested.items() if k not in props}
        sugg_options = ["— select suggestion —"] + [
            f"{k}  [{v}]" for k, v in sorted(available_sugg.items())
        ]

        sugg_w = pn.widgets.Select(
            value=sugg_options[0], options=sugg_options,
            height=26, sizing_mode="stretch_width", margin=0,
            stylesheets=_SELECT_SS,
        )
        new_key_w = pn.widgets.TextInput(
            placeholder="or type key (e.g. CB/energy_level)",
            sizing_mode="stretch_width", height=26, margin=0, stylesheets=_INPUT_SS,
        )
        new_val_w = pn.widgets.TextInput(
            placeholder="value", width=90, height=26, margin=0,
            stylesheets=_INPUT_SS,
        )
        new_unit_w = pn.widgets.TextInput(
            placeholder="unit", width=72, height=26, margin=0,
            stylesheets=_INPUT_SS,
        )
        add_btn = pn.widgets.Button(
            name="Add", height=26, width=44, margin=0,
            stylesheets=["""
                .bk-btn { background:#2d4a6e !important; color:#4a9eff !important;
                    border:1px solid #4a9eff !important; border-radius:3px !important;
                    font-size:12px !important; cursor:pointer !important; }
                .bk-btn:hover { background:#3a5e8a !important; }
            """],
        )

        key_hint_pane = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)

        def _update_key_hint(key: str):
            # "vth" is canonical; "v_th" is the deprecated alias still found in
            # older project files (see fem.spatial.KEY_ALIASES).
            if key.endswith("/vth") or key.endswith("/v_th"):
                T = _project_temperature(self.project)
                key_hint_pane.object = _vth_hint_html(T)
            elif key == "temperature" and not is_domain:
                key_hint_pane.object = _TEMP_WARNING_HTML
            else:
                key_hint_pane.object = ""

        def _on_sugg(event):
            sel = event.new
            if sel == sugg_options[0]:
                key_hint_pane.object = ""
                return
            # parse "key  [unit]"
            key = sel.split("  [")[0]
            unit = sel.split("[")[1].rstrip("]") if "[" in sel else ""
            new_key_w.value = key
            new_unit_w.value = unit
            _update_key_hint(key)

        def _on_key_typed(event):
            _update_key_hint(event.new.strip())

        sugg_w.param.watch(_on_sugg, "value")
        new_key_w.param.watch(_on_key_typed, "value")

        def _on_add(e):
            key = new_key_w.value.strip()
            if not key:
                return
            try:
                val = float(new_val_w.value or "0")
            except ValueError:
                val = new_val_w.value or ""
            props[key] = {"value": val, "unit": new_unit_w.value.strip()}
            # Reset form
            sugg_w.value = sugg_options[0]
            new_key_w.value = ""
            new_val_w.value = ""
            new_unit_w.value = ""
            key_hint_pane.object = ""
            _refresh_prop_rows()
            on_change()

        add_btn.on_click(_on_add)

        return pn.Column(
            pn.pane.HTML(
                '<div style="font-size:11px;color:#556070;font-weight:600;'
                'letter-spacing:0.06em;margin-top:10px;margin-bottom:4px;">PROPERTIES</div>'
            ),
            prop_rows_col,
            pn.pane.HTML(
                '<div style="font-size:11px;color:#7a8caa;margin-top:6px;'
                'margin-bottom:3px;">+ Add property</div>',
                margin=0,
            ),
            sugg_w,
            pn.Row(
                new_key_w,
                new_val_w,
                new_unit_w,
                add_btn,
                margin=0, styles={"gap": "4px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            key_hint_pane,
            sizing_mode="stretch_width",
            styles={"gap": "3px"},
        )

    # ── Layer detail ──────────────────────────────────────────────────────────

    def _build_layer_detail(self, i: int) -> pn.viewable.Viewable:
        layers = self.project.layers
        if i >= len(layers):
            return pn.pane.HTML("")
        layer = layers[i]

        # Doping — use TextInput so we can display/accept scientific notation
        doping_d    = layer.properties.get("poisson/static_rho", {})
        doping_abs  = abs(float(doping_d.get("value", 0.0))
                         if isinstance(doping_d, dict) else 0.0)
        doping_unit = (doping_d.get("unit", "cm^-3")
                       if isinstance(doping_d, dict) else "cm^-3")
        doping_sign = ("p-type"
                       if isinstance(doping_d, dict) and
                          float(doping_d.get("value", 0.0)) < 0
                       else "n-type")

        doping_in = pn.widgets.TextInput(
            value=_fmt_sci(doping_abs), placeholder="e.g. 1e18",
            width=120, height=26, margin=0, stylesheets=_INPUT_SS,
        )
        doping_unit_w = pn.widgets.Select(
            value=doping_unit, options=["cm^-3", "m^-3"],
            width=72, height=26, margin=0, stylesheets=_SELECT_SS,
        )
        doping_sign_w = pn.widgets.RadioButtonGroup(
            value=doping_sign, options=["n-type", "p-type"],
            stylesheets=_RADIO_SS, width=140,
        )

        def _save_doping(*_):
            try:
                val = abs(float(doping_in.value or "0"))
            except ValueError:
                return
            mult = -1 if doping_sign_w.value == "p-type" else 1
            layer.properties["poisson/static_rho"] = {
                "value": mult * val, "unit": doping_unit_w.value,
            }
            self.app.autosave()

        doping_in.param.watch(_save_doping, "value")
        doping_unit_w.param.watch(_save_doping, "value")
        doping_sign_w.param.watch(_save_doping, "value")

        # Mesh
        _lbl = '<span style="font-size:11px;color:#8a9bb5;white-space:nowrap;">'
        mesh_start_w = pn.widgets.FloatInput(
            value=layer.mesh.start if layer.mesh.start is not None else 0.005,
            name="", width=80, height=26, stylesheets=_INPUT_SS,
        )
        mesh_factor_w = pn.widgets.FloatInput(
            value=layer.mesh.factor if layer.mesh.factor is not None else 1.2,
            name="", width=80, height=26, stylesheets=_INPUT_SS,
        )
        def _save_mesh(*_):
            layer.mesh.start  = mesh_start_w.value
            layer.mesh.factor = mesh_factor_w.value
            self.app.autosave()
        mesh_start_w.param.watch(_save_mesh, "value")
        mesh_factor_w.param.watch(_save_mesh, "value")

        def _mesh_row(label_text, widget, unit_text):
            return pn.Row(
                pn.pane.HTML(f'{_lbl}{label_text}</span>', margin=0),
                widget,
                pn.pane.HTML(f'<span style="font-size:11px;color:#556070;">{unit_text}</span>', margin=0),
                pn.Spacer(sizing_mode="stretch_width"),
                margin=0, styles={"align-items": "center", "gap": "6px"},
            )

        _MESH_HDR_SS = ["""
            .bk-btn { background: transparent !important; border: none !important;
                color: #556070 !important; font-size: 11px !important;
                font-weight: 600 !important; letter-spacing: 0.06em !important;
                text-align: left !important; padding: 0 !important;
                cursor: pointer !important; width: 100% !important; }
            .bk-btn:hover { color: #8a9bb5 !important; }
        """]
        mesh_toggle = pn.widgets.Button(
            name="▶ MESH OPTIONS", sizing_mode="stretch_width",
            height=20, margin=(6, 0, 2, 0), stylesheets=_MESH_HDR_SS,
        )
        mesh_content = pn.Column(
            _mesh_row("Starting mesh spacing", mesh_start_w, "µm"),
            _mesh_row("Expansion factor", mesh_factor_w, "(dimensionless)"),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#556070;margin-top:2px;">'
                'Mesh spacing grows geometrically away from interfaces.</div>',
                margin=0,
            ),
            visible=False, sizing_mode="stretch_width", styles={"gap": "6px"},
        )
        def _toggle_mesh(*_):
            mesh_content.visible = not mesh_content.visible
            mesh_toggle.name = ("▼" if mesh_content.visible else "▶") + " MESH OPTIONS"
        mesh_toggle.on_click(_toggle_mesh)
        mesh_card = pn.Column(mesh_toggle, mesh_content,
                              sizing_mode="stretch_width", styles={"gap": "0px"})

        def _on_prop_change():
            self._missing_params_pane.object = self._build_missing_params_html()
            self.app.update_layers_badge()
            self.app.autosave()

        prop_editor = self._build_property_editor(layer.properties, _on_prop_change)

        subtitle = (
            f'<span style="display:inline-block;width:9px;height:9px;'
            f'border-radius:50%;background:{layer.color};margin-right:5px;'
            f'vertical-align:middle;"></span>'
            f'<span style="color:#c8d6e5;font-weight:600;">{layer.name}</span>'
            f'<span style="color:#556070;font-size:11px;margin-left:6px;">layer</span>'
        )
        return pn.Column(
            pn.pane.HTML(f'<div style="font-size:13px;margin-bottom:6px;">{subtitle}</div>'),
            mesh_card,
            pn.pane.HTML('<div style="font-size:11px;color:#556070;font-weight:600;'
                         'letter-spacing:0.06em;margin-top:6px;margin-bottom:4px;">DOPING</div>'),
            doping_sign_w,
            pn.pane.HTML('<div style="font-size:11px;color:#556070;margin-top:4px;'
                         'margin-bottom:2px;">Concentration</div>'),
            pn.Row(doping_in, doping_unit_w,
                   styles={"gap": "6px", "align-items": "center"}),
            prop_editor,
            sizing_mode="stretch_width",
            styles={"gap": "4px"},
        )

    # ── Overlay detail ────────────────────────────────────────────────────────

    def _build_overlay_detail(self, idx: int) -> pn.viewable.Viewable:
        overlays = self.project.overlay_regions
        if idx >= len(overlays):
            return pn.pane.HTML("")
        ov = overlays[idx]

        # Extent summary for the header
        if ov.is_domain:
            extent_str = "entire domain"
        elif ov.layer_names:
            extent_str = ", ".join(ov.layer_names)
        elif ov.start is not None and ov.end is not None:
            extent_str = f"{ov.start} – {ov.end} {ov.extent_unit or 'um'}"
        else:
            extent_str = "(extent not set)"

        # ── Doping (same UI as layers) ────────────────────────────────────────
        doping_d    = ov.properties.get("poisson/static_rho", {})
        doping_abs  = abs(float(doping_d.get("value", 0.0))
                         if isinstance(doping_d, dict) else 0.0)
        doping_unit = (doping_d.get("unit", "cm^-3")
                       if isinstance(doping_d, dict) else "cm^-3")
        doping_sign = ("p-type"
                       if isinstance(doping_d, dict) and
                          float(doping_d.get("value", 0.0)) < 0
                       else "n-type")

        doping_in = pn.widgets.TextInput(
            value=_fmt_sci(doping_abs), placeholder="e.g. 1e18",
            width=120, height=26, margin=0, stylesheets=_INPUT_SS,
        )
        doping_unit_w = pn.widgets.Select(
            value=doping_unit, options=["cm^-3", "m^-3"],
            width=72, height=26, margin=0, stylesheets=_SELECT_SS,
        )
        doping_sign_w = pn.widgets.RadioButtonGroup(
            value=doping_sign, options=["n-type", "p-type"],
            stylesheets=_RADIO_SS, width=140,
        )

        def _save_doping(*_):
            try:
                val = abs(float(doping_in.value or "0"))
            except ValueError:
                return
            mult = -1 if doping_sign_w.value == "p-type" else 1
            ov.properties["poisson/static_rho"] = {
                "value": mult * val, "unit": doping_unit_w.value,
            }
            self.app.autosave()

        doping_in.param.watch(_save_doping, "value")
        doping_unit_w.param.watch(_save_doping, "value")
        doping_sign_w.param.watch(_save_doping, "value")

        non_domain_idx = sum(
            1 for o in overlays[:idx] if not o.is_domain
        )
        color = (_OV_COLORS[non_domain_idx % len(_OV_COLORS)]
                 if not ov.is_domain else "#8a9bb5")

        def _on_prop_change():
            self._missing_params_pane.object = self._build_missing_params_html()
            self.app.update_layers_badge()
            self.app.autosave()

        prop_editor = self._build_property_editor(
            ov.properties, _on_prop_change, is_domain=ov.is_domain
        )

        subtitle = (
            f'<span style="display:inline-block;width:9px;height:9px;'
            f'border-radius:50%;background:{color};margin-right:5px;'
            f'vertical-align:middle;"></span>'
            f'<span style="color:#c8d6e5;font-weight:600;">{ov.name}</span>'
            f'<span style="color:#556070;font-size:11px;margin-left:6px;">'
            f'overlay · {extent_str}</span>'
        )
        return pn.Column(
            pn.pane.HTML(f'<div style="font-size:13px;margin-bottom:6px;">{subtitle}</div>'),
            pn.pane.HTML('<div style="font-size:11px;color:#556070;font-weight:600;'
                         'letter-spacing:0.06em;margin-bottom:4px;">DOPING</div>'),
            doping_sign_w,
            pn.pane.HTML('<div style="font-size:11px;color:#556070;margin-top:4px;'
                         'margin-bottom:2px;">Concentration</div>'),
            pn.Row(doping_in, doping_unit_w,
                   styles={"gap": "6px", "align-items": "center"}),
            prop_editor,
            styles={"gap": "4px"},
        )

    # ── Overlay regions ───────────────────────────────────────────────────────

    def _delete_overlay(self, idx: int):
        del self.project.overlay_regions[idx]
        if self._selected_overlay >= len(self.project.overlay_regions):
            self._selected_overlay = -1
        self._overlay_strip.object = self._build_overlay_strip_html()
        self._overlays_inner.objects = self._build_overlays()
        self._detail.objects = [self._build_detail()]

    def _build_overlay_item(self, ov_idx: int) -> pn.viewable.Viewable:
        ov = self.project.overlay_regions[ov_idx]

        sel = (self._selected_overlay == ov_idx)
        non_domain_idx = sum(1 for o in self.project.overlay_regions[:ov_idx]
                             if not o.is_domain)
        ov_color = (_OV_COLORS[non_domain_idx % len(_OV_COLORS)]
                    if not ov.is_domain else "#8a9bb5")
        _dot_border = "2px solid #fff" if sel else "2px solid rgba(255,255,255,0.15)"
        card_border = f"1px solid {ov_color}" if sel else "1px solid #2d3748"

        # Clickable dot (same pattern as layer rows)
        dot_btn = pn.widgets.Button(name="", width=18, height=18, margin=0,
            stylesheets=[f"""
                :host {{ width:18px !important; height:18px !important;
                         display:flex !important; align-items:center !important;
                         justify-content:center !important; flex-shrink:0; }}
                .bk-btn {{ width:100% !important; height:100% !important;
                    background:{ov_color} !important; border-radius:50% !important;
                    border:{_dot_border} !important; padding:0 !important;
                    cursor:pointer !important; min-width:0 !important;
                    box-sizing:border-box !important; }}
                .bk-btn:hover {{ filter:brightness(1.35) !important; }}
            """])
        dot_btn.on_click(lambda e, idx=ov_idx: self._select_overlay(idx))

        if ov.is_domain:
            # Domain: display-only (no edit fields, no delete button)
            return pn.Row(
                dot_btn,
                pn.pane.HTML(
                    '<span style="color:#8a9bb5;font-weight:600;font-size:12px;">'
                    'domain</span>'
                    '<span style="color:#556070;font-size:11px;margin-left:8px;">'
                    'entire domain</span>',
                    sizing_mode="stretch_width", height=18, margin=0,
                ),
                margin=(0, 0, 4, 0),
                styles={"align-items": "center", "gap": "6px",
                        "padding": "4px 6px",
                        "border": card_border,
                        "border-radius": "4px",
                        "background": "rgba(26,36,53,0.7)"},
                sizing_mode="stretch_width",
            )

        # ── Editable overlay ──────────────────────────────────────────────────
        name_in = pn.widgets.TextInput(
            value=ov.name, placeholder="region name",
            height=26, sizing_mode="stretch_width", margin=0,
            stylesheets=_INPUT_SS,
        )

        has_range = ov.start is not None and ov.end is not None
        if ov.layer_names:
            type_initial = "Layers"
        elif has_range:
            type_initial = "Coordinates"
        else:
            type_initial = "Add region by..."
        # TODO (future): add "Set operation" type so users can define a new
        # overlay as the intersection / union / difference of two existing
        # overlays (e.g. intersect a "by layers" region with a "by coordinates"
        # range to trim it).  For now only "Layers" and "Coordinates" are
        # implemented; "Add region by..." is the unset placeholder.
        type_sel = pn.widgets.Select(
            value=type_initial,
            options=["Add region by...", "Layers", "Coordinates"],
            width=110, height=26, margin=0, stylesheets=_SELECT_SS,
        )

        del_btn = pn.widgets.Button(
            name="✕", width=22, height=22, margin=0, stylesheets=_BTN_LIGHT_SS,
        )
        del_btn.on_click(lambda e, idx=ov_idx: self._delete_overlay(idx))

        # ── Layers mode: MultiChoice checkbox picker ──────────────────────────
        layer_options = [l.name for l in self.project.layers]
        layer_names_w = pn.widgets.MultiChoice(
            options=layer_options,
            value=[n for n in (ov.layer_names or []) if n in layer_options],
            placeholder="Select layers…",
            sizing_mode="stretch_width",
            stylesheets=_MULTICHOICE_SS,
        )

        # ── Coordinates mode: Left [val]  Right [val] [unit] ──────────────────
        _LBL = 'style="font-size:11px;color:#8a9bb5;white-space:nowrap;"'
        start_in = pn.widgets.FloatInput(
            value=ov.start or 0.0, width=68, height=26, margin=0,
            stylesheets=_INPUT_SS,
        )
        end_in = pn.widgets.FloatInput(
            value=ov.end or 0.0, width=68, height=26, margin=0,
            stylesheets=_INPUT_SS,
        )
        range_unit = pn.widgets.Select(
            value=ov.extent_unit or "um", options=["nm", "um", "mm"],
            width=50, height=26, margin=0, stylesheets=_SELECT_SS,
        )
        # Fixed-width labels keep everything tight; trailing spacer absorbs slack
        range_row = pn.Row(
            pn.pane.HTML(f'<span {_LBL}>Left</span>',
                         width=28, height=26, margin=0),
            start_in,
            pn.pane.HTML(f'<span {_LBL}>Right</span>',
                         width=34, height=26, margin=0),
            end_in,
            range_unit,
            pn.Spacer(sizing_mode="stretch_width", margin=0),
            margin=0,
            styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )

        extent_pane = pn.Column(margin=0, sizing_mode="stretch_width")

        def _save(*_):
            clean = _sanitize_ident(name_in.value)
            if clean != name_in.value:
                name_in.value = clean   # triggers another _save, which is a no-op
            ov.name = clean
            if type_sel.value == "Layers":
                ov.layer_names = list(layer_names_w.value)
                ov.start = ov.end = None
            elif type_sel.value == "Coordinates":
                ov.start       = start_in.value
                ov.end         = end_in.value
                ov.extent_unit = range_unit.value
                ov.layer_names = []
            # "Add region by..." → leave stored extent unchanged
            self._overlay_strip.object = self._build_overlay_strip_html()
            if self._selected_overlay == ov_idx:
                self._detail.objects = [self._build_detail()]

        def _update_extent(*_):
            if type_sel.value == "Layers":
                extent_pane.objects = [layer_names_w]
            elif type_sel.value == "Coordinates":
                extent_pane.objects = [range_row]
            else:   # "Add region by..."
                extent_pane.objects = []
            _save()

        name_in.param.watch(_save, "value")
        layer_names_w.param.watch(_save, "value")
        start_in.param.watch(_save, "value")
        end_in.param.watch(_save, "value")
        range_unit.param.watch(_save, "value")
        type_sel.param.watch(_update_extent, "value")
        _update_extent()   # initialise

        # Single-row header: [dot] [type_sel (110px)] [Name: label] [name_in ↔] [del]
        header_row = pn.Row(
            dot_btn,
            type_sel,
            pn.pane.HTML(
                '<span style="font-size:11px;color:#8a9bb5;white-space:nowrap;">'
                'Name:</span>',
                width=36, height=26, margin=0,
            ),
            name_in,
            del_btn,
            margin=0,
            styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )

        return pn.Column(
            header_row,
            extent_pane,
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
            styles={
                "border": card_border,
                "border-radius": "4px",
                "padding": "4px 6px",
                "background": "rgba(26,36,53,0.7)",
            },
        )

    def _build_overlays(self) -> list:
        items = [self._build_overlay_item(idx)
                 for idx in range(len(self.project.overlay_regions))]

        add_btn = pn.widgets.Button(
            name="+ Add overlay", stylesheets=_ADD_BTN_SS,
            sizing_mode="stretch_width", height=28,
        )
        def _on_add(e):
            self.project.overlay_regions.append(
                OverlayRegion(name=f"overlay_{len(self.project.overlay_regions)}")
            )
            self._overlay_strip.object = self._build_overlay_strip_html()
            self._overlays_inner.objects = self._build_overlays()
        add_btn.on_click(_on_add)
        items.append(add_btn)
        return items

    # ── Public view ───────────────────────────────────────────────────────────

    def view(self) -> pn.Row:
        return self._layout
