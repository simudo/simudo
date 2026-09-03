"""
Simudo GUI — Processes panel.

Each process has:
  • name
  • class (Select — discovered by AST-parsing electro_optical_process.py)
  • lower-energy band  (src_band in YAML)
  • higher-energy band (dst_band in YAML)
  • trap_band          (conditional — TrapEOPMixin subclasses only)
  • Include radiative recombination: Yes/No  (conditional — optical classes only)

EOP class discovery::

  The panel scans electro_optical_process.py at startup using ast.parse() (no
  import, no dolfin) to find all concrete subclasses of ElectroOpticalProcess
  and determine:
    - optical classes: inherit AbsorptionAndRadiativeRecombinationEOPMixin
                       -> show radiative toggle
    - trap classes:    inherit TrapEOPMixin
                       -> show trap band selector
  Cross-reference: see those mixin classes in electro_optical_process.py.

gui_hint attribute:
  Any EOP class may define a class-level string attribute:
      gui_hint = "Brief description visible in the GUI."
  The AST extractor reads this at startup and displays it as a styled note below
  the class selector whenever that class is selected. Classes without gui_hint
  show no note. This is how per-class usage instructions are added without
  modifying the GUI code.  The hint is plain text, not markup: it is escaped
  before display, so a placeholder like '<name>/alpha_function' survives.
"""

from __future__ import annotations
import ast, html, os
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import panel as pn

from simudo.gui.model import Process, CallableTopHat
from simudo.gui.panels.shared import (
    INPUT_SS, SELECT_SS, BTN_LIGHT_SS, RADIO_SS, ADD_BTN_SS, CHECKBOX_SS,
    sec_header, label,
)

if TYPE_CHECKING:
    from simudo.gui.app import SimudoApp

# ── EOP class discovery via AST ────────────────────────────────────────────────

_EOP_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "../../physics/electro_optical_process.py")
)

def _discover_eop_classes(
    path: str,
) -> Tuple[
        List[str], Set[str], Set[str],
        Dict[str, str],
        Dict[str, Dict[str, str]],
        Dict[str, Dict[str, Dict[str, str]]],
        Dict[str, bool]]:
    """AST-parse the EOP source and return metadata for all concrete subclasses.

    Returns:
        process_classes     — ordered list of concrete ElectroOpticalProcess subclass names
        optical_classes     — subset that inherit AbsorptionAndRadiativeRecombinationEOPMixin
        trap_classes        — subset that inherit TrapEOPMixin
        gui_hints           — {class_name: hint_string} for classes defining gui_hint = "..."
        required_params     — {class_name: {suffix: unit}} from required_spatial_params = {...}
        required_callables  — {class_name: {suffix: {meta_key: meta_value}}} from
                              required_spatial_callables = {...}.  Inherited from parent
                              classes within this file (e.g. IBBeerLambert overrides
                              BeerLambert's alpha_function with sigma_function); the
                              extractor honours the lexical override but DOES propagate
                              the attribute down through ancestors that don't redeclare it.
        rad_defaults        — {class_name: bool} effective default of the
                              enable_radiative_recombination class attribute (inherited
                              through ancestors; False when never declared).  Used by the
                              Layers panel to decide whether refractive_index is required.
    """
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read())

    # Collect every class definition: name → set of direct base names
    all_bases: Dict[str, Set[str]] = {}
    all_nodes: Dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            all_bases[node.name] = {b.id for b in node.bases if isinstance(b, ast.Name)}
            all_nodes[node.name] = node

    # Compute full ancestor sets transitively (within this file)
    def ancestors(name: str, seen: Set[str] | None = None) -> Set[str]:
        if seen is None:
            seen = set()
        for base in all_bases.get(name, set()):
            if base not in seen:
                seen.add(base)
                ancestors(base, seen)
        return seen

    process_classes:    List[str] = []
    optical_classes:    Set[str] = set()
    trap_classes:       Set[str] = set()
    gui_hints:          Dict[str, str] = {}
    required_params:    Dict[str, Dict[str, str]] = {}
    # Per-class direct (non-inherited) declaration of required_spatial_callables;
    # we resolve inheritance below so callers see the effective dict.
    direct_callables:   Dict[str, Dict[str, Dict[str, str]]] = {}
    # Per-class direct declaration of enable_radiative_recombination (bool).
    # Resolved through inheritance below; default False if never declared.
    direct_rad:         Dict[str, bool] = {}

    def _const_dict(d: ast.Dict) -> Dict[str, str]:
        """Return {key: value} from an ast.Dict whose entries are all Constant→Constant."""
        out: Dict[str, str] = {}
        for k, v in zip(d.keys, d.values):
            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                out[str(k.value)] = str(v.value)
        return out

    # Classify EOP subclasses (filtered to ElectroOpticalProcess descendants).
    for name, node in all_nodes.items():
        anc = ancestors(name)
        if "ElectroOpticalProcess" not in anc:
            continue
        process_classes.append(name)
        if "AbsorptionAndRadiativeRecombinationEOPMixin" in anc:
            optical_classes.add(name)
        if "TrapEOPMixin" in anc:
            trap_classes.add(name)

    # Extract class-level attributes from EVERY class in the file — including
    # mixins that are NOT ElectroOpticalProcess subclasses — so that
    # inheritance resolution below can pick up attributes declared on a mixin
    # (e.g. enable_radiative_recombination = True on
    # AbsorptionAndRadiativeRecombinationEOPMixin).
    for name, node in all_nodes.items():
        for stmt in node.body:
            if not (isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                continue
            attr = stmt.targets[0].id

            if attr == "gui_hint" and isinstance(stmt.value, ast.Constant):
                gui_hints[name] = stmt.value.value

            elif attr == "required_spatial_params" and isinstance(stmt.value, ast.Dict):
                rsp: Dict[str, str] = {}
                for k, v in zip(stmt.value.keys, stmt.value.values):
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                        rsp[k.value] = v.value
                required_params[name] = rsp

            elif attr == "required_spatial_callables" and isinstance(stmt.value, ast.Dict):
                rsc: Dict[str, Dict[str, str]] = {}
                for k, v in zip(stmt.value.keys, stmt.value.values):
                    if (isinstance(k, ast.Constant)
                            and isinstance(v, ast.Dict)):
                        rsc[str(k.value)] = _const_dict(v)
                direct_callables[name] = rsc

            elif (attr == "enable_radiative_recombination"
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, bool)):
                direct_rad[name] = stmt.value.value

    # Resolve inheritance of required_spatial_callables: a class that does NOT
    # redeclare the attribute inherits the dict from its nearest ancestor (in
    # this file) that does declare it.  This mirrors Python attribute lookup
    # without requiring imports.
    required_callables: Dict[str, Dict[str, Dict[str, str]]] = {}
    def _resolve_callables(cls_name: str) -> Dict[str, Dict[str, str]]:
        if cls_name in direct_callables:
            return direct_callables[cls_name]
        # Walk MRO in declaration order: depth-first through bases.
        for base in all_bases.get(cls_name, ()):
            if base in all_nodes:
                inherited = _resolve_callables(base)
                if inherited:
                    return inherited
        return {}

    for cls in process_classes:
        eff = _resolve_callables(cls)
        if eff:
            required_callables[cls] = eff

    # Resolve enable_radiative_recombination defaults through inheritance.
    # Use None as the "not declared anywhere" sentinel (False is a real value).
    rad_defaults: Dict[str, bool] = {}
    def _resolve_rad(cls_name: str):
        if cls_name in direct_rad:
            return direct_rad[cls_name]
        for base in all_bases.get(cls_name, ()):
            if base in all_nodes:
                r = _resolve_rad(base)
                if r is not None:
                    return r
        return None
    for cls in process_classes:
        rad_defaults[cls] = bool(_resolve_rad(cls))

    return (process_classes, optical_classes, trap_classes,
            gui_hints, required_params, required_callables, rad_defaults)


# Fallback used if the runner source tree is not found alongside the GUI.
_FALLBACK_CLASSES = ["SRHRecombination", "NonRadiativeTrap",
                      "ShockleyReadBand2BandTrap", "ShockleyReadTrap2Trap",
                      "NonOverlappingTopHatBeerLambert",
                      "NonOverlappingTopHatBeerLambertIB"]
_FALLBACK_OPTICAL  = {"NonOverlappingTopHatBeerLambert",
                       "NonOverlappingTopHatBeerLambertIB"}
_FALLBACK_TRAP     = {"NonRadiativeTrap", "NonOverlappingTopHatBeerLambertIB",
                       "ShockleyReadBand2BandTrap", "ShockleyReadTrap2Trap"}
_FALLBACK_HINTS: Dict[str, str] = {}
_FALLBACK_REQUIRED: Dict[str, Dict[str, str]] = {
    "NonOverlappingTopHatBeerLambert":   {"alpha": "1/cm"},
    "NonOverlappingTopHatBeerLambertIB": {"sigma_opt": "cm^2"},
    "SRHRecombination":    {"{dst_band}/tau": "s", "{src_band}/tau": "s", "energy_level": "eV"},
    "NonRadiativeTrap":    {"sigma_th": "cm^2", "vth": "cm/s"},
    "ShockleyReadBand2BandTrap": {"{dst_band}/tau": "s", "{dst_band}/capture_rate": "cm^3/s"},
    "ShockleyReadTrap2Trap":     {"{dst_band}/capture_rate": "cm^3/s"},
}
_FALLBACK_CALLABLES: Dict[str, Dict[str, Dict[str, str]]] = {
    "BeerLambert": {"alpha_function": {
        "arg_units": "eV", "return_units": "1/cm",
        "description": "optical absorption coefficient α(E)"}},
    "IBBeerLambert": {"sigma_function": {
        "arg_units": "eV", "return_units": "cm^2",
        "description": "IB optical cross-section σ(E)"}},
}
_FALLBACK_RAD_DEFAULTS: Dict[str, bool] = {
    "BeerLambert": False,                       # overrides mixin default to False
    "IBBeerLambert": True,
    "NonOverlappingTopHatBeerLambert": True,    # inherits mixin default True
    "NonOverlappingTopHatBeerLambertIB": True,
}

if os.path.exists(_EOP_FILE):
    (_PROCESS_CLASSES, _OPTICAL_CLASSES, _TRAP_CLASSES,
     _PROCESS_HINTS, _EOP_REQUIRED_PARAMS, _EOP_REQUIRED_CALLABLES,
     _EOP_RAD_DEFAULTS) = \
        _discover_eop_classes(_EOP_FILE)
else:
    _PROCESS_CLASSES        = _FALLBACK_CLASSES
    _OPTICAL_CLASSES        = _FALLBACK_OPTICAL
    _TRAP_CLASSES           = _FALLBACK_TRAP
    _PROCESS_HINTS          = _FALLBACK_HINTS
    _EOP_REQUIRED_PARAMS    = _FALLBACK_REQUIRED
    _EOP_REQUIRED_CALLABLES = _FALLBACK_CALLABLES
    _EOP_RAD_DEFAULTS       = _FALLBACK_RAD_DEFAULTS

_RAD_OPTIONS = ["Yes", "No"]

# Units offered for the SVR quadrature bounds.  Energy only: the window
# is defined as a fixed *width* above the absorption threshold, which has
# no meaning in wavelength, where the mapping is inverse.
_SVR_ENERGY_UNITS = ["eV", "meV", "J"]

# Short prefix used for auto-generating process names from class names.
_CLASS_PREFIX: dict[str, str] = {
    "SRHRecombination":                   "srh",
    "NonRadiativeTrap":                   "nr",
    "ShockleyReadBand2BandTrap":          "b2b",
    "ShockleyReadTrap2Trap":              "t2t",
    "NonOverlappingTopHatBeerLambert":    "opt",
    "NonOverlappingTopHatBeerLambertIB":  "opt_ib",
}


def _unique_process_name(base: str, existing_names: list[str]) -> str:
    """Return *base* if it is not in *existing_names*, else *base_2*, *base_3*, …

    Strips any trailing ``_<digits>`` suffix from *base* before generating
    candidates, so calling with ``base='srh_2'`` still produces ``srh_3``
    rather than ``srh_2_2``.
    """
    import re as _re
    root = _re.sub(r"_\d+$", "", base)
    if root not in existing_names:
        return root
    n = 2
    while True:
        candidate = f"{root}_{n}"
        if candidate not in existing_names:
            return candidate
        n += 1

# ── Stylesheet for hint pane ───────────────────────────────────────────────────
_HINT_STYLE = (
    "font-size:11px;color:#8ab4d0;background:#0d1a2a;"
    "border-left:2px solid #2d4a6e;padding:5px 8px;"
    "border-radius:0 3px 3px 0;margin-bottom:2px;"
)


def _hint_html(text: str) -> str:
    """Wrap a gui_hint for display.  Hints are plain text written by physicists,
    not markup: escape them, or a placeholder like '<name>/generation' is parsed
    as a tag and silently disappears from the pane."""
    return f'<div style="{_HINT_STYLE}">{html.escape(text)}</div>' if text else ""


class ProcessesPanel:
    def __init__(self, app: "SimudoApp"):
        self.app = app

        self._list_col = pn.Column(sizing_mode="stretch_width", styles={"gap": "2px"})
        self._add_btn = pn.widgets.Button(
            name="+ Add process", stylesheets=ADD_BTN_SS,
            sizing_mode="stretch_width", height=30, margin=(4, 0, 0, 0),
        )
        self._add_btn.on_click(lambda e: self._add_process())

        self._view = pn.Column(
            sec_header("PROCESSES"),
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
            self._build_row(i, p)
            for i, p in enumerate(self.app.project.processes)
        ]

    def _add_process(self):
        procs = self.app.project.processes
        band_names = self.app.project.band_names
        src = band_names[1] if len(band_names) > 1 else (band_names[0] if band_names else "VB")
        dst = band_names[0] if band_names else "CB"
        default_cls = "SRHRecombination"
        base = _CLASS_PREFIX.get(default_cls, "proc")
        existing = [p.name for p in procs]
        name = _unique_process_name(base, existing)
        procs.append(Process(name=name, cls=default_cls,
                              src_band=src, dst_band=dst,
                              radiative_recombination=None))
        self.app.autosave()
        self.app.refresh_layers_missing_params()
        self._refresh()

    def _delete_process(self, idx: int):
        self.app.project.processes.pop(idx)
        self.app.autosave()
        self.app.refresh_layers_missing_params()
        self._refresh()

    def _ib_band_name(self) -> str | None:
        """Return the name of the unique Sharp Intermediate Band, or None if 0 or >1 exist."""
        ib_bands = [b for b in self.app.project.bands if b.type == "intermediate"]
        return ib_bands[0].name if len(ib_bands) == 1 else None

    def _band_type(self, name: str) -> str | None:
        """Return the type string of the band with the given name, or None if not found."""
        for b in self.app.project.bands:
            if b.name == name:
                return b.type
        return None

    def _material_provided_callable_keys(self) -> set:
        """Full spatial keys (e.g. 'opt_cv/alpha_function') that are provided
        by a material applied to at least one layer.

        Reuses the mat_ast scanner, which now flags callable get_dict entries
        in the returned key set."""
        try:
            applied_mats = {l.material for l in self.app.project.layers if l.material}
            if not applied_mats:
                return set()
            lib_dirs = self.app.get_library_dirs()
            from simudo.gui.panels.mat_ast import get_property_keys_for_material
            keys: set = set()
            for mat in self.app.project.materials:
                if mat.name in applied_mats:
                    keys.update(get_property_keys_for_material(mat, lib_dirs))
            return keys
        except Exception:
            return set()

    def _region_options(self) -> list:
        """Region scopes a top-hat can target: 'domain' plus every layer name."""
        names = [l.name for l in self.app.project.layers if l.name]
        return ["domain"] + names

    def _build_callable_section(self, idx: int, proc: Process) -> pn.Column:
        """Build the alpha/sigma callable editor for a process card.

        Empty (invisible) for classes that declare no required callables.
        For each required callable shorthand, shows:
          - a header with the description,
          - whether a material already provides the key,
          - an editable list of per-region top-hat specs,
          - an '+ add region' button.
        """
        rsc = _EOP_REQUIRED_CALLABLES.get(proc.cls, {})
        container = pn.Column(margin=0, sizing_mode="stretch_width",
                              styles={"gap": "3px"})
        if not rsc:
            container.visible = False
            return container

        mat_keys = self._material_provided_callable_keys()
        procs = self.app.project.processes

        for shorthand, meta in rsc.items():
            ret_unit = meta.get("return_units", "1/cm")
            desc     = meta.get("description", shorthand)
            full_key = f"{proc.name}/{shorthand}"
            by_material = full_key in mat_keys
            # AST-derived value label: the callable's own shorthand minus the
            # conventional '_function' suffix (e.g. 'alpha_function' → 'alpha',
            # 'sigma_function' → 'sigma').  Extends automatically to new EOPs.
            val_label = (shorthand[:-len("_function")]
                         if shorthand.endswith("_function") else shorthand)

            # Ensure the model has a list to mutate for this shorthand.
            rows_model = proc.callable_top_hats.setdefault(shorthand, [])

            # Header + material status line.
            status_html = (
                f'<span style="color:#3fb98a;">✓ provided by a material '
                f'({full_key})</span>'
                if by_material else
                f'<span style="color:#8a9bb5;">set via top-hat below, or by a '
                f'material providing <code>{full_key}</code></span>'
            )
            header = pn.pane.HTML(
                f'<div style="font-size:11px;color:#8ab4d0;margin-top:3px;">'
                f'<b>{desc}</b><br>{status_html}</div>',
                sizing_mode="stretch_width", margin=0,
            )

            rows_col = pn.Column(margin=0, sizing_mode="stretch_width",
                                 styles={"gap": "2px"})

            def _make_row_widgets(row: CallableTopHat, shand=shorthand):
                region_sel = pn.widgets.Select(
                    value=(row.region if row.region in self._region_options()
                           else self._region_options()[0]),
                    options=self._region_options(),
                    width=70, height=24, margin=0, stylesheets=SELECT_SS)
                e_low_in = pn.widgets.FloatInput(
                    value=row.E_low, width=58, height=24, margin=0,
                    stylesheets=INPUT_SS)
                inf = row.E_high is None
                e_high_in = pn.widgets.FloatInput(
                    value=(0.0 if inf else row.E_high), width=58, height=24,
                    margin=0, stylesheets=INPUT_SS, disabled=inf)
                inf_chk = pn.widgets.Checkbox(
                    value=inf, name="∞", margin=(4, 0, 0, 0),
                    stylesheets=CHECKBOX_SS, width=34)
                val_in = pn.widgets.FloatInput(
                    value=row.value, width=72, height=24, margin=0,
                    stylesheets=INPUT_SS)
                unit_in = pn.widgets.TextInput(
                    value=row.unit or ret_unit, width=64, height=24, margin=0,
                    stylesheets=INPUT_SS)
                del_btn = pn.widgets.Button(
                    name="✕", width=22, height=24, margin=0,
                    stylesheets=BTN_LIGHT_SS)

                def _commit(_=None, r=row, rs=region_sel, el=e_low_in,
                            eh=e_high_in, ic=inf_chk, vi=val_in, ui=unit_in):
                    r.region = rs.value
                    r.E_low = float(el.value or 0.0)
                    r.E_high = None if ic.value else float(eh.value or 0.0)
                    r.value = float(vi.value or 0.0)
                    r.unit = ui.value or ret_unit
                    self.app.autosave()
                    self.app.refresh_layers_missing_params()

                def _on_inf(e, eh=e_high_in):
                    eh.disabled = bool(e.new)
                    _commit()

                region_sel.param.watch(_commit, "value")
                e_low_in.param.watch(_commit, "value")
                e_high_in.param.watch(_commit, "value")
                val_in.param.watch(_commit, "value")
                unit_in.param.watch(_commit, "value")
                inf_chk.param.watch(_on_inf, "value")

                def _delete(_=None, r=row, shd=shand):
                    lst = procs[idx].callable_top_hats.get(shd, [])
                    if r in lst:
                        lst.remove(r)
                    self.app.autosave()
                    self.app.refresh_layers_missing_params()
                    self._refresh()
                del_btn.on_click(_delete)

                return pn.Row(
                    label("E≥", 22), e_low_in, label("E<", 22), e_high_in,
                    inf_chk, label(f"{val_label}:", 44), val_in, unit_in,
                    label("in", 16), region_sel, del_btn,
                    margin=0, styles={"gap": "3px", "align-items": "center"},
                    sizing_mode="stretch_width")

            rows_col.objects = [_make_row_widgets(r) for r in rows_model]

            add_btn = pn.widgets.Button(
                name="+ add top-hat region", stylesheets=ADD_BTN_SS,
                height=24, margin=(2, 0, 0, 0), sizing_mode="stretch_width")

            def _add(_=None, shd=shorthand, ru=ret_unit):
                opts = self._region_options()
                default_region = opts[1] if len(opts) > 1 else opts[0]
                procs[idx].callable_top_hats.setdefault(shd, []).append(
                    CallableTopHat(region=default_region, E_low=0.0,
                                   E_high=None, value=0.0, unit=ru))
                self.app.autosave()
                self.app.refresh_layers_missing_params()
                self._refresh()
            add_btn.on_click(_add)

            container.append(header)
            container.append(rows_col)
            container.append(add_btn)

        return container

    def _build_row(self, idx: int, proc: Process) -> pn.Column:
        procs      = self.app.project.processes
        band_names = self.app.project.band_names or ["CB", "VB"]

        # ── Name ─────────────────────────────────────────────────────────────
        name_in = pn.widgets.TextInput(
            value=proc.name, placeholder="name",
            height=26, margin=0, stylesheets=INPUT_SS,
            sizing_mode="stretch_width",
        )
        name_error = pn.pane.HTML(
            '<div style="font-size:11px;color:#e07b39;padding:1px 0 1px 4px;">'
            '⚠ Process name must be unique.</div>',
            sizing_mode="stretch_width", margin=0, visible=False,
        )

        def _on_name(e, i=idx):
            new_name = e.new.strip()
            if not new_name:
                # Blank — revert display to current model value without saving
                name_error.visible = False
                name_in.value = procs[i].name
                return
            other_names = [p.name for j, p in enumerate(procs) if j != i]
            if new_name in other_names:
                name_error.visible = True
                # Do not commit the duplicate to the model
                return
            name_error.visible = False
            procs[i].name = new_name
            self.app.autosave()
            self.app.refresh_layers_missing_params()
        name_in.param.watch(_on_name, "value")

        # ── Class ─────────────────────────────────────────────────────────────
        cls_val = proc.cls if proc.cls in _PROCESS_CLASSES else _PROCESS_CLASSES[0]
        cls_sel = pn.widgets.Select(
            value=cls_val, options=_PROCESS_CLASSES,
            height=26, margin=0, stylesheets=SELECT_SS,
            sizing_mode="stretch_width",
        )

        # ── Delete ────────────────────────────────────────────────────────────
        del_btn = pn.widgets.Button(name="✕", width=24, height=26, margin=0,
                                     stylesheets=BTN_LIGHT_SS)
        del_btn.on_click(lambda e, i=idx: self._delete_process(i))

        header = pn.Column(
            pn.Row(
                name_in, cls_sel, del_btn,
                margin=0, styles={"gap": "4px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            name_error,
            margin=0, sizing_mode="stretch_width",
        )

        # ── Hint pane ─────────────────────────────────────────────────────────
        hint_text = _PROCESS_HINTS.get(cls_val, "")
        hint_pane = pn.pane.HTML(
            _hint_html(hint_text),
            sizing_mode="stretch_width", margin=0,
            visible=bool(hint_text),
        )

        # ── Band selectors ────────────────────────────────────────────────────
        def _safe_band(val):
            return val if val in band_names else (band_names[0] if band_names else "")

        src_sel = pn.widgets.Select(
            value=_safe_band(proc.src_band), options=band_names,
            height=26, width=90, margin=0, stylesheets=SELECT_SS,
        )
        dst_sel = pn.widgets.Select(
            value=_safe_band(proc.dst_band), options=band_names,
            height=26, width=90, margin=0, stylesheets=SELECT_SS,
        )
        bands_row = pn.Row(
            label("Lower E band:", 72), src_sel,
            label("Higher E band:", 80), dst_sel,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
        )

        # ── Trap band (conditional) ───────────────────────────────────────────
        trap_opts = ["(none)"] + band_names
        ib_name = self._ib_band_name()

        # Auto-fill trap band on initial build: if this is a trap class and exactly
        # one IB band exists, pre-select it when the stored trap_band is unset.
        stored_trap = proc.trap_band
        if stored_trap is None and proc.cls in _TRAP_CLASSES and ib_name is not None:
            stored_trap = ib_name
            procs[idx].trap_band = ib_name  # persist immediately

        trap_val = stored_trap if stored_trap in band_names else "(none)"
        trap_sel = pn.widgets.Select(
            value=trap_val, options=trap_opts,
            height=26, width=100, margin=0, stylesheets=SELECT_SS,
        )

        # Error indicators for the trap band
        def _trap_is_valid(tv: str) -> bool:
            """Trap band should be a Sharp Intermediate Band (or none)."""
            return tv == "(none)" or self._band_type(tv) == "intermediate"

        def _trap_matches_bands(tv: str, sv: str, dv: str) -> bool:
            """For trap classes, trap band must equal src or dst band."""
            return tv == "(none)" or tv == sv or tv == dv

        trap_error_type = pn.pane.HTML(
            '<div style="font-size:11px;color:#e07b39;padding:1px 0 1px 4px;">'
            '⚠ Trap band should be a Sharp Intermediate Band.</div>',
            sizing_mode="stretch_width", margin=0,
            visible=(proc.cls in _TRAP_CLASSES and not _trap_is_valid(trap_val)),
        )
        trap_error_match = pn.pane.HTML(
            '<div style="font-size:11px;color:#e07b39;padding:1px 0 1px 4px;">'
            '⚠ Trap band must equal either the Lower E or Higher E band.</div>',
            sizing_mode="stretch_width", margin=0,
            visible=(proc.cls in _TRAP_CLASSES
                     and not _trap_matches_bands(trap_val,
                                                 _safe_band(proc.src_band),
                                                 _safe_band(proc.dst_band))),
        )

        trap_row = pn.Column(
            pn.Row(
                label("Trap band:", 64), trap_sel,
                margin=0, styles={"gap": "4px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            trap_error_type,
            trap_error_match,
            sizing_mode="stretch_width",
            margin=0,
            visible=(proc.cls in _TRAP_CLASSES),
        )

        # ── Radiative recombination (optical classes only) ────────────────────
        if proc.radiative_recombination is False:
            rad_val = "No"
        else:
            rad_val = "Yes"   # True or None both map to Yes

        rad_rbg = pn.widgets.RadioButtonGroup(
            options=_RAD_OPTIONS, value=rad_val,
            height=26, margin=0, stylesheets=RADIO_SS,
        )
        rad_row = pn.Row(
            label("Include radiative recomb.:", 162), rad_rbg,
            margin=0, styles={"gap": "4px", "align-items": "center"},
            sizing_mode="stretch_width",
            visible=(proc.cls in _OPTICAL_CLASSES),
        )

        # ── SVR quadrature window ─────────────────────────────────────────────
        # The emission integrand carries a blackbody weight exp(-E/kT) that peaks
        # at the absorption threshold, so *where* this window starts matters as
        # much as how wide it is: at the runner's 0-4 eV default the threshold
        # falls partway through a bin, and that bin -- the largest in the sum --
        # is counted or dropped whole, worth tens of percent either way.  Start
        # at the threshold and run 23 kT (0.6 eV at 300 K) and a silicon cell is
        # accurate to ~0.2% at the same bin count.  Hence E_max tracks E_min
        # automatically, at the 300 K width, which the user must widen if the
        # device is hotter.
        svr_Emin_in = pn.widgets.FloatInput(
            value=proc.svr_E_min, step=0.01, start=0, height=26, width=70,
            margin=0, stylesheets=INPUT_SS, placeholder="auto",
        )
        svr_Emin_unit = pn.widgets.Select(
            options=_SVR_ENERGY_UNITS, height=26, width=58, margin=0,
            value=(proc.svr_E_min_unit if proc.svr_E_min_unit in _SVR_ENERGY_UNITS
                   else "eV"),
            stylesheets=SELECT_SS,
        )
        svr_Emax_in = pn.widgets.FloatInput(
            value=proc.svr_E_max, step=0.01, start=0, height=26, width=70,
            margin=0, stylesheets=INPUT_SS, placeholder="auto",
        )
        svr_Emax_unit = pn.widgets.Select(
            options=_SVR_ENERGY_UNITS, height=26, width=58, margin=0,
            value=(proc.svr_E_max_unit if proc.svr_E_max_unit in _SVR_ENERGY_UNITS
                   else "eV"),
            stylesheets=SELECT_SS,
        )
        svr_hint = pn.pane.HTML(
            _hint_html(
                "Energy window for the emission integral, split into 100 bins. "
                "Set the lower bound exactly at the absorption threshold (the "
                "band gap, or the bottom of the top hat) — emission peaks right "
                "there, so a bin straddling it can throw the rate off by tens "
                "of percent. The upper bound then follows automatically, 0.6 eV "
                "higher; nothing above that contributes at 300 K. Above 300 K "
                "the emission spectrum is broader, so raise the upper bound (it "
                "scales as kT: ~23 kT above the threshold). Leaving both blank "
                "uses the 0-4 eV default, which is not aligned to any "
                "threshold."),
            sizing_mode="stretch_width", margin=0,
        )
        svr_row = pn.Column(
            pn.Row(
                label("Emission integral:", 162),
                label("from", 26), svr_Emin_in, svr_Emin_unit,
                label("to", 16), svr_Emax_in, svr_Emax_unit,
                margin=0, styles={"gap": "4px", "align-items": "center"},
                sizing_mode="stretch_width",
            ),
            svr_hint,
            margin=0, sizing_mode="stretch_width",
            visible=(proc.cls in _OPTICAL_CLASSES
                     and proc.radiative_recombination is not False),
        )

        # Width of the default window above the absorption threshold, at 300 K.
        # That is ~23 kT, where the blackbody weight has fallen by e^-23 and the
        # tail left out is ~1e-8 of the integral.  Fixed rather than computed:
        # temperature is a spatial rule and need not be uniform, so the GUI has
        # no single T to scale by.  The hint tells the user to widen it if the
        # device runs hot.
        _SVR_WINDOW_EV = 0.6

        def _svr_default_max(v_min: float, unit: str) -> float:
            """E_max implied by E_min, expressed in `unit`."""
            per_eV = {"eV": 1.0, "meV": 1000.0, "J": 1.602176634e-19}.get(unit)
            if per_eV is None:      # unfamiliar unit: leave it to the user
                return None
            return round(v_min + _SVR_WINDOW_EV * per_eV, 6)

        # ── Callable (alpha / sigma) section ──────────────────────────────────
        callable_section = self._build_callable_section(idx, proc)

        # ── Callbacks ─────────────────────────────────────────────────────────
        def _refresh_trap_errors(tv, sv, dv, is_trap):
            trap_error_type.visible  = is_trap and not _trap_is_valid(tv)
            trap_error_match.visible = is_trap and not _trap_matches_bands(tv, sv, dv)

        def _on_cls(event, i=idx):
            new_cls = event.new
            procs[i].cls = new_cls

            # Suggest a unique name based on the new class prefix if the current
            # name still matches the old auto-generated pattern (safe to replace).
            base = _CLASS_PREFIX.get(new_cls, "proc")
            other_names = [p.name for j, p in enumerate(procs) if j != i]
            suggested = _unique_process_name(base, other_names)
            if name_in.value != suggested:
                # Only update if no duplicate error is currently shown —
                # don't override a name the user is actively editing.
                if not name_error.visible:
                    procs[i].name = suggested
                    name_in.value = suggested

            # Update hint pane
            h = _PROCESS_HINTS.get(new_cls, "")
            hint_pane.object  = _hint_html(h)
            hint_pane.visible = bool(h)

            # Show/hide conditional rows
            is_trap    = new_cls in _TRAP_CLASSES
            is_optical = new_cls in _OPTICAL_CLASSES
            trap_row.visible = is_trap
            rad_row.visible  = is_optical
            svr_row.visible  = is_optical and rad_rbg.value == "Yes"

            # When switching to a trap class, always apply smart defaults if an IB exists:
            #   lower E band = IB, higher E band = first non-IB, trap band = IB.
            # (No `is None` guard — a class switch is an intentional action that
            # should reset to sensible defaults regardless of prior state.)
            if is_trap:
                ib = self._ib_band_name()
                if ib is not None:
                    procs[i].trap_band = ib
                    trap_sel.value = ib
                    procs[i].src_band = ib
                    src_sel.value = ib
                    non_ib = next((n for n in band_names if n != ib), None)
                    if non_ib:
                        procs[i].dst_band = non_ib
                        dst_sel.value = non_ib

            _refresh_trap_errors(trap_sel.value, src_sel.value, dst_sel.value, is_trap)
            self.app.autosave()
            self.app.refresh_layers_missing_params()
            # The callable (alpha/sigma) section differs per class, so rebuild
            # the whole card list to show/hide and repopulate it.
            self._refresh()
        cls_sel.param.watch(_on_cls, "value")

        def _on_src(e, i=idx):
            procs[i].src_band = e.new
            _refresh_trap_errors(trap_sel.value, e.new, dst_sel.value,
                                  procs[i].cls in _TRAP_CLASSES)
            self.app.autosave()

        def _on_dst(e, i=idx):
            procs[i].dst_band = e.new
            _refresh_trap_errors(trap_sel.value, src_sel.value, e.new,
                                  procs[i].cls in _TRAP_CLASSES)
            self.app.autosave()

        src_sel.param.watch(_on_src, "value")
        dst_sel.param.watch(_on_dst, "value")

        def _on_trap(e, i=idx):
            procs[i].trap_band = None if e.new == "(none)" else e.new
            _refresh_trap_errors(e.new, src_sel.value, dst_sel.value,
                                  procs[i].cls in _TRAP_CLASSES)
            self.app.autosave()
        trap_sel.param.watch(_on_trap, "value")

        def _on_rad(e, i=idx):
            procs[i].radiative_recombination = (e.new == "Yes")
            svr_row.visible = (e.new == "Yes")
            self.app.autosave()
        rad_rbg.param.watch(_on_rad, "value")

        def _on_svr_emin(e, i=idx):
            procs[i].svr_E_min = e.new
            # Setting a lower bound fills in the upper bound, unless the user
            # has already chosen one themselves.
            if e.new is not None and procs[i].svr_E_max is None:
                implied = _svr_default_max(e.new, svr_Emin_unit.value)
                if implied is not None:
                    svr_Emax_unit.value = svr_Emin_unit.value
                    svr_Emax_in.value = implied   # its watcher stores it
            self.app.autosave()
        svr_Emin_in.param.watch(_on_svr_emin, "value")

        def _on_svr_emin_unit(e, i=idx):
            procs[i].svr_E_min_unit = e.new
            self.app.autosave()
        svr_Emin_unit.param.watch(_on_svr_emin_unit, "value")

        def _on_svr_emax(e, i=idx):
            procs[i].svr_E_max = e.new
            self.app.autosave()
        svr_Emax_in.param.watch(_on_svr_emax, "value")

        def _on_svr_emax_unit(e, i=idx):
            procs[i].svr_E_max_unit = e.new
            self.app.autosave()
        svr_Emax_unit.param.watch(_on_svr_emax_unit, "value")

        return pn.Column(
            header, hint_pane, bands_row, trap_row, rad_row, svr_row,
            callable_section,
            sizing_mode="stretch_width",
            styles={
                "background": "#1a2435",
                "border": "1px solid #2d3748",
                "border-radius": "4px",
                "padding": "8px",
                "gap": "4px",
            },
        )
