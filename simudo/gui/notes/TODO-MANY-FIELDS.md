# TODO: Many Optical Fields

See full design in `gui/notes/MANY_OPTICAL_FIELDS.md`.

## Done

- [x] 1. `spatial.add_callable_rule` / `get_callable` — `simudo/fem/spatial.py`
- [x] 2. `make_alpha_from_table`, `make_alpha_from_wavelength_table`, `make_alpha_from_formula` — `simudo/physics/optical_alpha.py`
- [x] 3. Export from `simudo/physics/__init__.py`
- [x] 4. `BeerLambert` EOP — `simudo/physics/electro_optical_process.py` (accumulates into DG2)
- [x] 5. Fix `optical_properties_table` return in `materials/silicon.py`
- [x] 6. `AM15Spectrum` — `simudo/util/am15.py` + `simudo/util/data/astm_g173_03.npz`
  - ASTM G-173-03 data, 280–4000 nm, 2002 points
  - `photon_flux_integral`, `photon_flux_weighted_energy`, `equal_flux_bin_edges`, `bin_parameters`
  - CDF-based integration (no `quad` warnings, fast)
  - Exported from `simudo/util/__init__.py`

## Done (Runner)

### Runner (gui/simudo_1d_runner.py)

- [x] 7. `optical_field_sets` expansion
  - Parse `optical_field_sets:` YAML key in `_create_problemdata`
  - `_expand_field_set`: call `AM15Spectrum.bin_parameters` (or `Blackbody`),
    then `optical.easy_add_field` + `ospatial.add_BC` for each bin
  - Supported spectra: `blackbody`, `am15g`, `am15d`
  - `concentration` multiplier on flux
  - Name collision check against existing `optical_fields`

- [x] 8. YAML wiring for `BeerLambert`
  - In the EOP registration loop, detect `class: BeerLambert`
  - `_parse_alpha_spec`: supports `constant`, `energies_eV`, `wavelengths_nm`, `table_file`
  - `_register_beer_lambert_alpha`: calls `spatial.add_callable_rule` per region
  - `domain` → PRIORITY_MATERIAL (10); layer names → PRIORITY_LAYER (5)

- [x] 9. Example YAML — `gui/many_fields_example.yaml`
  - FSF/p/n structure (no IB), Si-like band parameters
  - `opt_cv` uses `BeerLambert` with inline Si-like alpha table
  - FSF alpha overridden to 0 (transparent FSF)
  - `optical_field_sets` with N=5 bins, `am15g`, E_min=1.12 eV, E_max=4.0 eV

## Remaining

### BeerLambert physics

- [x] 10. **SVR / radiative recombination in `BeerLambert`** (v2)
  - Formula: `R_rad(x) = expm1(qΔμ(x)/kT) × B_0(x)`
    where `B_0(x) = Σ_k α(E_k, x) · K·n_r²·E_k²·exp(−E_k/kT(x)) · ΔE`
    with K = 2·Ω/(h³c²), Ω = `svr_solid_angle` (default 4π sr),
    n_r = `spatial.get('refractive_index')` (UFL, spatially varying)
  - Integration over a separate energy grid (independent of the optical field bins);
    Boltzmann approximation valid for E >> kT ≈ 0.026 eV
  - `B_0` pre-accumulated into DG2 function once (static α) or each Newton step (IB);
    `expm1(qΔμ/kT)` is live UFL — Newton assembles the correct Jacobian
  - YAML: `radiative_recombination: {integration_bins: 100, E_min: ..., E_max: ...}`
    Defaults: 100 bins, [0, 4] eV (safe but wasteful; narrow toward E_gap for speed)
  - `svr_solid_angle` class attribute (default 4π), overridable per instance or via YAML
  - `svr_static_alpha = True` (set False in IBBeerLambert for state-dependent α)
  - `refractive_index` is a material-level spatial property (key `'refractive_index'`,
    dimensionless); set it in YAML under `regions: domain: properties:` or per layer.
    Example: Si at bandgap n_r ≈ 3.5; the n_r² factor accounts for photon density of
    states inside the medium (Shockley-van Roosbroeck formula).
  - **Validated** against silicon B coefficient (validate_svr_B.py):
    With Si alpha table, n_r=3.5, n_i=1.45×10¹⁰ cm⁻³ (Sze): B = 2.08×10⁻¹⁵ cm³/s
    vs literature 1.8×10⁻¹⁵ cm³/s (Trupke 2003) — ratio 1.16, within 20%.
    N=100 bins on [0,4] eV gives ~2% integration error; N=500 fully converged.
    E_min has negligible effect below 1.0 eV (sub-gap alpha is tiny); cutting at
    E_min=1.12 eV discards ~49% of B_0 due to near-gap Urbach tail contributions.
  - **TODO (runner)**: when `E_min` is not specified in YAML, the runner should
    auto-detect the lowest energy at which alpha is nonzero and use that as the
    default instead of 0 eV. This is clearly knowable when alpha is given as an
    inline table or table_file (inspect the lowest-energy nonzero entry); for a
    `constant` or formula spec it falls back to 0 eV. Avoids wasting quadrature
    points on the guaranteed-zero region below the band gap.

- [x] 11. **`IBBeerLambert` — state-dependent alpha for intermediate-band cells**
  - `α(E, x) = σ(E, x) · f_IB(x)` where `f_IB` is the live IB filling factor:
    - `f_IB = IB.u`                        when IB and regular band have same sign
    - `f_IB = IB.number_of_states - IB.u`  otherwise
  - `σ(E)` registered via `spatial.add_callable_rule(f'{name}/sigma_function', ...)`
  - Subclasses `BeerLambert` + `TrapEOPMixin`; `svr_static_alpha = False` forces B_0
    to recompute each Newton step; `enable_radiative_recombination = True` by default
  - `trap_band` identifies the IB; `dst_band`/`src_band` follow BeerLambert convention
  - Sigma helpers added to `optical_alpha.py`:
    - `make_sigma_top_hat(E_low, E_high, sigma_0)` — step function with explicit bounds
      (GUI note: suggest E_low/E_high from band energy levels as defaults)
    - `make_sigma_from_table` / `make_sigma_from_wavelength_table` — tabulated σ(E)
    - `make_sigma_from_formula` — general callable / UFL expression
  - Runner YAML: `sigma:` block mirrors the `alpha:` block of `BeerLambert`;
    adds `top_hat:` key with `E_low`, `E_high`, `value`, `unit` sub-keys
  - Pre-accumulation lag: G at Newton step k+1 uses IB state from step k —
    acceptable for weakly-coupled alpha; may need verification for strongly coupled IB

### Runner / YAML

- [x] 12. **Auto alpha wiring from material file** — *implemented via §B of the
  2026-05-21 plan (uniform `get_dict()` callable convention).* DONE 2026-05-22.
  Original proposal (detect `optical_properties_table()` by name + a
  `alpha: {from_material: true}` YAML opt-in) was replaced by: a material's
  `get_dict()` may return a dict-shaped entry (with a `'type'` tag) under the
  full callable key (e.g. `opt_cv/alpha_function`); the runner dispatcher
  builds the callable and registers it via `spatial.add_material_callable_data`.
  Covers alpha, sigma, and any future callable; no YAML flag, no name heuristic.
  - Runner: `_register_material_dict` (now takes `U`), `_build_material_callable`,
    `_build_material_dict` pass-through — `gui/simudo_1d_runner.py`.
  - Spatial: `add_material_callable_data` — `simudo/fem/spatial.py`.
  - Recognised types: `top_hat`, `energy_table`, `wavelength_table` (v1);
    `formula` (v2, passes a Python callable through).
  - Example material: `simudo/materials/ib_test_material.py`.

- [partial] 13. **`bin_edges: uniform_energy` option in `optical_field_sets`** — *GUI
  + model + YAML side DONE 2026-05-22; runner expansion still TODO.*
  - The `OpticalFieldSet` model carries `bin_edges` (`equal_flux` |
    `uniform_energy`); the GUI exposes it and it round-trips through YAML.
  - **Runner still only implements `equal_flux`** in `_expand_field_set`
    (`gui/simudo_1d_runner.py`): add a `uniform_energy` branch using
    `np.linspace(E_min, E_max, N+1)` edges with bin-midpoint representative
    energies. Until then, a `uniform_energy` set silently bins by equal flux.

### GUI

- [x] 14. **GUI panel for `optical_field_sets`** — DONE 2026-05-22.
  - `OptFieldsPanel` now has two add buttons: "+ Add monochromatic field"
    and "+ Add field set (spectrum)" — `gui/panels/opt_fields.py`.
  - Field-set card: name prefix, direction, spectrum selector
    (`am15g`/`am15d`/`blackbody` + T_source when blackbody), N bins,
    concentration, bin-edge mode, E_min/E_max. Writes/reads the
    `optical_field_sets:` YAML block (`gui/yaml_io.py`).
  - Persisted as a *set* (`OpticalFieldSet` in `model.py`), not expanded to
    N rows; the runner expands at simulation time.
  - Deferred: a live read-only preview table of the N derived bin energies
    and per-bin fluxes (the helper exists in `simudo/util/am15.py`; not yet
    wired into the panel).

- [x] 15. **GUI support for `BeerLambert` / `IBBeerLambert` EOP + alpha/sigma config**
  — DONE 2026-05-22.
  - EOP class selector already lists `BeerLambert` / `IBBeerLambert` (AST
    discovery). Added `required_spatial_callables` to both
    (`simudo/physics/electro_optical_process.py`); AST-extracted with
    inheritance resolution in `gui/panels/processes.py`
    (`_EOP_REQUIRED_CALLABLES`).
  - Process card now shows a callable (α/σ) section for classes that require
    one: a material-provided status line (✓ when a layer's material declares
    the key) plus an editable list of per-region/per-domain **top-hat** specs
    (`E_low`, `E_high` with ∞ checkbox, value, unit). A top-hat with
    `E_high=∞` is the "constant" case.
  - Model: `CallableTopHat` + `Process.callable_top_hats`
    (`gui/model.py`); YAML round-trip emits the existing `alpha:`/`sigma:`
    block shape (`gui/yaml_io.py`).
  - Missing-callable surfacing: the Layers panel "PARAMETERS NOT SET IN ANY
    REGION" warning now lists unset α/σ callables, satisfied by either an
    inline top-hat or a material that declares the key
    (`gui/panels/layers.py`, `gui/panels/mat_ast.py`).
  - `radiative_recombination` toggle already present (item 10 is done).
  - Deferred to v2: inline-table editor and CSV file picker as additional
    α/σ source modes (only `top_hat` + material-provided in v1).

### Project structure

- [x] 16. **Move `materials/` into `simudo/materials/`** — fix hardcoded paths in
  `gui/gui/app.py` (`_BUNDLED_LIB`) and `gui/si_pn_jsc_sweep.py`
  - Remaining: add `simudo/materials/__init__.py` to make it importable as a package
  - Long-term: add to `find_packages()` in `setup.py` so
    `from simudo.materials.silicon import SiliconMaterial` works without `--library-dir`

- [x] 17. **Move `gui/` into `simudo/gui/`** — hardcoded paths updated
  - Remaining: add `simudo/gui/__init__.py` to enable normal imports in tests
  - Long-term: add `console_scripts` entry point in `setup.py`
    so `simudo-run my.yaml` works after `pip install`

## Implementation plan: GUI wiring for BeerLambert / IBBeerLambert / optical_field_sets (2026-05-21)

Touches items 12, 14, 15. Driven by the design choices captured in the
session of 2026-05-21 (after the IB convergence fix).

### A. Mark required callables on EOP classes

Mirror the existing `required_spatial_params` mechanism with a new
class-level attribute on `ElectroOpticalProcess`:

```python
class BeerLambert(...):
    required_spatial_callables = {
        'alpha_function': {
            'arg':         'photon_energy',
            'arg_units':   'eV',
            'return_units': '1/cm',
            'description': 'optical absorption coefficient α(E)',
        },
    }

class IBBeerLambert(...):
    required_spatial_callables = {
        'sigma_function': {
            'arg':          'photon_energy',
            'arg_units':    'eV',
            'return_units': 'cm^2',
            'description':  'IB optical cross-section σ(E)',
        },
    }
```

- AST-detectable (same approach as `required_spatial_params`), so
  `processes.py:_discover_eop_classes` can extract it without
  executing the class. Cache as `_EOP_REQUIRED_CALLABLES`.
- Spatial key written by the runner is `f'{proc_name}/{shorthand}'`
  (e.g. `opt_cv/alpha_function` for a BeerLambert instance named
  `opt_cv`). This matches the keys the runner already uses today —
  no rename needed.

### B. Material-side provision of callables

Materials stay region-agnostic and continue to expose their data
through the existing `get_dict()` contract. The dispatcher (runner
side, today `simudo_1d_runner.py`, possibly migrated into the
material's `register` method later) is what calls
`spatial.add_callable_rule`. The material's job is just to
*describe* the callable; the dispatcher knows which region to
attach it to (the layer(s) the material is applied to).

**Convention.** A scalar entry in `get_dict()` is a `pint.Quantity`.
A callable entry is a `dict` whose `'type'` key selects the
definition form:

```python
def get_dict(self):
    d = super().get_dict()
    U = self.unit_registry

    d['opt_cv/alpha_function'] = {
        'type':        'wavelength_table',
        'data':        self.optical_properties_table()[:, :2],
        'arg_unit':    'nm',
        'value_unit':  '1/cm',
    }

    d['opt_ci/sigma_function'] = {
        'type':   'top_hat',
        'E_low':  U('0.8 eV'),
        'E_high': U('inf eV'),
        'value':  U('3e-14 cm^2'),
    }
    return d
```

Definition forms recognised in v1:

| `type`               | Required keys                                                    | v1 / v2 |
|----------------------|------------------------------------------------------------------|---------|
| `top_hat`            | `E_low`, `E_high` (`pint` energies, `inf` allowed), `value`      | v1      |
| `energy_table`       | `data` (Nx2 ndarray-like), `arg_unit` (`eV`), `value_unit`       | v1      |
| `wavelength_table`   | `data` (Nx2 ndarray-like), `arg_unit` (`nm` etc.), `value_unit`  | v1      |
| `formula`            | `func` (Python callable `E → value` with units)                  | v2      |

The dispatcher (extended `_apply_material_to_layer` /
`_register_material_rules` in the runner) walks `get_dict()` and:

- For each entry whose value is a `pint.Quantity` → call
  `spatial.add_rule(key, region, value, priority=PRIORITY_MATERIAL)`
  (existing behaviour, unchanged).
- For each entry whose value is a `dict` with a `'type'` key →
  build a callable with the existing helpers in
  `simudo/physics/optical_alpha.py`
  (`make_alpha_from_table`, `make_alpha_from_wavelength_table`,
  `make_sigma_top_hat`, …) and call
  `spatial.add_callable_rule(key, region, func, priority=PRIORITY_MATERIAL)`.

The key in the material's `get_dict()` (e.g. `opt_cv/alpha_function`)
explicitly names the EOP instance. Renaming the EOP in the GUI is
therefore a breaking change for that material's link, symmetric to
how scalar keys like `nr_top/sigma_th` work today.

**GUI side (`mat_ast.py`).** Extend `get_property_keys_for_material`
to also detect callable entries. Concretely: when walking the
`d.update({...})` / `d[...] = ...` assignments inside `get_dict`,
record an entry's key as a "callable provided" iff its value is a
`dict` literal containing a `'type'` key whose value is one of the
recognised type strings. The scanner does not need to interpret
the inner data — just the key and the type tag. Result is exposed
alongside the existing scalar keys so the layers / processes
panels can show "material X provides callable `opt_cv/alpha_function`".

**Existing `optical_properties_table`.** Materials that already
have this method are *not* automatically wired into BeerLambert.
The expectation is that maintainers of those materials will add an
explicit `'opt_cv/alpha_function'` entry to `get_dict()` that uses
`self.optical_properties_table()` as the data source (one-line
change, see the Silicon example above). Item 12's
`alpha: {from_material: true}` YAML shorthand remains as a
separate runner-side convenience and is unchanged by this plan.

### C. Process-card sub-widget for α/σ

In `processes.py:_build_row`, append a new `callable_row` after
`bands_row` (and before `trap_row`/`rad_row`), shown when
`required_spatial_callables` is non-empty. One sub-block per
required callable. Each sub-block:

- **Status indicator** showing which regions currently have the
  callable resolved:
  - Resolved via material: `✓ opt_cv/alpha_function — provided by
    material 'silicon' in layers p, n`
  - Top-hat in process card: `✓ opt_cv/alpha_function — top-hat
    (regions: I)`
  - Unset for a region the process is active in: `⚠ opt_cv/alpha_function
    not set in region 'I'`
- **Source mode selector** (v1): `from_material | top_hat` only.
  No separate "constant" — top-hat with `E_high = ∞` subsumes it.
  Inline-table editor and CSV file picker are deferred.
- **Top-hat form** (when chosen):
  - Region selector: `domain | <layer_name>` (multi-select; one
    top-hat row per chosen scope, additive)
  - `E_low` (eV), `E_high` (eV, `∞` checkbox), `value`, `unit`
  - Persisted to YAML under the EOP's `alpha:` / `sigma:` block in
    the exact shape the existing runner already accepts
    (`_parse_alpha_spec` / `_parse_sigma_spec` top_hat).
- **from_material form**: read-only display of which assigned
  materials declare the matching key, derived from the material AST
  scan in §B. No selector needed — at most one material per layer
  applies.

### D. Surface unset callables in "PARAMETERS NOT SET IN ANY REGION"

Extend `layers.py:_compute_missing_params` and
`_build_missing_params_html` to include callables alongside scalar
params. For each process and each required callable, check whether
the spatial key `<proc>/<callable>` is provided by:
1. a top-hat entry in the process card for some region the process
   is active in, OR
2. a material assigned to a layer in the process's effective extent
   that declares the matching key in its AST.

If neither, list the key in the warning table with a "set in
Processes panel or material file" hint, distinct from the existing
"set in Layers/Materials panel" hint for scalars. Refresh hooks in
`app.py:refresh_layers_missing_params` already cover edits to
processes, layers, and materials — extend the trigger list to
include the new top-hat sub-widget edits.

### E. Optical-field-sets panel

In `opt_fields.py`, add a second button next to the existing
"Add monochromatic field":

```
[ + Monochromatic ]   [ + Spectrum (N bins) ]
```

The "Spectrum" button adds a new `OpticalFieldSet` row to the
model. Sub-widgets:

- `Spectrum:` selector — `am15g | am15d | blackbody | uniform_table`
  (last one if/when item 13 lands)
- `T_source` (only when `spectrum == blackbody`)
- `concentration` (multiplier — applies to all spectra, including
  AM1.5)
- `N` bins, `E_min`, `E_max`
- `Bin edges:` `equal_flux | uniform_energy` (item 13)
- Read-only preview: a small static table showing the N derived
  representative energies and per-bin photon fluxes, regenerated
  whenever the user edits any of the inputs above.

YAML emission writes `optical_field_sets:` (already supported by
the runner from item 7). YAML import reconstructs the
`OpticalFieldSet` object as a single row rather than N expanded
fields. Monochromatic fields entered explicitly under
`optical_fields:` remain a separate list, displayed above the field
sets.

### F. Order of work

1. Add `required_spatial_callables` attribute to BeerLambert and
   IBBeerLambert in `electro_optical_process.py`. AST-extract in
   `processes.py`.
2. Wire missing-callables into the layers panel warning surface.
   At this point the user can run the existing GUI, see the
   warning, and manually edit YAML — the GUI just *tells them*
   that the EOPs are misconfigured.
3. Add the per-process top-hat sub-widget. Hand-test against
   `claude_ibbl_test.yaml` (round-trip).
4. Extend `mat_ast.py` to detect `add_callable_rule` calls in
   material files. Add the "provided by material" status to the
   process card sub-widget.
5. Add the optical-field-sets panel button + form. Round-trip
   against `many_fields_example.yaml`.

Steps 1–2 are a small, self-contained PR that gives users a real
diagnostic without committing to the full new UI. Steps 3+ are
incremental on top.

### G. Open questions deferred

- Inline-table editor (E, α) vs. CSV file picker — neither in v1.
  Likely worth a small in-panel grid editor in v2.
- Item 13 `uniform_energy` binning interacts with the field-set
  preview; safe to land independently.
- Item 12's `alpha: {from_material: true}` YAML shorthand stays
  as a runner-only path for inline YAML editing; the GUI does not
  emit it (it emits the resolved per-region top-hat block, or
  nothing when the material AST scan covers the key).

### H. Implementation status (2026-05-22)

All of §A–§E landed. Deltas from the plan as written:

- **§B materials.** Implemented as planned via `get_dict()` dict-shaped
  entries, but the GUI-side detection reuses the existing
  `mat_ast.extract_classes_from_file` / `get_property_keys_for_material`
  path (extended `_parse_value_node` to flag a dict literal with a
  recognised `'type'` tag as `is_callable`). No new `add_callable_rule`-call
  scanning was needed — materials describe callables as data, not as
  `spatial.add_callable_rule` calls, matching the corrected design.
- **§C source modes.** Shipped `top_hat` editing + a material-provided
  status line. The explicit `from_material | top_hat` *selector* was not
  needed: a material that declares the key shows the ✓ status, and the
  user adds top-hat rows when no material covers it. Per-region scoping is
  a `region` Select on each top-hat row (`domain` or a layer name).
- **§F order.** Done in one pass rather than as separate PRs.

Files touched:
- `simudo/physics/electro_optical_process.py` — `required_spatial_callables`
  on base + BeerLambert (alpha) + IBBeerLambert (sigma); plus the
  `_update_sigma_phi` no-overlap guard from the earlier session.
- `simudo/fem/spatial.py` — `add_material_callable_data`.
- `gui/simudo_1d_runner.py` — `_register_material_dict(U)`,
  `_build_material_callable`, `_build_material_dict` callable pass-through.
- `gui/panels/processes.py` — AST extraction of `required_spatial_callables`
  (`_EOP_REQUIRED_CALLABLES`, with inheritance resolution; verified it does
  NOT propagate to NonRadiativeTrap or the NonOverlappingTopHat classes),
  callable sub-widget on the process card.
- `gui/panels/mat_ast.py` — callable detection (`is_callable`,
  `callable_type`).
- `gui/panels/layers.py` — missing-callable surfacing.
- `gui/panels/opt_fields.py` — field-set add/edit UI.
- `gui/model.py` — `CallableTopHat`, `Process.callable_top_hats`,
  `OpticalFieldSet`, `Project.optical_field_sets`.
- `gui/yaml_io.py` — round-trip for both.

Tests:
- `gui/test/test_material_callables.py` — 6 host-side tests (AST detect,
  EOP callable discovery + inheritance, runner dispatch split, missing-param
  logic for inline/material/unset, refractive_index-required-when-radiative).
  Run with `conda run -n claude_simudo_gui python test/test_material_callables.py`.
- `gui/test/claude_ibbl_test.yaml` — inline-σ IBBeerLambert; converges to
  1 sun (with and without SVR).
- `gui/test/claude_ibbl_material_test.yaml` + `simudo/materials/ib_test_material.py`
  — σ provided by a material's `get_dict()`; converges to 1 sun in the
  container.
- `gui/test/claude_ibbl_roundtrip.yaml` — `claude_ibbl_test.yaml` after a
  GUI `yaml_io` save/load round-trip; used to confirm the GUI-emitted
  `sigma:` block runs unchanged through the runner.

Follow-up polish (2026-05-22, after first GUI test pass):
- Simulation panel "No optical fields defined" warning now also counts
  `optical_field_sets` (`gui/panels/simulation.py`).
- Process-card α/σ value label is derived from the callable shorthand
  (`alpha_function`→`alpha:`, `sigma_function`→`sigma:`), so BeerLambert and
  IBBeerLambert show only their own symbol (`gui/panels/processes.py`).
- Runner `_parse_alpha_spec` now accepts a `top_hat` block (it previously only
  handled `constant`/tables), matching what the GUI emits for BeerLambert
  alpha (`gui/simudo_1d_runner.py`).
- `refractive_index` is surfaced as a required parameter in the Layers
  warning whenever any optical process has radiative recombination on. Driven
  by an AST-extracted `enable_radiative_recombination` default per EOP class
  (with mixin inheritance) — `_EOP_RAD_DEFAULTS` in `gui/panels/processes.py`,
  consumed by `_any_radiative_recombination` in `gui/panels/layers.py`. [(b)]
- `get_refractive_index` raises a clear `ValueError` naming the process and the
  fix when SVR is enabled but `refractive_index` is unset, instead of a bare
  `AssertionError` (`simudo/physics/electro_optical_process.py`). [(a)]

Still TODO (tracked above): runner `uniform_energy` binning (item 13),
field-set preview table, v2 inline-table / CSV α/σ source modes.

## Notes

- `_G_function` uses DG2 (not CG2): alpha is piecewise constant per region,
  Phi is CG2, so their product is degree-2 discontinuous. DG2 is exact; CG2 would
  smear jumps at material interfaces.
