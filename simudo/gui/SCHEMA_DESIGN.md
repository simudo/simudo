# Simudo GUI — Project Schema Design Notes
*Working document — April 2026*

This records key design decisions before writing the full schema.
It is intentionally not yet a complete specification.

---

## Fundamental Objects

A simulation is built from two kinds of fundamental objects: **regions** and **bands**.
Everything else (materials, processes, boundary conditions, optical fields) is defined
in terms of these.

---

## Regions

### Layers vs. overlay regions

The mesher requires a set of **layers**: disjoint regions whose union is the complete
domain. Each layer has its own mesh parameters and typically a base material.

On top of the layer structure, **overlay regions** (or subtraction regions) can be
defined by specifying coordinate extents. These do not need to align with layer
boundaries and can span multiple layers or sit inside one.

Example:
```
Layer:    emitter [0, 0.2]   absorber [0.2, 0.5]   base [0.5, 1.0]
Overlay:  degraded [0.1, 0.3]   (spans part of emitter + part of absorber)
          high_dope [0.0, 0.05] (interior sub-region of emitter)
```

Parameters (like `SRH/CB/tau`) can be specified on any region. When multiple regions
overlap, Simudo's priority system determines which value wins — higher-priority rules
override lower ones.

### The `domain` overlay region

There is always a built-in overlay region named **`domain`** that covers the entire
simulation domain. It is created automatically and cannot be deleted. Its priority
sits between material file defaults and per-layer settings:

```
material file defaults  <  domain  <  per-layer regions  <  other overlay regions
```

This lets `domain` act as a global fallback for any property not set per-layer (e.g.
a single `SRH/CB/tau` that applies everywhere unless a layer overrides it), while
still being overridden by any more-specific region.

All other overlay regions default to a priority higher than per-layer settings, so
they function as local overrides.

### Mesh parameter display names

The internal YAML / runner names are terse; the GUI uses friendlier labels:

| YAML key | GUI label |
|---|---|
| `mesh.max_edge_length` | Maximum mesh spacing (global) |
| `mesh.dimension` | Mesh type: 1 = true-1D interval mesh (default), 2 = 2D strip |
| `layer.mesh.start` | Starting mesh spacing |
| `layer.mesh.factor` | Mesh spacing expansion factor |

`max_edge_length` is shown in the top-level Mesh section of the sidebar.
Per-layer `start` and `factor` are in a collapsible "Mesh" subsection inside
each layer's detail panel, hidden by default.

### What the schema must encode per region

- Name
- Extent — one of:
  - A list of layer names (union of those layers' `CellRegion` objects) — **implemented in runner**
  - A coordinate range `{start: X, end: Y}` in mesh units — **planned; not yet implemented in runner**
    The GUI must support coordinate-range extents. When this is added to the GUI,
    the runner must be updated simultaneously to implement the coordinate-range
    path in `_create_problemdata` (currently raises `NotImplementedError`).
    Coordinate ranges do not need to align with layer boundaries and can span
    multiple layers or sit entirely within one.
- Whether it is a layer (contributes to the mesher) or an overlay
- Mesh parameters (layers only): `type`, `start`, `factor`
- Material reference (optional shorthand for loading a set of property defaults)
- Per-property overrides: any `key: {value, unit}` or advanced expression string

### Region set operations

For advanced use, regions can be defined by set operations on other named regions
using the `CellRegion` algebra (`|`, `&`, `-`, `^`). This is not needed for the
common case but should not be architecturally excluded.

**Safe dispatch without `eval`:** store the operation as a structured list in YAML:

```yaml
regions:
  - name: degraded_outside_emitter
    definition: [degraded, emitter, subtract]   # R = degraded - emitter
```

The runner builds a lookup dict of all named `CellRegion` objects, then dispatches:

```python
OPERATORS = {
    'union':     CellRegion.__or__,
    'intersect': CellRegion.__and__,
    'subtract':  CellRegion.__sub__,
    'xor':       CellRegion.__xor__,
}

def resolve_region(defn, region_map):
    name_a, name_b, op = defn
    return OPERATORS[op](region_map[name_a], region_map[name_b])
```

No `eval` required. Compound expressions (more than two operands) are handled by
nesting: `[result_of_ab, C, union]` where `result_of_ab` is itself a list. The
runner resolves these recursively. The GUI can expose a simple two-operand form
for common cases and show a tree editor for compound expressions.

---

## Bands

Each band needs:

| Field | YAML value | GUI label | Default |
|-------|-----------|-----------|---------|
| `name` | string (e.g. `CB`, `VB`, `IB`) | — | required |
| `type` | `nondegenerate` | "Boltzmann" | `nondegenerate` |
| | `degenerate` | "Fermi" | |
| | `intermediate` | "Sharp Intermediate Band" | |
| | `mixed_qfl_nondegenerate` | "Mixed QFL / Boltzmann" | |
| `sign` | `-1` (electrons) or `+1` (holes) | "Electrons (−)" / "Holes (+)" | `-1`; auto-set to `+1` if name is `VB` |
| `extent` | list of region names, or `all` | checkbox list of layers + overlays | `all` (displayed as "Entire device (domain)") |

**Note on internal naming:** The Simudo names `nondegenerate` and `degenerate` are
poor descriptions of the physics (they refer to the Boltzmann vs. Fermi–Dirac
density-of-states–to–QFL relation). The GUI labels correct this. `IntermediateBand`
is displayed as "Sharp Intermediate Band" for clarity.

**Default bands** when the GUI panel first opens: `CB` (Boltzmann, electrons,
extent = all) and `VB` (Boltzmann, holes, extent = all). These match the standard
semiconductor starting point.

The `extent` list is used to build the subdomain `CellRegion` passed to
`easy_add_band(..., subdomain=...)`. Zero-current BCs at subdomain boundaries are
applied automatically. Only named regions are valid here (no set-operation
expressions). Both layer names and overlay region names are valid.

**Band naming**: The GUI shows a soft warning if a band name does not match `CB`,
`VB`, or `IB[…]` and a library material is selected — because library material files
use these conventional names for their property keys. Multi-valley names like `CBG`,
`CBL`, `CBX` are legitimate (e.g. for multi-valley materials) and the warning text
acknowledges this.

Multiple intermediate bands (IB, IB2, …) are supported.

---

## Materials

Materials are **shortcuts** for loading a set of property defaults, not fundamental
objects. A material file is a YAML dict of property keys → `{value, unit}` pairs.

The runner must:
1. Load the material (inline or from library/file) and obtain its `{key: value}` dict
2. Call `spatial.add_material_data(key, material_name, value, priority=PRIORITY_MATERIAL)`
   for each property. This applies each property on the union of all layers referencing
   that material, at low priority. *(Note: the runner does **not** use `material.register()`,
   which always uses priority 0. Direct calls are used so priority can be controlled.)*
3. Apply per-layer `properties:` as `spatial.add_rule()` calls at `PRIORITY_LAYER` on each layer's region
4. Apply named overlay region `properties:` at `PRIORITY_OVERLAY`, and the
   `domain` overlay's `properties:` at `PRIORITY_DOMAIN`, on those regions

The numeric priorities are:
```
PRIORITY_OVERLAY  = 0   (highest — wins)
PRIORITY_LAYER    = 5
PRIORITY_DOMAIN   = 7
PRIORITY_MATERIAL = 10  (lowest)
```

**Priority order (highest wins):**
```
named overlay region  >  layer  >  domain overlay  >  material defaults
```

`domain` is an overlay region in the schema, but it means *the whole device*,
so it belongs below layers: it is how a user states a device-wide default, and
a value set on one layer has to be able to override it. Registering it in the
same band as a named overlay made every layer-level override silently inert —
see `doc/release0_7/06_known_issues.md` A62. `domain` is still registered
*before* the materials so a material's `get_dict()` can read `temperature` and
similar keys (A56); registration order and priority are independent.

There is no intermediate "material override" level. After `register()` is called, Simudo
has no persistent concept of which material is in which region — it's all just spatial rules
at different priorities. To override a material property, the user adds it to the
`properties:` of the specific layer or overlay region where the change should apply.
If the same change is needed in every layer using that material, it must be added to
each such layer (or to an overlay region that covers their union).

All property specifications in the schema — whether inline material parameters, layer
properties, or overlay region properties — are called `properties:`. The distinction is
only where they appear (material entry vs. layer entry vs. overlay region entry), which
determines the region and priority at which they are registered.

Because materials are just defaults, a layer without a `material:` reference is
valid as long as all required properties are eventually defined by some region rule.

**Three material kinds:**

- **Inline:** constant key→value pairs directly in the YAML. Sufficient for toy
  problems. Only properties with plain numeric values can be stored this way.
  Properties computed at runtime (e.g. temperature-dependent DOS,
  `energy_level` derived from `Eg_300K`) are **not** available and will not count
  as "set" in the required-property check.
- **Library reference:** a Python `Material` subclass, stored as
  `source: library://ClassName` or `source: file:///path/to/material.py::ClassName`.
  All properties — including computed ones — are available when the simulation runs.
  The GUI counts all keys from the class's `get_dict()` body as "set".
  No overrides are stored in the project YAML.
- **Derived:** a library class plus a set of numeric overrides. YAML schema:
  ```yaml
  materials:
    MyGaAs:
      source: library://GalliumArsenide
      properties:
        CB/energy_level: {value: 1.35, unit: eV}
  ```
  The runner applies the library class at `PRIORITY_MATERIAL` (10) and the
  inline overrides on the union of all layers using this material at
  `PRIORITY_MATERIAL - 1` (9). Lower number = higher priority in Simudo, so
  overrides always win over the library default. All library keys (including
  computed properties) count as "set" in the required-property check.

**When to use each kind:**
- **Library reference** — normal case; all computed properties work, no overrides needed.
- **Derived** — need to adjust one or two numeric values (e.g. a custom band gap)
  while keeping all temperature-dependent computed properties from the library class.
- **Inline** — simple toy materials with no computed properties.

**Material library directory layout:**
- `code/materials/` — bundled read-only library, sibling of `gui/` in the repo.
  Contains `.py` files (Python `Material` subclasses for library-reference materials).
- User library — optional directory configured in `~/.simudo_gui.yaml`. Scanned
  first, so user materials take precedence over bundled ones when class names conflict.
- The runner's `--library-dir` flag (repeatable) is set by the GUI to point at each
  configured library directory. User dir is listed first.

**GUI Materials panel — creating project materials:**
Three ways to create a project material from a library class:
- **"Add to project"** button: library reference — `source: library://ClassName`, no
  properties stored. All library keys count as "set". Recommended for most simulations.
- **"Derive from this"** button: creates a derived material with
  `source: library://ClassName` and an empty `properties` dict.
  The detail panel shows a unified editable table of all library properties:
  - **Numeric props** (extractable by AST): shown with their library value dimmed;
    typing a new value turns the row amber and saves the override to `mat.properties`.
  - **Computed props** (runtime expressions): shown with a "calculated" badge and
    empty inputs; typing a value creates an override entry.
  - **Overridden rows** (amber): a ✕ button appears on the right; clicking it removes
    the key from `mat.properties`, restoring the library default (or removing the row
    for custom keys).
  - **Custom keys** (not in the library): appear in a CUSTOM OVERRIDES section; ✕
    deletes them entirely.
  - An **ADD KEY** form at the bottom adds arbitrary extra properties.
- **"Create new material"** form with no library base: creates a plain inline material.

All three create a named entry in `project.materials` that can be assigned to layers
via the layer's `material:` field (strict dropdown, populated from project materials).

**Runner handling of derived materials** (`_build_material_dict` in `simudo_1d_runner.py`):
After registering the library class at `PRIORITY_MATERIAL` as usual, if the material
entry also has a `properties` dict, compute the union of all layers referencing this
material and call `spatial.add_rule(key, union_region, qty, priority=PRIORITY_MATERIAL-1)`
for each override property.

**AST introspection of library classes (GUI-side, no dolfin):**
Each `.py` material file is parsed with `ast.parse()` — no import, no dolfin needed.
The parser walks `get_dict()` bodies to find `d.update({…})` and `d[key] = value`
patterns. For each key, the value node is inspected:
- `U("1.42 eV")` or a plain numeric literal → display value and unit; mark copyable.
- Anything else (arithmetic, UFL, function calls) → display `"(computed at runtime)"`
  and mark not-copyable. These are shown with a `runtime` badge in the property table.

The class-level `name = "SomeName"` attribute (the material's display name) is also
extracted and shown in the library browser alongside the Python class name.

### Required-property checking

The runner should compute the set of properties that *must* be defined given:
- Which bands are active
- Which interband processes are in use
- Which contact models are selected

It should then verify that every required property is covered by at least one
spatial rule somewhere in the region stack, and warn the user about anything
missing before launching the solve. This replaces the opaque FEniCS error that
would otherwise appear mid-run.

**GUI-side check (coarse):** The Layers panel maintains a real-time "parameters not
set in any region" list. A parameter is considered set if it appears in:
1. Any layer's `properties:` dict, OR
2. Any overlay region's `properties:` dict (including `domain`), OR
3. The set of keys provided by a project material that is referenced by at least
   one layer.

For case 3, the key set is determined differently by material type:
- **Inline material**: only the keys explicitly listed in `properties:` count.
  Computed properties (e.g. `CB/energy_level`) are **not** counted, even if the
  material was copied from a library class that provides them.
- **Library-reference material**: all keys declared in the class's `get_dict()` body
  count, including those marked "computed at runtime" in the AST scan.

This means users who want `CB/energy_level` (and similar computed properties) to
disappear from the missing-params list must use a **library-reference** material
(`Add as ref`), not an inline copy.

### Material introspection for the GUI

The GUI runs on the host machine; dolfin/UFL are only available in the remote
or Docker environment. Material introspection must therefore work **without
importing dolfin**. The approach is **static AST analysis** of the material
source files using Python's standard-library `ast` module:

1. Parse the `.py` material file with `ast.parse()` — no import, no dolfin.
2. Walk the AST to find `d.update({...})` and `d[key] = value` calls inside
   `get_dict()`. String-literal keys are extracted directly.
3. For each key, inspect the value node:
   - `ast.Call` matching `U("1.5 eV")` or similar → extract and display the
     unit string.
   - Anything else (arithmetic, conditionals, function calls) → display
     "evaluated at runtime".
4. To read `required_spatial_params`, find the class-level assignment of that
   name — it is always a literal list, extractable without execution.

For **inline YAML materials**, no analysis is needed — keys and values are
directly in the YAML and trivially readable on the host.

| Material type | Key discovery | Value display |
|---|---|---|
| Inline YAML | Read from YAML directly | Show value + unit |
| Python library | AST-parse source file | Simple `U("…")` → show; expression → "evaluated at runtime" |

**Consequence for workflow**: the execution profile does not need to be
configured before the user can browse materials or define a device. Remote
access is only required when actually launching a simulation.

### Declaring required spatial inputs (`required_spatial_params`)

Some materials need spatial rules to be set *before* they can evaluate (e.g.
`AluminumGalliumArsenideAlloy` reads `MoleFractionX` from `spatial`). To let
the GUI prompt the user for these, add a class attribute to `Material`:

```python
class Material:
    required_spatial_params: list[str] = []
    # Subclass example:
    # required_spatial_params = ['MoleFractionX']
```

The GUI reads this attribute and adds the listed keys to the layer's
"required parameters" panel, prompting the user to set them per-layer.

### Declaring required layer parameters for interband processes (`required_spatial_params`)

Each interband process reads spatial rules whose keys are prefixed with the
process name (e.g. `nr_top/sigma_th`). To let the GUI know what per-layer
parameters each process type requires, add a class attribute to each process
class:

```python
class NonRadiativeTrap:
    required_spatial_params = ['sigma_th', 'v_th']

class NonOverlappingTopHatBeerLambert:
    required_spatial_params = ['alpha']

class NonOverlappingTopHatBeerLambertIB:
    required_spatial_params = ['sigma_opt']
```

The GUI prepends the process instance name: `nr_top` + `sigma_th` →
`nr_top/sigma_th`. This tells the GUI exactly which per-layer entries to prompt
for in the second pass of the Layers panel.

### Guiding users through the two-pass workflow

Because per-process parameters on layers can only be known after bands and
processes are defined, users must return to the Layers panel after completing
those sections. Three mechanisms guide this:

1. **Sidebar nav badge** — the "Layers" nav item shows a yellow `●` indicator
   whenever any required per-process parameter (`required_spatial_params`) is
   unset for any region. The badge turns green (or disappears) when all required
   parameters are filled in.

2. **Contextual footer hint** — the bottom of the Bands panel and the Processes
   panel each show a soft callout: *"Once bands and processes are defined, return
   to Layers to set per-region parameters for each process (e.g. `SRH/CB/tau`,
   `sigma_opt`)."* Dismissable but shown by default.

3. **Amber highlighting in the Layers detail panel** — in the second pass, any
   required parameter that is still unset is highlighted in amber. The layer's
   row in the table also carries a yellow indicator until all its required
   parameters are filled.

### Band naming convention

Library material files use `CB` and `VB` as the band name prefix for all
parameters (`CB/energy_level`, `VB/mobility`, etc.). Users should be encouraged
to name their conduction and valence bands `CB` and `VB` respectively. The GUI
shows a warning if a library material is selected for a layer whose device
definition uses different band names.

### SRH lifetimes

`SRH/CB/tau` and `SRH/VB/tau` are growth properties, not intrinsic material
properties. They may appear in a material file as a convenience, but the user
must be able to define or override them directly on any region without touching
the material file.

---

## Interband Processes (formerly "Electro-Optical Processes / EOPs")

Use Simudo class names exactly, not GUI shorthand. Current classes:

| Class | Dark/Optical | Trap selector | Key parameters |
|-------|-------------|:---:|----------------|
| `SRHRecombination` | dark | — | `src_band`, `dst_band` |
| `NonRadiativeTrap` | dark | ✓ | `src_band`, `dst_band`, `trap_band` |
| `ShockleyReadBand2BandTrap` | dark | — | `src_band`, `dst_band` |
| `ShockleyReadTrap2Trap` | dark | — | `src_band`, `dst_band` |
| `NonOverlappingTopHatBeerLambert` | optical | — | `src_band`, `dst_band`, `radiative_recombination` |
| `NonOverlappingTopHatBeerLambertIB` | optical | ✓ | `src_band`, `dst_band`, `trap_band`, `radiative_recombination` |

The "Trap selector" column indicates whether the GUI shows the Trap band field.
This is determined dynamically: the GUI shows the trap field for any class that
directly inherits `TrapEOPMixin`. `SRHRecombination` was refactored to inherit
`TrapEnergyLevelMixin` instead (it uses `get_u1` / `get_trap_energy_level` but
has no explicit trap Band object), so the trap selector is correctly hidden for it.

**Class names are never abbreviated in the GUI.** Names like
`NonOverlappingTopHatBeerLambert` must remain distinct from future variants
(overlapping, non-top-hat, etc.) that will be added as new classes. The class
name is always passed through to the YAML verbatim.

**Class discovery:** the GUI scans `electro_optical_process.py` via AST at
startup and finds all concrete subclasses of `ElectroOpticalProcess`. No
hardcoded class list is maintained. The GUI falls back to a hardcoded list only
if the runner source tree is not found alongside the GUI.
- **Optical classes** (`radiative_recombination` toggle shown): direct subclasses of
  `AbsorptionAndRadiativeRecombinationEOPMixin`, detected by AST.
- **Trap classes** (`trap_band` selector shown): direct subclasses of `TrapEOPMixin`,
  detected by AST.

**gui_hint attribute:** any EOP class may define a class-level string attribute:
```python
gui_hint = "Brief usage note shown in the GUI when this class is selected."
```
The AST extractor reads this at startup and displays it as a styled note below
the class selector. Classes without `gui_hint` show nothing. This is the
preferred way to add per-class usage instructions without modifying the GUI code.
All current EOP classes have `gui_hint` defined.

It must be a plain string literal — the extractor matches `ast.Constant`, so an
f-string or a name concatenation is skipped silently. (Adjacent literals inside
parentheses are joined by the parser and are fine.) The text is treated as plain
text, not markup, and escaped before display, so placeholders like
`'<name>/alpha_function'` appear as written.

**User-defined EOP classes** are supported via an optional `custom_eop_files`
key in the project YAML:

```yaml
custom_eop_files:
  - ../my_project/custom_processes.py   # relative to this YAML file
```

The GUI discovers classes in those files via AST; the runner imports them
dynamically via `importlib` using the same paths. Paths are stored in the
project YAML (not only in `~/.simudo_gui.yaml`) to keep the project
self-contained and portable across machines.

**src/dst convention:** `src_band` is always the **lower-energy** band;
`dst_band` is always the **higher-energy** band. The GUI labels these
"Lower E band" and "Higher E band".

**`radiative_recombination`** (optical classes only): boolean, defaults `true`.
The GUI exposes this as a Yes/No toggle, visible only for optical classes.
If omitted from the YAML, the runner uses the class-level default (`True`).

**Trap band validation:** for trap classes, the GUI enforces two rules:
1. The trap band must be of type `intermediate` (Sharp Intermediate Band).
2. The trap band must equal either the Lower E or Higher E band — this is
   required by `TrapEOPMixin.reg_band` which derives the non-trap band as
   whichever of src/dst is not the trap.
Both are shown as orange inline warnings; they do not block saving.

**Default bands when switching to a trap class:** if exactly one `intermediate`
band exists in the project, the GUI auto-fills: Lower E = IB, Higher E = first
non-IB band, Trap = IB. This matches the common `nr_top`-style process where
the IB is both the lower-energy participant and the trap.

**Name uniqueness:** every process must have a distinct `name`. The GUI
validates this on edit and auto-suggests names from the class prefix
(e.g. `opt`, `opt_2` for a second optical class; `nr`, `nr_2` for a second
`NonRadiativeTrap`).  Duplicate names show an inline orange error and are
not committed to the model until fixed.

Examples from the fourlayer case:
- `opt_cv` / `NonOverlappingTopHatBeerLambert` (CB↔VB): `src_band=VB`, `dst_band=CB`
- `nr_top` / `NonRadiativeTrap` (IB↔CB): `src_band=IB`, `dst_band=CB`, `trap_band=IB`
- `nr_bottom` / `NonRadiativeTrap` (VB↔IB): `src_band=VB`, `dst_band=IB`, `trap_band=IB`

---

## Optical Fields

**The Opt. Fields section is entirely optional.** A simulation with no optical
fields runs in dark mode (equilibrium or voltage sweep). Absorption EOPs
(e.g. `NonOverlappingTopHatBeerLambert`) can and should still be defined even
in dark mode because they contribute radiative recombination, which affects
the physics even without external illumination.

Each optical field is a monochromatic (or pseudo-monochromatic) photon flux
propagating in one direction through the device.

**The "NonOverlapping" constraint:** `NonOverlappingTopHatBeerLambert` and
`NonOverlappingTopHatBeerLambertIB` are physically constructed so that each
photon is absorbed by *at most one* transition. This means the spectral bands
assigned to CB↔VB, CB↔IB, and VB↔IB must be non-overlapping ranges. Each of
these transitions must therefore have its own optical field (at its own
representative photon energy). In the fourlayer example: `forward_cv`, `forward_ci`,
and `forward_iv` are three distinct fields even though ci and iv may have the same
photon energy, because they represent different non-overlapping spectral bands that
are each absorbed by only one process.

Two optical fields *may* coincide in photon energy if they correspond to
non-overlapping spectral sub-bands — each field carries the flux for its own band,
and each is exclusively consumed by its associated EOP.

### Intensity specification

Two modes:

**Manual:** user specifies intensity directly.
```yaml
intensity: {value: 100.0, unit: mW/cm^2}
```

**Blackbody band integral:** a single representative photon energy stands in for
the integrated flux from an entire spectral band of a blackbody source. Simudo
already has code to compute this integral; the runner passes the parameters through.
```yaml
intensity:
  mode: blackbody_band
  T_source: {value: 6000, unit: K}
  E_min: {value: 1.42, unit: eV}    # lower edge of this band
  E_max: infinity                    # or upper edge; use next band's E_min otherwise
  concentration: 1.0                 # solar concentration factor X
```
`concentration` is dimensionless (X suns) and may be non-integer (e.g. 0.1 for
sub-1-sun illumination).

`f_s` (the solid-angle factor = 1/46500 for sunlight on Earth) is a physical constant
already built into `Blackbody.photon_flux_integral_on_earth()`. It is **not** a user
parameter and should not appear in the schema.

### Direction

Currently `"+x"` or `"-x"` (left-to-right or right-to-left propagation).
**Future TODO**: generalise to an angle of incidence relative to the layer
normal. Even in a 1D simulation the light can propagate at an oblique angle,
which changes the effective path length through each layer. Reserve the field
for this extension — do not assume it will always be a simple ±x string.

### Future intensity modes

**TODO**: AM1.5G spectrum mode — divide the AM1.5G spectrum into N frequency
bins over a user-specified range, auto-generating one optical field per bin.
This is not in the runner yet; it would require a new intensity mode and
corresponding runner support.

---

## Boundary Conditions

The GUI sidebar section is called "Boundary Conditions". The YAML key remains
`contacts:` — that is a runner detail not surfaced in the UI.

### External contacts

Each exterior face of the device (left and right in 1D) has a boundary condition
for every band. Options per band per contact:

| Model | Simudo BC | Notes |
|-------|----------|-------|
| `ohmic` | carrier density fixed at thermal equilibrium | SRV = ∞; majority carrier contact |
| `zero_current` | zero carrier current | SRV = 0; minority carrier contact |
| *(future)* `srv` | finite surface recombination | not yet in Simudo; reserve field |

All bands are shown at both contacts regardless of extent — assigning a BC to a
band at a face it does not reach is costless. Default: `ohmic` for all bands.

One contact is `swept` (voltage is the sweep parameter; not user-specified) and
one is `reference` (always 0 V; not user-specified). No voltage fields are
exposed in the GUI — the swept voltage is controlled by the simulation stepper,
and the reference is fixed at 0 V.

### Band extent boundaries

For each band whose extent is not `domain` (i.e. the band is spatially confined),
there are two internal boundary conditions — one at each end of the band's extent.
These are exposed in the Boundary Conditions panel as a per-band sub-table with
independent `left boundary BC` and `right boundary BC` dropdowns. Default:
`zero_current` at both ends.

**Runner implementation — facet topology:**
Simudo BCs are applied to *facets* based on region topology, not spatial
coordinates. To apply a BC to the left boundary of band B's extent, the runner
calls `B.boundary(left_adjacent_region).both()`, where `left_adjacent_region`
is the region immediately to the left of B's extent:

- **Layer-list extent**: the adjacent region is the layer immediately before
  (left boundary) or after (right boundary) the extent layers in the stack order.
- **Coordinate-range extent**: the runner must determine which region contains
  each boundary coordinate, then use that region as the adjacency argument.

This means the runner must be aware of the full region topology (layer order,
overlay positions) when applying extent-boundary BCs — it cannot simply look up
a coordinate.

### Internal interfaces

Internal interface BCs are **not** auto-applied by default.  Every
`ThermionicHeterojunction` (or future BC type) that the user wants must be
stored explicitly in the project YAML.  If an interface is absent from the
YAML the runner applies no BC there.

Internal SRV: not yet in Simudo; not in scope for v1.

Internal interface BCs are configured by clicking the divider buttons between
layer blocks in the schematic (Layers panel).  They are **not** exposed in the
Boundary Conditions panel.

Band subdomain boundaries (where an `extent`-limited band meets a region
where it is not active) automatically receive zero-current BCs by default;
the Boundary Conditions panel allows these to be overridden.

#### YAML schema

```yaml
interfaces:
  - left: p_layer       # left layer name (Python identifier, matches R.<name>)
    right: n_layer      # right layer name
    bcs:
      - type: ThermionicHeterojunction
        bands: [CB, VB]               # bands that get this BC at this interface
        HJBC_enhancement:             # per-band dolfin.Constant weight; default 1; must be > 0
          CB: 1.0
          VB: 1000.0
```

- An interface entry is written **only** when ≥1 band has a BC enabled there.
  An interface with no active BCs is absent from the YAML entirely.
- The `bcs` list is extensible: future BC types (e.g. tunneling) add new entries
  alongside `ThermionicHeterojunction` without breaking old YAMLs.
- `HJBC_enhancement` is a **per-band dict** (band name → positive float).  It
  increases the weight of the BC in the weak form, penalizing deviations more
  strongly.  Suggested values: 1, 1e3, 1e6.  Bands absent from the dict default
  to 1.0.  Old YAMLs with a scalar value are loaded with that scalar broadcast
  to all listed bands (backwards compatible).

#### Interface identification

Interfaces are identified by ordered layer-name pair `(left, right)`.  This maps
directly to the Simudo topology call:

```python
facet = getattr(R, left).boundary(getattr(R, right))
ThermionicHeterojunction(band, facet, HJBC_enhancement=dolfin.Constant(enh)).register()
```

If a layer is renamed in the GUI, all `Interface.left` / `Interface.right`
references to the old name are updated automatically.

#### Runner handling

After all contact BCs are registered in `_create_problemdata`, iterate
`config.get('interfaces', [])`.  For each entry, for each BC spec:

```python
from simudo.physics.heterojunction import ThermionicHeterojunction
for iface in config.get('interfaces', []):
    left, right = iface['left'], iface['right']
    for bc_spec in iface.get('bcs', []):
        if bc_spec['type'] == 'ThermionicHeterojunction':
            enh_raw = bc_spec.get('HJBC_enhancement', {})
            # enh_raw may be a per-band dict or a legacy scalar
            if isinstance(enh_raw, dict):
                enh_dict = {k: float(v) for k, v in enh_raw.items()}
            else:
                scalar = float(enh_raw) if enh_raw else 1.0
                enh_dict = {b: scalar for b in bc_spec.get('bands', [])}
            facet = getattr(R, left).boundary(getattr(R, right))
            for bname in bc_spec.get('bands', []):
                band_obj = band_objects.get(bname)
                if band_obj is None:
                    log.warning(f'Interface {left}|{right}: unknown band {bname!r}')
                    continue
                enh = enh_dict.get(bname, 1.0)
                ThermionicHeterojunction(
                    band_obj, facet,
                    HJBC_enhancement=dolfin.Constant(enh),
                ).register()
```

### GUI representation of interfaces

The Layers panel is split into two columns below the full-width schematic:
- **Left column**: layer table, overlay regions, missing-parameters list.
- **Right column**: properties / detail panel (shows whichever of layer,
  overlay, or interface is currently selected; only one can be active at a time).

The schematic shows:
- **Internal interfaces** as 12 px clickable divider buttons between layer blocks
  (faint blue with blue border; solid blue when selected); hovering highlights
  them; clicking selects the interface and shows its configuration in the right
  column.
- **External boundaries** (left and right device edges): BC markers at the ends
  of the schematic (static labels; clicking them is not supported).

Interface detail panel (right column):
- Header showing `left_name|right_name interface` and their material assignments.
- Dismissable amber warning when both layers share the same material (unusual
  for a heterojunction BC).  Only shown when ≥1 band is enabled; dismissed by
  an ✕ button on the warning row; resets whenever a layer's material changes
  or a different interface is selected.
- Soft dismissable warning when `project.bands == default_bands()`: "Default
  bands are present; visit Bands panel to add/change before applying HJBCs."
- Error (no toggles) when `project.bands` is empty.
- Per-band rows: checkbox enabling `ThermionicHeterojunction` for that band,
  plus an **"Enhancement (optional):"** numeric input on the same row (default 1;
  must be > 0; scientific notation accepted).  Enhancement is stored per-band.
- Saving: toggling a band or changing enhancement autosaves via
  `app.autosave()`.  When all bands are unchecked the interface entry is removed
  from `project.interfaces`.

### GUI representation of overlay regions

Overlay regions are displayed in a collapsible **Overlay Regions** section
directly below the layer table, inside the Layers panel (not a separate sidebar
section). Two ways to add a new overlay:

1. **Click the schematic** — hovering over the schematic highlights the
   span interactively; clicking confirms the span and opens a name/confirm form.
2. **"+ Add overlay" button** — opens a form where start and stop coordinates
   can be typed directly.

The `domain` overlay is always the first entry in the list, visually
distinguished (e.g. a lock icon, greyed delete button), and cannot be removed.
Its property panel notes that it acts as a lower-priority global default.

Existing overlays are shown as dashed outlines on the schematic, colour-coded
to match their list entries.

---

## Project Open / New Workflow

- **Default on launch**: open immediately with a blank project. The output folder
  defaults to `out/YYYY[Mon]DD/a` relative to the directory from which the GUI
  was launched. No modal file-picker at startup.
- **Recent projects**: a persistent "Recent ▼" dropdown button in the header bar,
  always accessible regardless of which sidebar section is open or how much
  editing has been done. Clicking it drops down the recent-projects list without
  navigating away from the current work.
- **Two loading modes** when opening a past project:
  - *Settings only* — loads layers, materials, bands, processes, optical fields,
    contacts, and simulation config from the YAML. Does not associate any output
    folder or results. Use this to start a modified version of a past device.
  - *Settings + results* — loads the YAML and also points the GUI at the
    associated output folder, enabling the J-V and band-diagram views to show
    data from that run. Use this to inspect or continue a past simulation.
- The resolved YAML file path is shown prominently in the UI header at all times.
- Recent-project tracking requires a small per-user config file (location TBD).

---

## Output Folder Naming

Default: `out/YYYY[Mon]DD/a/` (relative to the project YAML file) where the trailing letter increments (`a`→`b`→…→`z`→`za`→…)
if the directory already exists. Implemented via `ensure_new_dir()` from
`JK_Helpers/helpers.py`. The runner should use the same logic.

Users can override with any path. The GUI suggests the date-based default but
allows free entry. The full resolved path is stored in the project file so that
reopening a project points to the correct data directory.

---

## Checkpoints

- Checkpoints are expensive (large files) and should not default to every step.
- GUI asks the user to specify checkpoint values explicitly (a list of voltage or
  intensity values at which to save), with an option "save at all output points".
- A suggested default: save at the final point only (guarantees one resumable state).
- The checkpoint directory lives inside the output folder.
- On project re-open, the GUI scans the checkpoint directory, lists available
  checkpoints, and offers to resume from a selected one.
- **Resuming requires an identically-constructed problem.** The project YAML must
  be unchanged (or changes must be clearly flagged as incompatible with existing
  checkpoints).

---

## Simulation Sweep

### Simulation mode

Four named modes control which stages run (`intensity_ramp.enabled`,
`voltage_sweep.enabled`). Simudo is always steady-state, so there is no
time-dependent mode; "Illuminated (V=0)" means a full illuminated intensity ramp
to steady state with no subsequent voltage sweep.

| GUI mode | `intensity_ramp` | `voltage_sweep` |
|---|---|---|
| Equilibrium only | off | off |
| Dark J-V | off | on |
| Illuminated (V=0) | on | off |
| Illuminated J-V (default) | on | on |

A warning note is shown if an illuminated mode is selected but no optical fields
are defined. The illuminated modes are not greyed out (just warned) so the user
can still set up the simulation config before adding optical fields.

### Self-consistent optics

`selfconsistent_optics` controls whether optical generation rates are
recomputed at each solver step.

- **Must be ON** if any EOP's absorption coefficient depends on band filling
  (e.g. `NonOverlappingTopHatBeerLambertIB`, where σ_opt varies with IB
  occupancy). Solving without self-consistency in this case is physically wrong.
- **Can be OFF** if all EOPs have fixed absorption coefficients
  (e.g. `NonOverlappingTopHatBeerLambert` with a constant `alpha`). Non-self-
  consistent optics is faster.

The GUI auto-detects which EOPs are defined and shows a contextual
recommendation. The user can always override.

### Voltage sweep values

The GUI provides a `linspace` helper (from / to / N steps) that expands to an
explicit list, plus a read/edit view of that list for users who need irregular
spacing. The YAML always stores the explicit expanded list.

**Bidirectional sweeps:** if the `values` list contains negative voltages, the
runner automatically splits it into a positive group (0, v₁, v₂, …) and a
negative group (0, −v₁, −v₂, …), both sorted by increasing magnitude from zero.
V = 0 is always prepended to each group. A second `ProblemData` instance is
created for the negative sweep, initialised from `full_problem` after the optical
ramp completes — so both sweeps begin from the same fully-illuminated V = 0 state
and the expensive optical ramp is only run once.

### Advanced stepper parameters

Exposed in a collapsible "Advanced" section in the GUI:
- **Initial step size** (`step_size`): separate values for the intensity ramp
  and voltage sweep steppers. Most users should leave these at the defaults.
- **Resume from checkpoint** (`resume_from`): file path (relative to the project
  YAML) of a checkpoint file to start from. When set, completed stages are
  skipped and the simulation continues from the saved state. Requires an
  identically-constructed problem (see Checkpoints section).

---

## Output Configuration

Beyond the folder path, a short `prefix` string (default `"sim"`) is prepended to all
output file names. This replaces the problem-specific prefix (e.g. `"pd"` in the
`fourlayer` example) with something generic. Override via `output.prefix:` in the YAML.

The `new_experiment` flag (CLI: `--resume`) controls whether `ensure_new_dir` is applied
to the output folder. Set it `True` (default) for a fresh run — the folder is
incremented if it already exists. Set it `False` when continuing from a checkpoint so
output lands back in the same directory as the original run.

### YAML copy in output folder

The runner copies the project YAML into the output folder immediately after resolving
the output directory (`shutil.copy2(yaml_path, PREFIX / yaml_path.name)`). This makes
every output directory self-documenting: the exact simulation configuration that produced
the results is always stored alongside them, and the batch-aware schematic can find it
via `glob("*.yaml")` in the output directory.

---

## YAML Save Strategy

The YAML file is written on every completed field edit (focus-out / Enter key),
keeping the on-disk file continuously current. A crash loses at most the last
few keystrokes, not a whole session of work. No separate draft or recovery file
is needed.

The **Save** button in the header serves as an explicit "I'm satisfied with this
state" action and also triggers a full validation pass. It does not gate the
continuous writes — those happen regardless.

Execution profile (`execution:`) is set on the execution panel (build step 2)
and is not part of the device-definition YAML. The device definition screen
saves a pure physics YAML that the runner consumes directly.

---

## Validation

Validation runs on the full YAML on every explicit Save. Schema completeness
is checked (required fields present, values in range, band/process name
uniqueness, etc.). Physics sanity checks (e.g. optical field energies consistent
with band gaps) are deferred to a future version.

**Error display:**
- Inline red border on the offending field, with a short error message below it.
- A count badge in the header (e.g. "3 errors") that links to a collapsible
  summary list of all current errors with jump-to links.

---

## Layers Panel — Implementation Notes
*Added May 2026 — records decisions made during the Panel (HoloViz) implementation.*

### Schematic flex layout and overlay strip alignment

The layer stack schematic is a `pn.Row` of `pn.widgets.Button` objects with
CSS `flex:{fv} 1 0px` (flex-grow = fv, **flex-basis = 0 px**) and `gap:2px`
between them. The overlay indicator strip below it is a `display:flex; gap:2px`
HTML row using the same flex values.

**Why flex-basis must be 0 px:** With any non-zero flex-basis `b`, each item
receives `b` px before proportional distribution, so its left-edge position
becomes `i·b + flex_cum[i]/total_flex·(W − n·b)` — a width-dependent formula.
Because the Python code cannot know the rendered width `W`, it cannot compute
exact overlay positions. With flex-basis = 0, positions collapse to the
width-independent fraction `flex_cum[i]/total_flex`, which is what `_um_to_frac`
returns for coordinate overlays and what the strip flex divs use by construction.
The residual error from `gap:2px` is `i·2px/W ≈ 1–3 px` and is imperceptible.

**`_SCHEMATIC_MIN_FLEX = 5.0`** ensures every layer — including zero-thickness
layers — gets at least 5/total_flex of the visual width. Both
`_build_schematic_objects` and `_build_overlay_strip_html` import this constant;
changing it in one place keeps them in sync.

### Overlay strip rendering strategy

- **Layer-based overlays** are rendered as a flex-mirror: one `<div>` per layer
  with the same flex value, coloured if that layer belongs to the overlay and
  transparent otherwise. This handles disjoint selections (e.g. p + n without I)
  automatically. A `position:absolute` name label is added for each contiguous
  run of selected layers.
- **Coordinate overlays** are rendered as `position:absolute` bars with
  `left = _um_to_frac(start)·100%` and `width = (_um_to_frac(end) − _um_to_frac(start))·100%`.
  `_um_to_frac` maps a µm coordinate to the flex fraction by interpolating within
  the containing layer using that layer's flex width, preserving sub-layer
  precision even when thin layers are visually expanded by `_SCHEMATIC_MIN_FLEX`.
- The strip is wrapped in a `padding-left:22px; padding-right:22px` div so it
  aligns with the inner layer area (excluding the 22 px BC contact markers).

### Overlay region extent types

Two extent types are implemented. A third is planned:

| Type | Storage | Runner support | GUI status |
|------|---------|---------------|-----------|
| Layer list (`layer_names`) | `OverlayRegion.layer_names: List[str]` | ✅ `CellRegion` union | ✅ MultiChoice picker |
| Coordinate range (`start`/`end`) | `OverlayRegion.start`, `.end`, `.extent_unit` | ❌ raises `NotImplementedError` | ✅ Left/Right inputs |
| Set operation (intersect/union/diff) | not yet in model | ❌ | ❌ future TODO |

The coordinate range path in the runner (`_create_problemdata`) currently raises
`NotImplementedError`. When the GUI's coordinate overlay UI is used, a warning
should inform the user that runner support is pending.

### Layer and region name validation

Layer and overlay region names are sanitised on every edit via `_sanitize_ident`:
spaces and hyphens → `_`, non-alphanumeric stripped, leading digit prefixed with
`_`, empty result → `_`. This ensures names are valid Python identifiers, which
the runner requires for `CellRegion` naming.

### Doping display

Doping concentration is stored as a signed float (positive = p-type,
negative = n-type) under `poisson/static_rho` in `layer.properties`. The GUI
separates sign (RadioButtonGroup n-type/p-type) and magnitude (TextInput with
scientific-notation formatting via `_fmt_sci`). `FloatInput` is deliberately not
used because Bokeh 3.x renders `1e19` as `10000000000000000000`.

### Contact BC markers

The schematic row is `pn.Row(contact_left, inner, contact_right)`. Two *separate*
`pn.pane.HTML` objects are created — Panel cannot render one object in two layout
slots simultaneously.  The markers show `BC` as static text.

### Finite-extent band BCs and disjoint regions

When a band has `extent` set to a list of regions rather than `"all"`, the runner
attaches boundary conditions at the edges of that band's domain.  A natural question
is whether a disjoint extent (e.g., IB assigned to two non-contiguous layers) requires
additional BCs for the internal interfaces.

**Key insight:** if both edge BCs are `zero_current` (the default), disjointness is
harmless.  The runner can collect all internal-interface facets with:

```python
(R.domain - R.IB).boundary(R.IB)
```

which works correctly whether or not the IB region is contiguous.

**The only problematic corner case** is a finite-extent band with a *non*-zero-current
BC (e.g., ohmic) on a *disjoint* region, because the single `left`/`right` BC pair
cannot unambiguously describe separate sub-region boundaries.  This situation is
physically unusual and should be caught by the runner with a clear error message.

The GUI does **not** validate this; the contiguity warning was deliberately removed
after determining that zero-current (the default) makes it a non-issue in practice.

---

### Property editor — second-pass layer editing

After bands and processes are defined, every region (layer or overlay) may need
per-process and per-band spatial parameters set on it. The GUI guides the user
through a "second pass" of the Layers panel.

#### `required_spatial_params` format

The attribute `required_spatial_params` on each EOP class is a `dict[str, str]`
mapping suffix → unit string (implemented in both `simudo/physics/electro_optical_process.py`
and the GUI's AST extractor in `panels/processes.py`):

```python
class NonRadiativeTrap:
    required_spatial_params = {'sigma_th': 'cm^2', 'v_th': 'cm/s'}

class SRHRecombination:
    required_spatial_params = {'{dst_band}/tau': 's', '{src_band}/tau': 's'}
```

Suffixes may contain `{dst_band}` / `{src_band}` as placeholders; the GUI
substitutes the actual band names from the process's `src_band` / `dst_band`
fields, then prepends the process instance name to form the full spatial key:

```
{proc.name}/{expanded_suffix}   →   e.g. "nr_ci/sigma_th", "SRH/CB/tau"
```

The base class declares `required_spatial_params: dict = {}`.

The `_discover_eop_classes` AST extractor in `processes.py` is extended to read
this dict (key + value literals) at startup and store it alongside the class
metadata. The layers panel imports this map to build suggested-property lists.

#### Band required parameters (hardcoded in GUI)

Each band type requires the following spatial params (prepended with the band
name at runtime):

| Band type | Suffix | Unit |
|-----------|--------|------|
| All bands | `energy_level` | `eV` |
| Boltzmann (`nondegenerate`) + Fermi (`degenerate`) | `effective_density_of_states` | `cm^-3` |
| Sharp IB (`intermediate`) | `number_of_states` | `cm^-3` |

These are hardcoded in the GUI; they do not appear in any Python class attribute.

#### Property editor UI

Inside each region's detail panel (layer or overlay), below the doping field:

- A labelled **Properties** section listing all currently-set non-doping properties
  as editable rows: `key` (read-only label), `value` (TextInput), `unit` (TextInput
  or Select), `✕` delete button.
- An inline **"+ Add property"** form (like "Add band" / "Add overlay"):
  - A `Select` dropdown pre-populated with all suggested properties derived from
    the current bands and processes, labelled as `{key}  [{unit}]`.
  - A free-text `TextInput` below it for entering an arbitrary key not in the list.
  - `Value` and `Unit` fields, with unit auto-filled when a suggestion is picked.
  - A `+` confirm button that appends the property to the region's `properties` dict.
- **Expression mode** (deferred): a small `ƒ` toggle per row that switches from
  value+unit fields to a free-text expression box. Add this later.

#### Suggested property derivation

On every panel refresh, the layers panel recomputes:

```python
suggested = {}   # {full_key: unit}
for band in project.bands:
    suggested[f"{band.name}/energy_level"] = "eV"
    if band.type in ("nondegenerate", "degenerate"):
        suggested[f"{band.name}/effective_density_of_states"] = "cm^-3"
    elif band.type == "intermediate":
        suggested[f"{band.name}/number_of_states"] = "cm^-3"
for proc in project.processes:
    rsp = EOP_REQUIRED_PARAMS.get(proc.cls, {})   # from _discover_eop_classes
    for suffix, unit in rsp.items():
        key = suffix.replace("{dst_band}", proc.dst_band).replace("{src_band}", proc.src_band)
        suggested[f"{proc.name}/{key}"] = unit
```

#### Missing-parameters global section

A collapsible section **at the bottom of the Layers panel** (below the overlay
regions list, always visible regardless of which region is selected) titled
**"Parameters not yet set in any region"**.

Logic: the set of "all required params" is `set(suggested.keys())`. A param is
considered *set* if it appears in `properties` of *any* layer or overlay region
(including `domain`). Params that are set nowhere are listed here. Once a param
is set in any region, it is removed from this list.

This list is informational only — it does not block saving or running. Some
params may be covered by material files (not yet implemented) and will
legitimately stay unset in every region; the user can ignore them in that case.

#### Sidebar badge

The "Layers" nav button shows a yellow `●` badge when the missing-parameters
list is non-empty. No count is shown — just the presence/absence indicator.
The badge is recalculated on every panel rebuild (when bands or processes change)
and after every property edit.

---

### Temperature — spatial property, not a physics field

**Temperature is a spatial property**, not a top-level physics setting. It is
set by the user in the `domain` overlay's properties under the key `temperature`
with unit `K` (or any Pint-parseable temperature unit).

**Default on new project:** `make_domain_overlay()` pre-populates
`properties={"temperature": {"value": 300.0, "unit": "K"}}` so every new project
starts with temperature visible and editable. It appears immediately in the domain
properties and is absent from the "Parameters not yet set" list until deleted.

**YAML schema:** there is no `physics: temperature:` field. Temperature is saved
and loaded as a region property under `regions: [{name: domain, properties:
{temperature: {value: 300, unit: K}}}]`. Old files that have `physics: temperature:`
are silently ignored on load.

**Runner:** the runner no longer calls `spatial.add_rule('temperature', R.domain, T)`
from a `physics:` key. Temperature enters through the normal overlay-region loop,
which now handles `name: domain` as a special case mapping to `R.domain`. The
domain region has no `extent` key; its presence is detected by `name == "domain"`.

**`_GLOBAL_REQUIRED`:** the GUI always includes `{"temperature": "K"}` in the
suggested/required params set, so `temperature` appears in the "Parameters not yet
set" list if it is deleted from the domain (or never set).

**Warning for non-domain regions:** if the user tries to add `temperature` as a
property of any region other than `domain`, a red warning box appears in the
Add-property form explaining that temperature set in only some regions will be
0 K elsewhere, likely causing solver errors.

**v_th formula helper:** when the user selects or types a property key ending in
`/v_th`, an amber info box appears below the Add-property form showing:
- Formula: `v_th = √(8 k_B T / π m* m_e)`
- Numerical value at the current project temperature (read from the domain
  overlay's `temperature` property, converted to K via Pint) with m* = 1
- Scaling note: scales as 1/√m* (e.g. m*=0.063 for GaAs CB → ×3.98)
- Note that a loaded material file sets v_th automatically

`_project_temperature(project)` reads the domain overlay's `temperature` property,
converts to K using `pint.UnitRegistry`, and falls back to 300 K if unset or
unparseable.

---

### YAML top-level `output.folder`

The runner reads output folder from `config["output"]["folder"]` (top-level `output`
key, not nested under `simulation`). The GUI saves this as:

```yaml
output:
  folder: out
```

and loads it from the same path. `simulation.output` (nested) is a separate key
holding `spatial_profiles` and `xdmf_mesh` flags, which the runner reads from
`sim_config["output"]`.

---

### Development / testing environment

All internal and testing simulations (including those launched from the GUI during
development) must be run inside the **`claude_simudo_gui`** conda environment. This
environment contains Panel, and all required dependencies. The Panel
server itself also runs in this environment:

```bash
conda run -n claude_simudo_gui panel serve gui/app.py --port 5007 --allow-websocket-origin='*'
```

The Run button in the Simulation panel uses `sys.executable` to launch the runner,
which — since the Panel server is already running in `claude_simudo_gui` — guarantees
the correct Python and all imports without any extra conda invocation.

This invocation will fail unless simudo/dolfin are installed locally, which is 
almost never the case.

---

### Simulation runner invocation

The runner is launched as a subprocess using `sys.executable` (the same Python
that runs the Panel server, i.e. the `claude_simudo_gui` conda environment):

```
sys.executable  simudo_1d_runner.py  <yaml_path>  [--resume]
```

`--resume` is passed only when `simulation.resume_from` is set in the project;
it tells the runner not to increment the output directory (`new_experiment=False`).
There is no `--output` flag — the runner determines its output directory from
`output.folder` in the YAML.

Output is streamed line-by-line in a daemon thread via `subprocess.Popen` with
`stdout=PIPE, stderr=STDOUT`, and appended to a log pane in real time. The log
is capped at 500 lines. A new run clears the previous log.

---

### dolfin.Constant wrapping for spatial parameters

All scalar spatial parameters passed to `spatial.add_rule()` or
`spatial.add_material_data()` are wrapped as `dolfin.Constant(float(value)) * U(unit)`
rather than `float(value) * U(unit)`.  FEniCS compiles finite-element forms symbolically
and caches them by form signature.  Wrapping a scalar in `dolfin.Constant` makes its
value a mutable coefficient rather than a literal in the expression tree; when the
constant's value changes (e.g. between batch sub-runs or resumed sweeps) the cached
compiled form can be reused without recompilation.

`parse_spatial_quantity(entry, U)` performs this wrapping.  `parse_quantity(entry, U)`
(the plain version) is kept for contexts where a Python float is required at parse time —
optical field computation, voltage sweep values, reference voltage.

---

## Batch Processing

*Implemented May 2026 — `gui/batch.py` + Simulation panel sweep card.*

### Motivation

Researchers commonly need to run the same device under a range of parameter values:
varying doping, layer thickness, IB energy, optical cross-section, etc.  The existing
`fourlayer_example.py` handles this with `multiplex_setup()` + `multiprocessing.Pool`,
placing each sub-run in a numbered subdirectory (`experiment_folder/0/`, `/1/`, …).
The GUI provides a comparable workflow without requiring the user to write Python.

### Sidecar file

The batch configuration is stored in a **sidecar file** next to the project YAML:

```
<project_stem>.batch.yaml
```

The runner (`simudo_1d_runner.py`) never reads or touches this file — it is consumed
exclusively by the GUI batch launcher.  Keeping it separate keeps the base project YAML
clean and directly runnable as a single simulation without modification.

### Sidecar YAML schema

```yaml
mode: cartesian          # "cartesian" (Cartesian product, default) or "zip" (paired)
max_parallel: 4          # maximum simultaneous runner processes
variations:
  - label: "p-layer / thickness (um)"
    target:
      kind: layer_thickness
      container_name: p-layer
      key: ""
    values: [0.5, 1.0, 1.5]
  - label: "absorber / CB/energy_level"
    target:
      kind: layer_property
      container_name: absorber
      key: CB/energy_level
    values: [1.1, 1.3, 1.5]
```

**`target.kind`** is one of six strings:

| kind | What it addresses | `container_name` | `key` |
|------|-------------------|-----------------|-------|
| `layer_thickness` | A layer's thickness value | layer name | `""` |
| `layer_property` | A layer property `{value, unit}` | layer name | property key |
| `overlay_property` | An overlay region property `{value, unit}` | region name | property key |
| `layer_mesh_start` | Per-layer mesh starting spacing (µm) | layer name | `""` |
| `layer_mesh_factor` | Per-layer mesh growth factor | layer name | `""` |
| `mesh_max_edge_length` | Global max mesh edge length (µm) | `""` | `""` |

Only the numeric `value` of a `{value, unit}` pair is varied; the unit remains fixed
from the base YAML.

**`mode: zip`** validation: if the swept value lists have unequal lengths, the GUI
shows an orange warning and uses the shortest list length.

### Parameter picker

`enumerate_sweep_params(project)` in `batch.py` walks the in-memory project model and
returns all sweepable parameters as `(display_label, BatchTarget)` pairs.  The GUI
presents these as a flat dropdown labelled with friendly strings such as
`"p-layer / thickness (um)"` or `"absorber / CB/energy_level"`.

A current-value hint is displayed below each parameter dropdown, showing the value
currently set in the project (e.g. `"1.0 um"`, `"300.0 K"`, `"(not set)"`).

### Sub-run output layout

```
out/2026May11/a/          ← base output folder (from project YAML; incremented if exists)
  0/                      ← first sub-run
    <project>.yaml        ← base YAML modified for this combination; output.folder = "."
    sim_V.csv             ← simulation output lands here
    info.log
    …
  1/
    <project>.yaml
    …
```

The `output.folder` key in each sub-run's YAML is set to `"."` so runner output lands
in the same numbered subdirectory as the YAML.  Sub-runs pass `--resume` to the runner
so it does not call `ensure_new_dir` again.

**Collision avoidance:** `generate_sub_yamls()` calls `ensure_new_dir(base_out_dir)`
before creating any subdirectories, using the same incrementing logic as the single
runner.  If `out/2026May11/a/` already exists, the batch will write to
`out/2026May11/b/`, then `b/0/`, `b/1/`, etc.

**Batch-aware schematic:** the Output panel detects the sub-YAML in the output
directory (`glob("*.yaml")`) and loads its layer thicknesses when rendering the
band-diagram and spatial-profile layer schematics, so the schematic reflects the
actual swept geometry rather than the base project's layers.

### Parallelism

`BatchLauncher` runs a supervisor daemon thread that manages all sub-runs.  A
`threading.BoundedSemaphore(max_parallel)` limits concurrency.  The semaphore is
acquired before each sub-run starts and released when it finishes (or errors), so at
most `max_parallel` runner processes run simultaneously.

The "Max number of cores to use" spinner in the GUI is initialised from
`backend.get_core_count()`:

| Backend | `get_core_count()` |
|---------|-------------------|
| Local | `os.cpu_count()` |
| Docker | `docker exec <container> nproc` |
| SSH | `nproc` over SSH |

**Docker:** multiple `docker exec` calls against the same container run as independent
OS processes.  FEniCS's JIT cache uses file locking and is safe under concurrent access.
Each sub-run writes to a distinct numbered subdirectory so there are no output conflicts.

**SSH (v1 limitation):** all sub-runs go to the same remote host.  The SSH backend
rewrites `output.folder`, so numbered subdirs produced by `generate_sub_yamls` will
not be the final output location for SSH runs.  A future version should stage each
sub-YAML independently and sync its output back to the correct local subdirectory.

### Comparison with SCAPS-1D batch mode

| Feature | SCAPS-1D | Simudo GUI (v1) |
|---------|----------|-----------------|
| Max parameters varied | 9 | Unlimited |
| Value spacing | linear / log / custom list | Explicit list (GUI has linspace helper) |
| Parallel execution | No (manual multi-instance) | Yes (`BoundedSemaphore`) |
| Batch definition saved | Yes (.bdf file) | Yes (`<project>.batch.yaml` sidecar) |
| Script language | Custom (SCAPS script) | None — GUI only |
| Cross-run comparison | Within SCAPS plots | Out of scope v1 (use Jupyter) |

The main improvement over SCAPS is native parallelism.  The main limitation vs. SCAPS
scripts is that the GUI does not support conditional logic or optimization loops —
those remain Python-only.
