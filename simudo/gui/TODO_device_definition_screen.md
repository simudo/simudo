# Device Definition Screen — Decisions To-Do

Decisions needed before and during building Step 1 of the build sequence:
the screen that lets a user define a complete Simudo device and saves it as a
valid project YAML. Checked off as each decision is made.

---

## Screen structure

- [x] **Overall layout** — Left-nav sidebar with named sections; clicking a section
  shows its panel on the right. Sections map onto YAML top-level keys. Chosen
  over tabs and single-scroll because sections are logically distinct and unequal
  in size, and this lets the user jump directly to what they need.

- [x] **New vs. open project workflow** — Opens immediately with a blank project
  (default output folder relative to launch directory); no modal file-picker at
  startup. App opens directly to the Layers panel. A persistent "Recent ▼"
  dropdown button in the header gives access to recent projects at all times —
  before and after starting work. Two load modes: *settings only* (for modifying
  a past device) or *settings + results* (to inspect or continue a past run).
  YAML path shown in header.

---

## Layer stack

- [x] **Layer stack visualization** — Horizontal coloured block diagram above a
  table. Schematic is proportional to thickness with a minimum block width so
  thin layers (e.g. FSF) remain visible; colour-coded, matching the table rows.
  Clicking a block or row opens its parameter panel on the right.
  - Interface boundaries between adjacent layers shown as clickable dividers in
    the schematic (and/or between rows in the table) to access interface
    properties (e.g. thermionic heterojunction BC toggle).
  - External boundaries (left and right contacts) shown as distinct markers at
    the ends of the schematic with a click target for contact BC configuration.

- [x] **Layer operations** — Drag handles on table rows for reordering (preferred
  over up/down arrows). Add layer button (inserts at bottom or after selected
  row). Delete button per row. Right-click context menu (duplicate, insert
  above/below) as a stretch goal.

- [x] **Mesh parameters per layer** — Global `max_edge_length` shown prominently
  in the top-level Mesh section. Per-layer mesh parameters (`start`, `factor`)
  live inside each layer's detail panel in a collapsible "Mesh" subsection,
  hidden by default. Exposed to users with friendlier names:
  - `start`  → "Starting mesh spacing"
  - `factor` → "Mesh spacing expansion factor"

---

## Materials

- [x] **Material library location** — Two-tier:
  - Bundled read-only library: `code/materials/` (sibling of `gui/`, not inside
    it), shipped with the repo.
  - Optional user library: directory path from a per-user config file
    (e.g. `~/.simudo_gui.yaml`). User materials take precedence over bundled
    ones when names conflict.
  The runner's `--library-dir` flag points at whichever library is relevant for
  a given run; the GUI resolves this automatically.

- [x] **Material selection workflow and layer parameter editing** — Decided:
  - User picks a material for each layer from a dropdown (library or inline).
  - GUI immediately introspects the material via **AST analysis** of the source
    file (host-side, no dolfin needed — see SCHEMA); simple `U("…")` values are
    displayed, UFL expressions show as "evaluated at runtime."
  - Per-layer overrides (doping, process parameters, etc.) are shown in the same
    layer detail panel, below the material summary.
  - **Doping** (`poisson/static_rho`) is prominent in the layer panel (not hidden):
    a numeric field + unit + p/n radio button (GUI handles the sign). Default:
    uniform. Advanced expression mode is a TODO (see below).
  - **Copy settings**: a "Copy properties from…" action on each layer lets the
    user clone per-layer overrides from another layer without re-entering them.
  - **Band naming convention**: GUI prompts/encourages naming the conduction band
    `CB` and valence band `VB` for compatibility with library material files.
    A warning is shown if the user uses different names while also selecting a
    library material.
  - **Two-pass layer editing**: Layers panel has two phases:
    1. First pass (before bands/processes defined): geometry, material, doping.
    2. Second pass (after bands and interband processes defined): per-process
       parameters are shown per layer, colour-coded by covered (by material) vs.
       missing. The sidebar guides the user through this order.
  - **Spatially varying doping** (advanced, TODO): add an expression-mode toggle
    to the doping field; provide examples in the help text. Ensure the YAML
    stores this as a string expression that the runner passes through `eval()`
    in a controlled namespace.

---

## Per-layer / per-region properties

- [x] **Value + unit entry** — Numeric input + context-aware unit dropdown. For
  known property keys the dropdown pre-selects the canonical unit and offers
  sensible alternatives; for unknown keys it falls back to a free-text unit
  field. The same widget is reused everywhere a `{value, unit}` pair appears.

- [x] **Advanced expression mode** — Included in v1 but unobtrusive: each
  property row has a small toggle (e.g. `ƒ` button) that switches that field
  from value+unit mode to a free-text expression box. Basic Python syntax
  checking only (no dolfin needed). A tooltip/link to examples makes it
  discoverable without cluttering the default view.

- [x] **Overlay regions** — Displayed as a collapsible section below the layer
  table, inside the Layers panel (not a separate sidebar section).
  - A built-in `domain` overlay is always present and cannot be deleted. It
    covers the entire simulation domain. Its properties have lower priority than
    per-layer settings but higher priority than material file defaults.
  - Additional overlay regions default to higher priority than both `domain` and
    per-layer settings, functioning as local overrides.
  - Two ways to add an overlay:
    1. Click on the schematic — hover highlights candidate spans; clicking
       confirms the span and opens a name/confirm popup inline.
    2. Click "+ Add overlay" and enter start/stop coordinates directly.
  - Extents are stored as coordinate ranges `{start, end}` in the YAML (runner
    support for coordinate-range extents must be implemented when this is built).

---

## Bands

- [x] **Band configuration UI** — Separate "Bands" section in the sidebar nav.
  - **Table layout**: one row per band; columns are name, type, carriers, extent
    summary. Click a row to open its detail panel. Trash-can delete button per row.
  - **Default state** when the panel first opens: two pre-populated bands —
    `CB` (Boltzmann, electrons, extent = domain) and `VB` (Boltzmann, holes,
    extent = domain).
  - **"+ Add band"** defaults to inserting a Sharp Intermediate Band (sign = −1).
    A secondary option adds a Fermi band (sign = −1).
  - **Type labels** (GUI display → internal YAML value):
    - "Boltzmann" → `nondegenerate`
    - "Fermi" → `degenerate`
    - "Sharp Intermediate Band" → `intermediate`
    - "Mixed QFL / Boltzmann" → `mixed_qfl_nondegenerate`
  - **Sign** shown as radio: "Electrons (−)" / "Holes (+)". Auto-set based on
    name (`VB` → Holes, anything else → Electrons); user can override.
  - **Extent** specified as a checkbox list of all defined layers and overlay
    regions in the detail panel, plus an "Entire device (domain)" shortcut at
    the top. If a needed region has not yet been defined, the user can navigate
    back to the Layers panel to add it, then return.
  - **Naming nudge**: show a soft warning if the band name doesn't match `CB`,
    `VB`, or `IB[…]` *and* a library material is selected — but multi-valley
    names like `CBG`, `CBL`, `CBX` are legitimate and the warning text should
    acknowledge this.

---

## Interband processes (formerly "EOPs")

- [x] **Process list UI** — Table of rows; one row per process. Columns: name,
  class (full class name, verbatim from code), bands summary (e.g. `VB → CB`
  or `VB → IB → CB`), radiative recombination checkbox, trash-can delete.
  - **Class names are never abbreviated** in the UI — `NonOverlappingTopHatBeerLambert`,
    `NonOverlappingTopHatBeerLambertIB`, etc. are shown verbatim to stay
    distinct as new variants are added to Simudo.
  - **ⓘ button** per row (and in the add-process form) shows the Python
    docstring for that class, extracted via AST — no dolfin needed.
  - **"+ Add process"** expands an inline form below the table (no modal):
    - Class dropdown (auto-discovered; see below)
    - Name field — auto-suggested from class + band selection (e.g. `opt_ci`,
      `opt_ci_2` for a second instance); validated for uniqueness on blur.
    - src\_band / dst\_band dropdowns (populated from defined band names).
      trap\_band dropdown appears only for 3-band classes.
    - Radiative recombination checkbox (visible only for absorption classes;
      defaults **on**).
  - **Live update** — no confirm/cancel; fields update state immediately.
    Any process can be revised or deleted at any time.
  - **Auto-discovery of classes** — two tiers:
    1. *Built-in*: GUI scans `electro_optical_process.py` via AST at startup
       and discovers all concrete subclasses of `ElectroOpticalProcess`
       automatically. No hardcoded class list.
    2. *User-defined*: optional `custom_eop_files` list in the project YAML
       (paths relative to the YAML file). GUI scans those files via AST;
       runner imports them dynamically via `importlib`. Paths live in the
       project YAML (not only in `~/.simudo_gui.yaml`) so the project is
       self-contained and portable. A small "Custom EOP files" list at the
       top of the Processes panel lets users add/remove paths; the class
       dropdown refreshes immediately.

---

## Optical fields

- [x] **Optical field UI** — The entire Opt. Fields panel is **optional**. Users
  running dark (equilibrium or voltage-sweep) simulations leave it empty.
  Absorption EOPs can still be defined without optical fields, since they
  contribute radiative recombination even under dark conditions.
  - **Table layout**: name, direction, photon energy, mode, intensity summary,
    trash-can delete. "+ Add field" expands an inline live-update form.
  - **Form fields**:
    - Name — free text, validated unique.
    - Direction — `+x` / `−x` toggle. Future TODO: generalise to angle-of-
      incidence (light propagating at an angle to the layer normal, which
      is physically meaningful even in 1D simulations).
    - Photon energy — value+unit widget. In Blackbody mode a "= E_min + ε"
      checkbox auto-sets this to E_min + 0.0001 eV (standard usage: field
      sits just inside the EOP absorption window).
    - Intensity mode — dropdown: *Explicit* or *Blackbody spectrum*.
      - Explicit: value+unit (default mW/cm²).
      - Blackbody spectrum: T_source (K), concentration (dimensionless,
        labelled "× suns", may be non-integer e.g. 0.1), E_min (eV),
        E_max (eV) + "No upper bound (∞)" checkbox.
  - Future TODO: AM1.5G spectrum mode (N frequency bins over a range).

---

## Boundary Conditions

- [x] **Boundary Conditions panel** — Sidebar section called "Boundary Conditions"
  (YAML key remains `contacts:` — that is a runner detail not shown in the UI).
  Two sections:

  **Section 1 — External contacts** (always exactly two in 1D):
  - Fixed two-column layout: left contact and right contact, side by side.
  - Each contact: role toggle (`Swept` / `Reference`). No voltage field exposed —
    the swept contact's voltage is the sweep parameter (controlled by the stepper);
    the reference contact is always 0 V.
  - Per-band BC table: one row per defined band (all bands shown regardless of
    extent — assigning a BC to a band at a face it doesn't reach is costless).
    BC dropdown: `Ohmic` / `Zero current` (future: `SRV = <value>`). Default: Ohmic.
  - Future SRV support: the dropdown is designed to accommodate a third option
    with an associated value field, even though only 0 and ∞ are currently
    implemented in Simudo.

  **Section 2 — Band extent boundaries**:
  - Shown only if any band has a non-`domain` extent.
  - One sub-table per finite-extent band with two rows: left boundary and right
    boundary, each with an independent BC dropdown (`Zero current` default,
    future: `Ohmic`, `SRV`).
  - **Runner implementation note**: BCs in Simudo are applied to *facets* based
    on region topology, not spatial coordinates. To apply a BC to the left
    boundary of band B's extent, the runner must call
    `B.boundary(left_adjacent_region).both()`, where `left_adjacent_region` is
    whichever region lies immediately to the left of B's extent.
    - For layer-list extents: the adjacent region is the layer immediately
      before/after the extent layers in the stack order.
    - For coordinate-range extents: the runner must determine which region
      contains the boundary coordinate, then use that region as the adjacency
      argument.

  **Internal interface BCs** (thermionic heterojunction):
  - Not in this panel. Accessed by clicking interface dividers in the layer
    schematic (Layers panel). A note at the bottom of the BC panel points there.

  **Guiding the user back to Layers** after bands/processes are defined:
  1. Sidebar nav badge — "Layers" nav item shows a yellow `●` when per-process
     parameters are still missing; turns green when all `required_spatial_params`
     are set for all regions.
  2. Contextual footer hint at the bottom of the Bands and Processes panels:
     *"Once bands and processes are defined, return to Layers to set per-region
     parameters for each process (e.g. `SRH/CB/tau`, `sigma_opt`)."*
  3. In the Layers detail panel (second pass): missing required parameters for
     a process are highlighted in amber; the layer row in the table gets a yellow
     indicator.

---

## Simulation settings

- [x] **Simulation mode** — Named mode selector with four options (sets
  `intensity_ramp.enabled` and `voltage_sweep.enabled`):
  - *Equilibrium only* — no ramp, no sweep
  - *Dark J-V* — no ramp, voltage sweep
  - *Illuminated (steady-state)* — intensity ramp only, no sweep
  - *Illuminated J-V* — intensity ramp then voltage sweep (default)
  The two illuminated options are greyed out if no optical fields are defined.

- [x] **Voltage sweep value entry** — `linspace` helper (from / to / N steps)
  that expands to an explicit list, plus a read/edit view of the resulting list
  for users who want irregular spacing. The YAML always stores the explicit list.

- [x] **Checkpoint value entry** — Editable list (add/remove individual voltage
  values) for `at_values`, plus `checkpoint_at_end` checkbox (on by default).
  A note explains that checkpoints allow reloading spatial data and continuing
  or branching a previous simulation. No "every N steps" helper — checkpoints
  are expensive and users typically specify a few specific voltages.

- [x] **`selfconsistent_optics`** — Shown in the main simulation panel (not
  hidden), with contextual guidance text auto-derived from the defined processes:
  - Any `NonOverlappingTopHatBeerLambertIB` present → *"Recommended: ON —
    IB absorption depends on band filling and must be solved self-consistently."*
  - No IB processes → *"Recommended: OFF — fixed absorption coefficients can
    be computed once, which is faster."*
  User can always override the recommendation.

- [x] **Advanced simulation parameters** — Collapsible "Advanced" section:
  - *Initial step size* (`step_size`) for intensity ramp and voltage sweep
    steppers separately.
  - *Resume from checkpoint* (`resume_from`): collapsed file-picker field.
    When set, simulation starts from that checkpoint file and skips already-
    completed stages. A note explains this use case.

---

## Output / execution

- [x] **Output folder entry** — Text field showing the relative path, a
  folder-picker button for the parent directory. The resolved absolute path is
  shown alongside in smaller text. A note explains the auto-increment behaviour:
  *"If the folder already exists, a suffix is appended automatically (a → b → …)
  to avoid overwriting previous results."*
  `spatial_profiles` and `xdmf_mesh` as labelled checkboxes (both on by default).

- [x] **Execution profile** — Not on the device definition screen. The
  `execution:` section is consumed by the launcher (build step 2), not the
  runner. Device definition saves a pure physics YAML; how to run it is
  configured separately on the execution panel.

---

## Validation and feedback

- [x] **When is the YAML validated?** — The YAML file is written on every
  completed field edit (focus-out / Enter), so it is always current. A crash
  loses at most a few keystrokes, not a whole session. The "Save" button in the
  header serves as an explicit confirmation and triggers a full validation pass.
  Schema completeness only for v1; physics sanity checks (e.g. band/optical
  energy consistency) deferred.

- [x] **How are errors shown?** — Inline red border on the offending field plus
  a short error message below it. A count badge in the header ("3 errors") links
  to a summary list. Both mechanisms together so errors are visible both locally
  and globally.
