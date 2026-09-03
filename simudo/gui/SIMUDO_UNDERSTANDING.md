# Simudo Codebase Understanding
*For GUI development planning — April 2026*

---

## 1. Problem Setup / Definition

### Overall Workflow

Setting up a Simudo problem means constructing a `ProblemData` object (`simudo/physics/problem_data.py`) that encodes geometry, materials, bands, physics processes, and boundary conditions. Problems are solved in three stages: local charge neutrality → thermal equilibrium → full coupled solution. Each stage initializes from the previous one.

The canonical architecture to follow is `simudo/example/fourlayer/fourlayer.py`, not `pn_diode/` (which predates many design patterns).

---

### Geometry and Mesh

#### 1D Layered Structure

The primary helper is `ConstructionHelperLayeredStructure` (`simudo/mesh/construction_helper.py`), which builds 1D layered device meshes. Each layer is a dict:

```python
layers = [
    dict(name="fsf",  material="fsf",          thickness=0.05, mesh=s),
    dict(name="p",    material="semiconductor", thickness=1.0,  mesh=s),
    dict(name="I",    material="IB",            thickness=0.3,  mesh=s),
    dict(name="n",    material="semiconductor", thickness=1.0,  mesh=s),
]
# s = {"type": "geometric", "start": 0.0005, "factor": 1.2}
```

where `mesh` controls the 1D mesh density: geometric progression starting at `start` (µm) and growing by `factor`. The `mesh_unit` parameter sets the physical unit for all dimensions (typically `1 micrometer`).

#### 2D Extension (for future reference)

The 2D case (`../photodetector/run_photodiode2d.py`) adds:
- `width_domain`, `width_emitter`, `width_contact` (x-direction extents)
- Separate mesh growth factors for x and y: `Lx0`, `Lx_factor`, `Ly0`, `Ly_factor`
- `refinement_iterations`, `debug_mesh_method` (BAMG vs Gmsh)
- Contact geometry (partial-width contacts)

The 1D GUI can be designed so that these 2D parameters are simply absent/hidden rather than architecturally excluded.

---

### Topology: CellRegions and FacetRegions

Defined in `simudo/mesh/topology.py`. After mesh construction, all logical regions and boundaries are expressed as `CellRegion` and `FacetRegion` objects that are later evaluated into mesh marker values.

#### CellRegion Operations

| Operator / Method | Returns | Meaning |
|-------------------|---------|---------|
| `A \| B` | `CellRegionOr` | Union of two cell regions |
| `A & B` | `CellRegionAnd` | Intersection |
| `A ^ B` | `CellRegionXor` | Symmetric difference |
| `A - B` | `CellRegionSub` | Subtraction (A minus B) |
| `A.boundary(B)` | `CellBoundaryFacetRegion` | Signed facets on the boundary between A and B |
| `A.internal_facets()` | `CellRegionInternalFacetRegion` | All interior facets of region A |
| `A.subdomain_internal_facets()` | `CellRegionSubdomainInternalFacetRegion` | Interior facets of A, excluding inter-subdomain boundaries |
| `CellRegion.null()` | null region | Matches nothing (static) |

The `.boundary(B)` method returns **signed** facets: the sign encodes orientation (which side of the facet is "inside" which region). The convention is that `A.boundary(B)` and `B.boundary(A)` return the same facets with opposite signs (i.e., `B.boundary(A) == A.boundary(B).flip()`).

#### FacetRegion Operations

| Operator / Method | Returns | Meaning |
|-------------------|---------|---------|
| `A \| B` | `FacetRegionOr` | Union |
| `A & B` | `FacetRegionAnd` | Intersection |
| `A ^ B` | `FacetRegionXor` | Symmetric difference |
| `A - B` | `FacetRegionSub` | Subtraction |
| `A.flip()` | `FacetRegionFlip` | Invert sign of all facets in A |
| `A.both()` | `FacetRegionBothSides` | Both orientations (see below) |
| `A.unsigned()` | `FacetRegionUnsigned` | Erase sign (set sign=+1 for all) |

#### The `.both()` Operator

Facets in Simudo are internally represented as `(facet_value, sign)` pairs, where `sign ∈ {−1, +1}`. A signed facet region is directional — it distinguishes "facing left" from "facing right". This matters when applying boundary conditions that depend on normal orientation.

`A.both()` creates a facet region containing **both sign orientations** of every facet in A:

```python
f.both() == (f | f.flip())
```

**Use case:** when you want to apply a boundary condition on a facet regardless of which side you're approaching from. The most common example is applying a zero-current (no-flux) condition on all non-conductive boundaries — you want to enforce it on both sides of every such facet, so you write `F.nonconductive.both()`. Without `.both()`, the BC would only apply to facets with the matching sign orientation, which can silently miss half the facets.

#### Typical topology setup (from `fourlayer.py`):

```python
def topology_standard_contacts(mdc):
    R = mdc.cell_regions
    F = mdc.facet_regions

    R.domain = R.p | R.I | R.n | R.fsf
    R.exterior = R.domain.complement()  # or defined separately

    F.left_contact  = R.exterior_left.boundary(R.domain)
    F.right_contact = R.domain.boundary(R.exterior_right)
    F.nonconductive = (F.left_contact | F.right_contact).flip()  # exterior-facing
    F.pn_junction   = R.p.boundary(R.n)
```

---

### Materials

Materials are Python classes subclassing `Material` (`simudo/physics/material.py`). The class implements `get_dict()` which returns a dict of pint-valued parameters.

#### Standard material parameter keys

| Key | Symbol | Typical unit |
|-----|--------|-------------|
| `CB/energy_level` | E_C | eV |
| `CB/effective_density_of_states` | N_C | cm⁻³ |
| `CB/mobility` | μ_n | cm²/Vs |
| `VB/energy_level` | E_V | eV |
| `VB/effective_density_of_states` | N_V | cm⁻³ |
| `VB/mobility` | μ_p | cm²/Vs |
| `IB/energy_level` | E_IB | eV |
| `IB/effective_density_of_states` | N_IB | cm⁻³ |
| `IB/mobility` | μ_IB | cm²/Vs |
| `poisson/permittivity` | ε_r ε_0 | F/m (or C²/N/m) |
| `SRH/CB/tau` | τ_n (SRH) | s |
| `SRH/VB/tau` | τ_p (SRH) | s |
| `opt_cv/alpha` | α_CV | cm⁻¹ |
| `opt_ci/sigma` | σ_CI | cm² |
| `opt_iv/sigma` | σ_IV | cm² |

After construction, `material.register(pdd.spatial)` pushes all parameters to the spatial manager as region-based rules.

**GUI material library approach:** Materials should be stored as YAML/JSON files with these keys. The GUI reads the library directory at startup and shows available materials. Users can load a material and then override individual parameters per layer.

---

### Spatial Rules and Boundary Conditions

The `Spatial` object (`simudo/fem/spatial.py`) maps physical quantities to regions and facets with a priority system (higher priority overrides lower):

```python
spatial.add_rule('temperature', R.domain, U('300 K'))
spatial.add_rule('poisson/static_rho', R.pSi, -doping * U('e/cm^3'))  # static charge = doping

spatial.add_BC('poisson/phi', F.p_contact, phi0)                        # Ohmic voltage BC
spatial.add_BC('CB/j',        F.nonconductive.both(), U('A/cm^2') * zerovec)  # zero current
spatial.add_BC('CB/u',        F.n_contact, CB.thermal_equilibrium_u)   # equilibrium carrier density
```

Doping is represented as a static charge density (`poisson/static_rho`): positive for donors (n-type), negative for acceptors (p-type).

---

### Bands

Bands are added to `pdd` via:

```python
CB = pdd.easy_add_band('CB')                              # Conduction band
VB = pdd.easy_add_band('VB')                              # Valence band
IB = pdd.easy_add_band('IB', band_type=IntermediateBand,
                        subdomain=R.IB_region)            # IB confined to subdomain
```

Band types (`simudo/physics/poisson_drift_diffusion.py`):

| Type | Description |
|------|-------------|
| `NondegenerateBand` | Standard parabolic band (default for CB, VB) |
| `IntermediateBand` | Sharp (delta-function DOS) intermediate band |
| `MixedQflNondegenerateBand` | Mixed quasi-Fermi-level formulation (numerical stability) |

The IB can be confined to a `subdomain` (a `CellRegion`), so it only exists in the IB layer.

---

### Electro-Optical Processes

All processes are in `simudo/physics/electro_optical_process.py`. They are added via:

```python
pdd.easy_add_electro_optical_process(ProcessClass,
    src_band=CB, dst_band=VB, name='opt_cv', ...)
```

#### Full Process Class Reference

**Dark (no optical field) processes:**

| Class | Parameters | Description |
|-------|-----------|-------------|
| `SRHRecombination` | `src_band`, `dst_band` [, `tau_from_spatial=True`] | Shockley-Read-Hall recombination. Typically `src_band=VB`, `dst_band=CB`. Lifetime τ read from spatial rules (`SRH/CB/tau`, `SRH/VB/tau`) when `tau_from_spatial=True`. |
| `NonRadiativeTrap` | `src_band`, `dst_band`, `trap_band` | SR trapping via an explicit trap-level band object. Use when IB acts as non-radiative trap. |
| `ShockleyReadBand2BandTrap` | `src_band` (R), `dst_band` (D) | Shockley-Read-like trapping between two regular (parabolic DOS) bands. |
| `ShockleyReadTrap2Trap` | `src_band` (R), `dst_band` (D) | Shockley-Read-like trapping between two sharp (single-level) trap bands. |

**Optical absorption / radiative recombination processes:**

| Class | Parameters | Description |
|-------|-----------|-------------|
| `NonOverlappingTopHatBeerLambert` | `src_band`, `dst_band` [, `enable_radiative_recombination=True`] | Beer-Lambert absorption between two regular bands (e.g., VB→CB). Uses constant absorption coefficient α (top-hat in photon energy). Includes radiative recombination (Strandberg formula) by default. The optical field and α must be registered via spatial rules. |
| `NonOverlappingTopHatBeerLambertIB` | `src_band`, `dst_band`, `trap_band` [, `enable_radiative_recombination=True`] | Beer-Lambert absorption where one band is the IB (trap-like). Absorption cross-section σ is modulated by IB filling fraction. Used for VB→IB and CB→IB transitions. |

**Mixin building blocks** (not instantiated directly, but determine available parameters):

| Mixin | Role |
|-------|------|
| `DarkEOPMixin` | Marks process as having no optical absorption |
| `TwoBandEOPMixin` | Provides `src_band` / `dst_band` convention |
| `TrapEOPMixin` | Adds trap capture coefficient, trap DOS, thermal velocity |
| `RadiativeEOPMixin` | Computes Strandberg radiative integral, absorption bounds |
| `AbsorptionAndRadiativeRecombinationEOPMixin` | Combines optical absorption + radiative recombination |

**Sign convention:** For `TwoBandEOPMixin`, the `dst_band` gains carriers when net generation is positive (i.e., when illuminated). For SRH, this means `dst_band=CB` (electrons are the destination of recombination in the conventional description).

---

### Optical Fields

Optical fields represent monochromatic (or narrow-band) illumination propagating through the device. Added via:

```python
optical.easy_add_field(
    key='forward_cv',
    photon_energy=U('1.5 eV'),
    direction='+x',               # '+x' = left-to-right; '-x' = right-to-left
    solid_angle=...,              # steradians; relates flux to intensity
)
```

The field enters the device at the illuminated face with a boundary condition on photon flux (set via `spatial.add_BC`). It decays exponentially according to the sum of all `get_alpha()` contributions from processes that reference it.

In the fourlayer example, three separate optical fields are defined:
- `forward_cv`: CB↔VB photons (above-gap)
- `forward_ci`: CB↔IB photons
- `forward_iv`: VB↔IB photons

With `selfconsistent_optics=True` in the stepper, optical intensities are recomputed at each Newton iteration.

---

### Unit System

All physical quantities use **pint** units (`simudo/util/pint.py`):

```python
U = make_unit_registry(("mesh_unit = 1 micrometer",))
U('300 K'), U('1e18 /cm^3'), U('1.5 eV'), U('100 mW/cm^2')
```

Every parameter passed to `add_rule()`, `add_BC()`, or `get_dict()` must be a pint `Quantity`. The GUI must collect units alongside numeric values; dropdowns for common unit choices (eV, cm⁻³, cm²/Vs, etc.) will reduce errors.

---

## 2. Running Simulations

### Three-Stage Initialization

Problems are always solved in three stages (internal to any run script):

1. **Local charge neutrality** (`goal='local charge neutrality'`): ignores Poisson, gives a simple initial guess based on charge balance in each region
2. **Thermal equilibrium** (`goal='thermal equilibrium'`): solves Poisson + equilibrium carrier distribution, no current
3. **Full coupled** (`goal='full'`): solves the full coupled PDD system with applied bias and/or illumination

Each stage calls `problem.pdd.initialize_from(previous_problem.pdd)` to warm-start from the previous one. The GUI should hide this from users and run all three automatically.

### Parameter Sweeps (Steppers)

Defined in `simudo/physics/steppers.py`:

| Class | Sweeps | Primary use |
|-------|--------|------------|
| `VoltageStepper` | Applied bias V | J-V curve |
| `OpticalIntensityAdaptiveStepper` | Optical intensity (0→1 sun) | Illumination ramp |
| `NonequilibriumCoupledStepper` | Base class | Custom sweeps |

Typical voltage sweep:

```python
V_ext = dolfin.Constant(0.0) * U.V
stepper = VoltageStepper(
    solution=problem,
    constants=[V_ext],
    parameter_target_values=np.linspace(0, 1.2, 25),
    output_writer=OutputWriter(filename_prefix='run1 V_sweep'),
    checkpoint_write_values=[0.0, 0.4, 0.8, 1.2],
    checkpoint_write_filename='ckpt/V',
)
stepper.solve()
```

The stepper uses adaptive step-size control: on Newton failure it backtracks and retries with a smaller step.

In the fourlayer example, a **two-stage sweep** is used: first ramp illumination from 0→1 sun (optical stepper), then sweep voltage (voltage stepper), using the illuminated equilibrium as initial condition for the voltage sweep. This is the recommended workflow for solar cell simulations.

### Newton Solver

`NewtonSolver` (`simudo/fem/newton_solver.py`): Newton-Raphson with line search, backed by MUMPS via PETSc.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `maximum_iterations` | ~200 | Max Newton steps before failure |
| `relative_tolerance` | 1e-4 | Convergence criterion |
| `absolute_tolerance` | 1 | Adjustable weight |

Variants: `NewtonSolverMaxDu` (caps solution step size), `NewtonSolverLogDamping`.

### Types of Simulation the GUI Should Support

1. **J-V curve** — voltage sweep, dark or illuminated (most common)
2. **Illumination ramp** — optical intensity sweep at fixed voltage (or zero bias)
3. **Band diagram at equilibrium** — thermal equilibrium solve only, no sweep
4. **Band diagram at bias** — voltage sweep stopped at one point, spatial output requested

All are covered by the existing stepper/solver framework.

### Logging

`TypicalLoggingSetup` (`simudo/util/logging.py`) creates `debug.log` and `info.log` in the output directory. Call before running.

---

## 3. Outputs and Post-Processing

### Primary Output Pattern: `plot_meta.yaml` + `sweep_extraction.py`

The preferred pattern (from `fourlayer/`) is:

1. The `OutputWriter` writes one `plot_meta.yaml` file per solved parameter value.
2. `sweep_extraction.py` is the **canonical post-processing module**. It does three things:
   - Reads all `*.plot_meta.yaml` files in a run directory and assembles scalar quantities (J, V, power, etc.) into a pandas DataFrame (`SweepData.jv`)
   - Retrieves **spatial data** for any given row in that DataFrame — reading either the old `.csv.0` line-cut files or the newer `.xdmf` checkpoint files depending on what is available (`SweepData.get_spatial_data(row)`)
   - Provides **plotting schemas** for spatial quantities (e.g., band diagrams), operating on the spatial objects returned by `get_spatial_data`

`simudo/plot/postprocessing.py` is not typically used.

### OutputWriter

`OutputWriter` (`simudo/io/output_writer.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `filename_prefix` | required | Prefix for all output files |
| `parameter_name` | `'parameter'` | Name of swept variable (appears in filenames) |
| `plot_iv` | `True` | Write summary CSV |
| `plot_1d` | `False` | Write spatial line-cut CSVs at each step |
| `plot_mesh` | `False` | Write XDMF mesh+solution files |
| `line_cut_resolution` | 5001 | Points along device for line-cut |
| `meta_extractors` | `[]` | List of data extractor classes |

### Output File Formats

#### Per-step YAML metadata (`*.plot_meta.yaml`)
- One file per solved parameter value (e.g., `pd_V=0.50000.plot_meta.yaml`)
- Contains scalar quantities: currents at contacts, band info, parameter value
- Primary data source for J-V analysis via `SweepData`

#### Summary CSV (`*_parameter.csv`)
- One row per solved parameter value
- Columns: sweep parameter, integrated currents at each contact, band quasi-Fermi levels, carrier densities

#### Line-Cut CSV (`*.csv.0`, `*.csv.1`, ...)
- Spatial profiles along the 1D device at each parameter step
- Columns: `coord_x`, `coord_y`, φ, quasi-Fermi levels, carrier densities, currents
- Used for band diagrams and spatial profiles at a given bias

#### XDMF + HDF5 checkpoints (`*.xdmf`, `*.h5`, `*.yaml`)
- Full mesh + solution fields, viewable in ParaView or VisIt
- Also used for **simulation checkpointing** (see below)
- `SpatialXdmf` class in `sweep_extraction.py` can load these and extract 1D line cuts

### `sweep_extraction.py` — SweepData Class

`SweepData(folder)` is the main post-processing interface:

| Property / Method | Returns | Description |
|-------------------|---------|-------------|
| `.jv` | DataFrame | J, V, power at each solved step (from `plot_meta.yaml` files) |
| `.mpp_row` | Series | Row at maximum power point |
| `.voc_row` | Series | Row at open-circuit voltage |
| `.v_row(V)` | Series | Row closest to specified voltage |
| `.params` | dict | Loaded from `submit.yaml` in run folder |
| `.get_spatial_data(row)` | object | Reads spatial profile for a given row (from CSV or XDMF) |
| `.IB_mask(spatial_data)` | bool array | Mask for IB region coordinates |

`SpatialXdmf` class (used for newer XDMF-based spatial data):
- `load_from_xdmf(attr, space_name)` — loads named field from XDMF checkpoint
- `line_cut(quantity, num_points, y, x)` — extracts 1D profile with `num_points` samples

The older `.csv.0` format and the newer `.xdmf` format are both supported transparently via `get_spatial_data(row)`, which selects the appropriate reader. The XDMF path is preferred when checkpoint files exist (it contains the full FEM solution, not just a sampled line cut).

### Meta-Extractors

| Class | Computes |
|-------|---------|
| `MetaExtractorBandInfo` | Quasi-Fermi levels, integrated carrier densities per band |
| `MetaExtractorIntegrals` | Current at specified facets; density integrals over cells |

---

## 4. Checkpointing

### How Checkpoints Work

Checkpoints save the full FEM solution state at specified parameter values and can be reloaded to resume an interrupted sweep.

**Implemented in:** `simudo/fem/adaptive_stepper.py` + `simudo/io/h5yaml.py`

**Checkpoint attributes on steppers:**

| Attribute | Meaning |
|-----------|---------|
| `checkpoint_write_values` | List of parameter values at which to save checkpoints |
| `checkpoint_write_filename` | Base path for checkpoint files |
| `checkpoint_reload_yaml` | Path to a `.yaml` metadata file to reload from |

In the fourlayer example, separate checkpoint settings exist for the optical sweep (`_opt`) and voltage sweep (`_V`), so runs can be resumed at any saved voltage point.

### Checkpoint File Set

For each saved parameter value, three files are written:

```
{prefix}__{param_name}={value:.14g}.xdmf     # XDMF mesh + field data
{prefix}__{param_name}={value:.14g}.h5        # Underlying HDF5 (auto by dolfin)
{prefix}__{param_name}={value:.14g}.yaml      # Metadata (units per band/field)
```

### What is Saved

- For each band: the main field `delta_w` (the FEM degrees of freedom)
- For `MixedQflNondegenerateBand`: also `mixedqfl_base_w`
- The thermal equilibrium electrostatic potential φ₀
- Units for every stored quantity (in the `.yaml` file)

### Loading a Checkpoint

**Critical constraint:** loading a checkpoint does not reconstruct the problem from scratch. The checkpoint only transfers the current FEM solution fields into the `problem` object. The `problem` (bands, processes, spatial rules, mesh, function spaces) must already be fully constructed identically to how it was when the checkpoint was saved. The checkpoint loader simply overwrites the solution values — if any band, field, or function space is missing or differently configured, loading will fail or silently produce wrong results.

In practice this means: the run script must call `create_problemdata(goal='full', ...)` with exactly the same device structure before reloading, then pass the resulting `problem` to the stepper with `checkpoint_reload_yaml` set.

```python
# Reconstruct problem identically, THEN pass checkpoint path to stepper
problem = create_problemdata(goal='full', V_ext=V_ext, phi_cn=phi_cn)

stepper = VoltageStepper(
    solution=problem,
    constants=[V_ext],
    parameter_target_values=np.linspace(0.5, 1.2, 15),  # resume from 0.5 V
    checkpoint_reload_yaml='ckpt/V__parameter=0.50000.yaml',
    checkpoint_write_filename='ckpt/V',
    checkpoint_write_values=[0.8, 1.0, 1.2],
)
stepper.solve()
```

On startup, the stepper calls `load_xdmf_checkpoint()`, which reads each field from the `.xdmf` file, applies unit conversion via `opportunistic_assign()`, and validates the solution (NaN check).

### GUI Implications for Checkpointing

- The GUI should ask the user how often checkpoints should be saved (e.g., every N voltage steps, or at specific voltages). Suggested defaults should be offered (e.g., every 5 steps, or at Voc and MPP). Checkpoints are large files — saving at every step is normally wasteful.
- On re-opening a project folder, the GUI should detect available checkpoint `.yaml` files and offer to resume from the latest one.
- Checkpoint filenames are deterministic and human-readable (`param=value`), so a directory listing is sufficient to enumerate them.
- Because resuming requires an identically constructed `problem`, the GUI's code-generation path must be fully reproducible from the saved project file — the same `submit.yaml` must produce the same device structure every time.

---

## 5. Reference Architecture: `fourlayer/`

The `fourlayer/` example is the recommended model for the GUI code generator. Key files:

| File | Role |
|------|------|
| `fourlayer.py` | Problem definition, solver orchestration, checkpoint management |
| `fourlayer_example.py` | High-level wrapper; loads `submit.yaml`, calls `fourlayer.py`, optional optimization |
| `sweep_extraction.py` | Post-processing: `SweepData`, `SpatialXdmf` |
| `IV_Analysis.py` | J-V curve analysis utilities |

### `submit.yaml` Parameter File

The fourlayer example is parameterized by a `submit.yaml` dict. All physics inputs are collected there, and `fourlayer.py` reads them at runtime. This is the pattern the GUI should follow: the GUI edits a `submit.yaml` (or equivalent) and then calls the run script.

Key parameter groups in `submit.yaml`:

- **Layer thicknesses**: `FSF_thickness`, `p_thickness`, `IB_thickness`, `n_thickness` (µm)
- **Band energies and DOS**: `VB_E`, `CB_E`, `IB_E` (eV); `NC`, `NV`, `NI` (cm⁻³)
- **Mobilities**: `mu_n`, `mu_p`, `mu_I` (cm²/Vs)
- **Optical cross-sections**: `sigma_opt_ci`, `sigma_opt_iv` (cm²); `alpha_cv` (cm⁻¹)
- **Thermal cross-sections**: `sigma_th_ci`, `sigma_th_iv` (cm²)
- **Doping**: `NA`, `ND`, `N_FSF` (cm⁻³)
- **Temperature**: `T` (K)
- **Permittivity**: `epsilon_r` (dimensionless)
- **Solar input**: `X` (sun concentration), `Ts` (sun temperature, K)
- **Sweep targets**: `V_exts` (list of voltages in V)
- **SRH lifetimes**: `tau_n`, `tau_p` (s)

---

## 6. Existing SimudoUI (Student Work, 2019)

**Technology:** PyQt5 desktop app, ~2600 lines, abandoned.

**What was implemented:**
- Region creation by coordinates or set operations, visualized as buttons in a graphics view
- Per-region parameter editor with collapsible sections (geometry, material properties, physics models)
- `SRH` and `Optical_Absorption` model selection with editable parameter tables
- YAML I/O utilities (written but never connected to anything)

**What was missing / broken:**
- No connection to Simudo backend whatsoever
- No save/load; no unit handling; no input validation
- Hardcoded Windows paths; deprecated pandas `.ix` API
- Region set-operation logic incomplete

**Useful patterns to carry forward:**
- Collapsible parameter sections (`collapsible.py`) — clean and compact
- Table-based parameter editing with Description/Value/Priority columns
- Region-as-button visual metaphor for 1D device layer stack

---

## 7. Comparable Tools: PC1D and SCAPS

### PC1D

Desktop app (Windows) for silicon solar cells. Multi-view interface:

- **Parameter View**: comprehensive parameter list + interactive schematic diagram; double-click on schematic element to edit
- **Four-Graph View** (during run): shows 4 physics quantities simultaneously, updates in real-time
- **Interactive Graph View**: zoom/export individual graphs

Device definition is hierarchical: emitter → base → BSF → contacts → optical coatings. Mesh auto-generated with refinement near junctions/surfaces.

Outputs: I-V curve, spatial profiles (carrier velocities, generation/recombination rates, currents, field); exportable as tab-separated data.

### SCAPS (Solar Cell Capacitance Simulator)

Desktop app for thin-film solar cells; up to 7 layers. Panel-based:

- **Solar Cell Definition Panel**: visual layer stack; click a layer button to open its property panel
- **Layer Properties Panel**: all material parameters for one layer; nearly all parameters can be "graded" (depth-dependent)
- **Defect Panel**: bulk and interface defects; charge type (mono/divalent/multivalent), energy distribution (Gaussian/uniform/single-level)
- **Contact Properties Panel**: front and back contact separately
- **Illumination Panel**: AM1.5 default; can change spectrum, intensity, illumination side

Layer management: right-click → Remove / Duplicate / Split.

Simulation types: I-V, C-V, C-f, quantum efficiency, spectral response. Batch parameter sweeps.

### Design Lessons for Simudo GUI

| Lesson | Source |
|--------|--------|
| Separate views for setup, monitoring, and results | PC1D |
| Visual device schematic + parameter list side-by-side | PC1D |
| Click layer in schematic to open its parameter panel | SCAPS |
| Hierarchical disclosure: overview → detail | Both |
| Right-click context menu for layer operations (add/remove/duplicate) | SCAPS |
| Support graded (spatially varying) parameters within a layer | SCAPS |
| Real-time parameter validation with warnings | needed (gap in both) |
| First-class batch/sweep mode with clear UI | Both |
| Multiple output quantities visible simultaneously | PC1D |
| Contextual help tooltips for every parameter | needed (gap in both) |
| Material library editable by users | needed (gap in both) |

**Common frustrations with SCAPS users report:** entering non-physical parameters without warning, difficulty understanding which parameters affect which outputs, complexity of defect state definition. These are all addressable in a new design.

---

## 8. Key File Map

| Concept | File |
|---------|------|
| Problem container | `simudo/physics/problem_data.py` |
| Poisson-drift-diffusion | `simudo/physics/poisson_drift_diffusion.py` |
| Material base class | `simudo/physics/material.py` |
| Electro-optical processes | `simudo/physics/electro_optical_process.py` |
| Optical fields | `simudo/physics/optical.py` |
| Heterojunction BCs | `simudo/physics/heterojunction.py` |
| Parameter steppers | `simudo/physics/steppers.py` |
| Spatial rules & BCs | `simudo/fem/spatial.py` |
| Newton solver | `simudo/fem/newton_solver.py` |
| Mesh construction | `simudo/mesh/construction_helper.py` |
| Topology (regions/facets) | `simudo/mesh/topology.py` |
| Mesh data container | `simudo/fem/mesh_data.py` |
| Adaptive stepper + checkpointing | `simudo/fem/adaptive_stepper.py` |
| Output writer | `simudo/io/output_writer.py` |
| CSV export (line cuts) | `simudo/io/csv.py` |
| XDMF export | `simudo/io/xdmf.py` |
| HDF5/YAML serialization | `simudo/io/h5yaml.py` |
| Unit registry | `simudo/util/pint.py` |
| Logging setup | `simudo/util/logging.py` |
| **Canonical example** | `simudo/example/fourlayer/fourlayer.py` |
| Post-processing | `simudo/example/fourlayer/sweep_extraction.py` |
| 2D reference | `../photodetector/run_photodiode2d.py` |

---

## 9. GUI Design Notes and Decisions

### Confirmed Decisions
- **1D only** for initial release; 2D parameters left as extension points internally
- **Simulations run on Linux/Ubuntu** (local or remote/Docker); GUI may run locally and pipe I/O over SSH
- **Target user**: physicist / collaborator with weak (not zero) Python knowledge; should not need to write code for normal use
- **Material library**: YAML files; GUI auto-discovers materials in a library directory; users can load + locally override parameters per layer
- **Checkpointing**: GUI asks the user for checkpoint frequency (with suggested defaults); offers to resume from latest checkpoint when re-opening a project folder
- **In-app output visualization**: at minimum J-V curve and band diagram / spatial profile at a selected voltage
- **Technology: web app** — Python web server; users open it in a browser. For remote/Docker access, forward one port over SSH (`ssh -L 8888:localhost:8888 ...`); no X11, no VNC, works on any OS. Framework: **Panel (HoloViz)**, staying in Python as much as possible (no JS). Port-forwarding workflow is already familiar to physicists using remote Jupyter.
- **Runner architecture: fixed generic runner + rich YAML** — `simudo_1d_runner.py` reads a project YAML that encodes everything (layers, bands, processes, BCs, sweep config). The GUI only writes YAML; the runner handles Simudo construction. No code generation. The project YAML schema IS the interface between the GUI and the runner.

### Build Sequence (agreed order of implementation)

Steps 1–4 give a usable tool. Steps 5–7 polish it.

1. **Device definition screen** — layer stack editor, material library, per-layer parameter forms. No simulation yet; just produces a valid project YAML.
2. **Execution panel** — launch `simudo_1d_runner.py` via subprocess (local) or SSH; stream log output into the GUI.
3. **J-V output view** — read `plot_meta.yaml` files as they appear, plot the J-V curve live.
4. **Spatial / band-diagram view** — select a voltage point, load the line-cut CSV or XDMF, plot band diagram and spatial profiles.
5. **Checkpoint management** — detect existing checkpoint files in the output folder, offer to resume from a selected one.
6. **Execution profiles** — SSH config, Docker exec, ssh+docker, etc.
7. **Further polish** — parameter validation, material library editor, graded parameters, etc.

---

### Execution Environment Options

Users run Simudo in several configurations; all use SSH public-key authentication (no passwords). The GUI must support each:

| Configuration | Description | GUI strategy |
|---------------|-------------|-------------|
| **Mac + Docker (local)** | FEniCS in a Docker container on the same Mac as the GUI | Connect to container via `docker exec` or expose an SSH daemon inside the container; mount a shared directory for project files |
| **Remote Linux machine** | SSH to a lab server or HPC node | SSH with key auth; rsync/SFTP to transfer project files and retrieve outputs |
| **Straight Linux** | FEniCS installed natively on a Linux desktop | SSH localhost or direct subprocess if GUI runs on the same machine |
| **Docker on Windows** | Less common but similar to Mac+Docker | Same Docker strategy |

**Recommended GUI approach — abstract over all cases with an "execution profile":**

Each project (or user preference) stores an **execution profile**:
```yaml
execution:
  type: ssh          # "local", "ssh", "docker", "ssh+docker"
  host: myserver.university.ca   # for ssh/ssh+docker
  user: jacob
  identity_file: ~/.ssh/id_ed25519
  remote_work_dir: /home/jacob/simudo_runs/
  # For docker:
  container: simudo_fenics         # container name or ID
  docker_exec: true                # use "docker exec" instead of SSH into container
```

For the **Mac+Docker local** case specifically, the cleanest approach is to mount a shared directory (e.g., the project folder) into the container at a known path, and run `docker exec <container> python run_script.py` from the host. The GUI already has access to the output files since it's a shared mount. No SSH needed in this case.

For **remote SSH**, the GUI:
1. `rsync`s the project directory to `remote_work_dir` before running
2. Launches the simulation via `ssh user@host "cd remote_work_dir && python run_script.py"` (non-blocking; streams stdout/stderr back for the log panel)
3. `rsync`s output files back on completion (or periodically during the run) for visualization

**Fabric** (Python SSH library) or **asyncssh** are good choices for the SSH layer, since they integrate naturally into a Python GUI event loop.

---

### Spatially-Varying Parameters and UFL Expressions

Simudo's `Spatial` / `add_rule` system accepts any UFL expression as a parameter value — including expressions that depend on coordinates, solution fields (carrier densities, potential), or combinations. The full generality includes:
- Uniform scalars (constant per region): the simple case
- Piecewise-constant graded parameters (stepped profile across sub-regions)
- Smooth spatial functions of `x` (e.g., linear doping gradient)
- Functions of solution variables (e.g., density-dependent mobility, concentration-dependent lifetime)

**GUI approach for v1: uniform per region with an "advanced expression" escape hatch**

For normal use, every parameter field shows a simple numeric input (value + unit). This covers the vast majority of cases.

For advanced users, each parameter also has an **"expression mode" toggle**. When enabled, the input field accepts a free-text UFL/Python expression string. The GUI includes this string verbatim in the generated run script, so users get full Simudo flexibility without the GUI needing to understand the expression. Example:

```
# GUI renders this in the generated script:
spatial.add_rule('CB/mobility', R.nSi,
    U('cm^2/V/s') * (1400 / (1 + (CB.u / U('1e17 /cm^3'))**0.9)))
```

This approach:
- Keeps the v1 GUI simple for the 90% case
- Does not artificially limit Simudo's capability
- Gives a clear upgrade path: expression inputs could later be given a structured editor (e.g., a piecewise-linear profile builder), but always fall back to the raw expression string

Expressions should be validated at script-generation time (basic syntax check) with a warning if something looks wrong, but not deeply parsed by the GUI.

---

### Boundary Conditions and Interface Physics

#### Exterior contacts (device edges)

Currently supported contact models in Simudo:

| Model | Description | GUI control |
|-------|-------------|------------|
| **Ohmic (SRV = ∞)** | Carrier density fixed at thermal equilibrium value; full recombination | Default; no extra parameters |
| **Blocking (SRV = 0)** | Zero carrier current at contact | Toggle per band per contact |

General SRV (finite, non-zero) is on the Simudo development roadmap but not yet implemented. The GUI should reserve a field for it (shown as "not yet available" or greyed out), so it can be activated without a redesign.

The GUI should expose, per contact and per carrier band:
- Contact type: Ohmic / Blocking (/ SRV — future)
- Applied voltage (for the swept contact) or fixed voltage (for the reference contact)

#### Internal boundaries between layers

- **Band subdomain confinement**: bands can be defined only in a subdomain (e.g., IB exists only in the IB layer). The boundary of that subdomain automatically gets a zero-current BC for the confined band. The GUI should allow the user to mark which layers a band is active in; zero-current BCs at the subdomain boundary are applied automatically.
- **Heterojunction BCs**: `ThermionicHeterojunction` (`simudo/physics/heterojunction.py`) applies thermionic emission at internal band discontinuities. This should be the **default** whenever two adjacent layers have different materials (i.e., a heterojunction is detected). Users should be able to disable it explicitly if they want a simpler model. No additional parameters are required beyond detecting the junction.
- **General internal SRV**: not yet implemented in Simudo. Not needed in v1 GUI.

---

### Remaining Open Questions
1. **Output visualization backend**: Panel has built-in Matplotlib and Bokeh integration — Bokeh is the natural choice for live-updating plots in a Panel app, but Matplotlib is fine for static plots. Decision deferred to implementation of step 3.
2. **Project file format**: directory (resolved — maps naturally to Simudo's run folder; single-file archive adds no benefit for v1).
3. **Execution profile storage**: per-user config file with per-project override (deferred to build sequence step 6).
