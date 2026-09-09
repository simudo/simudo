# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.7.1.0] - 2026-09-09

### Added
- True-1D simulations. `ConstructionHelperLayeredStructure` builds a dolfin
  interval mesh instead of extruding the layer stack into a one-cell-tall 2D
  strip; `params['mesh_dimension'] = 2` restores the strip. On a
  1D mesh the electric field and band currents use continuous Lagrange
  elements of the same degree (as 1-component vectors) in place of `BDM`,
  which is the exact 1D analogue of the H(div)-conforming mixed method, so
  the weak forms, boundary conditions, processes and outputs are unchanged.
  The degrees of freedom roughly halve and solves are correspondingly
  faster. **This is now the default**: the strip
  is built only when `mesh_dimension = 2` is asked for, or when a subclass
  has customized `product2d_Ys` (which only makes sense on an extruded
  mesh). Existing scripts that build a layered structure therefore get the
  interval mesh, and their currents move by the difference between two
  discretizations of the same problem. No user changes should be required to
  take advantage of this speed improvement.
- The yaml runner (`simudo_1d_runner.py`) now uses the true-1D mesh by
  default; set `mesh: {dimension: 2}` in a project to get the previous 2D
  strip. It also logs the wall time of each stage. Line-cut CSVs from 1D
  runs have no `coord_y` column, and `SpatialXdmf`/the GUI spatial
  extractor read 1D XDMF output. Checkpoints are not portable between the
  two mesh dimensions.
- The GUI's Simulation panel has a mesh-type selector, "1D (interval mesh)"
  or "2D strip (legacy)", stored as `mesh_dimension` in the project file.
- Optical field directions written for the 2D strip keep working on a 1D
  mesh. `adapt_direction_to_gdim` accepts a direction with more components
  than the mesh has dimensions when the extra ones are zero, so `(1, 0)` and
  `(-1, 0)` -- the only directions a layered device could use on the strip --
  need no editing. A non-zero transverse component now raises `ValueError`
  instead of being ignored, since it has no meaning on a 1D mesh.

### Fixed
- `sim_V.csv` could put values under the wrong headings when restarting
  from a checkpoint. `MetaCSVWriter` now follows the existing
  header and warns if the two column sets disagree. 
  This affects output written by all previous versions.
- Resuming a voltage sweep from a checkpoint failed when the checkpoint had
  been taken after the optical solve with `selfconsistent_optics = False`.

## [0.7.0.3] - 2026-09-02

### Changed
- The GUI modules now import each other by package path, so the interface no
  longer has to be started from inside its own directory. Start it from
  wherever you keep your projects — that is where its Open and Save dialogs
  begin. Previously the instructions had you launch from inside the installed
  package, which is where projects were then saved.

### Added
- `python -m simudo.gui` starts the GUI. Pass a project file to
  open it at startup; any other arguments go to `panel serve`.
- The Docker image accepts `--build-arg FOR_DOCS=1` to build a documentation
  image, which skips the DOLFIN rebuild.

### Fixed
- Minor fix in calculating absorption energy bounds for TopHatBeerLambert classes.
- Several scripts under `example/` can now be imported: some used script-style
  imports, and `argh` has removed the `expects_obj` decorator they used.

## [0.7.0.2] - 2026-09-01

Fixes to the 0.7.0.0 release. (0.7.0.1 was published to PyPI with the first
three of these; 0.7.0.2 adds the last.)

### Fixed
- On update to numpy 1.26.4, repeated warnings about to-be-deprecated features
  swamped the log. Each distinct warning is now shown once. Set
  `TypicalLoggingSetup(collapse_repeated_warnings=False)`, to see them all.
- The bundled `fourlayer_example.yaml` failed to run. Thermal velocity became
  a band property in 0.7.0.0, so it needs `CB/vth` and `VB/vth`.
- The `example/fourlayer` scripts likewise, plus a plotting crash on newer
  numpy.
- The characterization tests in `test_runner_characterization.py` pin reference
  currents to 1e-12, which is tighter than the Newton solver's own relative
  tolerance of 1e-6 defines them. They passed on aarch64 and failed on x86_64,
  differing in the seventh significant figure. They now run only on the
  architecture the references were captured on.

## [0.7.0.0] - 2026-09-01
First public release since 0.6.5.0 (2022-01-29). Four years of development,
so the list below is a summary rather than a full account.

### Added
- **Graphical interface** for setting up, running, and plotting results for 1D
  devices. Runs in a browser on your computer and sends jobs for execution into
  a Docker or remote machine. Most 1D simudo features implemented, with lots of
  new hinting to ensure problems are set up properly. Install with
  `pip install simudo[gui]`; there is a tutorial with screenshots in the user
  documentation.
- **Easy support for many optical fields:**
  - AM1.5 helper, using the ASTM G173 reference spectrum.
  - `BeerLambert` and `IBBeerLambert` — absorption between two dispersing
    bands and with an intermediate band. Both can take arbitrary α(E) or σ(E). 
    IB case has absorption dependent on IB filling fraction.
- **Surface-recombination-velocity boundary condition** with selectable
  methods.
- **Degenerate (Fermi-Dirac) bands**, using Joyce-Dixon formulation. Treats 
  parabolic bands.
- **`ShockleyReadTrap2Trap`** and **`ShockleyReadBand2BandTrap`** Shockley-Read
  (not SRH) type processes between two trap bands and between two dispersing bands.
- **Checkpoint and resume** — save and continue a bias or optical sweep.
- Solver work: efficiency improvements, solver reuse, better convergence diagnostics.
- **2-D anisotropic meshing** rewritten onto the official gmsh Python API.
- Material files are distributed on an as-is basis.
- **A Docker image recipe** (`docker/`), built on Ubuntu 24.04 LTS, with the
  DOLFIN patch described below applied.
- **A test suite** — 153 solver and unit tests plus 9 GUI tests, covering
  configuration construction, physical invariants, analytic optics, golden J(V)
  curves and checkpoint round-trips.  

### ⚠️ Results that change
Read this before comparing new results against old ones.

- **Thermionic emission across a heterojunction used the wrong velocity
  average — a factor of 4.** Material files export `<band>/vth` as the
  Maxwell-Boltzmann mean speed √(8kT/πm\*), which is what an SRH capture
  coefficient needs. A flux across an interface instead needs the
  one-directional average v_R = √(kT/2πm\*) = ⟨v⟩/4, so previous heterojunction
  simulations were too transparent at the
  interface. `ThermionicHeterojunction` now uses a separate `emission_velocity`
  property; `vth` is unchanged and remains correct for capture. 

### Fixed an upstream DOLFIN bug
Legacy DOLFIN misassembles forms that combine an interior facet measure (`dS`)
with a subdomain-restricted `dx`, which affects some 2-D problems and gives a
wrong answer rather than an error. We diagnosed it, submitted a fix upstream,
and ship the patch in `docker/patches/`; our Docker image has it applied. 1-D
simulations are unaffected. To patch a native installation, see
`docker/README.md`.

### Changed
- **Electro-optical processes now check their band types** and warn when a
  process is applied to a kind of band it was not derived for — for example
  `BeerLambert` between a sharp intermediate band and a standard
  band (should use `IBBeerLambert`).
- The default heterojunction boundary condition formulation changed
  (`use_alternative_BC` now defaults to `True`), and is better converged at the
  default `HJBC_enhancement`.
- **`v_th` is now spelled `vth`**, and thermal velocity is a band property
  rather than a process property. The old spelling still works and warns once.
- A missing absorption coefficient or optical cross-section now logs a warning
  instead of silently contributing no generation.
- The material library gained provenance documentation
  (`simudo/materials/alloy_optics/SOURCES.md`).

## [0.6.5.0] - 2022-01-29
### Added
- `fem.debug_probe` and `MeshUtil.get_debug_probe` to probe physical
  quantities albeit inefficiently.
- Add jupyter notebook with examples.
- Update user docs with instructions for Docker on Windows and
  Podman on GNU+Linux.

## [0.6.4.1] - 2020-12-14
### Changed
- Use absolute imports in `fourlayer_example` to help out users who
  just want to modify the geometry without messing with the material
  parameters much.

## [0.6.4.0] - 2020-11-13
### Added
- Physics: Thermionic heterojunction boundary condition.
- FEM: Utility method `MeshUtil.region_oriented_dS()` for getting the
  same-side and opposite-side dS facet measures from a FacetRegion.
- Mesh: Null cell and facet regions.
- Added changelog file.

### Changed
- Stepper: If the solver raises a RuntimeError, treat it as a failure
  rather than letting the exception bubble up.
- Stepper: Reduce step size more dramatically on failure.
