# Output Panel Enhancement — Plan

## Architecture decisions

- **J-V data**: read from `sim_V.csv` (summary CSV, no dolfin needed)
- **Spatial data**: read XDMF checkpoint files via `spatial_extractor.py` running
  inside the container using dolfin; data streamed as CSV to stdout
- **Execution**: uses the same execution-profile abstraction as the simulation
  panel (`runner_backends.backend_from_profile`); works with Docker, Local, SSH
- **Cache**: spatial data held in-memory only (`_spatial_cache` dict in
  `OutputPanel`); ~32 MB for a typical run; no files written to disk
- **Loading UX**: show "⟳ Extracting spatial data…" while the extractor runs
- **Bokeh**: all interactive figures use native Bokeh toolbar
  (pan, box_zoom, wheel_zoom, reset, save); embedded via `pn.pane.Bokeh`

## File layout

| File | Purpose |
|------|---------|
| `gui/spatial_extractor.py` | Script that runs inside the container; reads `_full.xdmf` checkpoint files with dolfin; streams CSV blocks to stdout |
| `gui/runner_backends.py` | Add `launch_extractor(extractor_host_path, output_dir_host_path, extra_args)` to each backend |
| `gui/panels/output.py` | All UI + Bokeh figure building |

## Tasks

- [x] 1. Interactive J-V plot — Bokeh figure with pan/box_zoom/wheel_zoom/reset/save toolbar
- [x] 2. Linear / Log|J| radio button — rebuild figure on toggle; plot |J| in log scale
- [x] 3. Band diagram
  - [x] 3a. Detect available voltages from `sim_V=*_full.xdmf` filenames
  - [x] 3b. Voltage dropdown (default V=0 or first available)
  - [x] 3c. "Load Spatial Data" button → launches `spatial_extractor.py` via execution profile
  - [x] 3d. Loading notification shown while extractor runs
  - [x] 3e. Layer schematic subplot (top): coloured blocks proportional to layer thicknesses, `x_range` shared with band diagram
  - [x] 3f. Band diagram subplot: `Ephi_*` solid black per band, `qfl_*` dashed coloured per band
- [x] 4. Add Plot dashboard
  - [x] 4a. "＋ Add Plot" button creates a new dashboard section below
  - [x] 4b. Each dashboard: "Add Line" → quantity dropdown (all non-coord columns), colour, linestyle, line width, y-scale (linear/log)
  - [x] 4c. Live-update Bokeh figure as lines are added / removed
  - [x] 4d. "✕ Remove Plot" button to delete a dashboard
- [x] 5. `spatial_extractor.py` — standalone script for container
  - [x] 5a. Accepts `<output_dir> [--n-points N]` arguments
  - [x] 5b. Uses `SpatialXdmf` from `simudo.example.fourlayer.sweep_extraction`
         (handles element-type parsing and vector vs scalar evaluation already)
  - [x] 5c. `SpatialXdmf.line_cut` evaluates via `dolfin.Point` — returns x-component for vector fields
  - [x] 5d. Streams `# BEGIN_VOLTAGE V` … CSV rows … `# END_VOLTAGE V` blocks to stdout
- [x] 6. `runner_backends.py` — `launch_extractor()` per backend
  - [x] 6a. LocalBackend: runs extractor with current Python interpreter
  - [x] 6b. DockerBackend: translates host→container paths, runs inside container
  - [x] 6c. SSHBackend: rsyncs local output dir to a remote temp dir, runs
         the extractor via SSH, streams CSV to stdout, and removes the temp dir
         on completion via `SSHExtractorHandle`.

## Future / known issues

- **Card resize**: CSS `resize: vertical` on a Panel/Bokeh column is non-functional
  because Bokeh's layout engine re-pins the height on every render cycle, overriding
  whatever the browser's drag handle changed.  Possible fixes:
  (a) A ResizeObserver JS shim injected via `pn.extension(js_files=...)` that calls
  `Bokeh.index[...].invalidate_layout()` after the CSS resize event settles.
  (b) An `IntSlider` widget (Python `param.watch`) driving `plot_box.height` — works
  but adds visual clutter.

## stdout protocol (spatial_extractor.py → GUI)

```
# BEGIN_VOLTAGE 0.0
coord_x,Ephi_CB,Ephi_VB,qfl_CB,qfl_VB,...
0.000,...
0.001,...
...
# END_VOLTAGE 0.0
# BEGIN_VOLTAGE 0.1
...
```

Warnings and dolfin log messages go to stderr only (not mixed into stdout).
