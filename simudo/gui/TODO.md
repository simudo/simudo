# Simudo GUI — Implementation TODO

## Layers panel
- [x] Layer stack schematic (proportional flex blocks, contact markers)
- [x] Add / delete layers
- [x] Reorder layers (▲ / ▼ buttons)
- [x] Inline editing: layer name, thickness (value + unit), material
- [x] Schematic updates live when thickness or name changes
- [x] Select layer by clicking the colored dot → opens detail side panel
- [x] Click schematic layer block to select it
- [x] Detail panel: doping (n-type / p-type radio, concentration + unit, scientific notation)
- [x] Detail panel: mesh parameters card (collapsed by default)
- [x] Overlay regions section with domain + custom overlays
- [x] "Add overlay" button
- [x] Overlay region editing: rename, set extent (layer MultiChoice or numeric Left/Right)
- [x] Overlay detail panel on the right (doping; extensible for future parameters)
- [x] Select overlay by clicking its dot → opens overlay detail panel
- [x] Overlay indicator strip in schematic (coloured bars + name labels showing region extents)
- [x] Overlay strip alignment: uses flex-basis:0 + same gap as schematic so positions
      are pixel-perfect for layer-based overlays and sub-pixel-accurate for coordinates
- [x] Disjoint layer-based overlays rendered as separate coloured segments
- [x] Dual BC contact markers (left + right) — each is a separate Panel pane object
- [x] Layer names enforced as valid Python identifiers (sanitise on edit)
- [x] Layer/overlay name scientific-notation display for doping concentration
- [x] Prominent save-location banner when no filepath is set
- [x] Property editor in layer/overlay detail panel: show all non-doping properties as
      editable key / value / unit rows with a delete button per row
- [x] "Add property" inline form: suggested properties dropdown derived from active bands
      and processes; free-text key field; unit auto-filled from suggestion; Add button +
      stretch spacer so the key box doesn't grow too wide
- [x] Sidebar badge: yellow ● on the "Layers" nav button whenever any required parameter
      is unset in every region
- [x] "Missing parameters" global section at the bottom of the Layers panel; entries
      disappear once set anywhere; updates when bands/processes are added or deleted
- [x] Properties autosave to YAML on every edit (same as bands/processes)
- [x] Temperature: global required param (always in missing list until set); defaults to
      300 K in the domain overlay on new project; removed from physics.yaml field
- [x] Temperature warning: red hint box when adding `temperature` to any non-domain region
- [x] v_th formula helper: amber hint box when key ends in `/v_th`, showing formula,
      numerical value at current project temperature (read from domain overlay, Pint
      unit conversion), and scaling note
- [x] Domain overlay properties save/load correctly (previously silently dropped on save)
- [ ] (future) Expression mode toggle (ƒ) per property field — deferred
- [x] Interface click → configure interface conditions (heterojunction BC):
      Layout: schematic spans full width; below it a Row with left col (layer
      table + overlay regions + missing params) and right col (detail panel —
      shows layer, overlay, or interface detail based on selection).
      Material name Select in layer table rows: fixed width with a stretch
      Spacer to its right so it doesn't expand with the window.
      Schematic: 12 px clickable divider Buttons between layer blocks (faint
      blue, solid on selection); hover highlight; click calls
      _select_interface(i).  Width is set via Panel width= param (not flex CSS)
      to guarantee correct orientation; height via flex-chain CSS:
      :host{display:flex;flex-direction:column;height:auto} →
      .bk-btn-group{flex:1 1 0;display:flex;flex-direction:column} →
      .bk-btn{flex:1 1 0} so height stretches to match surrounding layers.
      Overlay strip: transparent 12 px placeholder divs mirror each divider
      so layer-based overlay segments stay pixel-aligned.
      _selected_interface: int tracks which divider (left layer index = -1 when
      none); deselects layer/overlay selection.
      Interface detail panel (right col):
        - Header: "left_name|right_name interface" with material names shown below.
        - Dismissable amber warning when both layers share the same material
          (unusual for HJBC); shown only when ≥1 band is enabled; resets when
          a layer's material changes or a different interface is selected.
        - Soft warning (dismissable) if bands == default_bands(): "Default bands
          are present; visit Bands panel to add/change before applying HJBCs."
        - If project.bands is empty: error message, no toggles shown.
        - Per-band rows: checkbox + band name label + "Enhancement (optional):"
          input on the same row; enhancement stored per-band (default 1.0,
          must be > 0, scientific notation OK).
        - Save rule: if no bands checked → remove this interface from
          project.interfaces (nothing in YAML); if ≥1 band checked → store
          explicitly in project.interfaces.
      Data model (model.py):
        - InterfaceBCSpec(type, bands: List[str],
                          HJBC_enhancement: Dict[str, float])
        - Interface(left, right, bcs)   ← left/right are layer names
        - Project.interfaces: List[Interface]
      YAML schema (yaml_io.py):
        interfaces:
          - left: p_layer
            right: n_layer
            bcs:
              - type: ThermionicHeterojunction
                bands: [CB, VB]
                HJBC_enhancement: {CB: 1.0, VB: 1000.0}
        Backwards compat: scalar HJBC_enhancement is broadcast to all listed
        bands on load.
      Layer rename: auto-update Interface.left/right when a layer is renamed.
      Runner (simudo_1d_runner.py):
        - Import ThermionicHeterojunction from simudo.physics.heterojunction.
        - After contact BCs: for each interface entry, for each BC spec with
          type ThermionicHeterojunction, for each listed band call:
          ThermionicHeterojunction(
              band_objects[band_name],
              getattr(R, left).boundary(getattr(R, right)),
              HJBC_enhancement=dolfin.Constant(enh_dict.get(band_name, 1.0))
          ).register()
        - Log a warning and skip if band_name not in band_objects or layer
          name not a valid R attribute.
        - Handles scalar HJBC_enhancement for backwards compat.
- [ ] Overlay set operations: define a region as intersection / union / difference of two
      existing overlays (e.g. intersect a "by layers" region with a "by coordinates" range)
- [ ] TODO: Jacob wants to demonstrate DebugProbe to verify the spatial attribute priority
      system is working correctly — schedule a walkthrough session

## Materials panel  ✅ COMPLETE (v1)
- [x] Library directory config (bundled + user dir, persisted to ~/.simudo_gui.yaml)
- [x] Browseable catalogue of library classes (AST scan, read-only property display)
- [x] "New from this" — creates editable inline copy; warns about computed properties
- [x] Project materials list: inline (editable key/value/unit) + library-ref (read-only)
- [x] "Create new material" form with inherit-from dropdown
- [x] Convert library-ref material to inline editable copy
- [x] Layer material field → strict Select dropdown (options from project materials)
- [x] Validation warning: layers referencing undefined material names
- [x] Missing-params check counts properties provided by materials applied to ≥1 layer
- [x] Runner receives --library-dir for each valid library directory
- [ ] (future) Runner-based per-layer parameter coverage validation: once execution
      settings (remote runner) are configured, the GUI should be able to invoke Simudo
      to check whether each required parameter is defined in each region, replacing the
      current coarse global "not set anywhere" check.
- [x] "Derived" material type: library class + inline overrides stored in YAML as
      {source: library://ClassName, properties: {key: {value, unit}}}. Runner applies
      library at PRIORITY_MATERIAL (10), overrides at PRIORITY_MATERIAL-1 (9) on the
      union of layers using this material. GUI: "Derive from this" button in library
      browser; derived detail panel shows all library properties (numeric + computed) in
      a unified editable table: numeric props dimmed at library default / amber when
      overridden; computed props show a "calculated" badge and empty inputs until edited;
      a single ✕ button appears only on overridden rows and removes the override (restores
      library default for numeric/computed, removes entirely for custom keys); a separate
      CUSTOM OVERRIDES section for user-added keys; ADD KEY form at the bottom.
      Missing-params check counts all library keys plus any additional override keys.

## Bands panel  ✅ COMPLETE
- [x] List of bands with add / delete
- [x] Band type selector (Boltzmann / Fermi (parabolic) / Sharp Intermediate Band)
- [x] Sign selector (Electrons / Holes radio buttons)
- [x] Auto-set sign=+1 (Holes) when new band name starts with "VB"
- [x] Extent selector ("Domain (all layers)", or MultiChoice of layer + overlay region names)
- [x] Extent boundary BCs left / right (shown only when extent ≠ "all")
- [x] Note text: "Band parameters (energy levels, effective density of states, etc.)
      are set in the Layers or Materials panels."
- [x] Round-trips correctly through YAML (fourlayer example: CB/IB/VB)
- [ ] **Runner TODO**: catch the edge case of a finite-extent band with a non-zero-current
      BC on a disjoint region (zero_current + disjoint is fine; see SCHEMA_DESIGN.md for details)

## Processes panel  ✅ COMPLETE
- [x] List of processes with add / delete
- [x] Class selector: discovered at startup by AST-parsing electro_optical_process.py;
      falls back to hardcoded list if runner source not found
- [x] Lower E band / Higher E band selectors (src_band / dst_band in YAML)
- [x] trap_band selector (conditional — shown only for TrapEOPMixin subclasses;
      SRHRecombination excluded after TrapEnergyLevelMixin refactor)
- [x] "Include radiative recombination: Yes/No" toggle (conditional — shown only for
      AbsorptionAndRadiativeRecombinationEOPMixin subclasses); defaults to Yes
- [x] gui_hint attribute: each EOP class may define gui_hint = "..." which the AST
      extractor reads at startup and displays as a styled note below the class selector
- [x] Auto-fill trap/src/dst bands on class switch: when switching to a trap class and
      exactly one Sharp IB band exists → Lower E = IB, Higher E = first non-IB, Trap = IB
- [x] Error: trap band is not a Sharp Intermediate Band (orange warning)
- [x] Error: trap band does not equal either Lower E or Higher E band (orange warning)
- [x] Round-trips correctly through YAML (fourlayer example: 5 processes)
- [x] Process name uniqueness: duplicate names show inline red error; model not updated
      until name is valid; `+ Add process` and class-switch auto-suggest unique names
      using a short class prefix (srh, nr, b2b, opt, …)

## Optical Fields panel  ✅ COMPLETE
- [x] List of optical fields with add / delete
- [x] Direction selector (+x / −x)
- [x] Photon energy (value + unit)
- [x] Intensity mode: Explicit | Blackbody
- [x] Explicit sub-form: value + unit
- [x] Blackbody sub-form: T_source, concentration, E_min, E_max (∞ checkbox)
- [x] Round-trips correctly through YAML (fourlayer example: 3 blackbody fields)

## Boundary Conditions panel  ✅ COMPLETE
- [x] Left / right contact cards
- [x] Contact role selector (swept / reference)
- [x] Per-band BC type (ohmic / zero_current / SRV radio buttons); default: ohmic
- [x] Round-trips correctly through YAML (fourlayer example: left swept, right reference)
- [x] Surface recombination velocity (SRV) BC: third radio option; value + unit (cm/s / m/s)
      sub-row appears/disappears when SRV is selected.  Stored as a dict in band_bcs:
      `{type: surface_recombination, srv: {value: 1000.0, unit: cm/s}}`.
      yaml_io.py and model.py required no changes — band_bcs is already Dict[str, Any].
- [x] SRV values available for parameter sweep in Batch panel (contact_srv kind in batch.py):
      enumerate_sweep_params, get_current_param_value, _apply_variation_to_dict,
      compact_param_label all handle the new kind.

## Simulation panel  ✅ COMPLETE (v1)
- [x] 4-mode selector (Equilibrium only / Dark J-V / Illuminated (V=0) /
      Illuminated J-V) — controls intensity_ramp.enabled + voltage_sweep.enabled
- [x] Warning note when illuminated mode selected but no optical fields defined
- [x] Intensity Ramp section: self-consistent optics toggle, step_size
- [x] Voltage Sweep section: self-consistent optics, step_size, values textarea, linspace helper
- [x] Checkpoints: labeled "Save checkpoint after intensity/voltage sweep"; directory field
- [x] Resume-from path: dropdown scans the checkpoints dir + shows human-readable
      labels, fills a path box; secondary/advanced picker — see "Checkpoint / Resume
      UX" section below for the primary entry point
- [x] Output: spatial_profiles toggle, xdmf_mesh toggle, output folder
- [x] Run: saves project, streams runner output line-by-line; log capped at 500 lines
- [x] "📄 View info.log" button: opens scrollable fixed-position overlay with full info.log;
      detects output dir from log lines; nonce forces Panel re-render on repeated clicks
- [x] Stale output-folder date warning: amber banner when the date token in the folder
      path (e.g. `2026May07`) does not match today; one-click "Update to YYYY[Mon]DD"
      button rewrites the token in-place and autosaves

## Checkpoint / Resume UX  (redesign — 2026Aug11)
Design discussion: 2026Aug11 session. Replaces the per-panel picker as the
*primary* path for starting a resumed project; that picker (Simulation panel,
above) stays as a secondary/advanced way to switch checkpoints mid-project.

- [x] **Entry point**: "Checkpoint ▾" button in the header, next to New/Open/Save As
      — a third way to *start* a project, not a Layers-panel feature.
- [x] **Two-step picker**, one dropdown, step 2 replaces step 1's content:
      1. Pick the *project* to resume from the existing recents list (reused, not
         duplicated) + Browse (native file dialog). No separate "recent
         checkpoints" list.
      2. App resolves that project's `output_folder`, scans `<folder>/checkpoints/`
         (`_scan_checkpoint_dir`, moved to `panels/shared.py` so both app.py and
         panels/simulation.py use the same code) and shows the human-readable
         list, plus "Browse for a different output folder…" for auto-incremented
         sibling folders (`out/a`, `out/a1`, …) the scan might miss.
- [x] **On selection**: load the *frozen* project YAML that lives beside the
      checkpoint (not the live file picked in step 1 — that may have diverged
      since the run). Set `resume_from` to the picked checkpoint (relative path),
      set `filepath` to that frozen YAML's own location (see save-in-place,
      below), jump to the Simulation panel, show a confirmation banner naming
      the checkpoint and run folder.
- [x] **Save-in-place default**: resumed project's `filepath` defaults to the
      frozen YAML's location inside the run folder, not the original project's
      home location — avoids clobbering a live project file that may have kept
      changing since that run. `output.folder` stays pointed at the same run
      folder so new results append there. Save As still available for
      branching into a new location.
- [x] **Generic resume-state model**: represented as (stage, parameter name,
      parameter value, run folder) rather than anything voltage-specific, so the
      Simulation panel's resume banner/pruning logic keep working unchanged once
      the optical intensity ramp gets a configurable endpoint (>1 sun — separate
      follow-up, not part of this pass, but this keeps it a drop-in).
- [x] **Bidirectional resume** (`simudo_1d_runner.py`): generalized the existing
      "two ProblemData copies sharing one seed state" mechanism (previously only
      used to split a fresh run's sweep around V=0) to split around *any*
      checkpoint's own parameter value. Resuming from V=0.4 with targets
      `[0.3, 0.4, 0.5]` now produces two independently-seeded continuations, one
      down to 0.3 and one up to 0.5, both from the same loaded V=0.4 state.
      Removes the old "checkpoint can only continue in the direction its sign
      implies" restriction.
- [x] **Skip already-computed points**: before building each direction's target
      list, drop any value already present in that run folder's `sim_V.csv` /
      `sim_I.csv` history (tolerance-matched). Enforced in the runner (applies
      regardless of entry point) and mirrored in the Simulation panel as a live
      note listing what will be skipped — auto-drop with a visible note, not a
      blocking confirmation dialog, since the device structure is frozen on
      resume so there's no legitimate reason to recompute a known point.
      (Judgment call; revisit if users want the dialog instead.)
- [x] **`ensure_new_dir` records the actual folder used** (`simudo_1d_runner.py`):
      after resolving the actual output dir (fresh run: post-increment; resume:
      as given), write it back to two places — the frozen copy inside that
      folder gets `output.folder: .` (self-referential, correct given
      save-in-place above), and the *live* project YAML (wherever the user has
      it open) gets `output.folder` updated to the correct relative path, fixing
      the case where auto-increment silently diverges from the stored config.
      GUI-side: `app.notify_run_output_dir()` (already fires once the actual dir
      is known) also refreshes the in-memory `project.output_folder`, so the
      next autosave doesn't clobber the runner's write-back. SSH backend: the
      write-back happens on the remote copy — needs to sync back with results,
      or it's a no-op for that backend.
- [ ] **Resume-linked device structure warning**: once loaded from a checkpoint,
      editing layers/materials/bands/processes/opt_fields/BCs should show a
      persistent indicator ("checkpoint-linked to X — editing the device will
      break resuming") with an explicit "detach" action, rather than allowing
      silent drift. Deferred — not implemented this pass.

## Execution Settings  ✅ COMPLETE (v1)
- [x] Execution type selector (Local / Docker / SSH) in Simulation panel
- [x] Docker fields: container name, host root, container root
- [x] SSH fields: host, user, identity file, remote work dir, Simudo path, delete-after-sync toggle
- [x] "Test connection" button (docker exec echo ok / ssh echo ok)
- [x] "Set as default" / "Restore default" buttons — save/restore profile in ~/.simudo_gui.yaml
- [x] Per-type profile memory: switching types saves/restores last-used settings per type
- [x] Per-project override: project YAML stores execution profile; falls back to user default
- [x] DockerBackend: docker_host_root defaults to "" (no hardcoded path); guard raises
      clear error if left blank; forward-slash path support for Windows/Cygwin documented
- [x] SSH backend: rsync-based output sync; remote staging dir; library dir caching
- [x] Docker/SSH: stdin=DEVNULL + python -u to prevent SIGTTOU and buffer issues

## Output panel  ✅ COMPLETE (v1)
- [x] Output folder path display
- [x] Open folder button (uses OS native open: open/xdg-open/explorer)
- [x] Refresh plot button
- [x] Auto-detect J-V CSV in output folder
- [x] SVG J-V curve plot rendered inline (no matplotlib dependency)
- [x] Rewrite with Bokeh: linear/log Y-axis toggle, hover tooltips, zoom/pan
- [x] Band diagram section: voltage dropdown + spatial profile plot
- [x] "Add Plot" dashboard: quantity browser for CSV columns
- [x] Batch-aware layer schematic: when the output being viewed came from a batch
      sub-run (i.e. a `<project>.yaml` exists in the output directory), load that
      sub-YAML's layer thicknesses for the schematic instead of `app.project.layers`.
      Implemented via `_spatial_data_dir` stored on load; passed through
      `_rebuild_band_diagram` → `_build_schematic_figure` and
      `_PlotDashboard._rebuild` → `_build_custom_with_schematic` →
      `_build_schematic_figure`; globs for `*.yaml` in output dir and loads first match.

## Runner / simudo_1d_runner.py  ✅ COMPLETE (v1)
- [x] Reads project YAML, builds full Simudo ProblemData
- [x] Library material support via --library-dir and AST-scanned classes
- [x] Spatial priority system: OVERLAY=0, LAYER=5, MATERIAL=10
- [x] dolfin.Constant wrapping for all scalar spatial parameters — avoids FEniCS
      form recompilation between runs; verified bit-for-bit identical outputs vs
      reference run in gui/test/out/2026May07/g/
- [x] parse_spatial_quantity() used at all four spatial registration sites:
      _build_material_dict, domain overlay, layer properties, overlay regions
- [x] Bidirectional voltage sweeps: voltage list with negative values is split
      into positive (0 → +Vmax) and negative (0 → −Vmax) groups; a second
      ProblemData is created after the optical ramp so both sweeps start from
      the same fully-illuminated V=0 state; optical ramp paid only once
- [ ] Coordinate-range overlay extents (currently raises NotImplementedError)

## App shell
- [x] Dark theme global CSS
- [x] Sidebar navigation (all sections)
- [x] Header with file path (editable), Open button, Save button
- [x] Inline Open / Save As dropdowns with error feedback; autosave on every edit
      when filepath is set, so Save As is the only explicit save action needed
- [x] "unsaved" badge + save-location banner
- [x] Save (YAML); autosave when filepath already set
- [x] YAML round-trip (load_project / save_project)
- [x] New project button
- [x] Recent files: persistent ~/.simudo_gui_recents.json; "Recent ▼" dropdown in header
- [x] Auto-load: CLI arg > most-recent file > bundled fourlayer_example.yaml (example
      not added to recents); fourlayer_example.yaml updated to match fourlayer_example.py
- [x] Default output folder: model._default_output_folder() returns out/YYYY[Mon]DD/a;
      runner resolves relative paths against the YAML file's directory
- [x] Error toast / notification for load/save failures — dismissable red banner
      (`_error_banner`) in the app layout; `show_error(msg)` called by `autosave()`
      on exception; open/save dropdowns already showed inline errors
- [ ] About / version info

## Batch Processing  ✅ COMPLETE (v1)
See SCHEMA_DESIGN.md §Batch Processing for design notes.
- [x] Sidecar spec: `<project>.batch.yaml` next to the project file
- [x] Parameter picker: flat dropdown of all numeric `{value, unit}` nodes in the YAML
      tree, including layer thickness, layer properties, overlay properties, per-layer
      mesh (start / growth factor), and global mesh max edge length
- [x] Current-value hint shown below the parameter dropdown for each variation row
- [x] Value list: textarea — one value per line; auto-parsed on edit
- [x] Cartesian product (default) and Paired/zip mode toggle
- [x] Zip-mode validation: warning shown when swept parameter lists have unequal lengths
- [x] Live "→ N sub-runs will be generated" count
- [x] Generate sub-YAMLs: numbered subdirs under output_folder (0/, 1/, …);
      each sub-YAML has output.folder = "." so runner output stays co-located
- [x] Run Sweep: bounded parallel execution via threading.BoundedSemaphore;
      "Max number of cores to use" spinner; per-run log lines prefixed with run index
- [x] Stop Sweep button; sidecar autosaved on every edit
- [x] Docker: tested — multiple `docker exec` calls to one container are independent OS
      processes; FEniCS JIT cache is file-locked (safe); no per-run container needed
- [x] SSH: same remote host for all runs (v1 limitation; SSH backend rewrites
      output.folder so numbered subdirs won't be the output location for SSH runs)
- [ ] Skip-if-done: check for sim_V.csv in each subfolder before launching
- [x] Batch-aware output-panel schematic (see Output panel TODO above)
- [x] Per-run status table: compact table above the shared log stream; columns are
      Run #, Parameters (collapsed, e.g. "p-layer=0.5 µm, CB/E=1.2 eV"), Status badge
      (Queued/Running/Done/Failed/Skipped), and "▶ Open" button that navigates to the
      Output panel pointed at that subdirectory. Shared log stream kept below.
