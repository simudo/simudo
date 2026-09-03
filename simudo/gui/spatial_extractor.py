#!/usr/bin/env python3
"""
Spatial data extractor for Simudo output.

Runs inside the execution environment (Docker container or local) where
simudo and dolfin are installed.  Delegates all XDMF reading and function
evaluation to :class:`simudo.example.fourlayer.sweep_extraction.SpatialXdmf`.

Usage:
    python3 spatial_extractor.py <output_dir> [--n-points N]

stdout protocol (one block per sweep point):
    # BEGIN_POINT V=<v> I=<i>
    coord_x,field1,field2,...
    val,val,val,...
    ...
    # END_POINT V=<v> I=<i>

V= files use I=0; I= files use V=0.
Dolfin log messages and warnings go to stderr only.
"""

from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Extract Simudo spatial fields to CSV (stdout).")
    parser.add_argument("output_dir",
                        help="Path to simulation output directory inside container")
    parser.add_argument("--n-points", type=int, default=500,
                        help="Number of uniform x-grid points (default: 500)")
    args = parser.parse_args()

    try:
        import dolfin
        dolfin.set_log_level(dolfin.LogLevel.ERROR)
    except Exception:
        pass

    from simudo.example.fourlayer.sweep_extraction import SpatialXdmf

    v_files = sorted(glob.glob(os.path.join(args.output_dir, "sim_V=*_full.xdmf")))
    i_files = sorted(glob.glob(os.path.join(args.output_dir, "sim_I=*_full.xdmf")))

    if not v_files and not i_files:
        print(f"ERROR: No sim_V=*_full.xdmf or sim_I=*_full.xdmf files found in {args.output_dir}",
              file=sys.stderr)
        sys.exit(1)

    # The voltage sweep runs *after* the intensity ramp, at the intensity the
    # ramp finished on -- so a sim_V=* point is not dark unless the whole run
    # was.  Without this, every voltage-sweep point was reported as I=0, and
    # sim_V=0 collided with the dark sim_I=0 point in the GUI's selector.
    ramp_values = [v for v in (_parse_sweep_value(f, 'I') for f in i_files)
                   if v is not None]
    sweep_intensity = max(ramp_values) if ramp_values else 0.0

    emitted = set()
    for xdmf_path in i_files:
        pt = _process_one(xdmf_path, args.n_points, SpatialXdmf, sweep='I',
                          other_value=0.0)
        if pt is not None:
            emitted.add(pt)
    for xdmf_path in v_files:
        # The end of the ramp and the start of the sweep are the same state;
        # emit it once.
        val = _parse_sweep_value(xdmf_path, 'V')
        if val is not None and (val, sweep_intensity) in emitted:
            continue
        _process_one(xdmf_path, args.n_points, SpatialXdmf, sweep='V',
                     other_value=sweep_intensity)


def _parse_sweep_value(xdmf_path: str, sweep: str):
    """Numeric value out of a ``sim_<sweep>=<value>_full.xdmf`` filename."""
    basename = os.path.basename(xdmf_path)
    prefix = f"sim_{sweep}="
    try:
        return float(basename[len(prefix):-len("_full.xdmf")])
    except ValueError:
        return None


def _process_one(xdmf_path: str, n_points: int, SpatialXdmf, sweep: str = 'V',
                 other_value: float = 0.0):
    """Emit one point.  Returns its ``(V, I)`` pair, or None if it was skipped."""
    basename = os.path.basename(xdmf_path)
    val = _parse_sweep_value(xdmf_path, sweep)
    if val is None:
        print(f"WARN: Could not parse {sweep} value from {basename}", file=sys.stderr)
        return None

    point_v = val if sweep == 'V' else other_value
    point_i = val if sweep == 'I' else other_value

    try:
        spatialx = SpatialXdmf(xdmf_path)
    except Exception as exc:
        print(f"WARN: Could not load {basename}: {exc}", file=sys.stderr)
        return None

    func_names: list = spatialx.func_names["funcs"]
    data: dict[str, np.ndarray] = {}
    coord_set = False

    for func_name in func_names:
        try:
            func = getattr(spatialx, func_name)
            vals, x_arr = spatialx.line_cut(func, n_points)
            data[func_name] = np.asarray(vals, dtype=float)
            if not coord_set:
                data["coord_x"] = np.asarray(x_arr, dtype=float)
                coord_set = True
        except Exception as exc:
            print(f"WARN: Skipping {func_name}: {exc}", file=sys.stderr)

    if not data or not coord_set:
        print(f"WARN: No data extracted from {basename}", file=sys.stderr)
        return None

    cols = ["coord_x"] + [k for k in data if k != "coord_x"]
    try:
        arr = np.column_stack([data[c] for c in cols])
    except Exception as exc:
        print(f"WARN: Could not assemble array for {sweep}={val}: {exc}",
              file=sys.stderr)
        return None

    print(f"# BEGIN_POINT V={point_v} I={point_i}")
    print(",".join(cols))
    for row in arr:
        print(",".join(f"{v:.10g}" for v in row))
    print(f"# END_POINT V={point_v} I={point_i}")
    sys.stdout.flush()
    return (point_v, point_i)


if __name__ == "__main__":
    main()
