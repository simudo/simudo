# -*- coding: utf-8 -*-
"""
Reference run of the fourlayer IB solar cell using the original fourlayer.py
driver.  Output goes to gui/out/reference/ (relative to the working directory,
which in Docker is /home/user/simudo/code).

Run from the Docker container:
    python3 gui/fourlayer_reference.py

The output is used by gui/compare_jv.py to validate simudo_1d_runner.py against
the original driver.
"""

from simudo.example.fourlayer.fourlayer import multiplex_setup, run
import numpy as np
from pathlib import Path
import logging

# ── Voltage list — must match fourlayer_example.yaml exactly ──────────────────
V_exts = list(np.linspace(0, 1.7, 25 + 1))
V_exts = sorted(set(V_exts))

# ── Physics parameters — must match fourlayer_example.yaml exactly ────────────
# fmt: off
params = {
    "experiment_folder" : Path("gui/out/reference"),
    "text"              : "Reference",
    "simple_mobility_model": True,
    "mu_I"              : 500,
    "X"                 : 100,
    "sigma_opt_ci"      : 3e-14,
    "sigma_opt_iv"      : 3e-14,
    "sigma_th_ci"       : 5e-18,
    "sigma_th_iv"       : 5e-18,
    "vth_CB"            : 2e7,
    "vth_VB"            : 2e7,
    "NI"                : 2.5e17,
    "IB_E"              : 1.2,
    "CB_E"              : 2.0,
    "VB_E"              : 0.0,
    "NC"                : 2e19,
    "NV"                : 2e19,
    "mu_n"              : 500,
    "mu_p"              : 500,
    "alpha_cv"          : 1e4,
    "epsilon_r"         : 13,
    "T"                 : 300,
    "Ts"                : 6000,
    "NA"                : -1e17,
    "ND"                : 1e17,
    "NFSF"              : -1e19,
    "f_0"               : 0.5,
    "prefix"            : "reference",
    "multiplex_keys"    : None,
    "title_keys"        : ["NI", "mu_I", "X", "sigma_opt_ci"],
    # Geometry
    "FSF_thickness"     : 0.05,
    "p_thickness"       : 1.0,
    "n_thickness"       : 1.0,
    "IB_thickness"      : 3.0,
    # Mesh
    "mesh_start"        : 0.005,
    "mesh_factor"       : 1.2,
    "mesh_max"          : 0.02,
    "V_exts"            : V_exts,
    "mumps_icntl"       : (),
    "find_unconverged"  : False,
    # Analysis — disabled; comparison is done by compare_jv.py
    "sc_band_diagrams"  : False,
    "jv_curve"          : False,
    "optimize_key"      : None,
    "optimize_bounds"   : [0.1, 5],
    "optimize_xtol"     : 1e-2,
    "multiprocess"      : False,
    "multiprocess_pool" : 4,
    "output_filename"   : "data.yaml",
}
# fmt: on


def run_reference():
    params["experiment_folder"].mkdir(parents=True, exist_ok=True)
    submitfiles = multiplex_setup(params, params["experiment_folder"])
    submitfile = submitfiles[0] if isinstance(submitfiles, list) else submitfiles
    run(submitfile)
    print(f"Reference run complete. Output in {params['experiment_folder']}")


if __name__ == "__main__":
    run_reference()
