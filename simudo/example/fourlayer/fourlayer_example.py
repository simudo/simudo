# -*- coding: utf-8 -*-
"""
Construct a FSF-p-IB-n layered structure as described in ``fourlayer.py``

Functions:

:py:func:`simudo_multiplexer`:
    Given params dict, set up yaml files suitable for calling :py:func:`fourlayer.run`
    Run all of the requested simulations

    Results will be stored in the folder ``experiment_folder``.
    If multiplexing is requested with ``multiplex_keys``, results of each run are in ``experiment_folder/n`` for n=0,1,..
    If optimizing over parameter ``optimize_key``, :py:func:`scipy.fminbound` is used to optimize efficiency

    Produces short circuit band diagram and JV curve if ``sc_band_diagrams`` or ``jv_curve`` are ``True``.

    If ``multiprocess=True``, the various runs from the multiplexing will be sent to different processors

:py:func:`optimizer`:              Called by simudo_multiplexer to optimize efficiency over one parameter
:py:func:`efficiency_analysis`:    Find efficiency
:py:func:`shortcircuitbanddiagram`:Make band diagram at short circuit and save as png
:py:func:`jv_curve`:               Make JV curve and save as png

Parameters described in source below and in ``fourlayer.py``

"""

from .fourlayer import multiplex_setup, run
from scipy.optimize import fminbound
import shutil
from datetime import datetime, date
from pathlib import Path
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")  # don't display output. Only save plots.
from . import sweep_extraction as se
import logging
from simudo.util import blackbody
from simudo.util import make_unit_registry
from simudo.io import h5yaml
import warnings
import multiprocessing
import traceback

__all__ = [
    "simudo_multiplexer",
    "optimizer",
    "efficiency_analysis",
    "shortcircuitbanddiagram",
    "jv_curve",
]

# List of voltages to run
V_exts = list(np.linspace(0, 1.7, 25 + 1))  # For more precision, use 75
# V_exts.extend(np.linspace(1.15, 1.30, 30+1))
V_exts = list(set(V_exts))
V_exts.sort()


# fmt: off
params = {
    "experiment_folder" : Path("test/"), # Relative path. Where to put output files. Will be created.
    "text"              : "Example",     # Descriptive text, for this set of parameters
    "simple_mobility_model": True,  # True for constant IB mobility. False for fill-fraction dependent mobility
    "mu_I"              : 500,      # IB mobility0 cm^2/Vs
    "X"                 : 100,      # Solar concentration factor
    "sigma_opt_ci"      : 3e-14,    # optical cross section for CI process, cm^2
    "sigma_opt_iv"      : 3e-14,
    "sigma_th_ci"       : 5e-18,    # thermal (SR) cross section for CI process, cm^2
    "sigma_th_iv"       : 5e-18,
    "vth_CB"            : 2e7,      # thermal velocity in CB, cm/s, for SR process
    "vth_VB"            : 2e7,
    "NI"                : 2.5e17,   # IB DOS cm^-3
    "IB_E"              : 1.2,      # IB energy, eV
    "CB_E"              : 2.0,      # CB edge energy, eV
    "VB_E"              : 0.0,
    "NC"                : 2e19,     # CB effective DOS, cm^-3
    "NV"                : 2e19,
    "mu_n"              : 500,      # electron mobility, cm^2/Vs
    "mu_p"              : 500,
    "alpha_cv"          : 1e4,      # absorption coefficient in p- and n-regions, 1/cm
    "epsilon_r"         : 13,       # relative permittivity
    "T"                 : 300,      # Cell temperature, K
    "Ts"                : 6000,     # Sun temperature, K
    "NA"                : -1e17,    # cm^-3, doping in p region (negative indicates acceptors)
    "ND"                : 1e17,     # cm^-3, doping in n region
    "NFSF"              : -1e19,    # FSF doping
    "f_0"               : 0.5,      # equilibrium filling in IB for pure material
    "prefix"            : "fourlayer",
    "multiplex_keys"    : None,     # Which keys to use for multiplexing, eg ["NI"]
    "title_keys"        : ["NI","mu_I","X","sigma_opt_ci"], # Make an informative title for plots etc
}

# Thicknesses of the four layers in um
geometryparams = {
    "FSF_thickness"     : 0.05,
    "p_thickness"       : 1.0,
    "n_thickness"       : 1.0,
    "IB_thickness"      : 3,
}

# meshing parameters and list of desired voltages
simudoparams = {
    "mesh_start"        : 0.005,    # smallest mesh segment, in um
    "mesh_factor"       : 1.2,      # geometric growth factor
    "mesh_max"          : 0.02,     # largest mesh segment, in um
    "V_exts"            : V_exts,   # list of voltages at which to calculate
}

analysisparams = {
    "sc_band_diagrams"  : True,      # Make short circuit band diagrams for each output
    "jv_curve"          : True,      # Make JV curve for each output
    "optimize_key"      : None,      # Key of field to be optimized, eg, "IB_thickness". Set None for no optimization
    "optimize_bounds"   : [0.1, 5],  # range for fminbnd to use on optimize_key
    "optimize_xtol"     : 1e-2,      # for fminbound to find efficiency
    "multiprocess"      : False,     # Use multiprocessing
    "multiprocess_pool" : 4,         # Number of CPU"s to use for multiprocess
    "output_filename"   : "data.yaml"  # Filename for summary of efficiencies, if multiplexing
}
# fmt: on

params = {**params, **geometryparams, **simudoparams, **analysisparams}

# If we are multiplexing, params['multiplex_keys'] must be a list
if params["multiplex_keys"] and not isinstance(params["multiplex_keys"], list):
    raise TypeError


def optimizer(submitfile):
    curparams = h5yaml.load(str(submitfile))["parameters"]
    curparams["experiment_folder"] = submitfile.parent
    curparams["multiplex_keys"] = None  # Don't multiplex in optimization loop
    curparams["sc_band_diagram"] = False
    print("Optimizer begun here: " + str(submitfile.parent))
    val_opt, eff, ierr, numfunc = fminbound(
        tobeoptimized,
        *curparams["optimize_bounds"],
        args=[curparams],
        xtol=curparams["optimize_xtol"],
        disp=3,
        full_output=True
    )

    print(
        "Optimizer complete: "
        + str(-eff)
        + ", optimized "
        + str(curparams["optimize_key"])
        + ": "
        + str(val_opt)
    )
    logging.getLogger("main").info(
        (
            "Optimizer finished. "
            + curparams["optimize_key"]
            + ": {} , eff: {}"
        ).format(val_opt, eff)
    )

    # Run one more time, since the last val run by the optimizer may not have been at the optimal point
    tobeoptimized(val_opt, curparams)

    # Save output. Doing this for each run so data is saved even if not all runs complete
    single_run_output = {curparams["optimize_key"]: val_opt, "eff": -eff}
    h5yaml.dump(
        single_run_output,
        str(curparams["experiment_folder"] / "optimizer_summary.yaml"),
    )
    return val_opt, -eff


def tobeoptimized(val, params):
    curparams = params.copy()
    curparams[params["optimize_key"]] = val  # set parameters to current value
    submitfile = multiplex_setup(curparams, curparams["experiment_folder"])
    if isinstance(submitfile, list) and len(submitfile) == 1:
        submitfile = submitfile[0]
    run(submitfile)
    eff = efficiency_analysis(submitfile)
    logging.getLogger("main").info(
        (params["optimize_key"] + ": {} , eff: {}").format(val, eff)
    )

    return -eff


val_result = []
eff_result = []
# def multiprocess_optimize_callback(result):
#     # Callback for apply_async
#     val_result.append(result[0])
#     eff_result.append(result[1])


def multiprocess_map_callback(result):
    result = np.transpose(result)
    val_result.append(result[0, :])
    eff_result.append(result[1, :])


def multiprocess_error_callback(e):
    print("multiprocess_error_callback() ", e)
    traceback.print_exception(type(e), e, e.__traceback__)


def simudo_multiplexer(params):
    start_time = datetime.now()
    print(start_time)
    if params["multiprocess"]:
        print("Starting pool")
        pool = multiprocessing.Pool(params["multiprocess_pool"])

    runfiles = multiplex_setup(params, params["experiment_folder"])

    # container to hold results.
    data = {"params": params.copy()}
    if params["multiplex_keys"]:
        dimlengths = [len(params[k]) for k in params["multiplex_keys"]]
    else:
        dimlengths = 1

    numfiles = np.product(dimlengths)

    if params["optimize_key"] and params["multiprocess"]:
        print("Starting optimizers")
        # This multiprocessing call does not evaluate the callback to assemble the results until
        # all runs have successfully completed. If any return an error, callback does not complete
        pool.map_async(
            optimizer,
            runfiles,
            callback=multiprocess_map_callback,
            error_callback=multiprocess_error_callback,
        )

        # unpack output into data structure, if using output = pool.map instead of map_async
        # data[params['optimize_key']]=[v for v,e in output]
        # data['eff']=[e for v,e in output]
        # print(f"data = {data}")
        # print(f"output = {output}")
    else:
        data["eff"] = np.zeros(numfiles)
        if params["optimize_key"]:
            data[params["optimize_key"]] = np.zeros(numfiles)

        for counter, submitfile in enumerate(runfiles):
            if params["optimize_key"]:
                print(
                    "optimizing over "
                    + str(params["optimize_key"])
                    + " in "
                    + str(submitfile)
                )
                if params["multiprocess"]:
                    pass
                    # pool.apply_async(optimizer, (submitfile,),callback=multiprocess_optimize_callback,
                    #                  error_callback=multiprocess_error_callback)
                else:
                    val_opt, eff = optimizer(submitfile)
                    data["eff"][counter] = eff
                    data[params["optimize_key"]][counter] = val_opt
            else:
                if os.path.exists(
                    submitfile.parent / "mesh_Xs.csv"
                ):  # Check whether we've run already, in crude way
                    print(
                        "Already exists " + str(submitfile) + ". Not running."
                    )
                    continue  # We already did this run, so don't run it again. Just do postprocessing
                if params["multiprocess"]:
                    pool.apply_async(
                        run,
                        (submitfile,),
                        error_callback=multiprocess_error_callback,
                    )
                else:
                    run(submitfile)
                print(str(datetime.now()) + ": finished " + str(submitfile))

    if params["multiprocess"]:
        print("waiting for pool to finish")
        pool.close()
        pool.join()
        print("pool closed")
        if params["optimize_key"]:
            data["eff"] = np.array(eff_result)
            data[params["optimize_key"]] = np.array(val_result)
    finish_time = datetime.now()
    data["run_time"] = finish_time - start_time
    print("Finished simudo runs at ", finish_time)
    print("Run time: ", finish_time - start_time)

    print("post processing")
    if params["sc_band_diagrams"]:
        for submitfile in runfiles:
            shortcircuitbanddiagram(submitfile)
    if params["jv_curve"]:
        for submitfile in runfiles:
            jv_curve(submitfile)

    # Only perform efficiency analysis if there is a voltage sweep and if we haven't already populated data['eff']
    if (
        len(params["V_exts"]) > 1
        and params["X"] != 0
        and not np.any(data["eff"])
    ):
        for counter, submitfile in enumerate(runfiles):
            eff = efficiency_analysis(submitfile)
            data["eff"][counter] = eff
            print("{}. eff = {}".format(counter, eff))

    # reshape data to reasonable shapes if multiplexing
    if params["multiplex_keys"]:
        try:
            data["eff"].reshape(*dimlengths)
            data[params["optimize_key"]].reshape(*dimlengths) if params[
                "optimize_key"
            ] in data else None
        except:
            print("reshape failed. Saving output structure anyway")
    h5yaml.dump(
        data, str(params["experiment_folder"] / params["output_filename"])
    )

    return


def efficiency_analysis(submitfile):
    # Efficiency analysis.
    from .IV_Analysis import IV_params

    U = make_unit_registry()
    curparams = h5yaml.load(str(submitfile))["parameters"]
    if isinstance(curparams["Ts"] * 1.0, float) or len(curparams["Ts"]) == 1:
        bb = blackbody.Blackbody(temperature=curparams["Ts"] * U.K)
    else:
        warnings.warn(
            "Efficiency analysis assumes there is only one Ts. Using first value."
        )
        bb = blackbody.Blackbody(temperature=curparams["Ts"][0] * U.K)
    P_in = (
        curparams["X"] * bb.total_intensity_on_earth()
    )  # Pint quantity. W/m^2
    P_in.ito(U.mW / (U.cm ** 2))  # convert to mW/cm^2
    P_in = P_in.magnitude
    # if not isinstance(curparams['X']*1.0,float) and len(curparams['X'])>1:
    #     warnings.warn("Assuming only one value of X")

    curdata = se.SweepData(submitfile.parent)
    powersign = -1  # power-producing current is j<0
    try:
        power = IV_params(curdata.jv["j"], curdata.jv["v"], powersign)
        eff = power.pmax / P_in
    except:
        print("unable to extract power in " + str(curdata.folder))
        try:
            eff = (
                max(powersign * curdata.jv["p"]) / P_in
            )  # take the highest achieved value
        except:
            eff = float("nan")
    return eff


def shortcircuitbanddiagram(submitfile):
    # Example of how to use sweep_extraction to make band diagram plots
    from matplotlib import pyplot as plt

    today = date.today()
    datafolder = submitfile.parent
    savefolder = datafolder
    curfoldername = datafolder.name
    curparams = h5yaml.load(str(submitfile))["parameters"]

    curdata = se.SweepData(datafolder)
    titlesize = 8
    savefile = True

    plotter = se.IB_band_diagram
    name_prefix = "sc_band_"

    # extract spatial data at short circuit
    spatial = curdata.get_spatial_data(curdata.v_row(0.0))
    IB_mask = curdata.IB_mask(spatial)

    fig, Ax = plt.subplots(1, 1, figsize=(6, 4), dpi=120)
    plotter(spatial, IB_mask)
    plt.legend()
    plt.title(
        "SC " + str(submitfile) + " " + curparams["title"], fontsize=titlesize
    )
    filename = (
        name_prefix + curfoldername + "_" + today.strftime("%b%d") + ".png"
    )

    if savefile:
        plt.savefig(savefolder / filename)
        print("Saved: " + filename)


def jv_curve(submitfile):
    # Example of making a JV curve
    from matplotlib import pyplot as plt

    today = date.today()
    datafolder = submitfile.parent
    savefolder = datafolder
    curfoldername = datafolder.name
    curparams = h5yaml.load(str(submitfile))["parameters"]
    titlesize = 8
    savefile = True

    curdata = se.SweepData(datafolder)
    name_prefix = "JV_"

    fig, Ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4), dpi=120)
    se.jv_plot(curdata.jv)
    plt.title(
        "JV " + str(submitfile) + " " + curparams["title"], fontsize=titlesize
    )
    filename = (
        name_prefix + curfoldername + "_" + today.strftime("%b%d") + ".png"
    )

    if savefile:
        plt.savefig(savefolder / filename)
        print("Saved: " + filename)


if __name__ == "__main__":
    simudo_multiplexer(params)
