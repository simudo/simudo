import numpy as np

# import matplotlib
# matplotlib.use('Agg') # tells matplotlib not to load any gui display functions
import matplotlib.pyplot as plt
from os import path
from glob import glob
from .IV_Analysis import IV_params
import pandas as pd
import yaml
import attr
from cached_property import cached_property

try:
    from simudo.io import h5yaml

    loader = h5yaml.load
except:
    loader = yaml.load

__all__ = [
    "SweepData",
    "jv_plot",
    "IB_band_diagram",
    "IB_generation_diagram",
    "subgap_generation_mismatch_diagram",
    "subgap_mismatch",
]


@attr.s
class SweepData:
    """Extracts data from fourlayer simudo runs

Parameters
----------

folder: str
    Location of ``plot_meta`` files.
parameter_name: str, optional
    Sweep parameter name, usually ``pd_V`` for a voltage sweep (but
    was ``a parameter`` at some point in history). (default: ``"pd_V"``)

Notes
-----

The most important properties are:

    :py:attr:`jv`:                 Extract j, v, p as pandas dataframe

    :py:attr:`mpp_row`:            Determine which file contains data closest to max power point

    :py:attr:`voc_row`:            Determine which file contains data closest to open circuit

    :py:attr:`v_row`:              Determine which file contains data closest to specified voltage

    :py:attr:`get_spatial_data`:   Read in spatial data for plotting from specified file

    :py:attr:`IB_mask`:            Mask suitable for plotting properties only in IB region

The output of the :py:func:`SweepData.jv` can be given to :py:func:`IV_Analysis.IV_params`
to find the max power point and efficiency

Example
-------

::

    data = SweepData(filename)
    spatial = data.get_spatial_data(data.mpp_row)
    IB_mask = data.IB_mask(spatial)
    IB_band_diagram(spatial,IB_mask)
    """

    folder = attr.ib()
    parameter_name = attr.ib(default="pd_V")

    @cached_property
    def jv(self):
        """
        Create a dataframe with j, v, p and names of files that contain those data
        """

        # If there is a pd_V.csv, could just read that in.
        # Instead we parse all the individual V files, because this ensures we get the right filename for plots at fixed V
        # Otherwise, read j, v from plot_meta.yaml files for each voltage point
        # return them in a Pandas DataFrame as self.jv
        df_list = []

        # May need tweaks to target correct filenames, if fourlayer.run changes
        files = glob(
            path.join(self.folder, f"{self.parameter_name}=*.plot_meta.yaml")
        )
        if not files:
            print("no files found")
        for fname in files:
            # parse yaml data
            with open(fname) as f:
                data = loader(f)
            try:
                if self.parameter_name == "pd_I":
                    # This is cheating, storing intensity in the V variable
                    v = data["sweep_parameter:I"]["value"]
                else:
                    v = data["sweep_parameter:V"]["value"]
            except:
                # old version of names
                # v = data["parameter_value"]["value"]
                v = data["sweep_parameter:parameter"]["value"]
            j = (
                data["avg:current_CB:n_contact"]["value"]
                + data["avg:current_VB:n_contact"]["value"]
            )
            df_list.append({"j": j, "v": v, "p": j * v, "file": fname})
        df = pd.DataFrame(df_list)
        # Sort the dataframe by voltage and make the indices go in that order
        df.sort_values("v", inplace=True)
        df.reset_index(drop=True, inplace=True)
        # self.jv = df  #Uncomment this line if removing the @cached_property
        return df

    @cached_property
    def mpp_row(self):
        # Find the row closest to max power point, for plotting.
        # Not a good way to find efficiency, as it depends on the voltages calculated
        index = self.jv["p"].idxmin()
        return self.jv.loc[index]

    @cached_property
    def voc_row(self):
        # Find the row closest to Voc
        index = abs(self.jv["j"]).idxmin()
        return self.jv.loc[index]

    def v_row(self, V):
        # Find row closest to voltage V
        index = abs(self.jv["v"] - V).idxmin()
        return self.jv.loc[index]

    @cached_property
    def params(self):
        # Get params from submit.yaml file
        with open(path.join(self.folder, "submit.yaml")) as stream:
            par = loader(stream)["parameters"]
        return par

    def get_spatial_data(self, row):
        # Read in the spatial plot file corresponding to desired row (returned by mpp_row, voc_row, or other)
        spatial_file = row["file"].split(".plot_meta.yaml")[0] + ".csv.0"
        return np.genfromtxt(
            spatial_file, delimiter=",", names=True, deletechars=""
        )

    def IB_mask(self, spatial_data):
        # For spatial_data (as returned by get_spatial_data), return a mask showing where the IB is located
        if not "p_thickness" in self.params:
            # Old default values
            self.params["FSF_thickness"] = 0.05
            self.params["p_thickness"] = 1
            self.params["n_thickness"] = 1
        x0 = float(self.params["FSF_thickness"]) + float(
            self.params["p_thickness"]
        )
        x1 = x0 + float(self.params["IB_thickness"])
        return np.where(
            (spatial_data["coord_x"] > x0) & (spatial_data["coord_x"] < x1)
        )


# Some sample figures that can be made.
# Take as inputs the spatial data from SweepData.get_spatial_data and the mask from SweepData.IB_mask
def IB_band_diagram(spatial, IB_mask):
    # spatial, IB_mask as returned by data.get_spatial_data and data.IB_mask
    plt.plot(spatial["coord_x"], spatial["Ephi_CB"], color="k")
    plt.plot(spatial["coord_x"], (spatial["Ephi_VB"]), color="k")
    plt.plot(
        spatial["coord_x"][IB_mask], spatial["Ephi_IB"][IB_mask], color="k"
    )
    plt.plot(
        spatial["coord_x"],
        (spatial["qfl_CB"]),
        color="blue",
        linestyle="--",
        label=r"$E_{F,C}$",
    )
    plt.plot(
        spatial["coord_x"][IB_mask],
        spatial["qfl_IB"][IB_mask],
        color="orange",
        linestyle="--",
        label=r"$E_{F,I}$",
    )
    plt.plot(
        spatial["coord_x"],
        (spatial["qfl_VB"]),
        color="red",
        linestyle="--",
        label=r"$E_{F,V}$",
    )
    plt.xlabel(r"x ($\mu$m)")
    plt.ylabel(r"Energy (eV)")


# plt.legend()


def IB_generation_diagram(spatial, IB_mask):
    # spatial, IB_mask as returned by data.get_spatial_data and data.IB_mask
    plt.semilogy(
        spatial["coord_x"], (spatial["g_CB"]), color="blue", label=r"g$_{CB}$"
    )
    plt.semilogy(
        spatial["coord_x"][IB_mask],
        (spatial["g_IB"])[IB_mask],
        color="orange",
        label=r"g$_{IB}$",
    )
    plt.semilogy(
        spatial["coord_x"][IB_mask],
        -(spatial["g_IB"])[IB_mask],
        color="orange",
        linestyle="--",
        label=r"-g$_{IB}$",
    )
    plt.semilogy(
        spatial["coord_x"], (spatial["g_VB"]), color="red", label=r"g$_{VB}$"
    )
    plt.xlabel(r"X position ($\mu$m)")
    plt.ylabel(r"g (cm$^{-3}$s$^{-1}$)")


# plt.legend()


def subgap_generation_mismatch_diagram(spatial, IB_mask):
    # spatial, IB_mask as returned by data.get_spatial_data and data.IB_mask
    mismatch = (
        spatial["g_opt_ci_IB"] + spatial["g_opt_iv_IB"]
    )  # the CI term is always negative
    plt.semilogy(
        spatial["coord_x"][IB_mask],
        mismatch[IB_mask],
        color="blue",
        label=r"$g_{ci}-g_{iv}$",
    )
    plt.semilogy(
        spatial["coord_x"][IB_mask],
        -mismatch[IB_mask],
        color="red",
        label=r"$g_{iv}-g_{ci}$",
    )
    #  plt.xlabel(r"X position ($\mu$m)")
    plt.ylabel(r"g (cm$^{-3}$s$^{-1}$)")


# plt.legend()


def jv_plot(df):
    plt.plot(df["v"], df["j"])
    plt.xlabel(r"V (V)")
    plt.ylabel(r"J (mA/cm$^2$)")


def subgap_mismatch(spatial, IB_mask):
    # integrate the mismatch
    mismatch = (spatial["g_opt_ci_IB"] + spatial["g_opt_iv_IB"])[
        IB_mask
    ]  # the CI term is always negative
    dx = np.diff(spatial["coord_x"][IB_mask])
    subgap_mismatch = np.sum(mismatch[0:-1] ** 2 * dx)  # crude integral
    subgap_gen = np.sum(
        (abs(spatial["g_opt_iv_IB"]) + abs(spatial["g_opt_ci_IB"]))[IB_mask][
            0:-1
        ]
        ** 2
        * dx
    )
    return subgap_mismatch / subgap_gen
