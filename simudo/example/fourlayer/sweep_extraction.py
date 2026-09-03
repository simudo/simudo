import numpy as np

# import matplotlib
# matplotlib.use('Agg') # tells matplotlib not to load any gui display functions
import matplotlib.pyplot as plt
from os import path, scandir
from glob import glob
from .IV_Analysis import IV_params
import pandas as pd
import yaml
import attr
from cached_property import cached_property
from collections import defaultdict

try:
    from simudo.io import h5yaml
    loader = h5yaml.load
except:
    loader = yaml.load

from simudo.fem import expr
import dolfin

__all__ = [
    "SweepData",
    "jv_plot",
    "IB_band_diagram",
    "IB_generation_diagram",
    "subgap_generation_mismatch_diagram",
    "subgap_mismatch",
    "SpatialXdmf"
]


# @attr.s
class SweepData:
    """Extracts data from fourlayer simudo runs

Parameters
----------

folder: str
    Location of ``plot_meta`` files.
prefix: str, optional
    Sweep parameter name, usually something like``pd_V`` for a voltage sweep (but
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

    # folder = attr.ib()
    # prefix = attr.ib(default="pd_V")
    # contact_prefix = attr.ib(default="np")

    def __init__(self, folder, prefix="pd_V", contact_prefix="np", parameter_key = None):
        self.folder = folder
        self.prefix = prefix
        self.contact_prefix = contact_prefix
        self.cache = {}
        self.parameter_key = parameter_key



    @cached_property
    def jv(self,extra_fields=None):
        """
        Create a dataframe with j, v, p and names of files that contain those data

        If extra_fields is given, it should be a dicts of form {shortname: fullname}.
        df[shortname] will give data[fullname].
        """

        # If there is a pd_V.csv, could just read that in.
        # Instead we parse all the individual V files, because this ensures we get the right filename for plots at fixed V
        # Otherwise, read j, v from plot_meta.yaml files for each voltage point
        # return them in a Pandas DataFrame as self.jv
        df_list = []
        # May need tweaks to target correct filenames, if fourlayer.run changes
        files = glob(
            path.join(self.folder, f"{self.prefix}=*.plot_meta.yaml")
        )
        if not files:
            with scandir(str(self.folder)) as scan: # Additional scanning method since glob seems to find nothing if the run failed.
                for item in scan:
                    if item.is_file() and item.name.endswith(".plot_meta.yaml"):
                        files.append(item)
            if not files or len(files) == 0:
                raise FileNotFoundError(f"Failed to find any .plot_meta.yaml files in {self.folder}")

            return None
        for fname in files:
            # parse yaml data
            with open(fname) as f:
                data = loader(f)
    
            if self.parameter_key is not None:
                v = data[self.parameter_key]["value"]
            else:
                try:
                    if self.prefix[-1] == "I" or "sweep_parameter:V" not in data.keys():
                        # This is cheating, storing intensity in the V variable
                        v = data["sweep_parameter:I"]["value"]
                    else:
                        v = data["sweep_parameter:V"]["value"]
                except KeyError:
                    # old version of names
                    # v = data["parameter_value"]["value"]
                    v = data["sweep_parameter:parameter"]["value"]
            keys = list(data.keys())
            # `fname` handling kept from release_candidate; the multi-contact
            # loop below comes from [dbb88442cb].
            cur_dict = {"v": v,
                        "file": fname if isinstance(fname, str) else str(fname.path)}
            #Extract currents from all contacts
            for c in self.contact_prefix:
                contact_name = c + "_contact"
                #get currents from all bands (code modified from copilot)
                # find all the keys that start with "avg:current_" and ends with contact_name
                avg_current_keys = [k for k in keys if k.startswith("avg:current_") and k.endswith(contact_name)]
                j_band = [data[k]["value"] for k in avg_current_keys]
                j = np.sum(j_band)
                # j = (
                #     data["avg:current_CB:" + contact_name]["value"]
                #     + data["avg:current_VB:" + contact_name]["value"]
                # )
                cur_dict.update({"j_"+c: j, "p_"+c: j * v})
                if c == self.contact_prefix[0]:
                    cur_dict.update({"j": j, "p": j * v})
                # for shortname,fullname in extra_fields.items():
                #     cur_dict[shortname]=data[fullname]["value"]
            df_list.append(cur_dict)

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
        # print(path.join(self.folder, "submit.yaml"))
        with open(path.join(self.folder, "submit.yaml")) as stream:
            par = loader(stream)
            try:
                par = par["parameters"]
            except KeyError:
                # parameters were created by something other than fourlayer
                pass
                # par = yaml.load(stream, Loader=yaml.Loader)
        return par

    def get_spatial_data(self, row, num_points = 5001):
        # Read in the spatial plot file corresponding to desired row (returned by mpp_row, voc_row, or other)
        # row can be either a pandas row or the (int) index of a row
        if type(row) is int:
            row = self.jv.loc[row]
        spatial_file = row["file"].split(".plot_meta.yaml")[0] + ".csv.0"
        if path.isfile(spatial_file):
            print("genfromtxt")
            return np.genfromtxt(
                spatial_file, delimiter=",", names=True, deletechars=""
            )
        else:
            #try to get xdmf file information
            print("spatialXdmf")
            xdmf_filename = str(row["file"]).split(".plot_meta.yaml")[0] + "_full.xdmf"
            if not path.isfile(xdmf_filename):
                xdmf_filename = row["file"].split(".plot_meta.yaml")[0] + ".xdmf"
            spatialx = SpatialXdmf(xdmf_filename)
            d=dict()
            for f in spatialx.func_names["funcs"]:
                d[f] = spatialx.line_cut(getattr(spatialx,f),   num_points)[0]
            d["coord_x"]=spatialx.line_cut(getattr(spatialx,f), num_points)[1]
            return d

    def spatial_data(self,row_index, num_points = 5001):
        ''' implements a cache so data only needs to be extracted once for each row_number'''

        if row_index in self.cache:
            print("cached value")
            return self.cache[row_index]
        print("getting new value")
        new_data = self.get_spatial_data(row_index, num_points)
        self.cache[row_index] = new_data
        return new_data



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


class SpatialXdmf():

    def __init__(self, xdmf_filename,metadata_filename=None):
        self.xdmf_filename = xdmf_filename
        self.file = dolfin.XDMFFile(dolfin.MPI.comm_world, xdmf_filename)
        if metadata_filename is None:
            self.metadata_filename = path.splitext(xdmf_filename)[0] + "_func_names.yaml"
        self.func_names = loader(self.metadata_filename)    
        # read in the mesh
        self.mesh = dolfin.Mesh()
        self.file.read(self.mesh) 
        self.mesh_bbox = expr.mesh_bbox(self.mesh)

        for f in self.func_names["funcs"]:
            self.load_from_xdmf(f)

        # self.mu = mesh_util(self.mesh)

    # def __getattr__(self, attr):
    #     # if hasattr(self,attr):
    #     #     val = self.__getattribute__(attr)
    #     #     setattr(self, attr, val)
    #     # else:
    #     print("loading new " + str(attr))
    #     self.load_from_xdmf(attr)

    def load_from_xdmf(self, attr, space_name=None):
        '''load_from_xdmf(function_name, function_space)
        where function_name and function_space are strings
        e.g., load_from_xdmf("phi","DG0")
        '''
        #assert that attr is a string
        if space_name is None:
            space_name = self.func_names[attr]
        if space_name[0]=="v":
            space = dolfin.VectorFunctionSpace(self.mesh,space_name[1:-1], int(space_name[-1]))
        else:
            space = dolfin.FunctionSpace(self.mesh, space_name[:-1], int(space_name[-1]))
        # print(f"attr={attr}, space={space}")
        setattr(self, attr, dolfin.Function(space) )
        # print(getattr(self,attr))
        #TODO make work for non-checkpoint files, too. Can't load them into dolfin.Functions, though
        self.file.read_checkpoint(getattr(self,attr), str(attr))

    def line_cut(self,quantity,num_points=500, y=None, x=None):
        '''Line cut of ``quantity`` along the x axis, at points ``x_array`` 
        and position y (default: middle of the simulation area).
        For vector quantities, return the x-component

        Returns ``(values, x_array)``
        '''
        if y is None:
            y = np.mean(self.mesh_bbox[1]) #use the midpoint along the y-axis
        if x is None:
            x_box = self.mesh_bbox[0]
            x_array = np.linspace(x_box[0], x_box[1],num_points)
        else:
            x_array = x
        values = [quantity(dolfin.Point((x,y))) for x in x_array]
        # xhat = [1.,0.]
    
        if type(values[0]) is np.ndarray:
            # x-component of vector quantitites
            # values = [np.dot(v,xhat) for v in values]
            values = [v[0] for v in values]
        return (values, x_array)


        
    # def get_spatial(self,num_points=500):
        

# Some sample figures that can be made.
# Take as inputs the spatial data from SweepData.get_spatial_data and the mask from SweepData.IB_mask


def _has_column(spatial, name):
    """True if ``spatial`` carries a column called ``name``.

``get_spatial_data`` returns a numpy structured array for the CSV path and a
plain dict for the xdmf path. ``name in structured_array`` is an element-wise
comparison of a string against the rows, which numpy refuses:

    TypeError: Cannot compare structured or void to non-void arrays.

The column names of a structured array live in ``dtype.names``, so look there
when it is one and fall back to ordinary membership otherwise.
"""
    names = getattr(getattr(spatial, "dtype", None), "names", None)
    return name in (spatial if names is None else names)

def IB_band_diagram(spatial, IB_mask = True):
    # spatial, IB_mask as returned by data.get_spatial_data and data.IB_mask
    plt.plot(spatial["coord_x"], spatial["Ephi_CB"], color="k")
    plt.plot(spatial["coord_x"], (spatial["Ephi_VB"]), color="k")
    if _has_column(spatial, "Ephi_IB") and not (IB_mask is None or IB_mask is False):
        plt.plot(
            np.array(spatial["coord_x"])[IB_mask], np.array(spatial["Ephi_IB"])[IB_mask], color="k"
        )
    plt.plot(
        spatial["coord_x"],
        (spatial["qfl_CB"]),
        color="blue",
        linestyle="--",
        label=r"$E_{F,C}$",
    )
    if _has_column(spatial, "qfl_IB") and not (IB_mask is None or IB_mask is False):
        plt.plot(
            np.array(spatial["coord_x"])[IB_mask],
            np.array(spatial["qfl_IB"])[IB_mask],
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


def jv_plot(df, *args, **kwargs):
    p = plt.plot(df["v"], df["j"], *args, **kwargs)
    plt.xlabel(r"V (V)")
    plt.ylabel(r"J (mA/cm$^2$)")
    return p

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
