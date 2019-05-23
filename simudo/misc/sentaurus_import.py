
try:
    # FIXME: be more selective about import error
    # gracefully skip import on incompatible/missing dolfin
    from ..mesh.product2d import Product2DMesh
    from ..util.expr import mesh_bbox
except:
    Product2DMesh = None
    mesh_bbox = None

from ..util import parse_kv, xlistdir

import pandas as pd
import lzma
import numpy as np
from pint import UnitRegistry
import re
import os
from os import path as osp
import dolfin
import lzma
from scipy.interpolate import interp1d
from functools import partial


_ureg = UnitRegistry()

def assert_single_value(Xs):
    if not all(Xs[0] == Xs):
        raise ValueError("non-uniform array")
    return Xs[0]

def _regex_filenames_in_dir(regex, directory):
    for basename in os.listdir(directory):
        m = regex.match(basename)
        if not m: continue
        yield (os.path.join(directory, basename), m.groupdict())

def list_sentaurus_diode_1d_data(sentaurus_dir='data/sentaurus/'):
    r = []

    # 64-bit data
    # for basename, fullpath in xlistdir(osp.join(data_dir, 'sentaurus-study1')):
    #     before, _, after = basename.rpartition('.csv.xz')
    #     if after: continue
    #     d = parse_kv(before)
    #     d['minority_srv'] = 'inf'

    # 128-bit data, inf minority SRV
    for basename, fullpath in xlistdir(osp.join(sentaurus_dir, 'study2-spatial'), both=1):
        before, _, after = basename.rpartition('.csv.xz')
        if after: continue
        d = parse_kv(before)
        r.append((fullpath, d))

    # 128-bit data, zero minority SRV
    for basename, fullpath in xlistdir(osp.join(sentaurus_dir, 'study3-spatial'), both=1):
        before, _, after = basename.rpartition('.csv.xz')
        if after: continue
        d = parse_kv(before)
        r.append((fullpath, d))

    return r

def list_sentaurus_diode_1d_ivcurves(sentaurus_dir='data/sentaurus/'):
    r = []

    # 128-bit data, inf minority SRV
    for basename, fullpath in xlistdir(osp.join(sentaurus_dir, 'study2-iv'), both=1):
        before, _, after = basename.rpartition('.plt.csv')
        if after: continue
        d = parse_kv(before)
        r.append((fullpath, d))

    # 128-bit data, zero minority SRV
    for basename, fullpath in xlistdir(osp.join(sentaurus_dir, 'study3-iv'), both=1):
        before, _, after = basename.rpartition('.plt.csv')
        if after: continue
        d = parse_kv(before)
        r.append((fullpath, d))

    for _, d in r:
        d['device_length'] = (0.16670001 if (d['c'] in (
            '1e15', '1e18')) else 0.01667)
    return r

def read_df(handle, xspan=1.0, unit_registry=None):
    '''`filename` must be CSV as exported by Sentaurus

returns pandas DataFrame, with columns x,f,E,p suitable for
nondimensionalized version of the PDE (where diffusion and mobility
are set to 1).
'''
    U = unit_registry if unit_registry is not None else _ureg

    df = pd.read_csv(handle, skiprows=[1])

    devlength = (max(df['Y']) - min(df['Y']))/xspan*U.micrometer

    x_c = (U.cm / devlength).m_as('dimensionless')

    beta_e = (1/(U.k * 300*U.K).m_as(U.eV))
    mobility = assert_single_value(df['hMobility'].values)
    diffusivity = mobility / beta_e

    q_e = (1*U.elementary_charge).m_as(U.coulomb)

    u = df['hDensity']
    phi = df['ElectrostaticPotential'] * beta_e
    w = -np.log(u) - phi

    # u = exp(-(V + w))
    # -log(u) - V = w

    odf = pd.DataFrame({
        'x': df['Y'] / (devlength / U.micrometer).m_as('dimensionless'),
        'g': df['TotalRecombination'] / x_c**2 / diffusivity * -1,
        'E_x': df['ElectricField-Y'] / x_c * beta_e,
        'u': u,
        'V': phi,
        'w': w,
        'j_x': df['hCurrentDensity-Y'] / q_e / x_c / diffusivity}) # A/cm^2

    units = dict(
        x=devlength,
        g=U('1/cm^3') / (1 / x_c**2 / diffusivity),
        E_x=U('V/cm') / (1 / x_c * beta_e),
        u=U('1/cm^3'),
        V=U('V') / beta_e,
        w=U('V') / beta_e,
        j_x=U('A/cm^2') / (1 / q_e / x_c / diffusivity))

    odf.sort_values('x', inplace=True)
    odf.reset_index(drop=True, inplace=True)

    return (odf, units)

def read_df_unitful(handle, unit_registry):
    '''`filename` must be CSV as exported by Sentaurus

returns pandas DataFrame
'''
    U = unit_registry

    df = pd.read_csv(handle, skiprows=[1])

    odf = pd.DataFrame(dict(
        x=df['Y'],
        V=df['ElectrostaticPotential'],
        E_x=df['ElectricField-Y'],
        g=df['TotalRecombination'] * -1,
        CB_u=df['eDensity'],
        VB_u=df['hDensity'],
        CB_j_x=df['eCurrentDensity-Y']*1e3,
        VB_j_x=df['hCurrentDensity-Y']*1e3,
        CB_mobility=df['eMobility'],
        VB_mobility=df['hMobility'],
    ))

    density_unit = 1*U('1/cm^3')
    current_unit = 1*U('mA/cm^2')
    mob_unit = 1*U('cm^2/V/s')
    units = dict(
        x=1*U('micrometer'),
        V=1*U('V'),
        E_x=1*U('V/cm'),
        g=1*U('1/cm^3 / s'),
        CB_u=density_unit,
        VB_u=density_unit,
        CB_j_x=current_unit,
        VB_j_x=current_unit,
        CB_mobility=mob_unit,
        VB_mobility=mob_unit,
    )

    odf.sort_values('x', inplace=True)
    odf.reset_index(drop=True, inplace=True)

    return (odf, units)

def inplace_rescale_df_columns(df, units, new_units):
    actual_new_units = units.copy()
    actual_new_units.update(new_units)

    for k, unit in units.items():
        wanted_unit = actual_new_units[k]
        conv = (1 * unit / wanted_unit).m_as('dimensionless')
        if conv != 1:
            df[k] *= conv

    return actual_new_units

def interpolate_df(df, units, Xs):

    new_units = units.copy()
    new_units['x'] = Xs.units
    new_df = pd.DataFrame({'x': Xs.m})

    old_Xs = (df['x'].values * units['x'] / Xs.units).m_as('dimensionless')
    for k, u in units.items():
        if k == 'x': continue
        new_df[k] = interp1d(old_Xs, df[k].values)(Xs.m)

    return (new_df, new_units)

def interpolate(df, units, mesh, mesh_unit, function_space):
    assert mesh.geometry().dim() == 2
    ((xmin, xmax), (ymin, ymax)) = mesh_bbox(mesh)
    xscale = (units['x'] / mesh_unit).m_as('dimensionless')

    df = df.copy(deep=False)
    df['x'] = df['x'] * xscale

    # handle case where Sentaurus mesh is smaller than ours,
    # by duplicating the first and last rows as an extrapolation
    before = df.iloc[[0]]
    after  = df.iloc[[-1]]
    before['x'].values[0] = xmin
    after ['x'].values[0] = xmax

    objs = []
    if xmin < df['x'].values[ 0]: objs.append(before)
    objs.append(df)
    if df['x'].values[-1] < xmax: objs.append(after)

    df = pd.concat(objs, ignore_index=True)

    Xs = df['x']
    Ys = (xmin, xmax)

    pm = Product2DMesh(Xs=Xs, Ys=Ys)

    result = {}
    for k, unit in units.items():
        if k == 'x': continue
        unit = 1*unit
        pm_func = pm.create_function_from_x_values(
            df[k].values * unit.magnitude)
        func = dolfin.Function(function_space)
        dolfin.LagrangeInterpolator.interpolate(func, pm_func)
        result[k] = func * unit.units

    return result

def jam_sentaurus_data_into_solution(df, units, solution):
    from ..util.assign import opportunistic_assign
    pdd = solution.pdd
    md = pdd.mesh_data
    mu = pdd.mesh_util
    fsr = mu.function_subspace_registry
    R = interpolate(df, units, md.mesh, md.mesh_unit, mu.space.CG1)
    oa = partial(opportunistic_assign, function_subspace_registry=fsr)
    fsr.register(mu.space.CG1)
    oa(source=R['V'], target=pdd.poisson.phi)
    ikw = dict(function_subspace_registry=fsr,
               assign=oa)
    for k in ('CB', 'VB'):
        b = pdd.bands[k]
        b._initialize_from_custom_qfl(source=b.u_to_qfl(R[k+'_u']), **ikw)
        b._initialize_from_custom_j(source=R[k+'_j_x'] * mu.unitvec_x, **ikw)
        # oa(source=b.u_to_qfl(R[k+'_u']), target=b.qfl)
        # oa(source=R[k+'_j'] * mu.unitvec_x, target=b.j)

def remove_close_x_values(df, min_distance=1e-6):
    '''assumes df['x'] is already sorted'''

    keep_ = pd.Series(False, df.index)

    keep = keep_.values

    xs = df['x'].values
    min_x = xs[0] - 2*min_distance
    for i, x in enumerate(df['x'].values):
        if x >= min_x:
            keep[i] = True
            min_x = x + min_distance

    df.drop(df[keep_ == False].index, inplace=True)

    return df

def sampling_density(Xs, density):
    '''density is points/length'''
    xmin = Xs[0]
    xmax = Xs[-1]
    indices = np.searchsorted(Xs, np.arange(xmin, xmax, 1/density))
    indices = np.unique(indices)
    return indices

def df_enforce_sampling_density(df, X, cutpoints, densities):
    keep_ = pd.Series(False, df.index)
    keep = keep_.values

    Xs = X.values.view()

    if len(cutpoints) != len(densities) + 1: raise AssertionError()

    for i, density in enumerate(densities):
        a, b = cutpoints[i], cutpoints[i+1]
        if density is not None:
            keep[sampling_density(Xs[a:b], density) + a] = True
        else:
            keep[a:b] = True

    keep[-1] = True
    df.drop(df[keep_ == False].index, inplace=True)

    return df
