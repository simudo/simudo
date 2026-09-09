import csv
import logging
import warnings
import operator as OP
from contextlib import closing
from functools import reduce
import os
from cached_property import cached_property
import numpy as np

import dolfin

from ..io import h5yaml
from ..io.xdmf import XdmfPlot
from ..io.csv import LineCutCsvPlot
from ..util import SetattrInitMixin
from ..mesh import CellRegions


class ZeroAreaWarning(RuntimeWarning):
    pass


class OutputWriter(SetattrInitMixin):
    '''Write output data extracted from a solution object.

    The output can be any of a plot  on a mesh, on a 1D linecut,
    or in a csv file containing data from multiple solutions
    during a parameter sweep.

    parameter_name: label for the parameter being swept

    filename_prefix: a prefix to any filenames that will be saved.

    plot_mesh -- Save data on the original mesh. (default: False)

    plot_1d -- Extract data along a 1D line cut and save to a csv file. (default: False)

    plot_iv -- Save a csv file with terminal voltages, currents and other extracted
    data. (default: True)

    stepper -- (optional) give access to the stepper. Allows plotting quantities
    related to solution such as error metric (du).

    plot_du -- Plot du (only if stepper is also given, default: False)
    '''
    _meta_csv = None
    stepper = None
    plot_iv = True
    plot_1d = False
    plot_mesh = False
    plot_mesh_full = False #give a checkpoint output with DG outputs of all quantities
    plot_du = False #output the du vector from the last iteration
    clip_finite_extent = True # Set to zero all quantities outside the extent of the band
    parameter_name = 'parameter'
    line_cut_resolution = 5001
    hj_interface_condition = False
    hj_position = None
    hj_condition_filename = None
    hj_test_eps = 1e-10
    hj_test_y = 0.5
    vs_filename = "vs_quantities"
    vs_output_x = None
    vs_output_y = None
    vs_write_quantities = False

    @cached_property
    def meta_extractors(self):
        return []

    def format_parameter(self, solution, parameter_value):
        return '{:.14g}'.format(parameter_value)

    def get_plot_prefix(self, solution, parameter_value):
        return (self.filename_prefix +
                '_{}={}'.format(self.parameter_name,
                    self.format_parameter(solution, parameter_value)))

    def get_iv_prefix(self, solution, parameter_value):
        return self.filename_prefix

    def write_output(self, solution, parameter_value):
        if os.path.dirname(self.filename_prefix) != '':
            os.makedirs(os.path.dirname(self.filename_prefix), exist_ok=True)
        
        plot_prefix = self.get_plot_prefix(solution, parameter_value)
        
        meta = {}
        meta['sweep_parameter:{}'.format(self.parameter_name)] = dict(value=parameter_value)
        for extractor in self.meta_extractors:
            extractor(
                solution=solution, parameter_value=parameter_value,
                output_writer=self, meta=meta).call()

        if self.plot_mesh or self.plot_mesh_full:
            for ext in ['.xdmf', '_full.xdmf', '.h5', '_full.h5']:
                try:
                    os.remove(plot_prefix + ext)
                except FileNotFoundError:
                    pass
            if self.plot_mesh:
                with closing(XdmfPlot(plot_prefix + '.xdmf', None)) as mesh_plotter:
                    solution_plot(mesh_plotter, solution, 0, stepper=self.stepper, 
                                  plot_du=self.plot_du, clip_finite_extent=self.clip_finite_extent)

            if self.plot_mesh_full:
                with closing(XdmfPlot(plot_prefix + '_full.xdmf',
                            None, checkpoint=True)) as mesh_plotter:
                    solution_plot(mesh_plotter, solution, 0, stepper=self.stepper, 
                                  plot_du=self.plot_du, clip_finite_extent=self.clip_finite_extent)

        if self.plot_1d:
            with closing(LineCutCsvPlot(
                    plot_prefix + '.csv', None,
                    resolution=self.line_cut_resolution,
            )) as plotter:
                solution_plot(plotter, solution, 0, stepper=self.stepper, 
                              plot_du=self.plot_du, clip_finite_extent=self.clip_finite_extent)
        

        if self.plot_iv:
            meta_writer = self.get_meta_csv_file(
                self.filename_prefix + '_{}.csv'.format(self.parameter_name), meta)
            meta_writer.add_row(meta)
        
        h5yaml.dump(meta, plot_prefix + '.plot_meta.yaml')

        if self.hj_interface_condition:
            if self.hj_condition_filename is None:
                self.hj_condition_filename = self.filename_prefix + "_hj_condition"
            write_heterojunction_boundary_condition(
                solution = solution, 
                hj_position=self.hj_position, 
                filename_prefix = self.hj_condition_filename, 
                parameter_name = self.parameter_name, 
                parameter = parameter_value,
                eps = self.hj_test_eps,
                y_position = self.hj_test_y
            )
            
        if self.vs_write_quantities:
            logging.info("Writing valley scatter quantities to file")
            # Use LineCutCsvPlot to write the data to file
            with closing(LineCutCsvPlot(
                f"{self.filename_prefix}vs_quantities_{self.parameter_name}={parameter_value}.csv", None,
                resolution=self.line_cut_resolution,
            )) as plotter:
                vs_plot(plotter, solution, 0, stepper=self.stepper)
        
    def get_meta_csv_file(self, filename, meta):
        if self._meta_csv is None:
            self._meta_csv = MetaCSVWriter(filename, meta)
        return self._meta_csv

class MetaCSVWriter(SetattrInitMixin):
    def __init__(self, filename, meta):
        columns = []
        for k, d in meta.items():
            u = d.get('units', '')
            typ = d.get('type', 'float')
            columns.append(k)

        self.columns = columns

        # A second stepper stage (e.g. the negative-voltage branch, or a
        # resumed run) writes to the same filename via its own OutputWriter
        # instance. Append to any existing rows instead of truncating them,
        # so long as the file looks like it was written with this same
        # column layout.
        #
        # The order of `meta` is not stable across processes (it follows the
        # order in which the extractors' regions were registered, which has
        # come from sets), so an appended row must be written in the order
        # of the *existing* header, not in the new process's order.
        # Otherwise a resumed run silently writes, e.g., the left-contact
        # current under the right-contact column.
        append = os.path.exists(filename) and os.path.getsize(filename) > 0
        if append:
            with open(filename, 'rt', newline='') as fh:
                existing = next(csv.reader(fh), None)
            if existing and set(existing) == set(columns):
                self.columns = columns = list(existing)
            else:
                logging.getLogger('output_writer').warning(
                    "Appending to %s whose columns differ from this run's "
                    "(%d existing vs %d new); rows are written in this run's "
                    "column order and may not line up with the header.",
                    filename, len(existing or ()), len(columns))
        self.file = open(filename, 'at' if append else 'wt')
        self.writer = csv.writer(self.file)

        if not append:
            self.writer.writerow(columns)

            # add a '#' at the beginning of the units line so it can be
            # treated as a comment by numpy.genfromtxt(), pandas.read_csv() etc.
            units = [meta[c].get('units', '') for c in columns]
            units[0] = '# ' + units[0]
            self.writer.writerow(units)

    def add_row(self, meta):
        self.writer.writerow([meta[c]['value'] for c in self.columns])
        self.file.flush()


def _ensure_dict(d, k):
    v = d.get(k, None)
    if v is None:
        v = d[k] = {}
    return v


class MetaExtractorBandInfo(SetattrInitMixin):
    prefix = 'band_info:'

    def call(self):
        for k, band in self.solution.pdd.bands.items():
            pre = "{}{}:".format(self.prefix, band.name)
            self.meta[pre+"sign"] = dict(
                value=band.sign)


class MetaExtractorIntegrals(SetattrInitMixin):
    '''
Attributes
----------
prefix: str
    String to add to quantites.
facets: dict
    Facet regions where to extract quantities.
cells: dict
    Cell regions where to extract quantities.
solution:
    Solution object.
parameter_value:
    Parameter value.
meta:
    Metadata dictionary to write to.
'''
    prefix = ''

    def call(self):
        mu = self.pdd.mesh_util
        for k, b in self.pdd.bands.items():
            self.add_surface_flux(
                'avg:current_{}'.format(k), b.j, average=True,
                units='mA/cm^2')
            self.add_surface_flux(
                'tot:current_{}'.format(k), b.j, average=False,
                units='mA')
            self.add_volume_total(
                'avg:g_{}'.format(k), b.g, average=True,
                units='cm^-3/s')
            self.add_volume_total(
                'tot:g_{}'.format(k), b.g, average=False,
                units='1/s')
            for procname, proc in self.pdd.electro_optical_processes.items():
                self.add_volume_total('tot:g_{}_{}'.format(procname, k),
                    proc.get_generation(b), average=False, units='1/s' )

        for fieldname, field in self.solution.optical.fields.items():
            self.add_volume_total('tot:gother_{}'.format(fieldname),
                field.g, average=False, units='1/s')
            self.add_volume_total('tot:gabs_{}'.format(fieldname),
                field.alpha*field.Phi, average=False, units='1/s')

    @cached_property
    def pdd(self):
        return self.solution.pdd

    def add_quantity(self, name, location_name, value, units):
        self.meta[''.join((
            self.prefix, name, ':', location_name))] = dict(
                value=value, units=units)

    def add_surface_total(
            self, k, expr_ds, expr_dS, internal=True, external=True,
            average=False, units=None):
        mu = self.pdd.mesh_util
        for reg_name, reg in self.facets.items():
            ds, dS = mu.region_ds_and_dS(reg)
            dsS = ds + dS
            value = mu.assemble(ds*expr_ds + dS*expr_dS)
            if average:
                area = mu.assemble(dsS.abs()*mu.Constant(1.0))
                if area == 0:
                    value = (value * np.nan) / area.units
                    warnings.warn(ZeroAreaWarning(
                        "facet region {!r} has zero area".format(reg_name)))
                else:
                    value = value / area
            self.add_quantity(k, reg_name, value.m_as(units), units)

    def add_surface_flux(self, k, expr, **kwargs):
        mu = self.pdd.mesh_util
        self.add_surface_total(
            k,
            expr_ds=mu.dot(expr, mu.n),
            expr_dS=mu.dot(mu.avg(expr), mu.pside(mu.n)), **kwargs)

    def add_volume_total(self, k, expr, average=False, units=None):
        mu = self.pdd.mesh_util
        for reg_name, cregion in self.cells.items():
            dx = mu.region_dx(cregion)
            value = mu.assemble(dx*expr)
            if average:
                value = value / mu.assemble(dx*mu.Constant(1.0))
            self.add_quantity(k, reg_name, value.m_as(units), units)

def solution_plot(plotter, s, timestep, solver=None, stepper=None, plot_du=True, clip_finite_extent=True):
    plotter.new(timestep)
    pdd = s.pdd
    mesh = pdd.mesh_util.mesh
    mesh_data = pdd.mesh_data
    po = pdd.poisson
    ur = s.unit_registry
    mu = pdd.mesh_util

    CG1 = mu.space.CG1
    DG0 = mu.space.DG0
    DG1 = mu.space.DG1
    DG2 = mu.space.DG2
    VCG1 = mu.space.vCG1
    #Add name attributes for metadata
    CG1.name = "CG1"
    DG0.name = "DG0"
    DG1.name = "DG1"
    DG2.name = "DG2"
    VCG1.name = "vCG1"
    
    add = plotter.add

    Vunit = ur.V
    Eunit = ur.V/ur.mesh_unit

    eV = ur.eV
    conc = 1/ur.cm**3
    econc = ur.elementary_charge*conc
    junit = ur.mA/ur.cm**2
    fluxunit = 1/ur.cm**2/ur.s
    alphaunit = 1/ur.cm
    gunit = conc/ur.s

    add('mesh',None,mesh)
    add('E', Eunit, po.E, VCG1)
    add('phi', Vunit, po.phi, DG1)
    add('thmeq_phi', ur.V, po.thermal_equilibrium_phi, DG1)
    add('rho', econc, po.rho, DG1)
    add('static_rho', econc, po.static_rho, DG1)
    if hasattr(pdd, 'XMoleFraction'):
        add('XMole', ur.dimensionless, pdd.XMoleFraction, DG1)

    jays = []
    for k, band in pdd.bands.items():
        if clip_finite_extent:
            extent = band.extent
        else:
            extent = 1.0

        add('u_'+k, conc, band.u * extent, DG1)
        add('thmeq_u_'+k, conc, band.thermal_equilibrium_u * extent, DG1)
        add('qfl_'+k, eV, band.qfl * extent, DG2)
        add('g_'+k, gunit, band.g * extent, DG1)
        for procname, proc in pdd.electro_optical_processes.items():
            add('g_{}_{}'.format(procname, k), gunit,
                proc.get_generation(band), DG1)

        add('j_'+k, junit, band.j, VCG1)
        jays.append(band.j)
        add('mobility_'+k, ur('cm^2/V/s'), band.mobility * extent, DG1)

        if hasattr(band, 'energy_level'):
            E = band.energy_level
            ephi = po.phi*ur.elementary_charge
            add('E_'   +k, eV, E * extent, DG1)
            add('Ephi_'+k, eV, E - ephi * extent, DG1)
            del E, ephi
        if hasattr(band, 'mixedqfl_base_w'):
            add('w_{}_base'.format(k), eV, band.mixedqfl_base_w * extent, DG2)
            add('w_{}_delta'.format(k), eV, band.mixedqfl_delta_w * extent, DG2)
        if hasattr(band, 'number_of_states'):
            add('number_of_states_'+k, conc, band.number_of_states, DG2)
        if hasattr(band, 'effective_density_of_states'):
            add('effective_density_of_states_'+k, conc, band.effective_density_of_states, DG2)

    add('j_tot', junit, reduce(OP.add, jays), VCG1)

    omu = s.optical.mesh_util
    oDG1 = omu.space.DG1

    for k, o in s.optical.fields.items():
        add('opt_Phi_'+k, fluxunit, o.Phi, oDG1)
        add('opt_gother_'+k, gunit, o.g, oDG1)
        add('opt_alpha_'+k, alphaunit, o.alpha, oDG1)
        add('opt_gabs_'+k, gunit, o.alpha*o.Phi, oDG1)

    if plot_du and stepper is not None:
        split_du = pdd.mixed_function_helper.solution_mixed_space.split(stepper.du)

        # These units should match the trial units in poisson_drift_diffusion1.py1
        add('du_E', Eunit, split_du['poisson/E'], VCG1)
        add('du_phi', Vunit, split_du['poisson/phi'], DG1)

        for k, b in pdd.bands.items():
            add(f'du_{k}_delta_w', eV, split_du[f'{k}/delta_w'], DG1 )
            add(f'du_{k}_j', ur.A / ur.mesh_unit**2, split_du[f'{k}/j'], VCG1)

    #write metadata file
    if hasattr(plotter,"metadata_filename"):
        h5yaml.dump(plotter.funcs_plotted,plotter.metadata_filename)

def write_heterojunction_boundary_condition(
    solution, hj_position, filename_prefix, parameter_name, parameter, y_position, eps
):
    """ Write boundary condition for heterojunction interfaces, defined by hj_position"""

    pdd = solution.pdd
    mu = pdd.mesh_util

    from simudo.fem import expr

    def probe_interface_values(property_name, unit, interp, x, eps):
        probe = mu.get_debug_probe(property_name, interp)
        mesh_bbox = expr.mesh_bbox(mu.mesh)
        if len(mesh_bbox) == 1:
            y = 0.0  # true-1D mesh; DebugProbe ignores trailing coordinates
        else:
            y = (mesh_bbox[1][1] - mesh_bbox[1][0]) * y_position + mesh_bbox[1][0]
        # y = y_position
        left = probe(x - eps, y).m_as(unit)
        right = probe(x + eps, y).m_as(unit)
        return {
            "unit": unit,
            "left_value": left,
            "right_value": right,
        }
    
    def probe_interface_facet_value(probe, unit):
        if hasattr(probe(), "units"):
            value = probe().m_as(unit)
        else:
            value = probe()
            unit = "dimensionless"
        return {
            "unit": unit,
            "facet_value": value
        }
        

    out = {}
    for i, x in enumerate(hj_position):
        out[i] = {
            "interface_position": x,
        }
        for k, band in pdd.bands.items():
            
            if k == "CB" or k == "VB":
                out[i][f"current_{k}"] = probe_interface_values(band.j, "mA/cm^2", "vCG1", x, eps)
                out[i][f"u_{k}"] = probe_interface_values(band.u, "cm^-3", "DG1", x, eps)
                out[i][f"qfl_{k}"] = probe_interface_values(band.qfl, "eV", "DG2", x, eps)
                # out[i][f"w_{k}_base"] = probe_interface_values(band.mixedqfl_base_w, "eV", "DG2", x, eps)
                # out[i][f"w_{k}_delta"] = probe_interface_values(band.mixedqfl_delta_w, "eV", "DG2", x, eps)

                E = band.energy_level
                ephi = pdd.poisson.phi * solution.unit_registry.elementary_charge
                out[i][f"Ephi_{k}"] = probe_interface_values(E - ephi, "eV", "DG1", x, eps)
                out[i][f"E_{k}"] = probe_interface_values(E, "eV", "DG1", x, eps)
                out[i][f"phiqfl_{k}"] = probe_interface_values(band.phiqfl, "eV", "DG1", x, eps)

                out[i][f"effective_density_of_states_{k}"] = probe_interface_values(
                    band.effective_density_of_states, "cm^-3", "DG2", x, eps)

                out[i][f"kT_{k}"] = probe_interface_values(pdd.kT, "eV", "DG1", x, eps)

                vth = pdd.spatial.get(k + "/" + "vth")
                out[i][f"vth_{k}"] = probe_interface_values(vth, "cm/s", "DG1", x, eps)

                # # probe current at exactly the interface
                out[i][f"current_{k}_interface"] = probe_interface_values(band.j, "mA/cm^2", "vCG1", x, 0)

                # probe Delta_w_BC at facet
                # probe_Delta_w_BC_facet = getattr(band, f"Delta_w_BC")
                # print(probe_Delta_w_BC_facet)
                probe_Delta_w_BC_facet = getattr(band, "Delta_w_BC")[i]
                out[i][f"Delta_w_BC_{k}_interface"] = probe_interface_facet_value(probe_Delta_w_BC_facet, "eV")
                
                probe_Delta_w_BC_degen_facet = getattr(band, "Delta_w_BC_degen")[i]
                out[i][f"Delta_w_BC_degen_{k}_interface"] = probe_interface_facet_value(probe_Delta_w_BC_degen_facet, "eV")
                
                # probe Delta_w at facet
                probe_Delta_w_facet = getattr(band, "Delta_w")[i]
                out[i][f"Delta_w_{k}_interface"] = probe_interface_facet_value(probe_Delta_w_facet, "eV")
                
                # probe current at facet
                probe_j_band_facet = getattr(band, "j_band")[i]
                out[i][f"j_band_{k}_interface"] = probe_interface_facet_value(probe_j_band_facet, "mA/cm^2")

                # probe phiqfl at facet
                probe_phiqfl_facet = getattr(band, "phiqfl_facet")[i]
                out[i][f"phiqfl_{k}_interface"] = probe_interface_facet_value(probe_phiqfl_facet, "eV")
                
                # if use alternative BC, also probe lam1
                if hasattr(band, "lam1"):
                    try:
                        # probe lam at facet
                        probe_lam_facet = getattr(band, "lam1")[i]
                        out[i][f"lam1_{k}_interface"] = probe_interface_facet_value(probe_lam_facet, "dimensionless")

                        # probe lam_BC at facet
                        probe_lam_BC_facet = getattr(band, "lam1_BC")[i]
                        out[i][f"lam1_BC_{k}_interface"] = probe_interface_facet_value(probe_lam_BC_facet, "dimensionless")
                    except IndexError:
                        out[i][f"lam1_{k}_interface"] = {
                            "unit": None,
                            "facet_value": None
                        }
                        out[i][f"lam1_BC_{k}_interface"] = {
                            "unit": None,
                            "facet_value": None
                        }
                    
                    # # probe lam at facet
                    # probe_lam_facet = getattr(band, "lam2")[i]
                    # out[i][f"lam2_{k}_interface"] = probe_interface_facet_value(probe_lam_facet, "eV")

                    # # probe lam_BC at facet
                    # probe_lam_BC_facet = getattr(band, "lam2_BC")[i]
                    # out[i][f"lam2_BC_{k}_interface"] = probe_interface_facet_value(probe_lam_BC_facet, "eV")
                    

    yaml_filename = os.path.splitext(filename_prefix)[0] \
        + f"_{parameter_name}={parameter:.14g}" \
        + '.yaml'

    h5yaml.dump(out, yaml_filename)
    

def vs_plot(plotter, s, timestep, stepper):
    # add the vs quantities to plotter for later plotting
    plotter.new(timestep)
    pdd = s.pdd
    mesh = pdd.mesh_util.mesh
    mesh_data = pdd.mesh_data
    po = pdd.poisson
    ur = s.unit_registry
    mu = pdd.mesh_util
    
    cm = ur.cm
    second = ur.second
    kV = ur.kV
    
    DG1 = mu.space.DG1
    
    add = plotter.add
    
    add("mesh", None, mesh)
    
    # quantities = [
    #     ["vs_pervolume_rate_from", cm**-3/second],
    #     ["vs_rate_from", 1/second],
    #     ["vs_quasi_constant_rate_from", 1/second],
    #     ["vs_quasi_E", kV/cm],
    # ]
    
    for k, band in pdd.bands.items():
        if k == "CB" or k == "CBG":
            add(f"{k}/vs_pervolume_rate_from", cm**-3/second, getattr(band, "vs_pervolume_rate_from"), DG1)
            add(f"{k}/vs_rate_from", 1/second, getattr(band, "vs_rate_from"), DG1)
            add(f"{k}/vs_quasi_constant_rate_from", 1/second, getattr(band, "vs_quasi_constant_rate_from"), DG1)
            add(f"{k}/vs_quasi_E", kV/cm, getattr(band, "vs_quasi_E"), DG1)
            add(f"{k}/vs_net_pervolume_to", cm**-3/second, getattr(band, "vs_net_pervolume_to"), DG1)
    
    
            
        
            
        