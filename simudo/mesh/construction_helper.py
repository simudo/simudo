from __future__ import absolute_import, division, print_function

import tempfile
from builtins import bytes, dict, int, range, str, super
from collections import defaultdict
from functools import partial
from os import path as osp
from pathlib import Path
import warnings

import numpy as np
from cached_property import cached_property

import dolfin

from . import facet, refine
from ..fem.mesh_data import MeshData
from ..io.h5yaml import XLoader
from ..util import DictAttrProxy
from .facet import (
    FacetsManager, facet2d_angle, mark_boundary_facets, mesh_function_map)
from .interval1dtag import (
    CInterval, GeometricallyExpandingMeshInterval, Interval,
    Interval1DTag, clip_intervals)
from .mesh_entity_predicate import (
    DimensionAdapterPredicate, SubdomainCellPredicate)
from .product2d import Product2DMesh
from .pygmsh import PygmshMakeRegions
from .topology import CellRegion


def inplace_dict_map(mapping, func):
    for k, v in mapping.items():
        mapping[k] = func(v)

class InvertibleTransform(object):
    '''FIXME: make PlazaRefinementND use alternate definition of longest_edge

    This hack is instead used to temporarily modify the mesh
    coordinates temporarily to alter PlazaRefinementND's length metric

    v[i, j] is vertex i's jth coordinate'''

    def transform_coordinates(self, v):
        return v

    def untransform_coordinates(self, v):
        '''v[i, j] is vertex i's jth coordinate'''
        return v

    def transform(self, mesh):
        X = mesh.coordinates()
        X[:, :] = self.transform_coordinates(X)
        self.after_modification(mesh)

    def untransform(self, mesh):
        X = mesh.coordinates()
        X[:, :] = self.untransform_coordinates(X)
        self.after_modification(mesh)

    def after_modification(self, mesh):
        mesh.bounding_box_tree().build(mesh)

class LinearTransform(InvertibleTransform):
    def __init__(self, matrix):
        matrix = np.array(matrix)
        shape = matrix.shape
        if len(shape) == 1:
            matrix = np.diag(matrix)
        self.matrix = matrix
        self.matrix_inv = np.linalg.inv(matrix)

    def transform_coordinates(self, v):
        return self.matrix.dot(v.T).T

    def untransform_coordinates(self, v):
        '''v[i, j] is vertex i's jth coordinate'''
        return self.matrix_inv.dot(v.T).T

class BaseConstructionHelper(object):
    '''boilerplate'''

    def main(self, params, output_filename):
        dolfin.parameters["refinement_algorithm"] = 'plaza_with_parent_facets'
        self.params = params
        self.run()
        self.save(output_filename)

    @cached_property
    def p(self):
        return DictAttrProxy(self.params)

    @cached_property
    def robjs(self):
        return {}

    @cached_property
    def mesh_data(self):
        return MeshData(
            mesh=self.mesh,
            cell_function=self.cf,
            facet_function=self.ff,
            region_name_to_cvs=dict(self.cell_regions),
            facets_name_to_fvs=dict(self.facet_regions),
            material_to_region_map=self.material_to_region,
            facets_manager=self.facets,
            mesh_unit=self.params.get('mesh_unit', None))

    def run(self):
        self.generate_mesh()
        self.compute_facets()
        self.user_extra_definitions()
        self.util_define_internal()
        self.user_refinement()
        self.fix_facets_after_subdivision()

    def user_extra_definitions(self):
        '''override me'''

    def util_define_internal(self):
        fc = self.facets
        frs = self.facet_regions
        for k, cvs in self.cell_regions.items():
             frs['_internal_'+k] = fc.internal(cvs)

    @cached_property
    def dim(self):
        return self.mesh.topology().dim()

    @cached_property
    def gdim(self):
        return self.mesh.geometry().dim()

    def util_init_cf_from_domains(self):
        self.cf = dolfin.MeshFunction(
            "size_t", self.mesh, self.dim, self.mesh.domains())

    def allocate_subdomain_range(self, length):
        # TODO: make more efficient, currently O(num_cell_values)
        start = int(max(self.used_cell_values)) + 1
        self.used_cell_values.update(range(start, start+length))
        return start

    def user_mark_external_boundary_facets(
            self, mesh, boundary_facet_function):
        '''override me (maybe)

        this function is allowed to add new definitions and cell
        values to `self.cell_regions`

        by default this uses `facet.facet2d_angle` and defines new
        regions right/top/left/bottom

        this function MUST add an 'exterior' key for all external
        boundary facets
        '''

        cell_regions = self.cell_regions
        exterior = cell_regions['exterior'] = set()

        # TODO: abstract this further
        offset = self.allocate_subdomain_range(4)
        mark_boundary_facets(mesh, boundary_facet_function,
                             facet2d_angle, offset)
        for k, v in dict(right=0, top=1, left=2, bottom=3).items():
            cv = v+offset
            cell_regions['exterior_'+k] = set((cv,))
            exterior.add(cv)

    def compute_facets(self):
        mesh = self.mesh
        bff = dolfin.MeshFunction("size_t", mesh, self.dim-1, 1000)
        self.user_mark_external_boundary_facets(mesh, bff)
        self.facets = fc = FacetsManager(mesh, self.cf, bff)
        self.ff = fc.facet_mesh_function
        self.facet_regions = {}

    def refine_subdomains(self, subdomains, predicate):
        robjs = self.robjs

        def make_predicate():
            pred = (DimensionAdapterPredicate(
                SubdomainCellPredicate(robjs['cf'], subdomains),
                predicate.dim) & predicate)
            pred.prepare(robjs)
            return pred

        refine.refine_forever(robjs, make_predicate)

    def fix_facets_after_subdivision(self):
        self.facets.fix_undefined_facets_after_subdivision(
            self.mesh, self.cf, self.ff)

    def user_refinement(self):
        ''' override me '''

    def user_modify_meta(self, meta):
        ''' override me '''

    def debug_fvs_to_mf(self, fvs, signed=True):
        # TODO: probably move this to util
        fvs = set(fvs)
        s = -1 if signed else 1
        return mesh_function_map(
            self.ff, lambda fv: ((fv, 1) in fvs) + s*((fv, -1) in fvs),
            out_type='int')

    def debug_cvs_to_mf(self, cvs):
        # TODO: probably move this to util
        cvs = set(cvs)
        return mesh_function_map(
            self.cf, lambda cv: cv in cvs,
            out_type='size_t')

    def debug_plot(self):
        ro = self.robjs

        for k, v in self.facet_regions.items():
            dolfin.plot(self.debug_fvs_to_mf(v, signed=False), title=k)

        dolfin.plot(ro['cf'], title='cf')

        dolfin.interactive()

    @classmethod
    def _make_property_shortcuts(cls):
        def getter(attr, self): return self.robjs[attr]
        def setter(attr, self, value): self.robjs[attr] = value
        for attr in ['mesh', 'cf', 'ff']:
            setattr(cls, attr, property(partial(getter, attr),
                                        partial(setter, attr)))

    @classmethod
    def from_existing_mesh_cf(cls, mesh, cf, cell_regions,
                              run=True, params=None,
                              mesh_unit=None):
        self = cls()
        self.mesh = mesh
        self.cf = cf
        self.cell_regions = cell_regions
        self.params = {} if params is None else params
        if mesh_unit is not None:
            self.params['mesh_unit'] = mesh_unit
        if run:
            self.run()
        return self

    def generate_mesh(self):
        ''' by default does nothing. override me '''

    @cached_property
    def used_cell_values(self):
        return set(self.cf.array())

class ConstructionHelperMshr(BaseConstructionHelper):
    def user_define_mshr_regions(self):
        '''override me

        must return `{region_name: mshr_domain}`
        `regions['domain']` is overall domain'''
        raise NotImplementedError()

    def generate_mesh(self):
        mshr_regions = self.user_define_mshr_regions()
        domain = mshr_regions['domain']

        from .domaintag_mshr import MshrDomainTag
        dt = MshrDomainTag()

        self.cell_regions = cell_regions = dt.mshr_make_subdomains(
            domain, mshr_regions)

        for k, v in cell_regions.items():
            cell_regions[k] = set(v)

        self.mesh = mshr.generate_mesh(domain, 1)
        self.util_init_cf_from_domains()

        self.used_cell_values = set(
            cv for cvs in self.cell_regions.values() for cv in cvs)
        assert set(self.cf.array()).issubset(self.used_cell_values)


class ConstructionHelperManualCellTagging(BaseConstructionHelper):
    def user_just_generate_mesh(self):
        return self.params['existing_mesh']

    def user_tag_cell(self, cell):
        return self.params['user_cell_tag_function'](cell)

    def user_tag_cells(self, cell_function):
        cf_array = cell_function.array()
        tag_to_cvs = defaultdict(set)
        tagtuples = {}
        tagtuples_first_free_index = 0

        Cell = dolfin.Cell
        user_tag_cell = self.user_tag_cell

        for cell_entity in dolfin.entities(
                cell_function.mesh(), cell_function.dim()):
            cell = Cell(cell_entity.mesh(), cell_entity.index())
            tags = user_tag_cell(cell)
            tags.add('domain')
            tags = tuple(sorted(tags))
            cv = tagtuples.get(tags, None)
            if cv is None:
                tagtuples[tags] = cv = tagtuples_first_free_index
                tagtuples_first_free_index += 1
                for t in tags:
                    tag_to_cvs[t].add(cv)
            cf_array[cell.index()] = cv

        return dict(tag_to_cell_values=tag_to_cvs)

    def generate_mesh(self):
        mesh = self.user_just_generate_mesh()

        cf = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

        self.mesh = mesh
        self.cf = cf

        # initialize facets-cells mapping
        D = self.dim
        mesh.init(D-1, D)

        d = self.user_tag_cells(cf)

        self.cell_regions = d['tag_to_cell_values']


class ConstructionHelperPygmsh2(BaseConstructionHelper):
    """
    Example::

        from simudo.mesh import ConstructionHelperPygmsh2
        from simudo.mesh.gmsh import AnisotropicPygmshGeneratorHelper

        class MyMeshConstructor(ConstructionHelperPygmsh2):
            my_custom_r2_height = 1.5

            class GeneratorHelper(AnisotropicPygmshGeneratorHelper):
                def user_generate(self):
                    geo = self.geometry
                    tt = self.tag_tracker

                    # yuo can reference attributes of the ConstructionHelper
                    constructor = self.user_data
                    r2h = constructor.my_custom_r2_height

                    # construct three overlapping rectangles
                    r0 = tt(geo.add_rectangle([0.0, 0.0, 0.0], 1.0, 1.0))
                    r1 = tt(geo.add_rectangle([0.5, 0.5, 0.0], 1.0, 1.0))
                    r2 = tt(geo.add_rectangle([0.7, 0.2, 0.0], 0.1, r2h))

                    # no need to wait until the end to define physical tags!
                    self.regions["r0"] = r0
                    self.regions["core"] = r0 & r1 & r2

                    # merge the shapes together
                    tt.merge(r0, r1, r2)

                    # finish defining regions
                    self.regions["r1"] = r1
                    self.regions["r2"] = r2

                    # poke a hole in the mesh!
                    tt.remove((r2 & r0) - r1)

            def user_density_function(self, mesh, cell_function, tag_values):
                cf = cell_function
                r = tag_values
                dx = dolfin.dx()

                element = dolfin.VectorElement("DG", mesh.ufl_cell(), 1, 9)
                space = dolfin.FunctionSpace(mesh, element)
                u = dolfin.Function(space)
                v = dolfin.TestFunction(space)
                expr = -dolfin.inner(v, u) * dx

                a = r["r0"] - r["r1"] - r["r2"]
                b = r["r1"] - r["r0"] - r["r2"]
                c = (r["r0"] | r["r1"] | r["r2"]) - a - b

                A = 2000
                B = 20
                for cvs, vec in [
                    [a, [A, 0.0, 0.0, 0.0, B, 0.0, 0.0, 0.0, 1.0]],
                    [b, [B, 0.0, 0.0, 0.0, A, 0.0, 0.0, 0.0, 1.0]],
                    [c, [B,   A, 0.0,   A, B, 0.0, 0.0, 0.0, 1.0]],
                ]:
                    dx1 = dolfin.dx(subdomain_data=cf, subdomain_id=tuple(cvs))
                    expr += dolfin.inner(v, dolfin.Constant(vec)) * dx1

                solver = NewtonSolver(F=expr, u_=u, bcs=[])
                solver.solve()
                return u
    """

    GeneratorHelper: type = ...
    xdmf_output: str = None #set to save output mesh to location xdmf_output. 

    def user_density_function(self, mesh, cell_function, tag_values):
        """override me

        Parameters
        ----------
        mesh: FEM mesh
            Initial generated mesh.
        cell_function:
            Cell function with region information.
        tag_values:
            Dictionary mapping region names to sets of cell values.

        Returns
        -------
        function:
            Vector 9-component FEM function representing a 3x3 anisotropic
            metric matrix.
        """
        raise NotImplementedError

    def _generator(self):
        return self.GeneratorHelper.prepare(dim=self.dim, user_data=self)

    def generate_mesh(self):
        from .gmsh import generate_anisotropic_mesh

        
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            if self.xdmf_output is None:
                xdmf_output = tmp / "mesh.xdmf"
            else:
                #check that xdmf_output is a Path
                xdmf_output = Path(self.xdmf_output)
                #if xdmf_output is in a directory that doesn't exist, create it
                xdmf_output.parent.mkdir(parents=True, exist_ok=True)

            cvs = generate_anisotropic_mesh(
                xdmf_output=xdmf_output,
                mesh_generator=self._generator(),
                density_function=self.user_density_function,
                refinement_iterations=self.params.get('refinement_iterations', 0),
            )

            with dolfin.XDMFFile(str(xdmf_output)) as file:
                mesh = dolfin.Mesh()
                file.read(mesh)
                cf = dolfin.MeshFunction(
                    "size_t", mesh, mesh.topology().dim(), 0
                )
                file.read(cf)

        self.mesh = mesh
        self.cf = cf

        # initialize facets-cells mapping
        D = self.dim
        mesh.init(D - 1, D)

        # TODO: turn physical objects with dim-1 into facetfunction

        self.cell_regions = cvs

class ConstructionHelperIntervalProduct2DMesh(BaseConstructionHelper):
    Interval1DTag = Interval1DTag
    product2d_Ys = (0.0, 1.0)

    def user_define_interval_regions(self):
        '''override me

        must return `((domain_x0, domain_x1), list_of_intervals)`
        `domain` is overall domain'''

        # sensible default implementation
        return (self.params.domain, self.params.intervals)

    @cached_property
    def _property_user_define_interval_regions(self):
        (dx0, dx1), intervals = self.user_define_interval_regions()

        clip_intervals(dx0, dx1, intervals)

        return intervals

    @cached_property
    def interval_1d_tag(self):
        return self.Interval1DTag(
            self._property_user_define_interval_regions)

    def generate_mesh(self):
        o = self.interval_1d_tag
        o.product2d_Ys = self.product2d_Ys
        pm = o.product2d_mesh
        self.mesh = mesh = pm.mesh
        self.cf = pm.cell_function

        self.cell_regions = o.subdomains['tag_to_cell_values']
        self.used_cell_values = set(o.subdomains['used_cell_values'])

class ConstructionHelperLayeredStructure(
        ConstructionHelperIntervalProduct2DMesh):
    '''Mesh-construction helper for 1-D layered device structures (optionally
extruded to 2-D).

Layers are specified as a list of dicts, each containing at minimum
``'name'`` (str), ``'material'`` (str), and ``'thickness'`` (float, in
mesh units).  Pass the list as ``params['layers']``.  Additional mesh
refinement regions can be supplied as ``params['extra_regions']``.

See :py:class:`ConstructionHelperIntervalProduct2DMesh` for the underlying
mesh-generation machinery, and
:py:class:`~simudo.physics.problem_data.ProblemData` for the solver object
that consumes the resulting mesh.

Attributes
----------
material_to_region: dict
    Mapping from material name (str) to the union
    :py:class:`~simudo.mesh.topology.CellRegion` covering all layers with
    that material.
layers: list of dict
    The layer definitions as supplied in ``params['layers']``.
extra_regions: list
    Extra refinement regions, from ``params['extra_regions']``
    (formerly ``params['simple_overmesh_regions']``).
'''
    @cached_property
    def layers(self):
        return self.p.layers

    @cached_property
    def extra_regions(self):
        if 'extra_regions' in self.params:
            return self.p.extra_regions
        else:
            warnings.warn(
                "`simple_overmesh_regions` has been renamed to `extra_regions`, and "
                "will be removed eventually. Modify your code accordingly.",
                DeprecationWarning,
            )
            return self.p.simple_overmesh_regions

    @cached_property
    def material_to_region(self):
        d = {}
        for layer in self.layers:
            material = layer['material']
            region = CellRegion(layer['name'])
            if material not in d:
                d[material] = region
            else:
                d[material] = d[material] | region
        return d

    def user_layer_extra_intervals(
            self, intervals, name_interval):
        ''' override if necessary '''

    def user_define_interval_regions(self):
        intervals = []

        intervals.append(CInterval(
            -np.inf, np.inf,
            edge_length=self.p.edge_length))

        def mkinterval(meshing):
            meshing_type = meshing.pop("type", None)
            kw = {k: meshing[k] for k in ["x0", "x1", "tags"]}
            if meshing_type == "geometric":
                interval = GeometricallyExpandingMeshInterval(
                    edge_length_start=meshing["start"],
                    edge_length_expansion_factor=meshing["factor"],
                    **kw,
                )
            elif meshing_type == "constant":
                interval = CInterval(
                    edge_length=meshing["edge_length"],
                    **kw,
                )
            elif meshing_type == None:
                interval = Interval(**meshing)
            else:
                raise ValueError("bad meshing strategy {!r}".format(meshing_type))
            return interval

        name_interval = {}

        x00 = 0
        x0 = x1 = x00
        for layer in self.layers:
            w = layer['thickness']
            x0, x1 = x1, x1+w
            name = layer['name']
            material = layer['material']
            if 'mesh' not in layer:
                meshing = {}
                if 'edge_length' in layer:
                    warnings.warn(
                        "`edge_length` should be in a sub-dictionary under the layer "
                        "dictionary key `mesh`. See `example/pn_diode/pn_diode.py`.",
                        DeprecationWarning,
                    )
                    meshing['edge_length'] = layer['edge_length']
                    meshing['type'] = 'constant'
                else:
                    meshing['type'] = None
            else:
                meshing = layer['mesh'].copy()
            meshing["x0"] = x0
            meshing["x1"] = x1
            meshing["tags"] = (name, material)
            interval = mkinterval(meshing)
            intervals.append(interval)
            name_interval[name] = interval
        domain_extent = (x00, x1)

        def locate_extra_region_position(label):
            name, offset = label
            c = name[0]
            if   c == '-':
                return name_interval[name[1:]].x0 + offset
            elif c == '+':
                return name_interval[name[1:]].x1 + offset
            else:
                raise ValueError("must start with +/-")

        for meshing in self.extra_regions:
            meshing = meshing.copy()
            for k in ("x0", "x1"):
                meshing[k] = locate_extra_region_position(meshing[k])
            if "tags" not in meshing:
                meshing["tags"] = ()
            if "edge_length" in meshing and "type" not in meshing:
                warnings.warn(
                    "You must add explicit type=\"constant\" to `extra_regions` entry.",
                    DeprecationWarning,
                )
                meshing['type'] = "constant"
            # TODO: extend to support sreg['callback'] that generates
            # some number of intervals
            intervals.append(mkinterval(meshing))

        self.user_layer_extra_intervals(intervals, name_interval)

        intervals.append(mkinterval(dict(
            x0=domain_extent[0],
            x1=domain_extent[1],
            tags=('domain',),
            type=None,
        )))
        return (domain_extent, intervals)
