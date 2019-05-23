from __future__ import absolute_import, division, print_function

import tempfile
from builtins import bytes, dict, int, range, str, super
from collections import defaultdict
from functools import partial
from os import path as osp

import numpy as np
from cached_property import cached_property

import dolfin

from . import facet, refine
from ..fem.mesh_data import MeshData
from ..io.h5yaml import XLoader
from ..util import DictAttrProxy
from .facet import (
    FacetsManager, facet2d_angle, mark_boundary_facets, mesh_function_map)
from .interval1dtag import CInterval, Interval, Interval1DTag, clip_intervals
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

class ConstructionHelperPygmsh(BaseConstructionHelper):
    def user_define_pygmsh_geo(self):
        '''override me

Returns
-------
geo:
    `geo_object` that will be passed to :py:func:`pygmsh.generate_mesh`.
'''
        raise NotImplementedError()

    def pygmsh_generate_mesh(self, geo):
        import pygmsh

        p = self.params

        extra_gmsh_arguments = p.get('extra_gmsh_arguments', [])

        mesh_algorithm = p.get('mesh_algorithm', '7')
        if mesh_algorithm is not None:
            extra_gmsh_arguments.extend(
                ('-string', 'Mesh.Algorithm={};'.format(mesh_algorithm)))

        dim = self.params.get('dim', 2)

        generate_mesh_kwargs = p.get(
            'generate_mesh_kwargs', {})

        generate_mesh_kwargs.setdefault('prune_z_0', (dim == 2))
        generate_mesh_kwargs.setdefault('dim', dim)

        points, cells, point_data, cell_data, field_data = pygmsh.generate_mesh(
            geo,
            extra_gmsh_arguments=extra_gmsh_arguments,
            **generate_mesh_kwargs)
        return dict(points=points,
                    cells=cells,
                    point_data=point_data,
                    cell_data=cell_data,
                    field_data=field_data,
                    dim=dim)

    def generate_mesh(self):
        import meshio

        geo = self.user_define_pygmsh_geo()
        r = self.pygmsh_generate_mesh(geo)

        d = PygmshMakeRegions().process(
            cells=r['cells'],
            cell_data=r['cell_data'],
            field_data=r['field_data'],
            dim=r['dim'])

        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_file = osp.join(str(tmpdir), 'mesh.xdmf')
            meshio.write_points_cells(
                mesh_file, r['points'], r['cells'], cell_data=r['cell_data'])

            with dolfin.XDMFFile(mesh_file) as file:
                mesh = dolfin.Mesh()
                file.read(mesh)
                cf = dolfin.MeshFunction('size_t', mesh, r['dim'], 0)
                file.read(cf)

        self.mesh = mesh
        self.cf = cf

        # initialize facets-cells mapping
        D = self.dim
        mesh.init(D-1, D)

        # TODO: turn physical objects with dim-1 into facetfunction

        self.cell_regions = d['tag_to_cell_values']
        self.cell_regions['domain'] = domain_cvs = set()
        for cvs in d['tag_to_cell_values'].values():
            domain_cvs.update(cvs)

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
    '''
Attributes
----------
material_to_region: dict
    Mapping from material to abstract CellRegion.
'''
    @cached_property
    def layers(self):
        return self.p.layers

    @cached_property
    def simple_overmesh_regions(self):
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

        def mkinterval(*args, edge_length=None, **kwargs):
            if edge_length is None:
                return Interval(*args, **kwargs)
            else:
                return CInterval(*args, edge_length=edge_length, **kwargs)

        name_interval = {}

        x00 = 0
        x0 = x1 = x00
        for layer in self.layers:
            w = layer['thickness']
            x0, x1 = x1, x1+w
            name = layer['name']
            material = layer['material']
            edge_length = layer.get('edge_length', None)
            interval = mkinterval(x0, x1, (name, material),
                                  edge_length=edge_length)
            intervals.append(interval)
            name_interval[name] = interval
        domain_extent = (x00, x1)

        def locate_overmesh_label(label):
            name, offset = label
            c = name[0]
            if   c == '-':
                return name_interval[name[1:]].x0 + offset
            elif c == '+':
                return name_interval[name[1:]].x1 + offset
            else:
                raise ValueError("must start with +/-")

        for sreg in self.simple_overmesh_regions:
            x0 = locate_overmesh_label(sreg['x0'])
            x1 = locate_overmesh_label(sreg['x1'])
            # TODO: extend to support sreg['callback'] that generates
            # some number of intervals
            intervals.append(mkinterval(x0, x1,
                                        edge_length=sreg['edge_length']))

        self.user_layer_extra_intervals(intervals, name_interval)

        intervals.append(mkinterval(domain_extent[0],
                                    domain_extent[1], ('domain',)))
        return (domain_extent, intervals)
