
from cached_property import cached_property

import dolfin

from ..util import SetattrInitMixin

__all__ = ['MeshData']

class MeshData(SetattrInitMixin):
    '''Holds mesh and topology data relating to a particular dolfin mesh.

Attributes
----------
mesh: dolfin.Mesh
    The mesh itself.
cell_function: dolfin.CellFunction
    Subdomain cell markers.
facet_function: dolfin.FacetFunction
    Subdomain boundary facet markers.
region_name_to_cvs: dict
    {region_name: set_of_cell_values}
facets_name_to_fvs: dict
    {facets_name: {(facet_value, sign), ...}}
facets_manager: FacetsManager
    Relation information between cell and facet values.
material_to_region_map: dict
    Mapping from material to abstract CellRegion.
mesh_unit: pint.Quantity
    Mesh length unit.
'''
    def evaluate_topology(self, region):
        '''\
Evaluate a topological region.

Parameters
----------
region: :py:class:`.topology.CellRegion` or :py:class:`.topology.FacetRegion`
    Topological region.

Returns
-------
geometrical_values: set
    The return value depends on the type of the :code:`region` argument.

    - If the region is a :py:class:`.topology.CellRegion`, return a
      set of cell values (values in :py:attr:`~cell_function`).

    - If the region is a :py:class:`.topology.FacetRegion`, return a
      set of tuples :code:`(facet_value, sign)` where
      :code:`facet_value` is a facet value (value in
      :py:attr:`~facet_function`), and :code:`sign` is either
      :code:`+1` or :code:`-1` depending on whether the facet
      orientation matches the boundary orientation.
'''
        return region.evaluate(dict(
            region_name_to_cvs=self.region_name_to_cvs,
            facets_name_to_fvs=self.facets_name_to_fvs,
            facets_manager=self.facets_manager))

    @cached_property
    def cell_to_cells_adjacency_via_facet(self):
        mesh = self.mesh
        D = mesh.topology().dim()
        mesh.init(D-1, D)
        mesh.init(D, D)
        cell_neighbors = [[] for i in range(mesh.num_cells())]
        for facet in dolfin.facets(mesh):
            l = facet.entities(D)
            if len(l) == 2:
                a, b = l
                cell_neighbors[a].append(b)
                cell_neighbors[b].append(a)
        return cell_neighbors
