
import re

from cached_property import cached_property

import dolfin

from .. import pyaml
from ..util import SetattrInitMixin
from .delayed_form import DelayedForm
from .expr import CellFunctionFunction
from .function_space_cache import FunctionSpaceCache
from .function_subspace_registry import FunctionSubspaceRegistry

__all__ = [
    'MeshUtil',
]

PY = pyaml.load_res(__name__, 'mesh_util1.py1')

class SpaceShortcut(object):
    '''Convenience class for easily getting elements and function
spaces by name.

For example, ``space.vCG2`` or ``space.BDM2_element``.
'''

    def __init__(self, fsc, mesh, ufl_cell=None):
        '''
Arguments
---------
fsc: FunctionSpaceCache
    Function space cache, used to create the function space.
mesh: dolfin.Mesh
    Mesh on which to create the function space.
'''
        self.fsc = fsc
        self.mesh = mesh
        self.ufl_cell = mesh.ufl_cell() if ufl_cell is None else ufl_cell

    ELEMENT_NAME_RE = re.compile(r"^(v?)(.+?)(\d+)$")
    def get_element(self, fullname):
        m = self.ELEMENT_NAME_RE.fullmatch(fullname)
        if not m: raise ValueError(
                "bad element name {!r}".format(fullname))
        vector = m.group(1)
        name = m.group(2)
        degree = int(m.group(3))
        ufl_cell = self.ufl_cell
        return (dolfin.FiniteElement(name, ufl_cell, degree)
                if not vector else
                dolfin.VectorElement(name, ufl_cell, degree))

    def get_space_by_name(self, fullname):
        element = self.get_element(fullname)
        return self.fsc.FunctionSpace(self.mesh, element)

    def __getattr__(self, attr):
        before, sep, after = attr.partition('_')
        if after == 'element':
            return self.get_element(before)
        elif not sep:
            return self.get_space_by_name(attr)
        else:
            raise ValueError("space {!r}".format(attr))

class MeshUtil(PY.MeshUtilPY, SetattrInitMixin):
    '''Holds a plethora of PDE definition utilities, some of which are (needlessly) mesh-specific.

Notes
-----
Most of the methods and attributes are hidden away in the Pyaml file.
FIXME: Need to fix imports so that sphinx can see the pyaml modules.

Parameters
----------
mesh_data:
    See :py:attr:`mesh_data`.
function_space_cache:
    See :py:attr:`function_space_cache`.
function_subspace_registry:
    See :py:attr:`function_subspace_registry`.
unit_registry: optional
    See :py:attr:`unit_registry`.
\_sloppy:
    See :py:attr:`_sloppy`.

Attributes
----------
mesh_data: :py:class:`~.mesh_data.MeshData`
    Holds mesh and topology information.
function_space_cache: :py:class:`~.fem.function_space_cache.FunctionSpaceCache`
    Used to avoid creating redundant function spaces. Created
    automatically only if :py:attr:`_sloppy` is :code:`True`.
function_subspace_registry: :py:class:`~.fem.function_subspace_registry.FunctionSubspaceRegistry`
    Used for assignment and copying across subfunctions. Created
    automatically only if :py:attr:`_sloppy` is :code:`True`.
unit_registry: :py:class:`pint.UnitRegistry`
    Pint unit registry.
\_sloppy: bool
    Create some of the above attributes automatically. If you want the
    :py:class:`function_subspace_registry` to be shared, do not use
    this option.
'''
    _sloppy = False

    def _check_sloppy(self):
        if not self._sloppy:
            raise ValueError(
                "this attribute is optional only if "
                "`_sloppy=True` is passed to the constructor")

    @cached_property
    def function_space_cache(self):
        ''' create new one if not supplied '''
        self._check_sloppy()
        return FunctionSpaceCache()

    @cached_property
    def function_subspace_registry(self):
        ''' create new one if not supplied '''
        self._check_sloppy()
        return FunctionSubspaceRegistry()

    @cached_property
    def unit_registry(self):
        ''' get unit from :code:`self.mesh_data.mesh_unit` '''
        return self.mesh_data.mesh_unit._REGISTRY

    @cached_property
    def cell_function_function(self):
        return CellFunctionFunction(
            self.mesh_data.cell_function,
            self.space.DG0)

    @property
    def cell_function(self):
        return self.mesh_data.cell_function

    def region_dx(self, region):
        '''Return dx (volume) measure corresponding to topological
:py:class:`~.topology.CellRegion`.

Parameters
----------
region: :py:class:`~.topology.CellRegion`
    Volume region.

Returns
-------
measure: :py:class:`.expr.DelayedForm`
    Volume integration measure.
'''

        # TODO: this could benefit from some caching
        dx = self.dx
        dx = dx(subdomain_id=tuple(sorted(
            self.mesh_data.evaluate_topology(region))))

        return dx

    def region_ds_and_dS(self, region):
        '''Return ds and dS (surface) measures corresponding to
topological :py:class:`~.topology.FacetRegion`, for internal and
external facets respectively.

Parameters
----------
region: :py:class:`~.topology.FacetRegion`
    Facet region.

Returns
-------
measures: tuple of :py:class:`.expr.DelayedForm`
    Facet integration measures :code:`(ds, dS)`.
'''
        # TODO: this could benefit from some caching
        fv_pos = []
        fv_neg = []
        for fv, sign in self.mesh_data.evaluate_topology(region):
            if sign > 0:
                fv_pos.append(fv)
            else:
                fv_neg.append(fv)

        fv_pos = tuple(sorted(fv_pos))
        fv_neg = tuple(sorted(fv_neg))

        ds, dS = (d(subdomain_id=fv_pos) - d(subdomain_id=fv_neg)
                  for d in (self.ds, self.dS))

        return (ds, dS)

    @property
    def facet_function(self):
        return self.mesh_data.facet_function

    @property
    def mesh(self):
        return self.mesh_data.mesh

    @property
    def ufl_cell(self):
        return self.mesh.ufl_cell()

    @property
    def mesh_unit(self):
        # FIXME: assumes mesh_data.mesh_unit's unit registry is
        # self.unit_registry; not a problem if only one unit registry
        # is ever in play
        return self.mesh_data.mesh_unit

    def magnitude(self, x):
        return self.ensure_quantity(x).magnitude

    def units(self, x):
        return self.ensure_quantity(x).units

    def dless(self, x):
        return self.ensure_quantity(x).m_as('dimensionless')

    def ensure_quantity(self, x):
        return (x if hasattr(x, 'magnitude') else
                self.unit_registry.Quantity(x, 'dimensionless'))

    @property
    def space(self):
        return SpaceShortcut(fsc=self.function_space_cache, mesh=self.mesh)

    def element(self, name, *args, **kwargs):
        return dolfin.FiniteElement(name, self.ufl_cell, *args, **kwargs)
