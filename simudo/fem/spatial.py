
from collections import defaultdict
import warnings

import numpy as np
from cached_property import cached_property
from sortedcontainers import SortedList

import dolfin

from ..util import SetattrInitMixin
from .delayed_form import DelayedForm
from .expr import constantify_if_literal


class IdentityEq(object):
    def __init__(self, object):
        self.object = object

    def __hash__(self):
        return hash(id(self.object))

    def __eq__(self, other):
        return isinstance(other, IdentityEq) and self.object is other.object

class BoundaryCondition(SetattrInitMixin):
    '''Boundary condition class, used both for essential and natural BCs.

Parameters
----------
priority:
    Priority of this boundary condition. Lower values have higher
    priority.
region: :py:class:`.topology.FacetRegion`
    Facet region where this BC applies.
value: :py:class:`ufl.core.expr.Expr`
    Value of the boundary condition.
require_natural: bool, optional
    Does this BC require natural (as opposed to inflexible essential)
    application? By default false.
'''
    require_natural = False

class ValueRule(SetattrInitMixin):
    '''Rule for spatially-dependent value.

Parameters
----------
priority:
    Priority of this rule. Lower values have higher priority.
region: :py:class:`.topology.CellRegion`
    Cell region where this value applies.
value: :py:class:`ufl.core.expr.Expr`
    Value.
'''

class CombinedBoundaryCondition(SetattrInitMixin):
    '''Combination of multiple :py:class:`BoundaryCondition`.

Parameters
----------
bcs: SortedList
    Boundary conditions to combine.
value_extractor: callable, optional
    Applied to :py:class:`BoundaryCondition` objects to obtain the
    boundary condition value. See :py:attr:`map`.
mesh_util: :py:class:`.mesh_util.MeshUtil`
    Used to evaluate abstract topology into actual facet values, and
    other things.
essential_var: :py:class:`dolfin.Function`, optional
    Variable to apply BC on (used for essential BCs).
natural_ignore_sign: bool, optional
    Whether to ignore the facet sign when building a natural BC.

Attributes
----------
map: dict
    Mapping :code:`{(facet_value, sign): value_extractor(bc)}`.
reverse_map: dict
    Mapping :code:`{bc_value: {(facet_value, sign), ...}}`.
fv_set: set
    Set of referenced facet values. Useful to check that BCs don't
    overlap.
'''
    value_extractor = None

    @property
    def mesh_data(self):
        return self.mesh_util.mesh_data

    @cached_property
    def map(self):
        bcs = self.bcs
        func = self.value_extractor
        if func is None:
            func = lambda x: x.value

        mapping = {}
        for bc in bcs:
            region = bc.region
            fvss = self.mesh_data.evaluate_topology(region)
            for fvs in fvss:
                flipped = (fvs[0], -fvs[1])
                if flipped not in mapping:
                    mapping.setdefault(fvs, func(bc))
        return mapping

    @cached_property
    def reverse_map(self):
        r = defaultdict(set)
        for k, v in self.map.items():
            r[IdentityEq(v)].add(k)
        return r

    @cached_property
    def fv_set(self):
        return set(x[0] for x in self.map.keys())

    @cached_property
    def offset_partitioning_bc_cells(self):
        cells = self.fv_adjacent_cells
        cellavg = self.fv_adjacent_cellavg.magnitude.vector()[cells]
        return (cells, cellavg)

    @cached_property
    def fv_adjacent_cells(self):
        '''Return list of cells that are adjacent to the facet values.'''
        return self.algo_get_fv_adjacent_cells(
            facet_function=self.mesh_data.facet_function, fvs=self.fv_set)

    @cached_property
    def fv_adjacent_cells_function(self):
        '''Returns binary cellwise constant (DG0) function, where cell
has value 1 iff cell is in :py:attr:`fv_adjacent_cells`.'''
        func = dolfin.Function(self.mesh_util.space.DG0)
        func.vector()[:] = 0.0 # TODO: necessary?
        func.vector()[self.fv_adjacent_cells] = 1.0
        return func

    @cached_property
    def fv_adjacent_cellavg(self):
        '''DG0 function representing the average of the boundary
condition value in adjacent facets.'''
        return self.get_fv_adjacent_cellavg()

    def get_fv_adjacent_cellavg(self):
        return self.algo_fvs_to_cell_averages(
            mesh_util=self.mesh_util,
            bcdata=self.reverse_map,
            fv_adjacent_cells_function=self.fv_adjacent_cells_function)

    def algo_get_fv_adjacent_cells(self, facet_function, fvs):
        '''Get list of cells adjacent to facets that are marked with a
facet value in ``fvs`` inside ``facet_function``.'''
        indices = np.nonzero(np.isin(facet_function.array(), list(fvs)))[0]
        mesh = facet_function.mesh()
        D = mesh.topology().dim()
        Facet = dolfin.Facet
        cells = []
        for facet_index in indices:
            facet = Facet(mesh, facet_index)
            for cell in facet.entities(D):
                cells.append(cell)
        return list(sorted(frozenset(cells)))

    def algo_fvs_to_cell_averages(
            self, mesh_util, bcdata,
            fv_adjacent_cells_function, target_unit=None):
        '''Algorithm for turning BC data into cell averages.

Parameters
----------
mesh_util: :py:class:`.mesh_util.MeshUtil`
    Mesh utilities and data.
bcdata: dict
    Mapping `{expr: {(facet_value, sign), ...}}`.
target_unit: optional
    Unit to ``expr`` to in the ``bcdict`` argument.
fv_adjacent_cells_function: :py:class:`dolfin.Function`
    DG0 function containing ones and zeros; ones iff the cell is
    adjacent to a BC.
'''
        mu = mesh_util
        dless = mu.unit_registry('dimensionless')
        DG0 = mu.space.DG0
        dx = mu.dx
        u = dolfin.TrialFunction(DG0)
        v = dolfin.TestFunction(DG0)

        form = DelayedForm()
        for expr, fvss in bcdata.items():
            expr = dless * expr.object
            if target_unit is None:
                target_unit = expr.units
            expr_ = expr.m_as(target_unit)
            for fv, sign in fvss:
                # TODO: figure out what to do with the sign - relevant
                # for internal boundary conditions
                form += mu.ds(subdomain_id=fv)*v('+')*(u('+') - expr_)

        if target_unit is None:
            target_unit = dless

        form += dx*u*v*(1.0 - fv_adjacent_cells_function)
        form = form.delete_units().to_ufl().m

        u = dolfin.Function(DG0)
        dolfin.solve(dolfin.lhs(form) == dolfin.rhs(form), u, [])

        return target_unit*u

    def get_natural_bc(self, ignore_sign=None, bc_reverse_map=None):
        mu = self.mesh_util

        if ignore_sign is None:
            ignore_sign = self.natural_ignore_sign

        if bc_reverse_map is None:
            bc_reverse_map = self.reverse_map

        nbc_expr = 0
        nbc_ds = mu.ds # + mu.dS('+') # <--- *
        # *FIXME: should this include internal facets?
        for value, fvss in bc_reverse_map.items():
            value = value.object
            plus_fvs = set()
            minus_fvs = set()
            for fv, sign in fvss:
                if sign > 0:
                    plus_fvs.add(fv)
                else:
                    minus_fvs.add(fv)

            if ignore_sign:
                expr = nbc_ds(tuple(sorted(plus_fvs | minus_fvs)))
            else:
                expr = (nbc_ds(tuple(sorted(plus_fvs))) -
                        nbc_ds(tuple(sorted(minus_fvs))))

            nbc_expr += expr * value

        return nbc_expr

    def get_essential_bc(self, var=None, bc_reverse_map=None):

        if var is None:
            var = self.essential_var

        if bc_reverse_map is None:
            bc_reverse_map = self.reverse_map

        mu = self.mesh_util
        fsr = mu.function_subspace_registry

        ebc_space = fsr.get_function_space(
            var.m, collapsed=False)
        ebc_collapsed_space = fsr.get_function_space(
            var.m, collapsed=True)

        facet_function = mu.mesh_data.facet_function

        var_units = var.units

        essential_bcs = []
        for value, fvss in bc_reverse_map.items():
            value = value.object
            value_ = dolfin.project(value.m_as(var_units),
                                    ebc_collapsed_space)

            # TODO: determine when usage of geometric marking is necessary
            for fv, sign in fvss:
                essential_bcs.append(
                    dolfin.DirichletBC(ebc_space, value_, facet_function, fv))

        return essential_bcs


class Spatial(SetattrInitMixin):
    '''Contains spatially dependent quantities, such as material
parameters, as well as boundary conditions.

Parameters
----------
parent: object
    Attributes such as mesh_data are taken from here by default.
mesh_util: :py:class:`.mesh_util.MeshUtil`, optional
    Mesh-specific utilities. By default taken from parent.
material_to_region_map:
    Material to region map. By default taken from
    :py:attr:`~mesh_data`.

Attributes
----------
value_rules: defaultdict(SortedList)
    Rules for spatially dependent values. Must be dict of list of dicts.
bcs: defaultdict(SortedList)
    Boundary conditions.
'''
    @property
    def mesh_data(self):
        return self.mesh_util.mesh_data

    @property
    def material_to_region_map(self):
        return self.mesh_data.material_to_region_map

    @property
    def mesh_util(self):
        return self.parent.mesh_util

    @cached_property
    def value_rules(self):
        return defaultdict(lambda: SortedList(key=lambda x: x.priority))

    @cached_property
    def bcs(self):
        return defaultdict(lambda: SortedList(key=lambda x: x.priority))

    def get(self, key):
        '''Get spatially dependent quantity named `key`.'''

        mapping = {}
        for rule in self.value_rules[key]:
            region = rule.region
            cvs = self.mesh_data.evaluate_topology(region)
            for cv in cvs:
                mapping.setdefault(cv, rule.value)

        if not len(mapping):
            raise AssertionError(
                "no value rules for key {!r}, cannot deduce shape or units"
                .format(key))

        sample_value = rule.value
        if hasattr(sample_value, 'units'):
            units = sample_value.units
            mapping = {k: v.m_as(units) for k, v in mapping.items()}

        expr = self.mesh_util.cell_function_function.make_cases(
            {k: v for k, v in mapping.items()})

        if hasattr(sample_value, 'units'):
            expr = units * expr

        return expr

    def add_BC(self, key, region, value, priority=0,
               require_natural=False, **kwargs):
        if isinstance(region, str):
            raise AssertionError(
                "`region` argument should not be a string, have you "
                "swapped the first two arguments by mistake?")
        value = constantify_if_literal(value)
        bc = BoundaryCondition(region=region,
                               value=value,
                               priority=priority,
                               require_natural=require_natural, **kwargs)
        self.bcs[key].add(bc)
        return bc

    def add_rule(self, key, region, value, priority=0):
        '''Add a rule for a spatially-dependent value.

Parameters
----------
key: str
    Name of the spatial property, e.g., :code:`"temperature"` or
    :code:`"poisson/static_rho"`.
region: :py:class:`.topology.CellRegion`
    Cell region where this value applies.
value: :py:class:`ufl.core.expr.Expr`
    Value.
priority:
    Priority of this rule. Lower values have higher priority.
'''
        value = constantify_if_literal(value)
        self.value_rules[key].add(ValueRule(
            region=region,
            value=value,
            priority=priority))

    def add_value_rule(self, region, key, value, priority=0):
        '''Deprecated. Use :py:meth:`add_rule`.'''
        warnings.warn(FutureWarning(
            "`Spatial.add_value_rule` will be removed soon. Replace it "
            "with `Spatial.add_rule`. Note the different argument order!"))
        return self.add_rule(
            region=region, key=key, value=value, priority=priority)

    def add_material_data(self, key, material_name, value, priority=0):
        region = self.material_to_region_map.get(material_name, None)
        if region is None:
            return

        self.add_rule(
            key=key,
            region=region,
            value=value,
            priority=priority)

    def make_combined_bc(self, key, value_extractor=None, **kwargs):
        cbc = CombinedBoundaryCondition(
            bcs=self.bcs[key], value_extractor=value_extractor,
            mesh_util=self.mesh_util, **kwargs)
        return cbc

    def get_mixed_bc_pair(
            self,
            essential_key, essential_var,
            natural_key  , natural_var,
            natural_ignore_sign=False,
            essential_extract=None,
            natural_extract=None):
        '''Get mixed BC pair, ensuring that the essential and natural
BCs don't overlap.

Parameters
----------
essential_key: str
    Essential BC subfunction key.
natural_key: str
    Natural BC subfunction key.
natural_ignore_sign: bool, optional
    Whether to ignore facet sign/orientation in the natural
    BC. Default is false.
essential_extract: callable, optional
    Function that will be called to extract a BC value from a boundary
    condition object. Use this to implement any sort of transformation
    you want to apply to the boundary condition value.
natural_extract: callable, optional
    See `essential_extract` above.

Returns
-------
essential_bcs: list
    List of :py:class:`dolfin.DirichletBC`.
natural_bc_expr: UFL expr
    Natural BC expression.
'''
        if essential_key is not None:
            ebc = self.make_combined_bc(
                essential_key, essential_extract,
                essential_var=essential_var)
        else:
            ebc = None

        if natural_key is not None:
            nbc = self.make_combined_bc(
                natural_key, natural_extract,
                natural_ignore_sign=natural_ignore_sign)
        else:
            nbc = None

        if ebc and nbc and (ebc.fv_set & nbc.fv_set):
            raise AssertionError(
                "Essential and natural boundary conditions overlap. "
                "Keys {!r} and {!r}.".format(essential_key, natural_key))

        return (ebc, nbc)

    def get_single_essential_bc(
            self, key, var, extract=None):
        '''Get single essential BC.

Parameters
----------
key: str
    Essential BC key.
var: str
    Essential BC variable.
extract: callable, optional
    See :code:`essential_extract` under :py:meth:`get_mixed_bc_pair`.

Returns
-------
essential_bcs: list
    List of :py:class:`dolfin.DirichletBC`.
'''
        return self.get_mixed_bc_pair(
            essential_key=key,
            essential_var=var,
            natural_key=None,
            natural_var=None,
            essential_extract=extract)[0]

    def get_vanilla_bc_pair(self, variable_key):
        '''not implemented because all our methods are mixed methods'''
        raise NotImplementedError()
