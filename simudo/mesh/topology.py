
from functools import reduce

__all__ = ['CellRegion' , 'FacetRegion',
           'CellRegions', 'FacetRegions']

class XRegionsBase(object):
    def __init__(self, mapping=None):
        if mapping is None:
            mapping = dict()
        super().__setattr__('_mapping_', mapping)

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, name):
        if isinstance(name, (set, frozenset)):
            return {k: self[k] for k in name}

        mapping = self._mapping_
        try:
            return self._mapping_[name]
        except KeyError:
            self._mapping_[name] = value = self._create_object(name)
            return value

    def __setattr__(self, name, value):
        self[name] = value

    def __setitem__(self, name, value):
        self._mapping_[name] = value

    def __repr__(self):
        short_class_name = type(self).__name__.split('.')[-1]
        return '{}({{{}}})'.format(
            short_class_name, ''.join(
                '\n  {!r}: {!r}'.format(k, v)
                for k, v in self._mapping_.items()))

class CellRegions(XRegionsBase):
    '''Convenient container for cell region objects.

Attributes
----------
\_mapping\_: dict
    Dictionary of cell regions.

Methods
-------
\_\_getitem\_\_(name)
    I.e. :code:`obj[name]`. If :code:`name` is already in
    :py:attr:`_mapping_`, it is retrieved from there; otherwise a
    :py:class:`CellRegion` is created with that name, stored inside
    :py:attr:`_mapping_`, and returned.

    If :code:`name` is a :code:`set`, then a subset of the
    :py:attr:`_mapping_` dictionary is returned with those keys.
\_\_getattr\_\_(name)
    I.e. :code:`obj.$name`. Redirected to :code:`obj[name]`.
'''
    def _create_object(self, name):
        return CellRegion(name)

class FacetRegions(XRegionsBase):
    '''Convenient container for facet region objects.

See :py:class:`CellRegions`.
'''
    def _create_object(self, name):
        return FacetRegion(name)

class AbstractCellRegion(object):
    pass

class AbstractFacetRegion(object):
    pass

class AlgebraicMixin(object):
    def evaluate(self, context):
        raise NotImplementedError()

    def __hash__(self):
        return hash(tuple((k, getattr(self, k))
                          for k in self.kwarg_names))

    def __eq__(self, other):
        return type(self) == type(other) and all(
            getattr(self, k) == getattr(other, k)
            for k in self.kwarg_names)

class CellRegion(AlgebraicMixin, AbstractCellRegion):
    '''Abstract cell region.

Can be evaluated down to a set of subdomain markers using
:py:meth:`.MeshData.evaluate_topology`.

To access a cell region predefined (named) in the mesh
generator, instantiate this class directly with a :code:`name`
argument.
'''
    def __new__(cls, *args, **kwargs):
        if cls is CellRegion:
            return CellRegionByName(*args, **kwargs)

        return super().__new__(cls)

    def __and__(self, x):
        '''Return the intersection of this cell region and another.'''
        return CellRegionAnd((self, x))

    def __or__(self, x):
        '''Return the union of this cell region and another.'''
        return CellRegionOr((self, x))

    def __xor__(self, x):
        '''Return the symmetric difference of this cell region and another.'''
        return CellRegionXor((self, x))

    def __sub__(self, x):
        '''Return the subtraction of the other cell region from this one.'''
        return CellRegionSub((self, x))

    def boundary(self, x):
        ''' Signed boundary between `self` and `x`. '''
        return CellBoundaryFacetRegion((self, x))

    def internal_facets(self):
        ''' Get all internal facets. '''
        return CellRegionInternalFacetRegion((self,))

    def subdomain_internal_facets(self):
        ''' Get internal facets excluding boundaries across cell
        values (subdomains). '''
        return CellRegionSubdomainInternalFacetRegion((self,))

class CellRegionByName(CellRegion):
    kwarg_names = ('name',)

    def __init__(self, name):
        self.name = name

    def evaluate(self, context):
        return context['region_name_to_cvs'][self.name]

    def __repr__(self):
        return 'c{!r}'.format(self.name)

class OperatorMixin(object):
    kwarg_names = ('operands',)

    def __init__(self, operands):
        self.operands = tuple(operands)
        self.check_operands()

    def check_operands(self):
        pass

    def evaluate(self, context):
        return reduce(self.binary_operator, (
            o.evaluate(context) for o in self.operands))

    def __repr__(self):
        return '({} {})'.format(self.operator_name,
                                ' '.join(map(repr, self.operands)))

class CellRegionOperator(OperatorMixin, CellRegion):
    def check_operands(self):
        if not all(isinstance(x, AbstractCellRegion) for x in self.operands):
            raise TypeError("operands must derive from AbstractCellRegion")

class CellRegionAnd(CellRegionOperator):
    operator_name = '&'
    def binary_operator(self, x, y):
        return x & y

class CellRegionOr(CellRegionOperator):
    operator_name = '|'
    def binary_operator(self, x, y):
        return x | y

class CellRegionXor(CellRegionOperator):
    operator_name = '^'
    def binary_operator(self, x, y):
        return x ^ y

class CellRegionSub(CellRegionOperator):
    operator_name = '-'
    def binary_operator(self, x, y):
        return x - y


class FacetRegion(AlgebraicMixin, AbstractFacetRegion):
    '''Abstract facet region.

Can be evaluated down to a set of
:code:`(facet_marker_value, facet_sign)` using
:py:meth:`.MeshData.evaluate_topology`.

To access a facet region predefined (named) in the mesh
generator, instantiate this class directly with a :code:`name`
argument.
'''

    def __new__(cls, *args, **kwargs):
        if cls is FacetRegion:
            return FacetRegionByName(*args, **kwargs)

        return super().__new__(cls)

    def __and__(self, x):
        '''Return the intersection of this facet region and another.'''
        return FacetRegionAnd((self, x))

    def __or__(self, x):
        '''Return the union of this facet region and another.'''
        return FacetRegionOr((self, x))

    def __xor__(self, x):
        '''Return the symmetric difference of this facet region and another.'''
        return FacetRegionXor((self, x))

    def __sub__(self, x):
        '''Return the subtraction of the other facet region from this one.'''
        return FacetRegionSub((self, x))

    def flip(self):
        ''' Invert facet signedness. For example,

:code:`X.boundary(Y) == Y.boundary(X).flip()`
        '''
        return FacetRegionFlip((self,))

    def both(self):
        ''' Both facet sides, i.e.

:code:`f.both() == (f | f.flip())`'''
        return FacetRegionBothSides((self,))

    def unsigned(self):
        ''' Erase facet signedness information by setting
:code:`sign = 1` for every :code:`(facet_value, sign)` pair.
'''
        return FacetRegionUnsigned((self,))

class FacetRegionByName(FacetRegion):
    kwarg_names = ('name',)

    def __init__(self, name):
        self.name = name

    def evaluate(self, context):
        return context['facet_name_to_fvs'][self.name]

    def __repr__(self):
        return 'f{!r}'.format(self.name)


class FacetRegionOperator(OperatorMixin, FacetRegion):
    def check_operands(self):
        if not all(isinstance(x, AbstractFacetRegion) for x in self.operands):
            raise TypeError("operands must derive from AbstractFacetRegion")

class FacetRegionAnd(FacetRegionOperator):
    operator_name = '&'
    def binary_operator(self, x, y):
        return x & y

class FacetRegionOr(FacetRegionOperator):
    operator_name = '|'
    def binary_operator(self, x, y):
        return x | y

class FacetRegionXor(FacetRegionOperator):
    operator_name = '^'
    def binary_operator(self, x, y):
        return x ^ y

class FacetRegionSub(FacetRegionOperator):
    operator_name = '-'
    def binary_operator(self, x, y):
        return x - y

class FacetRegionFlip(FacetRegionOperator):
    operator_name = 'flip'

    def evaluate(self, context):
        arg, = self.operands
        return set(
            (fv, -sign) for fv, sign in arg.evaluate(context))

class FacetRegionBothSides(FacetRegionOperator):
    operator_name = 'both-sides'

    def evaluate(self, context):
        arg, = self.operands
        signs = (-1, 1)
        return set(
            (fv, sign)
            for fv, _ in arg.evaluate(context)
            for sign in signs)

class FacetRegionUnsigned(FacetRegionOperator):
    operator_name = 'unsigned'

    def evaluate(self, context):
        arg, = self.operands
        return set(
            (fv, 1) for fv, _ in arg.evaluate(context))

class OperandsAreCellRegionsMixin(object):
    def check_operands(self):
        if not all(isinstance(x, AbstractCellRegion) for x in self.operands):
            raise TypeError("operands must derive from AbstractCellRegion")

class CellBoundaryFacetRegion(OperandsAreCellRegionsMixin,
                              FacetRegionOperator):
    operator_name = 'boundary'

    def evaluate(self, context):
        X, Y = self.operands
        return context['facets_manager'].boundary(
            X.evaluate(context), Y.evaluate(context))

class CellRegionInternalFacetRegion(OperandsAreCellRegionsMixin,
                                    FacetRegionOperator):
    operator_name = 'internal'

    def evaluate(self, context):
        arg, = self.operands
        return context['facets_manager'].internal(
            arg.evaluate(context))

class CellRegionSubdomainInternalFacetRegion(OperandsAreCellRegionsMixin,
                                             FacetRegionOperator):
    operator_name = 'subdomain-internal'

    def evaluate(self, context):
        arg, = self.operands
        return context['facets_manager'].cell_value_internal(
            arg.evaluate(context))
