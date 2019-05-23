
from functools import reduce

import dolfin
import ufl.algebra
import ufl.constantvalue

from .function_subspace_registry import AssignmentError

__all__ = ['opportunistic_assign']

def opportunistic_assign(source, target, function_subspace_registry):
    ''' Assign the value of source `source` to target `target`, where
`target` may be a function or subfunction.

Use the fast
:py:meth:`~.function_subspace_registry.FunctionSubspaceRegistry.assign`
method if possible, otherwise py:func:`dolfin.project`. '''
    if target is source: # wew lad
        return

    src = source.m_as(target.units)
    tgt = target.magnitude

    src_offset, src_scale, src = interpret_as_linear_combination(src)
    tgt_offset, tgt_scale, tgt = interpret_as_linear_combination(tgt)

    # tgt * tgt_scale + tgt_offset = src * src_scale + src_offset
    # tgt = src * (src_scale/tgt_scale) + (src_offset - tgt_offset)
    # tgt = src * scale + offset
    scale = src_scale/tgt_scale
    offset = src_offset - tgt_offset

    fsr = function_subspace_registry
    try:
        fsr.assign(target=tgt, source=src,
                   coefficient=scale, offset=offset)
    except AssignmentError:
        pass
    else:
        return

    fsr.assign(target=tgt, source=dolfin.project(
        (src * scale + offset if offset else src * scale),
        fsr.get_function_space(tgt, collapsed=True)))
    return

class NotLinearCombination(Exception):
    pass

# TODO: desperately needs unit test
class InterpretAsLinearCombination(object):
    def interpret_constant(self, e):
        if isinstance(e, ufl.constantvalue.ConstantValue):
            return e.evaluate(None, {}, (), ())
        else:
            return None

    def interpret_as_linear_combination(self, expr):
        ''' Return :code:`(offset, scale, terminal)` such that
        :code:`expr=terminal*scale+offset`. '''
        try:
            return self._interpret_as_linear_combination(expr)
        except NotLinearCombination:
            return (0.0, 1.0, expr)

    def _interpret_as_linear_combination(self, expr):
        if isinstance(expr, ufl.algebra.Sum):
            terminal = None
            scale = 0.0
            offset = 0.0
            for x in expr.ufl_operands:
                c = self.interpret_constant(x)
                if c is not None:
                    offset += c
                else: # not a constant
                    offset_, scale_, terminal_ = (
                        self._interpret_as_linear_combination(x))
                    if terminal is not None and terminal != terminal_:
                        raise NotLinearCombination() # fail: multiple terminals
                    terminal = terminal_
                    offset += offset_
                    scale += scale_
            if terminal is None:
                raise NotLinearCombination()
            return (offset, scale, terminal)
        elif isinstance(expr, ufl.algebra.Product):
            terminal = None
            offset = 0.0
            scale = 1.0
            offset_scale = 1.0
            for x in expr.ufl_operands:
                c = self.interpret_constant(x)
                if c is not None:
                    offset_scale *= c
                    scale *= c
                else:
                    offset_, scale_, terminal_ = (
                        self._interpret_as_linear_combination(x))
                    if terminal is not None and terminal_ is not None:
                        raise NotLinearCombination() # fail: nonlinear
                    terminal = terminal_
                    offset = offset_
                    scale *= scale_
            offset *= offset_scale
            if terminal is None:
                raise NotLinearCombination()
            return (offset, scale, terminal)
        else:
            # TODO: extended check
            return (0.0, 1.0, expr)

interpret_as_linear_combination = (
    InterpretAsLinearCombination().interpret_as_linear_combination)
