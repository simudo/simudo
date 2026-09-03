"""
UFL has a bad design which prevents other modules from defining their own
operators/expressions after UFL has finished importing. This is stupid and
unnecessary. In particular,

https://github.com/FEniCS/ufl/blob/8cf81755da7f4a4e2ebf4ac8fa4c4f7898a9041e/ufl/algorithms/transformer.py#L42-L61

So yeah we have to recompute the sets of UFL classes, then clear out the
cache in `ufl.algorithms.transformer.Transformer`.

The code also happens to assume that sets will always be iterated in the
same order. Is that always true? I don't know.
"""

def update_ufl_expr_classes():
    from ufl.core.expr import Expr
    from ufl.algorithms.transformer import Transformer
    from ufl import classes

    def replace_set(s, xs):
        s.clear()
        s.update(xs)

    all_ufl_classes = classes.all_ufl_classes

    # fmt: off
    replace_set(classes.all_ufl_classes, Expr._ufl_all_classes_)
    replace_set(classes.abstract_classes, (c for c in all_ufl_classes if c._ufl_is_abstract_))
    replace_set(classes.ufl_classes, (c for c in all_ufl_classes if not c._ufl_is_abstract_))
    replace_set(classes.terminal_classes, (c for c in all_ufl_classes if c._ufl_is_terminal_))
    replace_set(classes.nonterminal_classes, (c for c in all_ufl_classes if not c._ufl_is_terminal_))
    # fmt: on

    Transformer._handlers_cache.clear()
