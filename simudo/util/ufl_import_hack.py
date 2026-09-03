import importlib
import re
import sys


def fix():
    try:
        import ufl_legacy
    except ImportError:
        return  # no need, we don't have to do the rename thing

    regex = re.compile("^ufl")

    # This list must name every `ufl.<submodule>` that Simudo imports anywhere.
    # A submodule that is missing here still resolves -- `sys.modules["ufl"]` is
    # the ufl_legacy package, so the import machinery finds it via
    # `ufl_legacy.__path__` -- but it gets *re-executed* under the name
    # `ufl.<submodule>`, producing a second module object whose classes are
    # distinct from those in `ufl_legacy.<submodule>`. Duplicated functions are
    # harmless; duplicated classes break `isinstance`/subclass checks. Listing a
    # submodule here instead aliases it to the identical object.
    #
    # To regenerate:
    #     grep -rhE '^\s*(from|import)\s+ufl\.' --include=*.py --include=*.py1 simudo \
    #         | grep -v trash
    for old in """
ufl.log
ufl.algebra
ufl.constantvalue
ufl.core.expr
ufl.core.ufl_type
ufl.equation
ufl.mathfunctions
ufl.operators
ufl.algorithms.transformer
    """.split():
        mod = ""
        for name in old.split("."):
            if not mod:
                mod = name
            else:
                mod = f"{mod}.{name}"

            new = regex.sub("ufl_legacy", mod)
            sys.modules[mod] = importlib.import_module(new)
