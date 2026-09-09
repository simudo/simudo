"""The materials' measured optical tables, and how they are exposed.

Why this exists, given that ``test_config_smoke.py`` already builds every
released material: that test *does* catch a broken table -- verified against the
real defect below -- but it takes about five seconds per material and fails
somewhere inside ``build_problems``, so it reports that the material would not
assemble rather than what is wrong with it. These run in well under a second for
the whole file and name the fix in the assertion message.

They also need no list to maintain. The signature check walks every material
module, so a file that grows an ``alpha_function`` block is covered the day it
does, without anyone remembering to register it.

The defect that prompted them: ``optical_properties_table`` is called as
``self.optical_properties_table()`` from ``get_dict``, so it must accept
``self``. Several unreleased materials declare it with **no** parameters
(``def optical_properties_table():``), which is self-consistent only while
nothing calls it through an instance. On 2026-09-03 a *clean* merge -- no
conflict reported -- combined trunk's new ``opt_cv/alpha_function`` block with a
branch's no-``self`` signature and broke ``IndiumArsenideMaterial``. Because the
two edits sat in different hunks, nothing flagged it.

Hence :func:`test_optical_table_is_callable_on_an_instance`, and hence
:func:`test_table_signature_accepts_self`, which reads the source with
:py:mod:`ast` instead of importing it -- that is what lets it cover every
material file rather than only the ones constructible in a test.

Nothing here checks physics. It checks that the tables are shaped the way every
consumer assumes: four columns in the order [wavelength_nm, alpha, n, k],
ascending in wavelength, finite and non-negative. ``_parse_alpha_spec`` and
``_build_material_callable`` both slice these blindly.
"""

import ast
import inspect
from pathlib import Path

import pytest

#: Released materials that ship a measured optical table, from 06 §B.4.
#: ``AluminumGalliumArsenideAlloy`` and ``IBTestMaterial`` are released but have
#: no table of their own -- the alloy's composition-dependent interpolator is
#: the unshipped ``alloy_optics`` (M15) -- so they are deliberately absent.
#:
#: This list governs the *runtime* tests only, which construct the class. The
#: unreleased materials are not all constructible without extra spatial
#: parameters (the alloys want ``MoleFractionX``) and ``alloy_optics`` is known
#: broken, so auto-discovering them would report failures that are not defects.
MATERIALS_WITH_TABLES = [
    ('galliumarsenide', 'GalliumArsenideMaterial'),
    ('aluminumarsenide', 'AluminumArsenideMaterial'),
    ('indiumarsenide', 'IndiumArsenideMaterial'),
    ('silicon', 'SiliconMaterial'),
]

#: Directory holding the material modules, for the source-reading tests.
_MATERIALS_DIR = Path(__file__).resolve().parents[2] / 'materials'


def _modules_defining_a_table():
    """Every material module that declares ``optical_properties_table``.

    Deliberately not filtered to the released set: reading the source costs
    nothing per file, and the unreleased materials are exactly where the
    no-``self`` signatures live, so they are the ones that would break on the
    day someone gives them an ``alpha_function`` block.
    """
    found = []
    for path in sorted(_MATERIALS_DIR.glob('*.py')):
        if path.stem == '__init__':
            continue
        if 'def optical_properties_table' in path.read_text(encoding='utf-8'):
            found.append(path.stem)
    return found


#: Discovered at import time, so a new material file needs no registration.
MODULES_DEFINING_A_TABLE = _modules_defining_a_table()

#: Column order every consumer assumes.
WAVELENGTH_NM, ALPHA, N, K = 0, 1, 2, 3


def _module_source(module_name):
    return (_MATERIALS_DIR / f'{module_name}.py').read_text(encoding='utf-8')


def _table_function(tree):
    """The ``optical_properties_table`` FunctionDef, or None."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == 'optical_properties_table'):
            return node
    return None


def _calls_table_on_self(tree):
    """Does this module call ``self.optical_properties_table()`` anywhere?"""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and node.attr == 'optical_properties_table'
                and isinstance(node.value, ast.Name)
                and node.value.id == 'self'):
            return True
    return False


def _load(class_name, module_name):
    """Import a material class, skipping if dolfin is absent.

    ``simudo.physics`` pulls in dolfin, so importing any material module needs
    a FEniCS install even though the table itself is plain numpy.
    """
    pytest.importorskip(
        'dolfin', reason='importing a material pulls in simudo.physics')
    module = __import__(f'simudo.materials.{module_name}',
                        fromlist=[class_name])
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Static: no dolfin required, so these run wherever pytest does
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('module_name', MODULES_DEFINING_A_TABLE)
def test_table_signature_accepts_self(module_name):
    """A material that calls its own table must declare a callable signature.

    This is the regression guard, and it covers every material file rather
    than a registered list. A no-parameter definition passes every import check
    and every AST scan, and fails only when something calls it on an instance
    -- which ``get_dict`` does, one line after declaring the
    ``alpha_function`` block that consumes it.

    Materials that define the table but never call it through ``self`` are
    left alone: several unreleased ones are written that way and are
    self-consistent. They start being checked the moment they gain a call.
    """
    source = _module_source(module_name)
    tree = ast.parse(source)
    func = _table_function(tree)
    assert func is not None, f'{module_name}: no optical_properties_table'

    if not _calls_table_on_self(tree):
        pytest.skip(
            f'{module_name} defines the table but never calls it via self')

    args = [a.arg for a in func.args.args]
    decorators = {getattr(d, 'id', None) for d in func.decorator_list}
    is_static = 'staticmethod' in decorators

    declared = args or 'no parameters'
    assert is_static or (args and args[0] == 'self'), (
        f'{module_name}.optical_properties_table declares {declared} and is '
        f'not a staticmethod, but this file calls it as '
        f'self.optical_properties_table(). Add self, or decorate it '
        f'@staticmethod.')


@pytest.mark.parametrize('module_name', MODULES_DEFINING_A_TABLE)
def test_alpha_function_block_is_a_literal_dict(module_name):
    """The ``alpha_function`` value must stay a literal dict.

    The GUI finds a material's alpha(E) by scanning the source with
    :py:mod:`ast` rather than by importing and calling it, so building the dict
    indirectly -- via a helper, a comprehension, or ``dict(...)`` -- hides it
    from the Processes panel while leaving the solver working. The comment in
    ``aluminumarsenide.py`` says as much; nothing enforced it.
    """
    tree = ast.parse(_module_source(module_name))
    literal_dict_keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                    and target.slice.value.endswith('alpha_function')):
                assert isinstance(node.value, ast.Dict), (
                    f'{module_name}: {target.slice.value} is assigned a '
                    f'{type(node.value).__name__}, not a literal dict; the '
                    f"GUI's AST scan will not see it")
                literal_dict_keys.add(target.slice.value)

    if not literal_dict_keys:
        pytest.skip(f'{module_name} publishes no alpha_function block')


# ---------------------------------------------------------------------------
# Runtime: needs the module importable, but not a built problem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('module_name,class_name', MATERIALS_WITH_TABLES)
def test_optical_table_is_callable_on_an_instance(module_name, class_name):
    """The call ``get_dict`` makes, without building a problem.

    ``problem_data`` is optional on :py:class:`~simudo.physics.Material`, so a
    bare instance is enough to exercise the binding. That keeps this in
    milliseconds where ``test_released_material_builds`` takes seconds.
    """
    cls = _load(class_name, module_name)
    table = cls().optical_properties_table()
    assert table.ndim == 2, f'{class_name}: table is {table.ndim}-dimensional'
    assert table.shape[1] == 4, (
        f'{class_name}: table has {table.shape[1]} columns, expected 4 '
        f'([wavelength_nm, alpha, n, k])')
    assert table.shape[0] > 0, f'{class_name}: table is empty'


@pytest.mark.parametrize('module_name,class_name', MATERIALS_WITH_TABLES)
def test_optical_table_columns_are_well_formed(module_name, class_name):
    """Shape assumptions every consumer makes, none of which was checked.

    ``get_dict`` hands ``table[:, 0:2]`` to a ``wavelength_table`` callable, so
    a table sorted the wrong way or carrying a NaN produces an interpolator
    that fails far from here -- inside a solve, as a convergence problem.
    """
    np = pytest.importorskip('numpy')
    cls = _load(class_name, module_name)
    table = cls().optical_properties_table()

    assert np.all(np.isfinite(table)), (
        f'{class_name}: table contains NaN or inf')

    wl = table[:, WAVELENGTH_NM]
    assert np.all(np.diff(wl) > 0), (
        f'{class_name}: wavelengths are not strictly increasing; '
        f'wavelength_table interpolation assumes sorted input')
    assert np.all(wl > 0), f'{class_name}: non-positive wavelength'

    assert np.all(table[:, ALPHA] >= 0), (
        f'{class_name}: negative absorption coefficient')
    assert np.all(table[:, N] > 0), (
        f'{class_name}: non-positive refractive index n')
    assert np.all(table[:, K] >= 0), (
        f'{class_name}: negative extinction coefficient k')


@pytest.mark.parametrize('module_name,class_name', MATERIALS_WITH_TABLES)
def test_alpha_function_slice_matches_the_table(module_name, class_name):
    """The declared block and the table it slices must stay in step.

    ``get_dict`` writes ``"data": self.optical_properties_table()[:, 0:2]``.
    If a table ever grew a leading column, that slice would silently start
    feeding the interpolator the wrong pair.
    """
    np = pytest.importorskip('numpy')
    cls = _load(class_name, module_name)
    table = cls().optical_properties_table()
    data = table[:, 0:2]

    assert data.shape == (table.shape[0], 2)
    assert np.array_equal(data[:, 0], table[:, WAVELENGTH_NM])
    assert np.array_equal(data[:, 1], table[:, ALPHA])
