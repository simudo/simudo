"""Loading materials from outside the package, and A20's shadowing hazard.

Two routes exist: ``library://ClassName`` searches the material directories,
and ``file:///abs/path.py::ClassName`` executes a file the user wrote. They are
not the same code path -- a packaged material is imported as a proper module,
an external one is executed by path -- and the difference is what A20 was
about.

**A20**: the loader used to put sibling imports into ``sys.modules`` under bare
names (``silicon``, ``galliumarsenide``), so a user file named ``silicon.py``
and the library's own ``silicon`` module competed for the same key and whichever
loaded first won. Fixed in ``5142b8ad97`` by dropping the ``sys.path``
manipulation, but nothing pinned it until now.

Construction-only, so the whole file runs in about a second.
"""

import sys
import textwrap
from pathlib import Path

import pytest

from .conftest import MATERIALS_DIR, make_pn_config

#: A minimal but complete material, written to a temporary file at test time.
#: External materials must use absolute imports -- a module executed by path has
#: no parent package, so a relative import raises.
EXTERNAL_MATERIAL = '''
from simudo.physics import Material


class {class_name}(Material):
    """A deliberately distinctive material, for testing the loader."""

    name = "{class_name}"

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry
        d.update({{
            "poisson/permittivity": U("{permittivity} vacuum_permittivity"),
            "CB/energy_level": U("1.42 eV"),
            "VB/energy_level": U("0.0 eV"),
            "CB/effective_density_of_states": U("4.0e17 cm^-3"),
            "VB/effective_density_of_states": U("1.0e19 cm^-3"),
            "CB/mobility": U("8000 cm^2/V/s"),
            "VB/mobility": U("400 cm^2/V/s"),
            "CB/vth": U("2.0e7 cm/s"),
            "VB/vth": U("2.0e7 cm/s"),
            "CB/mDOS": U("0.067"),
            "VB/mDOS": U("0.47"),
        }})
        return d
'''


def write_material(directory, filename, class_name, permittivity=13.0):
    path = directory / filename
    path.write_text(textwrap.dedent(
        EXTERNAL_MATERIAL.format(class_name=class_name,
                                 permittivity=permittivity)))
    return path


def source_file(cls):
    """Path of the file a material class was defined in.

    ``inspect.getfile`` cannot do this: an external material is executed by
    path and never registered in ``sys.modules``, so its ``__module__`` is not
    importable and inspect reports it as a built-in. The code object of one of
    its own methods carries the filename regardless.
    """
    return Path(cls.get_dict.__code__.co_filename).resolve()


def build_with(runner, source, tmp_path):
    config = make_pn_config()
    config['materials'] = {'GaAsLike': {'source': source}}
    _builder, problems = runner.build_problems(
        config, library_dir=MATERIALS_DIR, goals=('full',))
    return problems


# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_file_url_material_loads_and_builds(runner, tmp_path):
    """``file://`` materials work at all.

    The GUI writes this form whenever a user points at their own material file,
    so it is a supported entry point, not a developer convenience.
    """
    path = write_material(tmp_path, 'my_material.py', 'MyMaterial')
    build_with(runner, f'file://{path}::MyMaterial', tmp_path)


@pytest.mark.solver
def test_file_url_material_resolves_to_the_users_own_file(runner, tmp_path):
    """The class the loader returns must come from the file that was named.

    Checked on the class object rather than on a solved value: a spatial rule
    evaluates to a UFL conditional over regions, not a number, so reading the
    permittivity back out would test UFL rather than the loader.
    """
    path = write_material(tmp_path, 'odd.py', 'OddMaterial', permittivity=42.0)
    cls = runner._load_library_material_class(
        f'file://{path}::OddMaterial', library_dir=MATERIALS_DIR)

    assert cls.__name__ == 'OddMaterial'
    assert source_file(cls) == path.resolve(), (
        f'loaded OddMaterial from {source_file(cls)}, not {path}')


@pytest.mark.solver
def test_a_user_file_cannot_shadow_a_library_material(runner, tmp_path):
    """**The A20 regression test.**

    A user file named after a library material must not displace it. Both are
    resolved in the same session, and each must come from its own file --
    previously the loser of the ``sys.modules`` race was silently replaced by
    the winner, so a user's ``galliumarsenide.py`` could be handed out in place
    of the library's, or vice versa.
    """
    impostor = write_material(
        tmp_path, 'galliumarsenide.py', 'GalliumArsenideMaterial',
        permittivity=99.0)

    user_cls = runner._load_library_material_class(
        f'file://{impostor}::GalliumArsenideMaterial', library_dir=MATERIALS_DIR)
    library_cls = runner._load_library_material_class(
        'library://GalliumArsenideMaterial', library_dir=MATERIALS_DIR)

    assert source_file(user_cls) == impostor.resolve(), (
        "the user's own galliumarsenide.py was not the one returned for its "
        "own file:// source")
    library_file = source_file(library_cls)
    assert library_file == (Path(MATERIALS_DIR) / 'galliumarsenide.py').resolve(), (
        f'the library GaAs resolved to {library_file}, not the packaged file; '
        f'a user file of the same name displaced it -- this is A20')
    assert user_cls is not library_cls, (
        'the two sources returned the same class object')


@pytest.mark.solver
def test_loading_an_external_material_does_not_pollute_sys_modules(runner, tmp_path):
    """No bare module names left behind.

    The mechanism behind A20: a module registered under a bare name like
    ``galliumarsenide`` is visible to every later import in the process. This
    asserts the loader leaves no such key, which is the property that makes the
    shadowing impossible rather than merely unlikely.
    """
    before = set(sys.modules)
    path = write_material(tmp_path, 'silicon.py', 'SiliconMaterial')
    build_with(runner, f'file://{path}::SiliconMaterial', tmp_path)

    leaked = {
        name for name in set(sys.modules) - before
        if '.' not in name and name in ('silicon', 'galliumarsenide',
                                        'germanium', 'aluminumarsenide',
                                        'SiliconMaterial')}
    assert not leaked, f'loader left bare names in sys.modules: {sorted(leaked)}'


@pytest.mark.solver
def test_missing_external_material_file_is_reported(runner, tmp_path):
    """A path that does not exist must fail loudly."""
    with pytest.raises((FileNotFoundError, ImportError, OSError)):
        build_with(runner, f'file://{tmp_path}/absent.py::Absent', tmp_path)


@pytest.mark.solver
def test_wrong_class_name_in_a_file_url_is_reported(runner, tmp_path):
    """The file exists but does not define that class."""
    path = write_material(tmp_path, 'present.py', 'PresentMaterial')
    with pytest.raises(AttributeError):
        build_with(runner, f'file://{path}::NotTheClassName', tmp_path)
