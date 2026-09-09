#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Claude Sonnet 4.6
"""
simudo_1d_runner.py
===================
Generic runner that reads a Simudo project YAML file and executes a
1D layered-structure Poisson / drift-diffusion simulation.

Usage (inside the Docker container or environment with Simudo installed)::

    cd ~/simudo/code
    python gui/simudo_1d_runner.py gui/fourlayer_example.yaml

The project YAML format is documented and exemplified in
``gui/fourlayer_example.yaml``.

The ``execution:`` section of the project YAML is consumed by the GUI
launcher layer (which decides whether to run locally, via ``docker exec``,
or over SSH).  This script always runs locally inside the target
environment and never reads the ``execution:`` key.

Design notes
------------
Priority levels for spatial rules (lower number = higher priority; wins
via ``setdefault`` in :py:meth:`~simudo.fem.Spatial.get`)::

    PRIORITY_OVERLAY  = 0   named overlay-region properties — highest priority
    PRIORITY_LAYER    = 5   layer-specific properties
    PRIORITY_DOMAIN   = 7   the `domain` overlay — a device-wide default
    PRIORITY_MATERIAL = 10  material default properties — lowest priority

`domain` is deliberately *below* layers.  It is how a user says "unless I say
otherwise, this holds everywhere", so a value set on a single layer has to be
able to beat it; before this it shared PRIORITY_OVERLAY with named overlays and
was registered first, which made every layer-level override silently inert.

Material properties are registered by calling
``spatial.add_material_data`` directly with ``priority=PRIORITY_MATERIAL``,
so that layer / overlay rules always beat material defaults regardless of
registration order.

Bidirectional voltage sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simudo's Newton solver ramps the parameter (I or V) slowly from a known
solution.  Thermal equilibrium gives the solution at I = V = 0.  A
unidirectional sweep from 0 → +Voc is straightforward.  ``voltage_sweep.values``
is split around an *initial_v* -- 0 for a fresh run, or the resumed checkpoint's
own value (see below) -- into an "up" group (values above initial_v) and a
"down" group (values below initial_v), each run against its own problem
instance sharing one seed state, so the (possibly expensive) state that
produced that initial_v point is only computed once.

Checkpoint resumption
~~~~~~~~~~~~~~~~~~~~~~
``simulation.resume_from`` (a checkpoint ``.yaml`` path relative to the
output folder) resumes a previous run instead of starting fresh.  The stage
the checkpoint belongs to is inferred from its filename (``..._V=<v>.yaml``
or ``..._I=<i>.yaml``, written by
:py:meth:`~simudo.fem.adaptive_stepper.AdaptiveStepper.write_xdmf_checkpoint`):

* An **optical-ramp** (``I``) checkpoint continues the intensity ramp from
  that intensity (or, if it is already at I=1, is used directly) and then
  runs the voltage sweep(s) normally (initial_v = 0) from the resulting V=0
  state — this is how a run can branch into a fresh voltage sweep without
  recomputing the optical ramp.
* A **voltage** checkpoint (``V`` of either sign) resumes with initial_v = that
  checkpoint's own value, and sweeps *both* directions from it as needed --
  e.g. resuming from V=0.4 with targets ``[0.3, 0.4, 0.5]`` computes both 0.3
  and 0.5, each from its own copy of the loaded V=0.4 state. The optical
  ramp is skipped either way (the checkpoint already reflects its outcome).

Either way, any requested value already present in this output folder's
``sim_V.csv`` / ``sim_I.csv`` (from the run that produced the checkpoint) is
dropped rather than recomputed -- see :py:func:`_already_computed_values`.

See :py:func:`_parse_checkpoint_stage` and the "Stage 1/2/3" comments in
:py:func:`run` for the exact logic.  Resuming still requires the project
YAML to describe an identically-constructed problem (same layers, bands,
processes, mesh) — the checkpoint only restores solution *values*, not the
problem structure.
"""

import sys
import os
import re
import math
import time
import logging
import importlib.util
from datetime import date
from functools import partial
from pathlib import Path

import yaml
import numpy as np
import dolfin

import simudo.physics as _simudo_physics
from simudo.physics import (
    Material,
    ProblemData,
    SRHRecombination,
    NonRadiativeTrap,
    BeerLambert,
    IBBeerLambert,
    NonOverlappingTopHatBeerLambert,
    NonOverlappingTopHatBeerLambertIB,
    ShockleyReadBand2BandTrap,
    ShockleyReadTrap2Trap,
    make_alpha_from_table,
    make_alpha_from_wavelength_table,
    make_sigma_top_hat,
    make_sigma_from_table,
    make_sigma_from_wavelength_table,
)
from simudo.physics.heterojunction import ThermionicHeterojunction
from simudo.physics.boundary_conditions import SurfaceRecombination
from simudo.mesh import (
    ConstructionHelperLayeredStructure,
    CellRegions,
    FacetRegions,
)
from simudo.fem import setup_dolfin_parameters
from simudo.util import make_unit_registry, TypicalLoggingSetup, Blackbody, AM15Spectrum
from simudo.physics import VoltageStepper, OpticalIntensityAdaptiveStepper
from simudo.io.output_writer import (
    OutputWriter,
    MetaExtractorBandInfo,
    MetaExtractorIntegrals,
)
from simudo.io import h5yaml


# ---------------------------------------------------------------------------
# Priority constants
# ---------------------------------------------------------------------------

#: Temperature assumed when a project names none anywhere.  Every material
#: reads it during get_dict(), and an unset spatial key resolves to zero, so
#: without a default a project that forgets it is solved at 0 K -- which does
#: not converge, and says nothing useful about why.
DEFAULT_TEMPERATURE_K = 300.0

PRIORITY_OVERLAY  = 0   # named overlay-region rules — highest priority
PRIORITY_LAYER    = 5   # layer-specific rules
PRIORITY_DOMAIN   = 7   # the `domain` overlay: a device-wide default
PRIORITY_MATERIAL = 10  # material-default rules — lowest priority

# ---------------------------------------------------------------------------
# EOP class registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# EOP class registry and lookup
# ---------------------------------------------------------------------------
# EOP_CLASSES is a backward-compatibility / override table.  It is consulted
# as a fallback when a class name is not found directly in simudo.physics.
# You should NOT need to add entries here for new EOP classes — see
# _get_eop_class() below.
EOP_CLASSES = {
    # Keep entries here only for classes that have been renamed or moved
    # and whose old name must still be accepted by the runner.
    # Current published names (these are also present in simudo.physics, so
    # the table is redundant but harmless and serves as documentation):
    'SRHRecombination':                  SRHRecombination,
    'NonRadiativeTrap':                  NonRadiativeTrap,
    'NonOverlappingTopHatBeerLambert':   NonOverlappingTopHatBeerLambert,
    'NonOverlappingTopHatBeerLambertIB': NonOverlappingTopHatBeerLambertIB,
    'ShockleyReadBand2BandTrap':         ShockleyReadBand2BandTrap,
    'ShockleyReadTrap2Trap':             ShockleyReadTrap2Trap,
}


def _get_eop_class(cls_name):
    """Return the EOP class for *cls_name*, or raise a clear KeyError.

    Lookup order
    ------------
    1. ``simudo.physics`` namespace — any EOP class exported there is found
       automatically without touching this file.
    2. ``EOP_CLASSES`` dict — backward-compat overrides / renamed classes.

    Adding a new EOP class to simudo
    ---------------------------------
    If the new class is exported from ``simudo.physics.__init__`` (i.e.,
    ``from simudo.physics import NewClass`` works), **no change to this
    runner is needed** — the class will be resolved automatically at step 1.

    Only add an entry to ``EOP_CLASSES`` if:
    * the class was renamed and you want both the old and new name to work, or
    * the class lives in a sub-module that is not re-exported by
      ``simudo.physics.__init__``.
    """
    cls = getattr(_simudo_physics, cls_name, None)
    if cls is not None:
        return cls
    cls = EOP_CLASSES.get(cls_name)
    if cls is not None:
        return cls
    raise KeyError(
        f"Unknown EOP class {cls_name!r}.  "
        f"Either export it from simudo.physics.__init__ (preferred) or add "
        f"it to EOP_CLASSES in simudo_1d_runner.py."
    )

# Direction string → (dx, dy) unit vector for optical fields (2D strip mesh).
# Use _direction_vector() to get the vector for the mesh actually in use.
DIRECTION_MAP = {
    '+x': (1.0, 0.0),
    '-x': (-1.0, 0.0),
    '+y': (0.0, 1.0),
    '-y': (0.0, -1.0),
}

#: Default mesh dimension for the runner.  1 = true-1D interval mesh (new
#: route, about half the degrees of freedom); 2 = the historical 2D strip
#: (one cell layer of triangles between y=0 and y=1).  Override with
#: ``mesh: {dimension: 2}`` in the project YAML, e.g. to compare the two.
DEFAULT_MESH_DIMENSION = 1


def _optical_gdim(optical):
    """Geometric dimension of the mesh an ``Optical`` object lives on.

    Falls back to :py:data:`DEFAULT_MESH_DIMENSION` when the object does not
    expose a mesh -- e.g. the lightweight stand-ins used by the field-set
    unit tests.
    """
    mu = getattr(optical, 'mesh_util', None)
    gdim = getattr(mu, 'gdim', None)
    if gdim is None:
        md = getattr(optical, 'mesh_data', None)
        mesh = getattr(md, 'mesh', None)
        if mesh is not None:
            gdim = mesh.geometry().dim()
    return DEFAULT_MESH_DIMENSION if gdim is None else int(gdim)


def _direction_vector(direction_str, gdim):
    """Unit propagation vector for an optical field on a mesh of geometric
    dimension *gdim* (1 for the true-1D route, 2 for the strip)."""
    if direction_str not in DIRECTION_MAP:
        raise ValueError(f"Unsupported optical field direction {direction_str!r}")
    vec = DIRECTION_MAP[direction_str]
    if gdim == 1:
        if direction_str not in ('+x', '-x'):
            raise ValueError(
                f"Optical field direction {direction_str!r} is not possible "
                f"on a 1D mesh; use '+x' or '-x'.")
        return (vec[0],)
    return vec

# Large photon energy used in place of "infinity" (eV).
# Must be modest (≤ ~20 eV) so scipy.quad can resolve the peak of the
# Planck distribution near E ~ k_B*T_source.  Simudo's own blackbody code
# uses 20 eV as the conventional upper bound (see Blackbody.total_intensity_on_earth
# and non_overlapping_energy_ranges).
ENERGY_INFINITY_EV = 20.0


# ===========================================================================
# Helper utilities
# ===========================================================================

def _thickness_um(t) -> float:
    """Return layer thickness in microns from either a plain float or a {value, unit} dict."""
    if isinstance(t, dict):
        val = float(t['value'])
        unit = t.get('unit', 'um')
        if unit in ('um', 'µm'):
            return val
        elif unit == 'nm':
            return val / 1000.0
        elif unit == 'mm':
            return val * 1000.0
        elif unit == 'm':
            return val * 1e6
        else:
            raise ValueError(f"Unknown thickness unit {unit!r}")
    return float(t)


# Keys whose YAML values are specified as number densities (1/volume) but whose
# Simudo counterpart expects charge density (elementary_charge/volume).
# The runner multiplies by e automatically so users can write 1e18 cm^-3.
_CHARGE_DENSITY_KEYS = frozenset({"poisson/static_rho"})


def _coerce_quantity(key: str, qty, U):
    """Multiply *qty* by elementary_charge if *key* expects charge density
    but the value was given as a plain number density."""
    if key not in _CHARGE_DENSITY_KEYS:
        return qty
    if not hasattr(qty, 'dimensionality'):
        return qty
    # pint dimensionality for 1/cm^3 is {'[length]': -3}; for C/cm^3 it includes [current][time].
    dim = dict(qty.dimensionality)
    if dim == {'[length]': -3}:
        return qty * U('elementary_charge')
    return qty


def parse_quantity(entry, U):
    """Parse a YAML property value into a pint Quantity.

    Accepted forms:

    * ``{value: X, unit: "cm^2"}``  — returns ``X * U(unit)``
    * a bare scalar (int or float)  — returned as-is (dimensionless)
    * ``True`` / ``False``          — returned as-is (Python bool)

    Extra keys (such as ``mode``) are silently ignored, so a full intensity
    dict can be passed directly.
    """
    if isinstance(entry, dict):
        value = entry['value']
        unit  = entry['unit']
        return float(value) * U(unit)
    elif isinstance(entry, bool):
        return entry
    else:
        return entry


def parse_spatial_quantity(entry, U):
    """Like parse_quantity but wraps numeric magnitudes in dolfin.Constant.

    Use for quantities going into spatial.add_rule / spatial.add_material_data.
    dolfin.Constant avoids FEniCS form recompilation when the value changes
    between runs.
    """
    if isinstance(entry, dict):
        value = entry['value']
        unit  = entry['unit']
        return dolfin.Constant(float(value)) * U(unit)
    elif isinstance(entry, bool):
        return entry
    else:
        return entry


def _increment_dir(cur_dir):
    """Increment the last alphanumeric character of *cur_dir*.

    Mirrors the logic in ``JK_Helpers/helpers.py`` (inlined here so
    the runner has no dependency outside Simudo).
    """
    last_pos = -1
    for i, c in reversed(list(enumerate(cur_dir))):
        if c.isalnum():
            last_pos = i
            last_c   = c
            break
    if last_pos == -1:
        return cur_dir + '_1'

    if last_c == 'z':
        return cur_dir[:last_pos] + 'za' + cur_dir[last_pos + 1:]
    elif last_c == 'Z':
        return cur_dir[:last_pos] + 'ZA' + cur_dir[last_pos + 1:]
    elif last_c.isdigit():
        numbers = re.findall(r'\d+', cur_dir)
        if numbers:
            last_num     = int(numbers[-1])
            last_num_pos = cur_dir.rfind(numbers[-1])
            num_len      = len(numbers[-1])
            return (cur_dir[:last_num_pos]
                    + str(last_num + 1)
                    + cur_dir[last_num_pos + num_len:])
        return cur_dir
    else:
        return (cur_dir[:last_pos]
                + chr(ord(last_c) + 1)
                + cur_dir[last_pos + 1:])


def ensure_new_dir(cur_dir):
    """Return *cur_dir* (or an incremented variant) that does not yet exist."""
    while os.path.isdir(cur_dir):
        cur_dir = _increment_dir(cur_dir)
    return cur_dir


def standard_outdir(base_dir):
    """Return ``<base_dir>/out/YYYY[Mon]DD/a`` with auto-increment if the dir exists."""
    today  = date.today()
    outdir = str(Path(base_dir) / 'out' / today.strftime('%Y%b%d') / 'a')
    return ensure_new_dir(outdir)


# ===========================================================================
# Material helpers
# ===========================================================================

# Keys that are handled by the runner and must not be forwarded to Simudo's
# spatial system (they are not valid spatial-rule keys).
_RUNNER_HANDLED_KEYS = frozenset({
    'IB/simple_mobility_model',  # sets band.use_constant_mobility, not a rule
})


def _build_material_dict(mat_def, U):
    """Convert an inline YAML material definition to a ``{key: value}``
    dict, filtering out keys the runner handles separately.

    Scalar entries are parsed to pint quantities via ``parse_spatial_quantity``.
    Callable-spec entries (dicts containing a ``'type'`` key whose value is
    one of the recognised callable type tags) are passed through unchanged
    so that ``_register_material_dict`` can dispatch them to
    ``spatial.add_material_callable_data``.
    """
    result = {}
    for key, entry in mat_def.items():
        if key in _RUNNER_HANDLED_KEYS:
            continue
        if (isinstance(entry, dict)
                and entry.get('type') in
                    ('top_hat', 'energy_table', 'wavelength_table', 'formula')):
            result[key] = entry
        else:
            result[key] = parse_spatial_quantity(entry, U)
    return result


def _register_material_dict(spatial, mat_name, mat_dict,
                             U, priority=PRIORITY_MATERIAL):
    """Dispatch every entry in *mat_dict* to ``spatial.add_material_*``.

    Scalar entries (pint quantities, dolfin Constants, numbers) → add_material_data.
    Callable entries (``dict`` with a recognised ``'type'`` tag —
    ``top_hat``, ``energy_table``, ``wavelength_table``, ``formula``) →
    built via :py:func:`_build_material_callable` and registered via
    ``add_material_callable_data``.
    """
    for key, value in mat_dict.items():
        if isinstance(value, dict) and 'type' in value:
            func = _build_material_callable(value, key, U)
            spatial.add_material_callable_data(
                key=key, material_name=mat_name,
                func=func, priority=priority)
        else:
            spatial.add_material_data(
                key=key, material_name=mat_name,
                value=value, priority=priority)


def _build_material_callable(spec, key, U):
    """Build a callable f(E)→pint.Quantity from a material-provided spec dict.

    ``spec['type']`` selects the form.  Recognised values:

    - ``top_hat``: ``{E_low, E_high, value, [unit]}`` — ``value`` is a
      pint Quantity (preferred) or plain float with separate ``unit``.
    - ``energy_table``: ``{data: Nx2 array-like (E, value), arg_unit, value_unit}``
    - ``wavelength_table``: ``{data: Nx2 array-like (λ, value), arg_unit, value_unit}``
    - ``formula``: ``{func: callable f(E)→pint.Quantity}`` — v2, currently
      passes the callable through unchanged.

    Raises ``ValueError`` if the spec is malformed.
    """
    t = spec.get('type')

    def _as_quantity(v, default_unit):
        """Accept a pint Quantity or a (float, unit) pair."""
        if hasattr(v, 'units'):
            return v
        return float(v) * U(default_unit)

    if t == 'top_hat':
        # value carries the return unit, used as the default if missing
        unit = spec.get('unit')
        val  = spec['value']
        if hasattr(val, 'units'):
            sigma_or_alpha = val
            value_unit = str(val.units)
        else:
            value_unit = unit or '1/cm'
            sigma_or_alpha = float(val) * U(value_unit)
        E_low  = _as_quantity(spec['E_low'],  'eV')
        E_high = _as_quantity(spec['E_high'], 'eV')
        # Reuse the existing top-hat builder (used for sigma but unit-agnostic).
        return make_sigma_top_hat(E_low, E_high, sigma_or_alpha,
                                  sigma_unit=value_unit)

    if t == 'energy_table':
        data        = np.asarray(spec['data'])
        arg_unit    = spec.get('arg_unit',   'eV')
        value_unit  = spec.get('value_unit', '1/cm')
        # make_alpha_from_table expects energies in eV; if a different unit
        # is declared, pre-convert.
        E_col = data[:, 0]
        v_col = data[:, 1]
        if arg_unit != 'eV':
            E_q = U.Quantity(E_col, arg_unit).to('eV').magnitude
        else:
            E_q = E_col
        return make_alpha_from_table(E_q, v_col, alpha_unit=value_unit)

    if t == 'wavelength_table':
        data        = np.asarray(spec['data'])
        arg_unit    = spec.get('arg_unit',   'nm')
        value_unit  = spec.get('value_unit', '1/cm')
        wl_col = data[:, 0]
        v_col  = data[:, 1]
        if arg_unit != 'nm':
            wl_q = U.Quantity(wl_col, arg_unit).to('nm').magnitude
        else:
            wl_q = wl_col
        return make_alpha_from_wavelength_table(
            wl_q, v_col, alpha_unit=value_unit)

    if t == 'formula':
        func = spec.get('func')
        if not callable(func):
            raise ValueError(
                f"Material callable {key!r}: type='formula' requires a "
                f"'func' entry that is a Python callable.")
        return func

    raise ValueError(
        f"Material callable {key!r}: unknown type {t!r}. "
        f"Expected one of: top_hat, energy_table, wavelength_table, formula.")


def _to_dir_list(library_dir) -> list:
    """Normalise library_dir (None / str / list) to a list of existing dirs."""
    if library_dir is None:
        return []
    if isinstance(library_dir, (str, os.PathLike)):
        return [str(library_dir)]
    return [str(d) for d in library_dir]


def _packaged_material_module_name(fpath):
    """Return the ``simudo.materials.*`` module name for *fpath*, else None.

    Files that live inside the installed :py:mod:`simudo.materials` package are
    imported as package modules rather than executed by path. That is what lets
    the alloys import their parent binaries with ordinary relative imports
    (``from .galliumarsenide import ...``), and it keeps them out of
    ``sys.modules`` under bare names where a user's own ``silicon.py`` could
    shadow them.
    """
    import simudo.materials
    pkg_dir = os.path.abspath(os.path.dirname(simudo.materials.__file__))
    fpath = os.path.abspath(fpath)
    if os.path.dirname(fpath) != pkg_dir:
        return None
    stem = os.path.splitext(os.path.basename(fpath))[0]
    return 'simudo.materials.' + stem


def _exec_material_module(class_name, fpath):
    """Load *class_name* from the material file *fpath*.

    Packaged materials are imported normally. Anything else -- a user's own
    file given via ``file://`` or an out-of-tree ``--library-dir`` -- is
    executed by path, and must use absolute imports: a by-path module has no
    parent package, so relative imports raise, and bare sibling imports are
    not resolvable.
    """
    modname = _packaged_material_module_name(fpath)
    if modname is not None:
        module = importlib.import_module(modname)
    else:
        spec = importlib.util.spec_from_file_location(class_name, fpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return getattr(module, class_name)


def _load_library_material_class(source_str, library_dir=None):
    """Load a Material subclass from a library source string.

    Supported forms:

    * ``library://ClassName``  — searches *library_dir* recursively for a
      ``.py`` file containing ``class ClassName``
    * ``file:///abs/path/to/material.py::ClassName``  — loads directly

    *library_dir* may be a single path string, a list of path strings, or None.
    All listed directories are searched in order.
    """
    if source_str.startswith('library://'):
        class_name = source_str[len('library://'):]
        dirs = _to_dir_list(library_dir)
        if not dirs:
            raise ValueError(
                f"library:// source requires --library-dir to be set "
                f"(source was {source_str!r})")
        for search_dir in dirs:
            for root, _, files in os.walk(search_dir):
                for fname in files:
                    if not fname.endswith('.py'):
                        continue
                    fpath = os.path.join(root, fname)
                    with open(fpath, encoding='utf-8') as fh:
                        content = fh.read()
                    if f'class {class_name}' not in content:
                        continue
                    return _exec_material_module(class_name, fpath)
        raise ImportError(
            f"Could not find class {class_name!r} in library dirs: {dirs!r}")

    elif source_str.startswith('file://'):
        rest       = source_str[len('file://'):]
        path, _, class_name = rest.rpartition('::')
        return _exec_material_module(class_name, path)
    else:
        raise ValueError(f"Unknown material source format: {source_str!r}")


# ===========================================================================
# Mesh construction
# ===========================================================================

def _build_mesh(config, U):
    """Construct the 1D mesh and return *mesh_data*."""
    mesh_config  = config.get('mesh', {})
    max_edge     = float(mesh_config.get('max_edge_length', 0.02))

    layers = []
    for layer in config['layers']:
        m = layer.get('mesh', {})
        mesh_spec = dict(
            type   = m.get('type', 'geometric'),
            start  = float(m.get('start', 0.005)),
            factor = float(m.get('factor', 1.2)),
        )
        layers.append(dict(
            name      = layer['name'],
            material  = layer.get('material', ''),
            thickness = _thickness_um(layer['thickness']),
            mesh      = mesh_spec,
        ))

    dimension = int(mesh_config.get('dimension', DEFAULT_MESH_DIMENSION))

    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(
        edge_length    = max_edge,
        layers         = layers,
        extra_regions  = [],
        mesh_unit      = U.um,
        mesh_dimension = dimension,
    )
    ls.run()
    n_pts = len(ls.interval_1d_tag.coordinates["coordinates"])
    log = logging.getLogger('runner')
    log.info(f'Mesh points (along x): {n_pts}')
    log.info(f'Mesh dimension: {dimension} '
             f'({"true-1D interval mesh" if dimension == 1 else "2D strip"}); '
             f'{ls.mesh.num_vertices()} vertices, {ls.mesh.num_cells()} cells')
    return ls.mesh_data


# ===========================================================================
# Topology
# ===========================================================================

def _topology_standard_contacts(R, F):
    """Set up left/right contact facets and the nonconductive exterior."""
    F.left_contact  = R.exterior_left.boundary(R.domain)
    F.right_contact = R.domain.boundary(R.exterior_right)
    F.exterior      = R.domain.boundary(R.exterior)
    F.contacts      = F.left_contact | F.right_contact
    F.nonconductive = F.exterior - F.contacts.both()


def _build_topology(config, R, F):
    """Populate *R* and *F* with standard contact regions."""
    _topology_standard_contacts(R, F)
    # Note: R.domain already represents the full simulation domain;
    # there is no need for a separate R.cell alias.


# ===========================================================================
# Optical flux helpers
# ===========================================================================

def _compute_optical_flux(field_config, U):
    """Compute photon flux for one optical field.

    Returns a pint Quantity (typically in ``1/cm^2/s`` or equivalent).
    """
    intensity = field_config.get('intensity', {})
    mode      = intensity.get('mode', 'explicit')

    if mode == 'explicit':
        # Photon flux passed directly; parse_quantity ignores 'mode' key.
        return parse_quantity(intensity, U)

    elif mode == 'explicit_power_density':
        # Power density (W/m², mW/cm², suns, …) → photon flux via E_photon.
        power_val  = float(intensity.get('value', 0.0))
        power_unit = intensity.get('unit', 'mW/cm^2')
        # "suns" is not a pint unit: 1 sun ≈ 100 mW/cm²
        if power_unit.lower() in ('suns', 'sun'):
            power = power_val * 100.0 * U('mW/cm^2')
        else:
            power = power_val * U(power_unit)
        photon_e = parse_quantity(field_config['photon_energy'], U)
        return (power / photon_e).to('1/(cm^2 * s)')

    elif mode == 'blackbody_band':
        T_src = parse_quantity(intensity['T_source'], U)
        conc  = float(intensity.get('concentration', 1.0))

        E_min_entry = intensity['E_min']
        E_max_entry = intensity['E_max']

        E_min = (ENERGY_INFINITY_EV * U.eV if E_min_entry == 'infinity'
                 else parse_quantity(E_min_entry, U))
        E_max = (ENERGY_INFINITY_EV * U.eV if E_max_entry == 'infinity'
                 else parse_quantity(E_max_entry, U))

        bb   = Blackbody(temperature=T_src)
        flux = bb.photon_flux_integral_on_earth(E_min, E_max) * conc
        return flux

    else:
        raise ValueError(f"Unknown optical intensity mode: {mode!r}")


def _flux_contact(direction_str, F):
    """Return the facet region where the incoming photon flux BC is applied."""
    if direction_str == '+x':
        return F.left_contact    # light enters from the left face
    elif direction_str == '-x':
        return F.right_contact   # light enters from the right face
    else:
        raise ValueError(f"Unsupported optical field direction {direction_str!r}")


def _expand_field_set(fset_config, U, optical, ospatial, F, existing_names):
    """Expand one ``optical_field_sets`` entry into N individual optical fields.

    Parameters
    ----------
    fset_config : dict
        One entry from the ``optical_field_sets:`` YAML list.
    U : pint.UnitRegistry
    optical, ospatial : Simudo optical and optical-spatial objects
    F : FacetRegions
    existing_names : set[str]
        Names of fields already registered (for collision detection).
    """
    log = logging.getLogger('runner')

    name_prefix = fset_config['name_prefix']
    direction   = fset_config.get('direction', '+x')
    N           = int(fset_config['N_bins'])
    E_min       = parse_quantity(fset_config['E_min'], U)
    E_max       = parse_quantity(fset_config['E_max'], U)

    spec_config = fset_config.get('spectrum', {})
    # Accept bare string shorthand: `spectrum: am15g` == `spectrum: {type: am15g}`
    if isinstance(spec_config, str):
        spec_config = {'type': spec_config}
    spec_type   = spec_config.get('type', 'am15g')
    conc        = float(fset_config.get('concentration', spec_config.get('concentration', 1.0)))

    if spec_type in ('am15g', 'am15d'):
        variant = 'global' if spec_type == 'am15g' else 'direct'
        spec    = AM15Spectrum(U, variant=variant, concentration=conc)
    elif spec_type == 'blackbody':
        T_source = parse_quantity(spec_config['T_source'], U)
        bb       = Blackbody(temperature=T_source)
        # Wrap Blackbody in a thin adapter with the same interface as AM15Spectrum
        class _BlackbodyAdapter:
            def bin_parameters(self, e0, e1, n):
                edges = np.linspace(e0.m_as('eV'), e1.m_as('eV'), n + 1) * U.eV
                E_rep = np.array([
                    0.5 * (edges[i].m_as('eV') + edges[i+1].m_as('eV'))
                    for i in range(n)]) * U.eV
                flux = np.array([
                    bb.photon_flux_integral_on_earth(edges[i], edges[i+1]).magnitude * conc
                    for i in range(n)]) * U('1/cm^2/s')
                return edges, E_rep, flux
        spec = _BlackbodyAdapter()
    else:
        raise ValueError(
            f"Unknown spectrum type {spec_type!r} in optical_field_sets. "
            f"Valid types: 'am15g', 'am15d', 'blackbody'.")

    edges, E_rep, flux = spec.bin_parameters(E_min, E_max, N)

    d_vec        = _direction_vector(direction, _optical_gdim(optical))
    flux_contact = _flux_contact(direction, F)

    for i in range(N):
        fname = f'{name_prefix}_{i:03d}'
        if fname in existing_names:
            raise ValueError(
                f"optical_field_sets: name collision — field {fname!r} already exists")
        existing_names.add(fname)

        photon_e = E_rep[i]
        phi      = flux[i]
        optical.easy_add_field(fname, photon_energy=photon_e, direction=d_vec)
        ospatial.add_BC(f'{fname}/Phi', flux_contact, phi)
        log.info(f'Optical field set {name_prefix!r} bin {i:03d}: '
                 f'E={photon_e:.4f}, flux={phi:.3e}')


def _parse_alpha_spec(spec, U):
    """Parse a YAML alpha spec dict into a callable ``alpha_func(E) -> pint Quantity``.

    Supported keys (mutually exclusive):
    ``top_hat``      — ``{E_low: qty, E_high: qty, value: X, unit: '1/cm'}`` —
                       constant alpha between E_low and E_high (E_high may be ∞)
    ``constant``     — ``{value: X, unit: '1/cm'}`` — energy-independent alpha
    ``energies_eV``  — inline table paired with ``alphas`` and optional ``unit``
    ``wavelengths_nm`` — inline wavelength table paired with ``alphas`` and optional ``unit``
    ``table_file``   — path to a two-column CSV (energy_eV, alpha); ``unit`` overrides default
    """
    if 'top_hat' in spec:
        th = spec['top_hat']
        E_low  = parse_quantity(th['E_low'],  U)
        E_high = parse_quantity(th['E_high'], U)
        unit   = th.get('unit', '1/cm')
        alpha_0 = float(th['value']) * U(unit)
        # make_sigma_top_hat is unit-agnostic — reuse it for alpha.
        return make_sigma_top_hat(E_low, E_high, alpha_0, sigma_unit=unit)

    elif 'constant' in spec:
        val  = float(spec['constant']['value'])
        unit = spec['constant'].get('unit', '1/cm')
        def alpha_func(E):
            return dolfin.Constant(val) * U(unit)
        return alpha_func

    elif 'energies_eV' in spec:
        return make_alpha_from_table(
            spec['energies_eV'], spec['alphas'],
            alpha_unit=spec.get('unit', '1/cm'))

    elif 'wavelengths_nm' in spec:
        return make_alpha_from_wavelength_table(
            spec['wavelengths_nm'], spec['alphas'],
            alpha_unit=spec.get('unit', '1/cm'))

    elif 'table_file' in spec:
        data     = np.loadtxt(spec['table_file'], delimiter=',', comments='#')
        col_e    = int(spec.get('column_energy', 0))
        col_a    = int(spec.get('column_alpha',  1))
        return make_alpha_from_table(
            data[:, col_e], data[:, col_a],
            alpha_unit=spec.get('unit', '1/cm'))

    else:
        raise ValueError(
            f"Cannot parse alpha spec; expected 'top_hat', 'constant', 'energies_eV', "
            f"'wavelengths_nm', or 'table_file'. Got keys: {list(spec.keys())}")


def _register_beer_lambert_alpha(proc_name, alpha_config, spatial, R, U, layer_names):
    """Register per-region alpha callables for a BeerLambert EOP.

    ``alpha_config`` is the dict under ``alpha:`` in the process YAML entry.
    Keys are region names (layer names or ``'domain'``); values are alpha specs
    as parsed by ``_parse_alpha_spec``.

    Priority convention:
    - ``'domain'`` → PRIORITY_DOMAIN (7)  — a device-wide default
    - any layer name → PRIORITY_LAYER (5)  — overrides domain in that layer
    """
    key = f'{proc_name}/alpha_function'
    for region_name, alpha_spec in alpha_config.items():
        if region_name == 'domain':
            region   = R.domain
            priority = PRIORITY_DOMAIN
        elif region_name in layer_names:
            region   = getattr(R, region_name)
            priority = PRIORITY_LAYER
        else:
            raise ValueError(
                f"BeerLambert alpha: unknown region {region_name!r}. "
                f"Valid names: 'domain' or one of {layer_names}.")
        func = _parse_alpha_spec(alpha_spec, U)
        spatial.add_callable_rule(key, region, func, priority=priority)
        logging.getLogger('runner').info(
            f'BeerLambert {proc_name!r}: registered alpha for region {region_name!r}')


def _parse_sigma_spec(spec, U):
    """Parse a YAML sigma spec dict into a callable ``sigma_func(E) -> pint Quantity``.

    Supported keys (mutually exclusive):
    ``top_hat``       — ``{E_low: qty, E_high: qty, value: X, unit: 'cm^2'}``
    ``constant``      — ``{value: X, unit: 'cm^2'}`` — energy-independent sigma
    ``energies_eV``   — inline table paired with ``sigmas`` and optional ``unit``
    ``wavelengths_nm``— inline wavelength table paired with ``sigmas`` and optional ``unit``
    ``table_file``    — path to a two-column CSV (energy_eV, sigma)
    """
    if 'top_hat' in spec:
        th = spec['top_hat']
        E_low  = parse_quantity(th['E_low'],  U)
        E_high = parse_quantity(th['E_high'], U)
        unit   = th.get('unit', 'cm^2')
        sigma_0 = float(th['value']) * U(unit)
        return make_sigma_top_hat(E_low, E_high, sigma_0, sigma_unit=unit)

    elif 'constant' in spec:
        val  = float(spec['constant']['value'])
        unit = spec['constant'].get('unit', 'cm^2')
        def sigma_func(E):
            return dolfin.Constant(val) * U(unit)
        return sigma_func

    elif 'energies_eV' in spec:
        return make_sigma_from_table(
            spec['energies_eV'], spec['sigmas'],
            sigma_unit=spec.get('unit', 'cm^2'))

    elif 'wavelengths_nm' in spec:
        return make_sigma_from_wavelength_table(
            spec['wavelengths_nm'], spec['sigmas'],
            sigma_unit=spec.get('unit', 'cm^2'))

    elif 'table_file' in spec:
        data   = np.loadtxt(spec['table_file'], delimiter=',', comments='#')
        col_e  = int(spec.get('column_energy', 0))
        col_s  = int(spec.get('column_sigma',  1))
        return make_sigma_from_table(
            data[:, col_e], data[:, col_s],
            sigma_unit=spec.get('unit', 'cm^2'))

    else:
        raise ValueError(
            f"Cannot parse sigma spec; expected 'top_hat', 'constant', 'energies_eV', "
            f"'wavelengths_nm', or 'table_file'. Got keys: {list(spec.keys())}")


def _register_ib_beer_lambert_sigma(proc_name, sigma_config, spatial, R, U, layer_names):
    """Register per-region sigma callables for an IBBeerLambert EOP.

    ``sigma_config`` is the dict under ``sigma:`` in the process YAML entry.
    Keys are region names (layer names or ``'domain'``); values are sigma specs
    as parsed by ``_parse_sigma_spec``.
    """
    key = f'{proc_name}/sigma_function'
    for region_name, sigma_spec in sigma_config.items():
        if region_name == 'domain':
            region   = R.domain
            priority = PRIORITY_DOMAIN
        elif region_name in layer_names:
            region   = getattr(R, region_name)
            priority = PRIORITY_LAYER
        else:
            raise ValueError(
                f"IBBeerLambert sigma: unknown region {region_name!r}. "
                f"Valid names: 'domain' or one of {layer_names}.")
        func = _parse_sigma_spec(sigma_spec, U)
        spatial.add_callable_rule(key, region, func, priority=priority)
        logging.getLogger('runner').info(
            f'IBBeerLambert {proc_name!r}: registered sigma for region {region_name!r}')


def _setup_beer_lambert_svr(proc, rr_config, U):
    """Configure the SVR (Shockley-van Roosbroeck) integration grid on a
    BeerLambert EOP instance.

    ``rr_config`` is the value of ``radiative_recombination:`` from the YAML:
    either ``True`` (use defaults) or a dict with optional keys:

    ``integration_bins``  int   — quadrature points (default 100)
    ``E_min``             qty   — lower bound (default 0 eV)
    ``E_max``             qty   — upper bound (default 4 eV)

    The integrand carries a blackbody weight exp(-E/kT) that peaks at the
    absorption threshold, so accuracy is governed by the bin width relative to
    kT.  The 0-4 eV default at 100 bins gives dE/kT = 1.55 at 300 K, wide enough
    that the threshold falls partway through a bin; that bin is the largest in
    the sum and gets counted or dropped whole, which is worth tens of percent in
    either direction depending on where the gap happens to sit.  Put ``E_min``
    *at* the threshold to remove that, and ``E_max`` ~23 kT above it (0.6 eV at
    300 K, more when hotter) to shrink the residual midpoint bias; a silicon
    cell goes from -8.5% to -0.2% at the same cost, and the tail discarded is
    ~1e-8 of the integral.  The process warns when dE/kT exceeds 0.3.
    ``emission_solid_angle``  float or '4pi'/'2pi'  — steradians (default 4π)
    """
    if not isinstance(rr_config, dict):
        rr_config = {}

    N = int(rr_config.get('integration_bins', 100))
    E_min = parse_quantity(rr_config.get('E_min', {'value': 0.0, 'unit': 'eV'}), U)
    E_max = parse_quantity(rr_config.get('E_max', {'value': 4.0, 'unit': 'eV'}), U)

    # Set the process's own public knobs; it builds the quadrature grid itself,
    # so a project written in Python gets the same behaviour without going
    # through YAML.
    proc.svr_integration_bins = N
    proc.svr_E_min = E_min
    proc.svr_E_max = E_max

    E_min_eV = E_min.m_as('eV')
    E_max_eV = E_max.m_as('eV')

    if 'emission_solid_angle' in rr_config:
        sa = rr_config['emission_solid_angle']
        if sa == '4pi':
            proc.svr_solid_angle = 4.0 * np.pi
        elif sa == '2pi':
            proc.svr_solid_angle = 2.0 * np.pi
        else:
            proc.svr_solid_angle = float(sa)

    logging.getLogger('runner').info(
        f"BeerLambert '{proc.name}': SVR enabled, "
        f"N={N} bins, E=[{E_min_eV:.3f}, {E_max_eV:.3f}] eV, "
        f"solid_angle={proc.svr_solid_angle:.4g} sr")


# ===========================================================================
# create_problemdata
# ===========================================================================

def _create_problemdata(config, R, F, U, mesh_data,
                        goal='full', V_ext=None, phi_cn=None,
                        library_dir=None, dark_mode=False):
    """Build and return a :py:class:`~simudo.physics.ProblemData`.

    Parameters
    ----------
    config : dict
        Parsed project YAML.
    R, F : CellRegions, FacetRegions
    U : pint UnitRegistry
    mesh_data : simudo mesh data
    goal : str
        ``'full'``, ``'local charge neutrality'``, or
        ``'thermal equilibrium'``.
    V_ext : pint-wrapped dolfin.Constant, optional
        Applied voltage (required when *goal* is ``'full'``).
    phi_cn : dolfin Function, optional
        LCN potential (required when *goal* is ``'thermal equilibrium'``).
    library_dir : str, Path, or list, optional
        Directory (or list of directories) for ``library://`` material lookups.
    """
    root    = ProblemData(goal=goal, mesh_data=mesh_data, unit_registry=U)
    pdd     = root.pdd
    spatial = pdd.spatial

    # ------------------------------------------------------------------
    # Bands
    # ------------------------------------------------------------------
    band_objects = {}          # name → band object
    band_extents = {}          # name → CellRegion, or None for whole domain
    band_extent_layers = {}    # name → frozenset of layer names, or None for 'all'

    for band_config in config['bands']:
        bname       = band_config['name']
        band_type   = band_config.get('type', None)    # "nondegenerate", "intermediate", …
        band_sign   = band_config.get('sign', 'auto')  # -1, +1, or "auto"
        extent_spec = band_config.get('extent', 'all')

        if extent_spec == 'all':
            subdomain = None
            band_extents[bname] = None
            band_extent_layers[bname] = None          # None → spans full domain
        else:
            if isinstance(extent_spec, str):
                extent_spec = [extent_spec]
            region = None
            for lname in extent_spec:
                r = getattr(R, lname)
                region = r if region is None else (region | r)
            subdomain = region
            band_extents[bname] = region
            band_extent_layers[bname] = frozenset(extent_spec)

        band_obj = pdd.easy_add_band(
            bname,
            band_type = band_type,
            sign      = band_sign,
            subdomain = subdomain,
        )
        band_objects[bname] = band_obj

    # ------------------------------------------------------------------
    # Overlay regions: register before the materials, so that a library
    # material's get_dict() can read them.  A material resolves its spatial
    # keys over the whole mesh, so a key it needs (temperature,
    # MoleFractionX, ...) must already have at least one rule by this point,
    # even if that rule covers only part of the device.
    #
    # Order matters here, but only among equal priorities: rules are kept in
    # a SortedList by priority and `Spatial.get` takes the first rule to claim
    # each cell.  `domain` goes in at PRIORITY_DOMAIN, below layers, so a
    # layer-level value overrides it; named overlays keep PRIORITY_OVERLAY and
    # beat everything.  `domain` is still registered first so that a material's
    # get_dict() can read temperature and friends.  The later overlay pass
    # re-adds these same rules; the duplicates sort after these and never claim
    # a cell.
    # ------------------------------------------------------------------
    _overlays = [c for c in config.get('regions', []) if isinstance(c, dict)]
    for ovl_config in (sorted(_overlays, key=lambda c: c.get('name') != 'domain')):
        props = ovl_config.get('properties') or {}
        if not props:
            continue
        if ovl_config.get('name') == 'domain':
            ovl_region = R.domain
        else:
            ovl_extent = ovl_config.get('extent', [])
            if not isinstance(ovl_extent, list):
                continue          # coordinate extents handled in the later pass
            ovl_region = None
            for lname in ovl_extent:
                r = getattr(R, lname)
                ovl_region = r if ovl_region is None else (ovl_region | r)
            if ovl_region is None:
                continue
        _prio = (PRIORITY_DOMAIN if ovl_config.get('name') == 'domain'
                 else PRIORITY_OVERLAY)
        for key, entry in props.items():
            qty = _coerce_quantity(key, parse_spatial_quantity(entry, U), U)
            spatial.add_rule(key, ovl_region, qty, priority=_prio)

    # ------------------------------------------------------------------
    # Temperature fallback.
    #
    # This has to happen here, after the overlays and before the materials: a
    # material's get_dict() reads `temperature` while computing band gaps and
    # densities of states, so the rule must already exist by then.  It goes in
    # at PRIORITY_MATERIAL, which is below everything a project can say, so any
    # temperature the project does set -- on a layer, a named overlay or
    # `domain` -- still wins.
    # ------------------------------------------------------------------
    if 'temperature' not in spatial.value_rules:
        spatial.add_rule('temperature', R.domain,
                         DEFAULT_TEMPERATURE_K * U.K,
                         priority=PRIORITY_MATERIAL)
        logging.getLogger('runner').info(
            'No temperature set in any region; assuming %g K.',
            DEFAULT_TEMPERATURE_K)

    # ------------------------------------------------------------------
    # Materials: register at PRIORITY_MATERIAL
    # ------------------------------------------------------------------
    materials_config = config.get('materials', {})

    # Per-material flags for runner-side post-processing
    # (currently only IB/simple_mobility_model)
    material_flags = {}   # mat_name → {'simple_mobility_model': bool}

    for mat_name, mat_def in materials_config.items():
        if isinstance(mat_def, dict) and 'source' in mat_def:
            # Library reference or derived material
            mat_cls = _load_library_material_class(
                mat_def['source'], library_dir=library_dir)
            mat_obj      = mat_cls(problem_data=root)
            mat_obj.name = mat_name
            mat_dict     = mat_obj.get_dict()
            _register_material_dict(spatial, mat_name, mat_dict, U,
                                    priority=PRIORITY_MATERIAL)

            # Derived material: apply inline property overrides at a higher
            # priority so they win over the library defaults.
            # PRIORITY_MATERIAL - 1 = 9 (lower number → higher priority).
            overrides = mat_def.get('properties') or {}
            if overrides:
                # Compute the union of all layers referencing this material.
                override_region = None
                for lc in config.get('layers', []):
                    if lc.get('material') == mat_name:
                        lr = getattr(R, lc['name'])
                        override_region = lr if override_region is None else (override_region | lr)
                if override_region is not None:
                    for key, entry in overrides.items():
                        qty = _coerce_quantity(
                            key, parse_spatial_quantity(entry, U), U)
                        spatial.add_rule(key, override_region, qty,
                                         priority=PRIORITY_MATERIAL - 1)
        else:
            # Collect runner-handled flags before building the spatial dict
            simple_mob = mat_def.get('IB/simple_mobility_model', False)
            if simple_mob:
                material_flags[mat_name] = {'simple_mobility_model': True}

            mat_dict = _build_material_dict(mat_def, U)
            _register_material_dict(spatial, mat_name, mat_dict, U,
                                    priority=PRIORITY_MATERIAL)

    # ------------------------------------------------------------------
    # Layer-specific properties: register at PRIORITY_LAYER
    # ------------------------------------------------------------------
    for layer_config in config['layers']:
        lname    = layer_config['name']
        region   = getattr(R, lname)
        props    = layer_config.get('properties', {})
        for key, entry in props.items():
            if key in _RUNNER_HANDLED_KEYS:
                continue
            qty = _coerce_quantity(key, parse_spatial_quantity(entry, U), U)
            spatial.add_rule(key, region, qty, priority=PRIORITY_LAYER)

    # ------------------------------------------------------------------
    # Overlay regions: register at PRIORITY_OVERLAY
    # ------------------------------------------------------------------
    for ovl_config in config.get('regions', []):
        if not isinstance(ovl_config, dict) or not ovl_config:
            continue
        props = ovl_config.get('properties', {})
        if not props:
            continue
        if ovl_config.get('name') == 'domain':
            # Domain overlay: applies to the entire mesh domain
            ovl_region = R.domain
        else:
            ovl_extent = ovl_config.get('extent', [])
            if isinstance(ovl_extent, list):
                ovl_region = None
                for lname in ovl_extent:
                    r = getattr(R, lname)
                    ovl_region = r if ovl_region is None else (ovl_region | r)
            else:
                raise NotImplementedError(
                    "Overlay extent specified by coordinate range is not yet "
                    "implemented.  Use a list of layer names.")
        _prio = (PRIORITY_DOMAIN if ovl_config.get('name') == 'domain'
                 else PRIORITY_OVERLAY)
        for key, entry in props.items():
            qty = _coerce_quantity(key, parse_spatial_quantity(entry, U), U)
            spatial.add_rule(key, ovl_region, qty, priority=_prio)

    # ------------------------------------------------------------------
    # IB band attribute: use_constant_mobility
    # ------------------------------------------------------------------
    for layer_config in config['layers']:
        mat_name = layer_config.get('material', '')
        layer_props = layer_config.get('properties', {})
        layer_simple = layer_props.get('IB/simple_mobility_model', False)
        if isinstance(layer_simple, dict):
            layer_simple = bool(layer_simple.get('value', False))
        if material_flags.get(mat_name, {}).get('simple_mobility_model', False) or layer_simple:
            ib_band = band_objects.get('IB')
            if ib_band is not None:
                ib_band.use_constant_mobility = True

    # ------------------------------------------------------------------
    # Mesh utilities and reusable zero vectors
    # ------------------------------------------------------------------
    mu     = pdd.mesh_util
    zero_j = U('A/cm^2') * mu.zerovec
    zeroE  = F.exterior

    # ------------------------------------------------------------------
    # Goal-specific setup
    # ------------------------------------------------------------------
    if goal == 'full':
        # ---- Zero-current BCs for all band currents --------------------
        # Bands confined to a sub-region: zero current at extent boundary
        for bname, extent_region in band_extents.items():
            if extent_region is None:
                continue
            extent_bounds = (
                extent_region.boundary(R.domain - extent_region)
                | extent_region.boundary(R.exterior)
            )
            spatial.add_BC(f'{bname}/j',
                           F.nonconductive | extent_bounds, zero_j)

        # Full-domain bands: zero current on non-conductive exterior faces
        for bname, extent_region in band_extents.items():
            if extent_region is not None:
                continue
            spatial.add_BC(f'{bname}/j', F.nonconductive, zero_j)

        # ---- Optical fields ----------------------------------------
        optical  = root.optical
        ospatial = optical.spatial

        _registered_optical_names = set()

        for field_config in ([] if dark_mode else config.get('optical_fields', [])):
            fname     = field_config['name']
            direction = field_config.get('direction', '+x')
            photon_e  = parse_quantity(field_config['photon_energy'], U)
            d_vec     = _direction_vector(direction, _optical_gdim(optical))

            if fname in _registered_optical_names:
                raise ValueError(f"Duplicate optical field name: {fname!r}")
            _registered_optical_names.add(fname)

            optical.easy_add_field(fname, photon_energy=photon_e,
                                   direction=d_vec)

            flux         = _compute_optical_flux(field_config, U)
            flux_contact = _flux_contact(direction, F)
            ospatial.add_BC(f'{fname}/Phi', flux_contact, flux)
            logging.getLogger('runner').info(
                f'Optical field {fname!r}: flux = {flux}')

        for fset_config in ([] if dark_mode else config.get('optical_field_sets', [])):
            _expand_field_set(fset_config, U, optical, ospatial, F,
                              _registered_optical_names)

        # ---- Electro-optical processes (EOPs) --------------------------
        for proc_config in config.get('processes', []):
            cls_name  = proc_config['class']
            proc_name = proc_config['name']
            eop_cls   = _get_eop_class(cls_name)

            kwargs = dict(name=proc_name)
            for key in ('dst_band', 'src_band', 'trap_band'):
                if key in proc_config:
                    kwargs[key] = band_objects[proc_config[key]]

            proc = pdd.easy_add_electro_optical_process(eop_cls, **kwargs)

            if cls_name == 'BeerLambert':
                # Register per-region alpha callables
                if 'alpha' in proc_config:
                    _register_beer_lambert_alpha(
                        proc_name, proc_config['alpha'],
                        spatial, R, U,
                        [l['name'] for l in config['layers']])
                # Radiative recombination (Shockley-van Roosbroeck).  Default
                # True, matching BeerLambert.enable_radiative_recombination:
                # spontaneous emission is always present physically, and Simudo
                # has no photon recycling, so leaving it off silently drops a
                # real loss.  Set `radiative_recombination: false` to disable.
                rr = proc_config.get('radiative_recombination', True)
                if rr:
                    proc.enable_radiative_recombination = True
                    _setup_beer_lambert_svr(proc, rr, U)
                else:
                    proc.enable_radiative_recombination = False

            elif cls_name == 'IBBeerLambert':
                layer_names = [l['name'] for l in config['layers']]
                # Register per-region sigma callables
                if 'sigma' in proc_config:
                    _register_ib_beer_lambert_sigma(
                        proc_name, proc_config['sigma'],
                        spatial, R, U, layer_names)
                # Radiative recombination — enabled by default; configure SVR grid
                rr = proc_config.get('radiative_recombination', True)
                if rr:
                    proc.enable_radiative_recombination = True
                    _setup_beer_lambert_svr(proc, rr if isinstance(rr, dict) else {}, U)
                else:
                    proc.enable_radiative_recombination = False
            else:
                # Honour radiative_recombination flag for other EOP classes.
                # AbsorptionAndRadiativeRecombinationEOPMixin exposes this flag.
                if 'radiative_recombination' in proc_config:
                    if hasattr(proc, 'enable_radiative_recombination'):
                        proc.enable_radiative_recombination = bool(
                            proc_config['radiative_recombination'])

        # ---- Contact band BCs ------------------------------------------
        contacts_config  = config.get('contacts', {})
        contact_face_map = {
            'left':  F.left_contact,
            'right': F.right_contact,
        }

        # For each extent-limited band, determine whether it actually reaches
        # the left and/or right device contact.  In a layered 1D structure the
        # left contact is at the boundary of the first layer and the right
        # contact is at the boundary of the last layer, so it suffices to check
        # whether those layers are in the band's extent.  If a band's extent
        # does NOT reach a contact we must not register a BC for it there:
        # registering IB/u at a contact outside the band's domain causes
        # rebase_w to evaluate thermal_equilibrium_u in cells with no IB DOFs,
        # injecting NaN into the solution before the first Newton assembly.
        layer_names_ordered = [l['name'] for l in config['layers']]
        first_layer = layer_names_ordered[0]
        last_layer  = layer_names_ordered[-1]

        def _band_reaches(bname, contact_key):
            """True iff band *bname* physically reaches the named contact."""
            ext_layers = band_extent_layers.get(bname)
            if ext_layers is None:          # full-domain band
                return True
            if contact_key == 'left':
                return first_layer in ext_layers
            else:                           # 'right'
                return last_layer in ext_layers

        for contact_key, contact_config in contacts_config.items():
            face  = contact_face_map[contact_key]
            bands_explicit = contact_config.get('bands', {})
            # Default full-domain bands to 'ohmic' at contacts when not
            # explicitly listed.  Extent-limited bands that do not reach this
            # contact get no default BC (see _band_reaches above).
            bands_resolved = {bname: 'ohmic'
                              for bname in band_objects
                              if _band_reaches(bname, contact_key)
                              and band_extents.get(bname) is None}
            bands_resolved.update(bands_explicit)
            for bname, bc_spec in bands_resolved.items():
                band_obj = band_objects.get(bname)
                if band_obj is None:
                    logging.getLogger('runner').warning(
                        f'Contact {contact_key!r} references unknown band '
                        f'{bname!r} — skipping')
                    continue
                # bc_spec may be a plain string ('ohmic', 'zero_current') or a
                # dict with a 'type' key for parameterised BCs such as
                # surface_recombination.
                if isinstance(bc_spec, dict):
                    bc_type = bc_spec['type']
                else:
                    bc_type = bc_spec
                if bc_type == 'ohmic':
                    if not _band_reaches(bname, contact_key):
                        # Band is extent-limited and does not reach this
                        # contact.  Registering a BC here would inject NaN
                        # via rebase_w (see fourlayer_example_nan_analysis.md).
                        logging.getLogger('runner').warning(
                            f'Contact {contact_key!r} band {bname!r}: ohmic BC '
                            f'requested but band extent does not reach this '
                            f'contact — skipping to prevent NaN injection')
                    else:
                        spatial.add_BC(f'{bname}/u', face,
                                       band_obj.thermal_equilibrium_u)
                elif bc_type == 'zero_current':
                    spatial.add_BC(f'{bname}/j', face, zero_j)
                elif bc_type == 'surface_recombination':
                    # Robin BC: j·n = srv * q * (u - u_eq).
                    # Imposed as a weak-form surface integral; no essential BC
                    # on j or u is registered (that would override this term).
                    # Optional 'method' selects the weak-form placement:
                    #   'drift_diffusion'     (option 2, default) — ξ-row natural BC, recommended
                    #   'continuity_corrected'                    — v-row, double-count fixed
                    #   'continuity'          (option 1)          — v-row naive, does NOT converge
                    srv = parse_quantity(bc_spec['srv'], U)
                    srv_method = bc_spec.get('method', 'drift_diffusion')
                    SurfaceRecombination(band=band_obj, boundary=face,
                                         srv=srv, method=srv_method).register()
                else:
                    raise ValueError(
                        f'Unknown band BC type {bc_type!r} at contact '
                        f'{contact_key!r} band {bname!r}')

        # ---- Poisson BCs -----------------------------------------------
        phi0       = pdd.poisson.thermal_equilibrium_phi
        swept_face = ref_face = None
        ref_voltage = 0.0 * U.V

        for contact_key, contact_config in contacts_config.items():
            role = contact_config.get('role', 'reference')
            face = contact_face_map[contact_key]
            if role == 'swept':
                swept_face = face
            elif role == 'reference':
                ref_face    = face
                ref_voltage = parse_quantity(
                    contact_config.get('voltage',
                                       {'value': 0.0, 'unit': 'V'}), U)

        if swept_face is not None:
            if V_ext is None:
                raise ValueError("V_ext must be provided for goal='full'")
            spatial.add_BC('poisson/phi', swept_face, phi0 + V_ext)
            zeroE -= swept_face.both()
        if ref_face is not None:
            spatial.add_BC('poisson/phi', ref_face, phi0 + ref_voltage)
            zeroE -= ref_face.both()

    elif goal == 'thermal equilibrium':
        all_contacts = F.left_contact | F.right_contact
        spatial.add_BC('poisson/phi', all_contacts, phi_cn)
        zeroE -= all_contacts.both()

    # Zero electric field on all remaining exterior facets
    spatial.add_BC('poisson/E', zeroE, U('V/m') * mu.zerovec)

    # ---- Internal interface BCs (ThermionicHeterojunction etc.) ------------
    # Only on the 'full' goal.  The 'local charge neutrality' and 'thermal
    # equilibrium' problems build bands without the mixedqfl machinery, so
    # band.mixedqfl_xi -- which ThermionicHeterojunction.register() reads --
    # does not exist and registering there raises AttributeError.
    # example/heterojunction/het1d_example.py guards the same way.
    _log = logging.getLogger('runner')
    for iface in (config.get('interfaces', []) if goal == 'full' else []):
        left  = iface.get('left', '')
        right = iface.get('right', '')
        if not left or not right:
            _log.warning('Interface entry missing left/right names — skipping')
            continue
        facet = getattr(R, left).boundary(getattr(R, right))
        for bc_spec in iface.get('bcs', []):
            bc_type = bc_spec.get('type', '')
            if bc_type == 'ThermionicHeterojunction':
                enh_raw = bc_spec.get('HJBC_enhancement', {})
                if isinstance(enh_raw, dict):
                    enh_dict = {k: float(v) for k, v in enh_raw.items()}
                else:
                    # Backwards compat: scalar applies to all bands equally
                    scalar = float(enh_raw) if enh_raw else 1.0
                    enh_dict = {b: scalar for b in bc_spec.get('bands', [])}
                for bname in bc_spec.get('bands', []):
                    band_obj = band_objects.get(bname)
                    if band_obj is None:
                        _log.warning(
                            f'Interface {left}|{right}: unknown band {bname!r} '
                            f'for ThermionicHeterojunction — skipping')
                        continue
                    enh = enh_dict.get(bname, 1.0)
                    ThermionicHeterojunction(
                        band_obj, facet,
                        HJBC_enhancement=dolfin.Constant(enh),
                    ).register()
                    _log.info(
                        f'Registered ThermionicHeterojunction: band={bname} '
                        f'{left}|{right} enhancement={enh}')
            else:
                _log.warning(f'Interface {left}|{right}: unknown BC type '
                             f'{bc_type!r} — skipping')

    return root


# ===========================================================================
# Problem construction
# ===========================================================================

class ProblemBuilder:
    """Everything a project config needs before solving: units, mesh, topology.

    ``_create_problemdata`` takes ``(config, R, F, U, mesh_data, ...,
    dark_mode=...)`` on every call, and :py:func:`run` builds up to four
    problems from the same five objects.  This class owns that bundle so it is
    constructed once, and so callers other than :py:func:`run` -- notably the
    test suite -- can build a problem **without running a simulation**.

    Constructing a builder has no side effects outside dolfin's global
    parameters: it creates no directories, writes no files, configures no
    logging, and solves nothing.

    Parameters
    ----------
    config : dict
        Parsed project YAML.
    library_dir : str, Path, or list, optional
        Passed through to ``library://`` material resolution.
    setup_dolfin : bool, optional
        Call :py:func:`setup_dolfin_parameters`.  On by default; pass ``False``
        when the caller has already configured dolfin.

    Examples
    --------
    Check that a configuration assembles, without solving it::

        builder = ProblemBuilder(config, library_dir=materials)
        problem = builder.problem('full')
    """

    def __init__(self, config, library_dir=None, setup_dolfin=True):
        if setup_dolfin:
            setup_dolfin_parameters()

        self.config = config
        self.library_dir = library_dir

        # Unit registry (mesh_unit = 1 micrometer)
        self.U = make_unit_registry(('mesh_unit = 1 micrometer',))

        self.mesh_data = _build_mesh(config, self.U)

        self.R = CellRegions()
        self.F = FacetRegions()
        _build_topology(config, self.R, self.F)

        # Dark J-V mode: a voltage sweep with no optical ramp excludes the
        # optical fields entirely.
        sim_config = config.get('simulation', {})
        do_iramp = sim_config.get('intensity_ramp', {}).get('enabled', True)
        do_vsweep = sim_config.get('voltage_sweep', {}).get('enabled', True)
        self.dark_mode = do_vsweep and not do_iramp

    def problem(self, goal, **kwargs):
        """Build the :py:class:`ProblemData` for *goal*.  Does not solve.

        *kwargs* are forwarded to ``_create_problemdata`` -- ``V_ext`` for the
        ``'full'`` goal, ``phi_cn`` for ``'thermal equilibrium'``.
        """
        return _create_problemdata(
            self.config, self.R, self.F, self.U, self.mesh_data,
            goal=goal, library_dir=self.library_dir,
            dark_mode=self.dark_mode, **kwargs)


def build_problems(config, library_dir=None, goals=('full',), V_ext=None):
    """Build the problems named in *goals* without solving any of them.

    Returns ``(builder, {goal: problem})``.

    This is the entry point for tests that need to know whether a
    configuration *assembles* -- which is where a large class of runner bugs
    lives.  ``ThermionicHeterojunction`` was registered on every goal for
    months, and raised ``AttributeError`` on the ``'local charge neutrality'``
    problem long before any solve; a construction-only check would have caught
    it in about a second.

    ``'thermal equilibrium'`` needs a ``phi_cn`` from the local-charge-neutrality
    problem.  That is a dolfin ``Function`` which exists whether or not it has
    been solved for, so the dependency is satisfied here by building the LCN
    problem and handing over its (unsolved) ``phi``.  The resulting equilibrium
    problem is therefore well-formed but not physically meaningful -- which is
    all a construction test needs.
    """
    builder = ProblemBuilder(config, library_dir=library_dir)
    problems = {}
    for goal in goals:
        kwargs = {}
        if goal == 'full':
            kwargs['V_ext'] = (
                builder.U.V * dolfin.Constant(0.0) if V_ext is None else V_ext)
        elif goal == 'thermal equilibrium':
            lcn = problems.get('local charge neutrality')
            if lcn is None:
                lcn = builder.problem('local charge neutrality')
            kwargs['phi_cn'] = lcn.pdd.poisson.phi
        problems[goal] = builder.problem(goal, **kwargs)
    return builder, problems


# ===========================================================================
# Checkpoint resume helpers
# ===========================================================================

def _parse_checkpoint_stage(checkpoint_yaml_path):
    """Identify which stepper stage a checkpoint belongs to, and its value.

    Checkpoint files are named ``<prefix>_<ParamName>=<value>.yaml`` by
    :py:meth:`~simudo.fem.adaptive_stepper.AdaptiveStepper.write_xdmf_checkpoint`
    (e.g. ``sim_V=0.4.yaml`` or ``sim_I=1.yaml``).  The parameter name is not
    stored in the metadata itself, so it is parsed from the filename; the
    parameter value is read from the metadata (more precise than the
    filename's truncated representation).

    Returns
    -------
    (param_name, param_value) : (str, float)
        ``param_name`` is ``'V'`` or ``'I'`` (whatever the stepper that wrote
        the checkpoint used as its ``parameter_name``).
    """
    basename = os.path.basename(checkpoint_yaml_path)
    m = re.search(r'_([A-Za-z]+)=[^=/\\]+\.yaml$', basename)
    if not m:
        raise ValueError(
            f'Cannot determine which stage checkpoint {basename!r} belongs '
            f'to — expected a filename like "sim_V=0.4.yaml" or '
            f'"sim_I=1.yaml" (written by AdaptiveStepper.write_xdmf_checkpoint).')
    param_name = m.group(1)
    metadata = h5yaml.load(os.path.splitext(checkpoint_yaml_path)[0] + '.yaml')
    return param_name, float(metadata['parameter'])


def _already_computed_values(prefix, parameter_name):
    """Return the set of *parameter_name* values already recorded in
    ``<prefix>/sim_<parameter_name>.csv``.

    Used so a resumed run doesn't recompute (and re-append a duplicate row
    for) a point that a prior run in this same output folder already solved
    -- e.g. resuming from a V=0.4 checkpoint with target list
    ``[0, 0.1, 0.2, 0.3, 0.4, 0.5]`` should only compute 0.5; 0..0.4 are
    already in ``sim_V.csv`` from the run that produced the checkpoint.
    """
    csv_path = Path(prefix) / f'sim_{parameter_name}.csv'
    if not csv_path.exists():
        return set()
    try:
        with open(csv_path, newline='', encoding='utf-8') as fh:
            lines = fh.readlines()
    except OSError:
        return set()
    if not lines:
        return set()
    headers = [h.strip() for h in lines[0].split(',')]
    col = f'sweep_parameter:{parameter_name}'
    if col not in headers:
        return set()
    idx = headers.index(col)
    values = set()
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if idx < len(parts):
            try:
                values.add(float(parts[idx]))
            except ValueError:
                pass
    return values


def _prune_computed(targets, computed):
    """Drop values from *targets* already present in *computed* (within tolerance)."""
    if not computed:
        return list(targets)
    return [v for v in targets
            if not any(math.isclose(v, c, rel_tol=1e-9, abs_tol=1e-9) for c in computed)]


def _load_checkpoint_into(stepper_cls, solution, checkpoint_yaml, **stepper_kwargs):
    """Load *checkpoint_yaml*'s saved solution fields into *solution*.

    This performs just the "load" half of what
    :py:meth:`AdaptiveStepper.do_loop` does when ``checkpoint_reload_yaml`` is
    set, without running any solver iterations — used when a checkpoint's
    state needs to be in place *before* it can be copied into another
    ``ProblemData`` instance (e.g. seeding the negative-voltage-sweep problem
    from a checkpoint taken during the positive sweep).
    """
    stepper = stepper_cls(solution=solution, **stepper_kwargs)
    stepper.solver = stepper.user_make_solver(stepper.solution)
    stepper.checkpoint_reload_yaml = checkpoint_yaml
    stepper.load_xdmf_checkpoint()
    return stepper


# ===========================================================================
# Top-level runner
# ===========================================================================

def run(yaml_path, library_dir=None, new_experiment=True):
    """Read the project YAML at *yaml_path* and execute the simulation.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the project YAML file.
    library_dir : str, Path, or list, optional
        Directory (or list of directories) containing Python Material
        subclasses for ``library://`` material sources.
    new_experiment : bool, optional
        When ``True`` (default), the output directory is checked for
        existence and incremented if necessary (``ensure_new_dir``), so
        that each new run gets its own folder.  Set to ``False`` when
        continuing from a checkpoint so that output lands back in the
        same folder as the original run.
    """
    yaml_path = Path(yaml_path).resolve()
    yaml_dir  = yaml_path.parent
    with open(yaml_path, encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    # ------------------------------------------------------------------
    # Output directory
    # Relative paths are resolved against the YAML file's directory so
    # that output always lands near the project file, regardless of the
    # working directory from which the runner is invoked.
    # ------------------------------------------------------------------
    out_config = config.get('output', {})
    folder_raw = out_config.get('folder', None)

    def _resolve(p):
        """Make *p* absolute, anchoring relative paths to yaml_dir."""
        p = Path(p)
        return p if p.is_absolute() else yaml_dir / p

    if folder_raw:
        resolved = str(_resolve(folder_raw))
        PREFIX = Path(ensure_new_dir(resolved) if new_experiment else resolved)
    else:
        PREFIX = Path(standard_outdir(yaml_dir))
    PREFIX.mkdir(parents=True, exist_ok=True)
    print(f'Remote output dir: {PREFIX}', flush=True)

    # Copy the project YAML into the experiment directory for provenance,
    # with output.folder corrected to "." so this frozen copy correctly
    # records its own location -- it lives inside the very folder it names,
    # which also happens to be exactly where a project resumed from a
    # checkpoint in this folder should save itself back to (no extra path
    # bookkeeping needed on the GUI side).
    frozen_path = PREFIX / yaml_path.name
    try:
        frozen_config = dict(config)
        frozen_config['output'] = dict(frozen_config.get('output', {}))
        frozen_config['output']['folder'] = '.'
        with open(frozen_path, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(frozen_config, fh, sort_keys=False)
    except Exception:
        pass

    # Also fix up the *live* project YAML (wherever the user has it open) so
    # its stored output.folder reflects the actual folder this run used,
    # rather than silently diverging from it after ensure_new_dir increments
    # (e.g. out/a -> out/b). Skipped when yaml_path already *is* the frozen
    # copy above (resuming a project whose filepath is inside the run folder
    # itself) -- same file, already handled by the write above.
    try:
        if yaml_path.resolve() != frozen_path.resolve():
            actual_rel = os.path.relpath(PREFIX, yaml_dir)
            if config.get('output', {}).get('folder') != actual_rel:
                live_config = dict(config)
                live_config['output'] = dict(live_config.get('output', {}))
                live_config['output']['folder'] = actual_rel
                with open(yaml_path, 'w', encoding='utf-8') as fh:
                    yaml.safe_dump(live_config, fh, sort_keys=False)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Output file prefix  (replaces the hard-coded "pd" in fourlayer.py)
    # ------------------------------------------------------------------
    file_prefix = out_config.get('prefix', 'sim')
    filename_prefix = str(PREFIX / file_prefix)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logsetup = TypicalLoggingSetup(filename_prefix=str(PREFIX) + os.sep)
    logsetup.delta_time = True
    logsetup.setup()

    # Suppress verbose Newton / stepper console output
    console_filter = logsetup.stream_console.filters[-1]
    for noisy_logger in ('newton.optical', 'newton.ntrl', 'newton.thmq',
                         'stepper'):
        console_filter.name_levelno_rules.insert(0, (noisy_logger, logging.ERROR))
    # NOTE: 'newton' left at INFO to diagnose convergence

    log = logging.getLogger('runner')
    log.info(f'Project: {yaml_path}')
    log.info(f'Output:  {PREFIX}')

    # Wall-clock timing of the main stages, reported at the end (and useful
    # for comparing the true-1D and 2D-strip mesh routes).
    _t_start = time.perf_counter()
    _timings = []

    def _mark(stage, t0):
        dt = time.perf_counter() - t0
        _timings.append((stage, dt))
        log.info(f'Timing: {stage} took {dt:.2f} s')

    # ------------------------------------------------------------------
    # Dolfin setup, unit registry, mesh and topology
    # ------------------------------------------------------------------
    _t0 = time.perf_counter()
    builder = ProblemBuilder(config, library_dir=library_dir)
    U    = builder.U
    R, F = builder.R, builder.F
    _mark('mesh+topology', _t0)

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    sim_config  = config.get('simulation', {})
    iramp_config  = sim_config.get('intensity_ramp', {})
    vsweep_config = sim_config.get('voltage_sweep', {})

    do_iramp  = iramp_config.get('enabled', True)
    do_vsweep = vsweep_config.get('enabled', True)
    # Dark J-V mode: voltage sweep with no optical ramp → exclude optical fields
    dark_mode = builder.dark_mode

    iramp_step = float(iramp_config.get('step_size', 1e-25))
    iramp_sc   = iramp_config.get('selfconsistent_optics', True)

    vsweep_raw  = list(vsweep_config.get('values', [0.0]))
    vsweep_step = float(vsweep_config.get('step_size', 1e-4))
    vsweep_sc   = vsweep_config.get('selfconsistent_optics', True)

    # ------------------------------------------------------------------
    # Checkpoint configuration
    # ------------------------------------------------------------------
    checkpoint_config   = sim_config.get('checkpoints', {})
    checkpoint_dir      = str(PREFIX / checkpoint_config.get('directory', 'checkpoints'))
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _checkpoint_values(stage_key, end_value):
        """Build the list of parameter values at which to write checkpoints.

        The stepper accepts a ``checkpoint_write_values`` list of parameter
        values (e.g. intensity fractions or voltages).  When the stepper
        reaches one of those values it writes an XDMF + HDF5 + YAML triplet
        that can later be used to resume the run.

        This helper reads the YAML configuration for *stage_key* (either
        ``'intensity_ramp'`` or ``'voltage_sweep'``) and assembles the list
        from two sources:

        * ``at_values`` — explicit intermediate checkpoints specified by the
          user (e.g. ``[0.8, 1.2]`` to checkpoint at 0.8 V and 1.2 V during
          the voltage sweep).
        * If ``checkpoint_at_end: true``, *end_value* is appended
          (e.g. the final intensity of 1.0, or the last voltage in the list).
        """
        stage_cfg = checkpoint_config.get(stage_key, {})
        vals = list(stage_cfg.get('at_values', []) or [])
        if stage_cfg.get('checkpoint_at_end', False):
            vals.append(end_value)
        return vals

    # resume_from is a path *relative to the output folder*
    resume_from = sim_config.get('resume_from', None)
    resume_yaml = str(PREFIX / resume_from) if resume_from else None

    # resume_stage tells the three stages below which one of them the
    # checkpoint applies to:
    #   'optical' — checkpoint from the intensity-ramp stage (I, at V=0)
    #   'up_v'    — checkpoint from the positive voltage-sweep branch
    #   'down_v'  — checkpoint from the negative voltage-sweep branch
    resume_stage = None
    resume_param_value = None
    if resume_yaml is not None:
        resume_param_name, resume_param_value = _parse_checkpoint_stage(resume_yaml)
        if resume_param_name == 'I':
            resume_stage = 'optical'
        elif resume_param_name == 'V':
            resume_stage = 'up_v' if resume_param_value >= 0 else 'down_v'
        else:
            raise ValueError(
                f'Checkpoint {resume_yaml!r} has unrecognized parameter '
                f'name {resume_param_name!r} (expected "V" or "I").')
        log.info(f'Resuming from checkpoint: {resume_param_name}='
                 f'{resume_param_value:.6g}  (stage={resume_stage}, {resume_yaml})')

    # ------------------------------------------------------------------
    # Voltage targets, split around an initial_v: 0 for a fresh run or an
    # optical-ramp resume (the sweep starts from V=0), or the checkpoint's
    # own value for a voltage-branch resume (V >= 0 for 'up_v', V < 0
    # for 'down_v'). Splitting -- rather than only ever continuing
    # outward in the checkpoint's own direction -- lets a resume from V=0.4
    # with targets [0.3, 0.4, 0.5] walk both back to 0.3 and on to 0.5, each
    # via its own ProblemData copy seeded from the same loaded V=0.4 state
    # (see "up"/"down" problem setup below).
    initial_v = 0.0 if resume_stage in (None, 'optical') else resume_param_value
    already_v = _already_computed_values(PREFIX, 'V')

    # initial_v itself needs an explicit target (and so an output row) only if
    # it isn't already on record: a resumed checkpoint's own value always
    # is (whatever run wrote the checkpoint also wrote an output row for
    # that same point), but a fresh run's V=0 equilibrium point is not yet
    # in sim_V.csv and must appear in the curve. Added to "up" (and to
    # "down" too, but only when a down sweep is happening anyway) rather
    # than solved+written standalone, reusing the stepper's own
    # solve-and-write path instead of a bespoke one.
    initial_v_needs_anchor = not any(
        math.isclose(initial_v, c, rel_tol=1e-9, abs_tol=1e-9) for c in already_v)

    up_targets = sorted(v for v in vsweep_raw if v > initial_v)
    if initial_v_needs_anchor:
        up_targets = sorted(set(up_targets) | {initial_v})
    down_targets = sorted((v for v in vsweep_raw if v < initial_v), reverse=True)
    if initial_v_needs_anchor and down_targets:
        down_targets = sorted(set(down_targets) | {initial_v}, reverse=True)

    up_targets   = _prune_computed(up_targets, already_v)
    down_targets = _prune_computed(down_targets, already_v)

    # ------------------------------------------------------------------
    # Three-stage initialisation
    # ------------------------------------------------------------------
    V_ext        = U.V * dolfin.Constant(initial_v)
    full_problem = builder.problem('full', V_ext=V_ext)

    if resume_yaml is None:
        _t0 = time.perf_counter()
        lcn_problem = builder.problem('local charge neutrality')
        lcn_problem.pdd.easy_auto_pre_solve()
        _mark('local charge neutrality', _t0)

        _t0 = time.perf_counter()
        eqm_problem = builder.problem(
            'thermal equilibrium', phi_cn=lcn_problem.pdd.poisson.phi)
        eqm_problem.pdd.initialize_from(lcn_problem.pdd)
        eqm_problem.pdd.easy_auto_pre_solve()
        _mark('thermal equilibrium', _t0)

        full_problem.pdd.initialize_from(eqm_problem.pdd)
    elif resume_stage == 'up_v':
        # Load directly into full_problem now (rather than lazily via
        # checkpoint_reload_yaml on its VoltageStepper below) so the state
        # is available immediately -- both for that stepper's own "up" sweep
        # and, if needed, to seed full_problem_neg's "down" sweep from it.
        _load_checkpoint_into(
            VoltageStepper, full_problem, resume_yaml,
            constants=[V_ext], parameter_unit=U.V, selfconsistent_optics=False)
    # resume_stage == 'optical': full_problem is loaded by its own resuming
    # OpticalIntensityAdaptiveStepper in Stage 1 below.
    # resume_stage == 'down_v': full_problem is not populated here --
    # the checkpoint belongs to the down branch (full_problem_neg, set up
    # further below); full_problem only gets seeded from that if an "up"
    # sweep is also needed.

    # ------------------------------------------------------------------
    # Output writer meta-extractors
    # ------------------------------------------------------------------
    out_sim  = sim_config.get('output', {})
    plot_1d  = out_sim.get('spatial_profiles', True)
    # 'xdmf_mesh' drives the *full* output (DG fields of every quantity),
    # because that is what gui/spatial_extractor.py reads for the band diagram
    # and the spatial plots.  The plain per-voltage .xdmf is not written: the
    # same fields are already saved with each checkpoint, and writing both
    # doubles the output for no gain.
    plot_xdmf = out_sim.get('xdmf_mesh', True)

    layer_names = [layer['name'] for layer in config['layers']]

    meta_extractors = (
        MetaExtractorBandInfo,
        partial(
            MetaExtractorIntegrals,
            facets={k: F[k] for k in ('left_contact', 'right_contact')},
            cells={k: R[k] for k in layer_names},
        ),
    )

    # ------------------------------------------------------------------
    # Stage 1: optical intensity ramp (0 → 1)
    # ------------------------------------------------------------------
    has_optics = bool(config.get('optical_fields', []) or
                      config.get('optical_field_sets', []))

    checkpoint_values_opt   = _checkpoint_values('intensity_ramp', 1)
    checkpoint_filename_opt = (str(Path(checkpoint_dir) / file_prefix)
                               if checkpoint_values_opt else None)

    # Resuming an optical-ramp checkpoint continues the ramp in-place;
    # otherwise (not resuming) it runs the ramp fresh from I=0. A checkpoint
    # from a *voltage*-sweep stage means the ramp already completed in an
    # earlier run, so it is skipped here.
    if has_optics and do_iramp and resume_stage in (None, 'optical'):
        if resume_stage == 'optical':
            opt_start   = resume_param_value
            # Strictly beyond opt_start (not prefixed with it) — do_loop()
            # loads the checkpoint unconditionally before this loop runs, so
            # re-listing opt_start here would just re-solve and re-write an
            # output row for a point we already have. Also drop any target
            # already recorded in sim_I.csv from an earlier partial resume.
            opt_targets = _prune_computed(
                sorted(v for v in (0, 1) if v > opt_start),
                _already_computed_values(PREFIX, 'I'))
            opt_reload  = resume_yaml
        else:
            opt_start   = 0
            opt_targets = [0, 1]
            opt_reload  = None
        stepper = OpticalIntensityAdaptiveStepper(
            solution=full_problem,
            parameter_target_values=opt_targets,
            parameter_start_value=opt_start,
            step_size=iramp_step,
            stepper_rel_tol=1e-6,
            output_writer=OutputWriter(
                filename_prefix=filename_prefix,
                parameter_name='I',
                plot_1d=plot_1d,
                plot_iv=False,
                plot_mesh=False,
                plot_mesh_full=plot_xdmf,
                plot_du=False,
                meta_extractors=meta_extractors,
            ),
            selfconsistent_optics=iramp_sc,
            checkpoint_write_values=checkpoint_values_opt or [],
            checkpoint_write_filename=checkpoint_filename_opt,
            checkpoint_reload_yaml=opt_reload,
        )
        stepper.solver_parameters = {'extra_iterations': 2}
        _t0 = time.perf_counter()
        stepper.do_loop()
        _mark('intensity ramp', _t0)

    # ------------------------------------------------------------------
    # "Down" problem (V < initial_v): created and initialised HERE, after the
    # optical ramp, so that -- when it needs to be seeded from full_problem
    # rather than loaded directly -- it inherits the fully-illuminated state
    # (I = 1, V = initial_v) rather than the dark thermal-equilibrium state.
    # Doing this before the "up" sweep runs is essential — once that sweep
    # runs, full_problem's state has moved on to its final voltage and can
    # no longer be used as a clean V = initial_v starting point.
    #
    # Whichever of full_problem / full_problem_neg the checkpoint belongs to
    # holds that state directly (loaded above, for 'up_v'; loaded
    # below, for 'down_v'); the other one, if needed at all, is seeded
    # from it via ProblemData.pdd.initialize_from + Optical.initialize_from
    # (both required -- Optical is a sibling of pdd on ProblemData, not one
    # of its children, so pdd.initialize_from alone leaves the copy's photon
    # flux at its never-solved default, inconsistent with the copied-in
    # illuminated carrier state).
    # ------------------------------------------------------------------
    full_problem_neg = None
    V_ext_neg        = None
    need_up   = bool(up_targets) and do_vsweep
    need_down = bool(down_targets) and do_vsweep

    if do_vsweep and not need_up and not need_down:
        log.warning(
            'Nothing to compute for the voltage sweep: every requested '
            'value is either missing from voltage_sweep.values or already '
            'present in this output folder\'s sim_V.csv. This run will '
            'finish immediately without solving anything new.')

    if resume_stage == 'down_v':
        if need_down or need_up:
            V_ext_neg        = U.V * dolfin.Constant(initial_v)
            full_problem_neg = builder.problem('full', V_ext=V_ext_neg)
            _load_checkpoint_into(
                VoltageStepper, full_problem_neg, resume_yaml,
                constants=[V_ext_neg], parameter_unit=U.V,
                selfconsistent_optics=False)
        if need_up:
            # The checkpoint belongs to the down branch here, so full_problem
            # (used for the "up" sweep) is seeded from full_problem_neg --
            # the reverse of the usual direction.
            full_problem.pdd.initialize_from(full_problem_neg.pdd)
            full_problem.optical.initialize_from(full_problem_neg.optical)
    elif need_down:
        # resume_stage in (None, 'optical', 'up_v'): full_problem
        # already holds the correct V=initial_v state (solved fresh, resumed
        # via the optical stepper above, or loaded directly above).
        V_ext_neg        = U.V * dolfin.Constant(initial_v)
        full_problem_neg = builder.problem('full', V_ext=V_ext_neg)
        full_problem_neg.pdd.initialize_from(full_problem.pdd)
        full_problem_neg.optical.initialize_from(full_problem.optical)

    # ------------------------------------------------------------------
    # Stage 2: "up" voltage sweep  (initial_v → higher V)
    # ------------------------------------------------------------------
    if need_up:
        checkpoint_values_V   = _checkpoint_values('voltage_sweep', up_targets[-1])
        checkpoint_filename_V = (str(Path(checkpoint_dir) / file_prefix)
                                 if checkpoint_values_V else None)
        stepper = VoltageStepper(
            solution=full_problem,
            constants=[V_ext],
            step_size=vsweep_step,
            stepper_rel_tol=1e-6,
            parameter_target_values=up_targets,
            parameter_start_value=initial_v,
            parameter_unit=U.V,
            output_writer=OutputWriter(
                filename_prefix=filename_prefix,
                parameter_name='V',
                meta_extractors=meta_extractors,
                plot_1d=plot_1d,
                plot_mesh=False,
                plot_mesh_full=plot_xdmf,
                plot_iv=True,
            ),
            selfconsistent_optics=(vsweep_sc and has_optics and not dark_mode),
            checkpoint_write_values=checkpoint_values_V or [],
            checkpoint_write_filename=checkpoint_filename_V,
            # Never re-load here: full_problem already holds the correct
            # V=initial_v state by this point, however it got there (fresh
            # solve, optical-stepper resume, or the direct load above).
            checkpoint_reload_yaml=None,
        )
        _t0 = time.perf_counter()
        stepper.do_loop()
        _mark('voltage sweep (up)', _t0)

    # ------------------------------------------------------------------
    # Stage 3: "down" voltage sweep  (initial_v → lower V)
    # ------------------------------------------------------------------
    if need_down:
        log.info(f'Switching to the down sweep: V={initial_v:.6g} -> '
                 f'{down_targets[-1]:.6g}.')
        checkpoint_values_negV   = _checkpoint_values('voltage_sweep', down_targets[-1])
        checkpoint_filename_negV = (str(Path(checkpoint_dir) / file_prefix)
                                    if checkpoint_values_negV else None)
        stepper_neg = VoltageStepper(
            solution=full_problem_neg,
            constants=[V_ext_neg],
            step_size=vsweep_step,
            stepper_rel_tol=1e-6,
            parameter_target_values=down_targets,
            parameter_start_value=initial_v,
            parameter_unit=U.V,
            output_writer=OutputWriter(
                filename_prefix=filename_prefix,
                parameter_name='V',
                meta_extractors=meta_extractors,
                plot_1d=plot_1d,
                plot_mesh=False,
                plot_mesh_full=plot_xdmf,
                plot_iv=True,
            ),
            selfconsistent_optics=(vsweep_sc and has_optics and not dark_mode),
            checkpoint_write_values=checkpoint_values_negV or [],
            checkpoint_write_filename=checkpoint_filename_negV,
            # Never re-load: already loaded directly above (down_v) or
            # seeded from full_problem just above.
            checkpoint_reload_yaml=None,
        )
        _t0 = time.perf_counter()
        stepper_neg.do_loop()
        _mark('voltage sweep (down)', _t0)

    total = time.perf_counter() - _t_start
    log.info('Timing summary: ' + ', '.join(
        f'{k}={v:.2f}s' for k, v in _timings) + f', total={total:.2f}s')
    log.info(f'Run complete.  Output in {PREFIX}')


# ===========================================================================
# Command-line entry point
# ===========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run a Simudo 1D simulation from a project YAML file.')
    parser.add_argument(
        'yaml_file',
        help='Path to the project YAML file (e.g., gui/fourlayer_example.yaml)')
    parser.add_argument(
        '--library-dir', action='append', dest='library_dirs', default=None,
        metavar='DIR',
        help='Directory to search for Python Material subclasses used by '
             'library:// material sources. May be given more than once.')
    parser.add_argument(
        '--resume', action='store_true',
        help='Continue an existing run (do not increment the output directory).')
    args = parser.parse_args()

    run(args.yaml_file,
        library_dir=args.library_dirs or None,
        new_experiment=not args.resume)
