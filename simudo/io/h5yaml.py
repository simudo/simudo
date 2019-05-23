from __future__ import absolute_import, division, print_function

import atexit
import copy
import os
import shutil
import tempfile
import unittest
from builtins import bytes, dict, int, range, str, super
from decimal import Decimal
from functools import partial, wraps

import pint
import pint.quantity
import yaml
from future.utils import PY2, PY3, native
from yaml.representer import SafeRepresenter

import dolfin

from ..util import generate_base32_token


def xattr(loader_or_dumper):
    K = 'x_attr_dict'
    if not hasattr(loader_or_dumper, K):
        setattr(loader_or_dumper, K, {})
    return getattr(loader_or_dumper, K)

class BaseCustomTag(object):
    yaml_classes = ()
    yaml_tag = None
    @classmethod
    def register_in_loader(cls, loader):
        yt = cls.yaml_tag
        if yt is not None:
            loader.add_constructor(yt, cls.from_yaml)
    @classmethod
    def register_in_dumper(cls, dumper):
        for klass in cls.yaml_classes:
            dumper.add_multi_representer(klass, cls.to_yaml)

# FIXME: add class for pint Unit object

class PintQTag(BaseCustomTag):
    yaml_classes = (pint.quantity._Quantity,)
    yaml_tag = '!pintq'

    @classmethod
    def from_yaml(cls, loader, node):
        l = loader.construct_sequence(node)
        ur = xattr(loader)['pint_unit_registry']
        magnitude, units = l[:2]
        return ur.Quantity(magnitude, units)

    @classmethod
    def to_yaml(cls, dumper, data):
        qty = data
        return dumper.represent_sequence(
            cls.yaml_tag,
            [qty.magnitude, str(qty.units)])

class DolfinObjectTag(BaseCustomTag):
    load_pass = 0
    needs_load_into = True

    def __init__(self, dolfin_h5_manager, key, **kwargs):
        self.dolfin_h5_manager = dolfin_h5_manager
        self.key = key
        super().__init__(**kwargs)

    @classmethod
    def get_object_mpi_comm(cls, obj):
        comm = getattr(obj, 'mpi_comm', None)
        if comm is not None:
            comm = comm()
        elif hasattr(obj, 'mesh'):
            return cls.get_object_mpi_comm(obj.mesh())
        elif hasattr(obj, 'function_space'):
            return cls.get_object_mpi_comm(obj.function_space())

        return comm

    def _actual_load(self, hdf5file, hdf_path, target):
        hdf5file.read(target, hdf_path)

    def load_into(self, obj, comm=None):
        man = self.dolfin_h5_manager
        if comm is None:
            comm = self.get_object_mpi_comm(obj)
        (key, h) = man.open(comm, self.key, 'r')
        self._actual_load(h, '/obj', obj)
        h.close()

    @classmethod
    def from_yaml(cls, loader, node):
        key = loader.construct_scalar(node)
        return cls(dolfin_h5_manager=xattr(loader)['dolfin_h5_manager'],
                   key=key)

    @classmethod
    def to_yaml(cls, dumper, data):
        comm = cls.get_object_mpi_comm(data)
        man = xattr(dumper)['dolfin_h5_manager']

        (key, h) = man.open(comm, None, 'w')
        h.write(data, '/obj')
        h.close()

        return dumper.represent_scalar(cls.yaml_tag, key)

    def _copy(self):
        new = copy.copy(self)
        return new

    def __repr__(self):
        return "<{} key={!r}>".format(type(self), self.key)

class ScaleMixin(object):
    def __init__(self, scale=1.0, **kwargs):
        self.scale = scale
        super().__init__(**kwargs)

    def __imul__(self, scale):
        self.scale = self.scale * scale

    def __mul__(self, scale):
        new = self._copy()
        new *= scale
        return new

    def __rmul__(self, scale):
        return self.__mul__(scale)

class ScaleDolfinObjectTag(ScaleMixin, DolfinObjectTag):
    pass

class DolfinMeshTag(DolfinObjectTag):
    load_pass = -10
    yaml_classes = (dolfin.cpp.mesh.Mesh,)
    yaml_tag = '!dolfin/mesh'

    def _actual_load(self, hdf5file, hdf_path, target):
        hdf5file.read(target, hdf_path, False)

class DolfinFunctionTag(ScaleDolfinObjectTag):
    yaml_classes = (dolfin.Function,)
    yaml_tag = '!dolfin/function'

class ClassProperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class BaseDolfinMeshFunctionTag(DolfinObjectTag):
    @ClassProperty
    @classmethod
    def yaml_tag(cls):
        return "!dolfin/meshfunction/" + cls.meshfunction_type_name

class DolfinMeshFunctionSizetTag(BaseDolfinMeshFunctionTag):
    meshfunction_type_name = "size_t"
    yaml_classes = (dolfin.cpp.mesh.MeshFunctionSizet,)

class DolfinMeshFunctionBoolTag(BaseDolfinMeshFunctionTag):
    meshfunction_type_name = "bool"
    yaml_classes = (dolfin.cpp.mesh.MeshFunctionBool,)

class DolfinMeshFunctionIntTag(BaseDolfinMeshFunctionTag):
    meshfunction_type_name = "int"
    yaml_classes = (dolfin.cpp.mesh.MeshFunctionInt,)

class DolfinMeshFunctionDoubleTag(ScaleMixin, BaseDolfinMeshFunctionTag):
    meshfunction_type_name = "double"
    yaml_classes = (dolfin.cpp.mesh.MeshFunctionDouble,)

class FrozenSetTag(BaseCustomTag):
    yaml_tag = '!frozenset'
    yaml_classes = (frozenset,)

    @classmethod
    def from_yaml(cls, loader, node):
        return frozenset(loader.construct_sequence(node))

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_sequence(cls.yaml_tag, sorted(data))

class DolfinH5Manager(object):
    def __init__(self, basepath):
        self.basepath = basepath

    def filename_from_key(self, key):
        # TODO: use subdirectories a la git
        return ''.join((self.basepath, '.a.', key, '.dolfin.h5'))

    def ensure_dir(self):
        try:
            os.makedirs(os.path.dirname(self.basepath))
        except OSError:
            pass

    def open(self, comm, key, mode):
        '''returns (key, dolfin.HDF5File)

pass key=None to have a new one randomly generated'''
        self.ensure_dir()
        if key is None:
            if mode != 'w': raise AssertionError()
            while True:
                key = generate_base32_token(26)
                if not os.path.exists(self.filename_from_key(key)):
                    break
        filename = self.filename_from_key(key)
        return (key, dolfin.HDF5File(comm, filename, mode))

class XDumper(yaml.Dumper):
    def __init__(self, *args, **kwargs):
        self.x_attr_dict = kwargs.pop('x_attr_dict', {})
        super().__init__(*args, **kwargs)

    def ignore_aliases(self, data):
        if isinstance(data, int): # py2 vs future int
            return True

class XLoader(yaml.Loader):
    def __init__(self, *args, **kwargs):
        self.x_attr_dict = kwargs.pop('x_attr_dict', {})
        super().__init__(*args, **kwargs)

if PY2:
    XDumper.add_representer(
        int,
        lambda dumper, data: SafeRepresenter.represent_int(
            dumper, native(data)))
    XDumper.add_representer(
        dict,
        lambda dumper, data: SafeRepresenter.represent_dict(
            dumper, native(data)))

for kl in [DolfinMeshTag, DolfinFunctionTag,
           DolfinMeshFunctionSizetTag,
           DolfinMeshFunctionIntTag,
           DolfinMeshFunctionDoubleTag,
           DolfinMeshFunctionBoolTag,
           FrozenSetTag, PintQTag]:
    kl.register_in_dumper(XDumper)
    kl.register_in_loader(XLoader)

def _load(stream, Loader=None, x_attr_dict=None,
          pint_unit_registry=None, **kwargs):

    filename = stream.name if hasattr(stream, 'name') else None

    xa = x_attr_dict if x_attr_dict is not None else {}
    if pint_unit_registry is not None:
        xa['pint_unit_registry'] = pint_unit_registry
    if filename is not None:
        xa['dolfin_h5_manager'] = DolfinH5Manager(filename)

    if Loader is None: Loader = XLoader

    return yaml.load(stream, Loader=partial(XLoader, x_attr_dict=xa))

def _dump(data, stream=None, Dumper=None, x_attr_dict=None,
         pint_unit_registry=None, **kwargs):

    filename = stream.name if hasattr(stream, 'name') else None

    xa = x_attr_dict if x_attr_dict is not None else {}
    if pint_unit_registry is not None:
        xa['pint_unit_registry'] = pint_unit_registry
    if filename is not None:
        xa['dolfin_h5_manager'] = DolfinH5Manager(filename)

    if Dumper is None: Dumper = XDumper

    return yaml.dump(data, stream, Dumper=partial(XDumper, x_attr_dict=xa))

@wraps(_load)
def load(stream, *args, **kwargs):
    if isinstance(stream, str):
        with open(stream, 'rt') as handle:
            return _load(handle, *args, **kwargs)
    else:
        return _load(stream, *args, **kwargs)

@wraps(_dump)
def dump(data, stream=None, *args, **kwargs):
    if isinstance(stream, str):
        with open(stream, 'wt') as handle:
            return _dump(data, handle, *args, **kwargs)
    else:
        return _dump(data, stream, *args, **kwargs)

def delete_attached(filename):
    dirname, name = os.path.split(filename)
    partial_prefix = name + '.a.'

    for f in [os.path.join(dirname, x) for x in os.listdir(dirname)
              if x.startswith(partial_prefix)]:
        try:
            os.remove(f)
        except OSError:
            pass

class SolutionIOTest(unittest.TestCase):
    def test_roundtrip(self):
        self.helper_test_roundtrip(dedup=False)

    def test_roundtrip_with_dedup(self):
        self.helper_test_roundtrip(dedup=True)

    def helper_test_roundtrip(self, dedup):
        mesh0 = dolfin.UnitSquareMesh(3, 5)
        uc0 = mesh0.ufl_cell()
        el0 = dolfin.FiniteElement('CG', uc0, 2)
        W0 = dolfin.FunctionSpace(mesh0, el0)
        u0 = dolfin.Function(W0)

        mf0 = dolfin.MeshFunction("size_t", mesh0, 1, 0)
        mf0.array()[:] = range(len(mf0.array()))

        x = dolfin.SpatialCoordinate(mesh0)
        dolfin.project(x[0] - x[1]**2, W0, function=u0)

        outdir = tempfile.mkdtemp('.solution_io')
        # atexit.register(partial(shutil.rmtree, outdir))
        outfile = os.path.join(outdir, 'test.yaml')

        man = DolfinH5Manager(outfile)

        with open(outfile, 'wt') as stream:
            yaml.dump(
                dict(mesh=mesh0, u=u0, mf=mf0),
                stream,
                Dumper=partial(XDumper, x_attr_dict=dict(
                    dolfin_h5_manager=man)))

        with open(outfile, 'rt') as h: print(h.read())

        if dedup:
            from h5dedup.dedup import DedupRepository
            repo = DedupRepository(outdir)
            repo.deduplicate_file_tree(outdir)

        with open(outfile, 'rt') as stream:
            r = yaml.load(stream, Loader=partial(XLoader, x_attr_dict=dict(
                dolfin_h5_manager=man)))

        mesh1 = dolfin.Mesh()
        r['mesh'].load_into(mesh1)

        W1 = dolfin.FunctionSpace(mesh1, el0)
        u1 = dolfin.Function(W1)
        r['u'].load_into(u1)

        mf1 = dolfin.MeshFunction("size_t", mesh1, 1, 0)
        r['mf'].load_into(mf1)

        u1_ = dolfin.project(u1, W0)

        self.assertLess(dolfin.assemble((u0-u1_)**2*dolfin.dx), 1e-26)
        self.assertEqual(tuple(mf0.array()), tuple(mf1.array()))
