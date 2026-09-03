from __future__ import absolute_import, division, print_function

import os
import re
import shlex
import xml.etree.ElementTree as ET
from builtins import bytes, dict, int, range, str, super
from collections import defaultdict
from io import StringIO
from os import path as osp

from future.utils import PY2, PY3, native

import dolfin

from ..fem import force_ufl
from ..util import generate_base32_token
from . import h5yaml
import logging


def magnitude(x):
    return x.magnitude if hasattr(x, 'magnitude') else x

class PlotAddMixin(object):
    def add(self, name, units, expr, space=None, subdomain=None):
        fsr = self.function_subspace_registry

        if hasattr(expr,"geometry"):
            #expr is actually a mesh
            self._add_func(expr)
            return

        if space is None:
            try:
                space = fsr.get_function_space(magnitude(expr), new=True)
            except:
                pass

        try:
            space_name = space.name
        except AttributeError:
            space_name = None

        e = (expr/units)
        if hasattr(e, 'm_as'):
            e = e.m_as('dimensionless')

        # TODO: actually use fsr
        # try:
        #     fsr.assign(func, e)
        # except: # output a warning or something

        e = force_ufl(e)
        try:
            func = dolfin.project(e, space)
            func.rename(name, name)
            self._add_func(func, space_name)
        except RuntimeError:
            logging.info("xdmf PlotAddMixin could not add integral {} with value {}, units {}".format(name,e, units))
            pass

class XdmfPlot(PlotAddMixin):
    timestamp = 0.0

    def __init__(self, filename, function_subspace_registry=None, checkpoint=False):
        self.function_subspace_registry = function_subspace_registry
        self.checkpoint = checkpoint
        self.xdmf_file = dolfin.XDMFFile(dolfin.MPI.comm_world, filename)
        self.xdmf_file.parameters['rewrite_function_mesh'] = False
        self.xdmf_file.parameters['functions_share_mesh'] = True
        self.metadata_filename = os.path.splitext(filename)[0] + "_func_names.yaml"
        self.funcs_plotted = {"funcs": []}

    def _add_func(self, func, space_name = None):
        if hasattr(func,"geometry"):
            #func is actuallly a mesh
            self.xdmf_file.write(func)
        elif self.checkpoint:
            # write_checkpoint() is needed to
            # properly visualize discontinuous datasets and datasets of degree>1.
            self.xdmf_file.write_checkpoint(func, func.name(), 0., append=True)            
            self.funcs_plotted["funcs"].append( func.name())
            if space_name:
                self.funcs_plotted[func.name()] = space_name 
        else:
            self.xdmf_file.write(func, 0.)
            self.funcs_plotted["funcs"].append( func.name())
            if space_name:
                self.funcs_plotted[func.name()] = space_name 

    def _add_meshfunction(self, name, meshfunction):
        self.xdmf_file.write(meshfunction)

    def new(self, timestamp):
        self.timestamp = timestamp

    def close(self):
        self.xdmf_file.close()

    def __del__(self):
        self.close()


if __name__ == '__main__':
    import pint
    from tempfile import TemporaryDirectory
    from os.path import join
#    import dolfin
    mesh = dolfin.UnitSquareMesh(2, 2)
    space = dolfin.FunctionSpace(mesh, "CG", 1)
    space2 = dolfin.FunctionSpace(mesh, "CG", 2)
    ur = pint.UnitRegistry()
    x = dolfin.SpatialCoordinate(mesh)
    c = dolfin.Constant(0.0)
    e1 = dolfin.sin(x[0] + c) * ur.dimensionless
    e2 = dolfin.sin(x[1] + c) * ur.dimensionless
    with TemporaryDirectory() as d:
        xp = XdmfPlot(join(d, "zz.xdmf"), function_subspace_registry=None)
        for i in range(2):
            c.assign(i/10.)
            t = float(i)
            xp.new(t)
            f1 = xp.add("FOO", 1, e1, space=space)
            if True or i % 2 == 0:
                f2 = xp.add("BAR",  1, e2, space=space2)
        xp.close()
