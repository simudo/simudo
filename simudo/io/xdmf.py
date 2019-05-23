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


def magnitude(x):
    return x.magnitude if hasattr(x, 'magnitude') else x

class PlotAddMixin(object):
    def add(self, name, units, expr, space=None):
        fsr = self.function_subspace_registry

        if space is None:
            try:
                space = fsr.get_function_space(magnitude(expr), new=True)
            except:
                pass

        func = dolfin.Function(space, name=name)
        e = (expr/units)
        if hasattr(e, 'm_as'):
            e = e.m_as('dimensionless')

        # TODO: actually use fsr
        # try:
        #     fsr.assign(func, e)
        # except: # output a warning or something

        e = force_ufl(e)

        dolfin.project(e, space, function=func)

        self._add_func(func)

class XdmfPlot(PlotAddMixin):
    partial_xdmf_file = None
    timestamp = 0.0

    def __init__(self, filename, function_subspace_registry=None):
        self.base_filename = filename
        self.function_subspace_registry = function_subspace_registry

        self.delete_partials(self.base_filename)

    @staticmethod
    def delete_partials(filename):
        dirname, name = osp.split(osp.abspath(filename))
        partial_prefix = name + '.p.'

        for f in [osp.join(dirname, x) for x in os.listdir(dirname)
                  if x.startswith(partial_prefix)]:
            try:
                os.remove(f)
            except OSError:
                pass

    def mpi_comm(self):
        return dolfin.mpi_comm_world()

    def new(self, timestamp):
        self.close()
        self.timestamp = timestamp

    def _add_func(self, func):
        self.ensure_open()
        self.partial_xdmf_file.write(func, float(self.timestamp))

    def open(self):
        if self.partial_xdmf_file is not None:
            self.close()

        while True:
            base = '{:s}.p.{:s}'.format(
                self.base_filename, generate_base32_token(20))
            xdmf_filename = base + '.xdmf'
            try:
                with open(xdmf_filename, 'x'):
                    pass
            except FileExistsError:
                continue
            else:
                break

        with open(base + '.incomplete', 'w'): pass

        self.partial_base_filename = base
        self.partial_xdmf_filename = xdmf_filename
        self.partial_xdmf_file = dolfin.XDMFFile(self.mpi_comm(), xdmf_filename)

    def ensure_open(self):
        if self.partial_xdmf_file is None:
            self.open()

    def close(self):
        xf = self.partial_xdmf_file
        if xf is None:
            return
        self.partial_xdmf_file = None
        xf.close()
        os.remove(self.partial_base_filename + '.incomplete')
        xdmf_combine(self.base_filename)

    def __del__(self):
        self.close()

def xdmf_combine(filename):
    dirname, name = osp.split(osp.abspath(filename))
    name_ = name + '.p.'

    files = [osp.join(dirname, head) for head, sep, tail
             in (f.rpartition('.xdmf') for f in os.listdir(dirname))
             if head.startswith(name_) and sep and not tail]

    result_grids = []

    for fn in files:
        if osp.exists(fn+'.incomplete'):
            continue

        tree = ET.parse(fn+'.xdmf')
        root = tree.getroot()

        domain = root.find('Domain')
        top_grids = domain.findall('Grid')
        att = top_grids[0].attrib

        grids = defaultdict(list)

        for g in (g
                  for top_grid in top_grids
                  for g        in top_grid.findall('Grid')):
            grids[float(g.find('Time').attrib['Value'])].append(g)

        assert len(grids) == 1

        grid_ts, gs = next(iter(grids.items()))

        g0 = gs[0]
        for g in gs[1:]:
            g0.append(g.find('Attribute'))

        result_grids.append((grid_ts, g0))

    result_grids.sort(key=lambda kv: kv[0])

    result = ET.Element("Grid")
    result.attrib['Name'] = 'TimeSeries'
    result.attrib['GridType'] = 'Collection'
    result.attrib['CollectionType'] = 'Temporal'

    new_et = ET.parse(StringIO('''\
<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
<Domain></Domain></Xdmf>'''))

    for g in result_grids:
        result.append(g[1])

    new_et.getroot().find('Domain').append(result)

    with open(filename, 'wb') as h:
        new_et.write(h, encoding="utf-8", xml_declaration=True)

if __name__ == '__main__':
    from ..util.function_subspace_registry import FunctionSubspaceRegistry
    import pint
    fsr = FunctionSubspaceRegistry()
    mesh = dolfin.UnitSquareMesh(2, 2)
    space = dolfin.FunctionSpace(mesh, "CG", 1)
    space2 = dolfin.FunctionSpace(mesh, "CG", 2)
    ur = pint.UnitRegistry()
    x = dolfin.SpatialCoordinate(mesh)
    c = dolfin.Constant(0.0)
    e1 = dolfin.sin(x[0] + c) * ur.dimensionless
    e2 = dolfin.sin(x[1] + c) * ur.dimensionless
    xp = XdmfPlot("out/zz.xdmf", fsr)
    for i in range(2):
        c.assign(i/10.)
        t = float(i)
        xp.new(t)
        f1 = xp.add("FOO", 1, e1, space=space)
        if True or i % 2 == 0:
            f2 = xp.add("BAR",  1, e2, space=space2)
    xp.close()
