
from functools import partial

import numpy as np
from cached_property import cached_property

import dolfin

from .product2d import Product2DMesh

__all__ = [
    'clip_intervals',
    'Interval', 'CInterval',
    'Product2DMeshMixin', 'MinimumCoordinateDistanceMixin',
    'BaseInterval1DTag', 'Interval1DTag']

def clip_intervals(x0, x1, intervals):
    ''' in-place modifies intervals '''

    for I in intervals:
        I.x0 = max(I.x0, x0)
        I.x1 = min(I.x1, x1)

    new = [I for I in intervals
           if I.x0 != I.x1] # remove degenerate intervals

    intervals[:] = new

    return intervals

class Interval(object):
    ''' Represents a single interval, possibly with tags. '''

    def __init__(self, x0, x1, tags=()):
        self.x0 = x0
        self.x1 = x1
        self.tags = frozenset(tags)

    def __repr__(self):
        return "{}(x0={}, x1={}{})".format(self.__class__.__name__,
                                           self.x0, self.x1,
                                           self._supplementary_repr())

    def _supplementary_repr(self):
        return ', tags={{{}}}'.format(', '.join(repr(t) for t in self.tags))

class CInterval(Interval):
    ''' Represents an interval with constant edge length (in the x
    direction).

    Note: you do not need to derive from this class to implement a
    custom edge length. Instead, you can subclass from
    :py:class:`Interval` and implement a custom
    :code:`local_edge_length` method. '''

    def __init__(self, x0, x1, tags=(), edge_length=np.inf):
        self.edge_length = edge_length
        super().__init__(x0, x1, tags)

    def local_edge_length(self, x):
        return self.edge_length

    def _supplementary_repr(self):
        return '{}, edge_length={!r}'.format(
            super()._supplementary_repr(), self.edge_length)

class BaseInterval1DTag(object):
    ''' Class that turns a bunch of arbitrary overlapping intervals
    into a mesh. '''
    first_cell_value = 1

    def __init__(self, intervals):
        self.intervals = intervals

    @cached_property
    def subdomains(self):
        return self.intervals_to_subdomains(intervals=self.intervals)

    @cached_property
    def coordinates(self):
        sub = self.subdomains
        return self.make_coordinates(
            subdomain_to_intervals=sub['subdomain_to_intervals'],
            endpoints=sub['endpoints'])

    @cached_property
    def tag_to_cell_values(self):
        ''' Mapping from a tag to a set of cell values (e.g. values
        inside `self.product2d_mesh.cell_function`). '''
        return self.subdomains['tag_to_cell_values']

    def intervals_to_subdomains(self, intervals):
        all_tags = set()
        all_tags_update = all_tags.update
        endpoints = set()
        endpoints_add = endpoints.add
        for interval in intervals:
            endpoints_add(interval.x0)
            endpoints_add(interval.x1)
            all_tags_update(interval.tags)
        del all_tags_update, endpoints_add

        endpoints = list(endpoints)
        endpoints.sort()
        all_tags = list(all_tags)
        all_tags.sort()

        endpoint_to_index = {x: i for i, x in enumerate(endpoints)}

        # subdomains form a partition of the mesh space
        N = (len(endpoints)-1) # number of subdomains
        subdomain_to_tags      = [set() for i in range(N)]
        subdomain_to_intervals = [set() for i in range(N)]

        for interval in intervals:
            i0 = endpoint_to_index[interval.x0]
            i1 = endpoint_to_index[interval.x1]
            for i in range(i0, i1):
                subdomain_to_intervals[i].add(interval)
                subdomain_to_tags[i].update(interval.tags)

        # freeze tags into tuples so they're hashable
        for i, tags in enumerate(subdomain_to_tags):
            subdomain_to_tags[i] = tuple(sorted(tags))

        cell_values_start = self.first_cell_value
        tagset_to_cell_value = {
            ts: cv for cv, ts in enumerate(
                sorted(set(subdomain_to_tags)), cell_values_start)}
        cell_values_end = cell_values_start + len(tagset_to_cell_value)

        subdomain_to_cell_value = [
            tagset_to_cell_value[ts] for ts in subdomain_to_tags]

        tag_to_cell_values = {tag: set() for tag in all_tags}
        for tagset, cell_value in tagset_to_cell_value.items():
            for tag in tagset:
                tag_to_cell_values[tag].add(cell_value)

        # freeze cell value sets
        for k, v in tag_to_cell_values.items():
            tag_to_cell_values[k] = frozenset(v)

        return dict(tag_to_cell_values=tag_to_cell_values,
                    subdomain_to_cell_value=subdomain_to_cell_value,
                    subdomain_to_intervals=subdomain_to_intervals,
                    used_cell_values=range(cell_values_start,
                                           cell_values_end),
                    endpoints=endpoints)

    def make_coordinates(self, subdomain_to_intervals, endpoints):
        def local_edge_length_function(intervals, x):
            return min(interval.local_edge_length(x)
                       for interval in intervals)

        coordinates = []
        subdomain_to_coordinate_range = []
        x1 = endpoints[0]
        last_coordinate_index = 0
        for i, intervals in enumerate(subdomain_to_intervals):
            x0 = x1
            x1 = endpoints[i+1]
            lcf = partial(local_edge_length_function,
                          tuple(iv for iv in intervals
                                if hasattr(iv, 'local_edge_length')))
            coords = self.make_interval_coordinates(x0, x1, lcf)
            coordinates.extend(coords)

            prev_last_coordinate_index = last_coordinate_index
            last_coordinate_index = len(coordinates)
            subdomain_to_coordinate_range.append(
                range(prev_last_coordinate_index,
                      last_coordinate_index))
        coordinates.append(x1)

        return dict(
            coordinates=coordinates,
            subdomain_to_coordinate_range=subdomain_to_coordinate_range)

    def make_interval_coordinates(
            self, x0, x1, local_edge_length_function):
        '''Note: this excludes second endpoint'''
        coords = []
        x = x0
        while True:
            coords.append(x)
            delta = local_edge_length_function(x)
            x += delta
            if x >= x1:
                break
        return coords


class Product2DMeshMixin(object):
    product2d_Ys = (0.0, 1.0)
    Product2DMesh = Product2DMesh

    @cached_property
    def product2d_mesh(self):
        ''' Use this to get a readily-constructed Product2D
        object. Note that an attribute `cell_function` has been added
        to it. '''

        s = self.subdomains
        c = self.coordinates
        return self.make_product2d_mesh(
            coordinates=c['coordinates'],
            subdomain_to_coordinate_range=c['subdomain_to_coordinate_range'],
            subdomain_to_cell_value=s['subdomain_to_cell_value'],
            Ys=self.product2d_Ys)

    def make_product2d_mesh(self, coordinates,
                            subdomain_to_coordinate_range,
                            subdomain_to_cell_value, Ys):
        pm = self.Product2DMesh(coordinates, Ys)
        mf = pm.cell_function = dolfin.MeshFunction("size_t", pm.mesh, 2)
        mfw = mf.array()
        for crange, cell_value in zip(subdomain_to_coordinate_range,
                                      subdomain_to_cell_value):
            for ix in crange:
                mfw[pm.cells_at_ix(ix)] = cell_value

        return pm


class MinimumCoordinateDistanceMixin(object):
    minimum_coordinate_distance = 1e-10
    def make_interval_coordinates(self, x0, x1, *args, **kwargs):
        coords = super().make_interval_coordinates(x0, x1, *args, **kwargs)
        minimum_coordinate_distance = self.minimum_coordinate_distance

        coords = np.array(coords, dtype='object')
        keep = np.zeros(len(coords), dtype='bool')

        keep[0] = True
        min_x = x0
        max_x = x1 - minimum_coordinate_distance

        for i, x in enumerate(coords):
            if x >= min_x:
                if x > max_x:
                    break
                keep[i] = True
                min_x = x + minimum_coordinate_distance

        return list(coords[keep])


class Interval1DTag(Product2DMeshMixin, MinimumCoordinateDistanceMixin,
                    BaseInterval1DTag):
    pass
