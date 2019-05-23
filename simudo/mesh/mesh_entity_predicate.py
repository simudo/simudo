from __future__ import absolute_import, division, print_function

from builtins import bytes, dict, int, range, str, super

import numpy as np

import dolfin
from dolfin import entities as dolfin_entities
from dolfin import Cell, Edge, Face, Facet, MeshEntity, Vertex

from ..fem import point_to_array

__all__ = [
    'BaseMeshEntityPredicate',
    'CombiningMeshEntityPredicate',
    'AndMeshEntityPredicate',
    'OrMeshEntityPredicate',
    'NandMeshEntityPredicate',
    'SubdomainCellPredicate',
    'DimensionAdapterPredicate',
    'InRadiusCellPredicate',
    'MaxEdgeLengthCellPredicate',
    'DirectionalEdgeLengthPredicate',
    'AlwaysTruePredicate']

def as_Cell(mesh_entity):
    return Cell(mesh_entity.mesh(), mesh_entity.index())

def as_Edge(mesh_entity):
    return Edge(mesh_entity.mesh(), mesh_entity.index())

def as_Face(mesh_entity):
    return Face(mesh_entity.mesh(), mesh_entity.index())

def as_Facet(mesh_entity):
    return Facet(mesh_entity.mesh(), mesh_entity.index())

def as_Vertex(mesh_entity):
    return Vertex(mesh_entity.mesh(), mesh_entity.index())

class BaseMeshEntityPredicate(object):
    dim = None

    def __call__(self, mesh_entity):
        raise NotImplementedError()

    def prepare(self, dictionary):
        pass

    @classmethod
    def assert_compatible_predicates(cls, predicates):
        if len(frozenset(p.dim for p in predicates)) > 1:
            raise ValueError("different mesh entity dimensions")

    def __and__(self, other):
        return AndMeshEntityPredicate(self, other)

    def __or__(self, other):
        return OrMeshEntityPredicate(self, other)

    def __neg__(self):
        return NandMeshEntityPredicate(self)

class CombiningMeshEntityPredicate(BaseMeshEntityPredicate):
    def __init__(self, *predicates):
        self.assert_compatible_predicates(predicates)
        self.children = predicates
        self.dim = next(iter(predicates)).dim

    def prepare(self, dictionary):
        for p in self.children:
            p.prepare(dictionary)

class AndMeshEntityPredicate(CombiningMeshEntityPredicate):
    def __call__(self, mesh_entity):
        return all(c(mesh_entity) for c in self.children)

class OrMeshEntityPredicate(CombiningMeshEntityPredicate):
    def __call__(self, mesh_entity):
        return any(c(mesh_entity) for c in self.children)

class NandMeshEntityPredicate(CombiningMeshEntityPredicate):
    def __call__(self, mesh_entity):
        return not all(c(mesh_entity) for c in self.children)

class SubdomainCellPredicate(BaseMeshEntityPredicate):
    def __init__(self, cell_function, subdomains):
        self.cell_function = cell_function
        self.subdomains = subdomains
        self.dim = self.cell_function.dim()

    def __call__(self, mesh_entity):
        return self.cell_function[as_Cell(mesh_entity)] in self.subdomains

class DimensionAdapterPredicate(BaseMeshEntityPredicate):
    def __init__(self, predicate, dim):
        self.predicate = predicate
        self.dim = dim
        self.is_identity = self.dim == predicate.dim

    def __call__(self, mesh_entity):
        predicate = self.predicate
        if self.is_identity:
            return predicate(mesh_entity)
        else:
            mesh = mesh_entity.mesh()
            dim = predicate.dim
            return any(predicate(MeshEntity(mesh, dim, i)) for i in
                       mesh_entity.entities(dim))

    def prepare(self, dictionary):
        dictionary['mesh'].init(self.dim, self.predicate.dim)
        self.predicate.prepare(dictionary)

class InRadiusCellPredicate(BaseMeshEntityPredicate):
    def __init__(self, threshold, dim):
        self.threshold = threshold
        self.dim = dim

    def __call__(self, mesh_entity):
        return as_Cell(mesh_entity).inradius >= self.threshold

class MaxEdgeLengthCellPredicate(BaseMeshEntityPredicate):
    def __init__(self, threshold, dim):
        self.threshold = threshold
        self.dim = dim

    def __call__(self, mesh_entity):
        return as_Cell(mesh_entity).h() >= self.threshold

def edge_to_vector(edge):
    v0, v1 = edge.entities(0)
    mesh = edge.mesh()
    v0 = Vertex(mesh, v0)
    v1 = Vertex(mesh, v1)
    return point_to_array(v1.point() - v0.point())

def normalize_vector(v):
    return v/np.sqrt(v.dot(v))

class DirectionalEdgeLengthPredicate(BaseMeshEntityPredicate):
    # TODO: tell plaza not to subdivide edges we really don't care about
    dim = 1

    def __init__(self, direction, threshold=None):
        if threshold is None:
            v = direction
        else:
            v = normalize_vector(direction) / threshold

        self.directional_threshold = v

    def __call__(self, mesh_entity):
        v = self.directional_threshold
        return abs(edge_to_vector(mesh_entity).dot(v)) >= 1.0

    def prepare(self, dictionary):
        dictionary['mesh'].init(1, 0)

class AlwaysTruePredicate(BaseMeshEntityPredicate):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, mesh_entity):
        return True
