from __future__ import absolute_import, division, print_function

import itertools as it
import operator
import unittest
from builtins import bytes, dict, int, range, str, super
from functools import reduce

import dolfin

try:
    import mshr
except:
    mshr = None

REL_DISJOINT = 0
REL_OVERLAPPING  = 1
REL_A_SUBSET_OF_B = 2 | REL_OVERLAPPING
REL_B_SUBSET_OF_A = 4 | REL_OVERLAPPING
REL_EQUAL = REL_A_SUBSET_OF_B | REL_B_SUBSET_OF_A

rel_inverse = {
    REL_DISJOINT: REL_DISJOINT,
    REL_OVERLAPPING: REL_OVERLAPPING,
    REL_A_SUBSET_OF_B: REL_B_SUBSET_OF_A,
    REL_B_SUBSET_OF_A: REL_A_SUBSET_OF_B,
    REL_EQUAL: REL_EQUAL}

def powerset(iterable):
    lst = tuple(iterable)
    return it.chain.from_iterable(
        it.combinations(lst, n) for n in range(len(lst)+1))

def _generate_domain_relationship_table():
    d = {}
    for cvs in powerset((1, 2, 3)):
        overlap = 1 in cvs
        AnotB = 2 in cvs
        BnotA = 3 in cvs
        if not overlap:
            v = REL_DISJOINT
        elif not AnotB and not BnotA:
            v = REL_EQUAL
        elif AnotB and BnotA:
            v = REL_OVERLAPPING
        elif AnotB:
            v = REL_B_SUBSET_OF_A
        else:
            v = REL_A_SUBSET_OF_B
        d[frozenset(cvs)] = v
    return d

_domain_relationship_table = _generate_domain_relationship_table()


class DomainTag(object):
    first_cell_value = 1

    def domain_relationship(self, A, B):
        ''' must return one of {REL_DISJOINT, REL_OVERLAPPING, ...} '''
        raise NotImplementedError()

    def domain_relationship_tabulate(self, items):
        # TODO: be clever and exploit e.g. transitivity of subset relation
        r = {}
        for k0, v0 in items:
            for k1, v1 in items:
                key = (k0, k1)
                a = r.get((k1, k0), None)
                if a is not None:
                    r[key] = rel_inverse[a]
                else:
                    r[key] = self.domain_relationship(v0, v1)
        return r

    def generate_intersection_regions(self, tagitems):
        '''returns frozenset of `tagset`

        region described by `tagset` is
        `intersection(tagset) - union(all_tags - tagset)`
        '''

        rel = self.domain_relationship_tabulate(tagitems)

        # entry = (included, excluded)
        # -> intersection(included) - union(excluded)

        N = len(tagitems)

        def entries(i, included, excluded):
            if i == N:
                if len(included):
                    yield (included, excluded)
                return

            k0, v0 = tagitems[i]

            inc = True
            exc = True

            if any(rel[(k0, k)] == REL_DISJOINT for k in included):
                inc = False

            if any(rel[(k0, k)] & REL_A_SUBSET_OF_B == REL_A_SUBSET_OF_B
                   for k in excluded):
                inc = False

            if any(rel[(k, k0)] & REL_A_SUBSET_OF_B == REL_A_SUBSET_OF_B
                   for k in included):
                exc = False

            if inc:
                for e in entries(i+1, included.union((k0,)), excluded):
                    yield e

            if exc:
                for e in entries(i+1, included, excluded.union((k0,))):
                    yield e

        return entries(0, frozenset(), frozenset())

    def region_to_subdomain(self, tags, included, excluded):
        return reduce(operator.sub, (tags[t] for t in excluded),
                      reduce(operator.add, (tags[t] for t in included)))

    def make_subdomains(self, tags):
        '''
        dict must be {tag_name: region}

        returns (tag_to_cv_dict, subdomains)

        `tag_to_cv_dict` will be {tag_name: cell_values}
        where all cell_values >= start_value

        `subdomains` is list of (subdomain, cell_value) where
        `subdomain` is as returned by the self.region_to_subdomain
        method
        '''
        tagitems = tuple(sorted(tags.items()))
        tagkeys = tuple(x[0] for x in tagitems)

        regions = tuple(generate_intersection_regions(tagitems))

        tag_to_cv = {tag: set() for tag in tagkeys}

        subdomains = []

        def region_to_csg_domain(region):
            inc, exc = region
            return reduce(operator.sub, (tags[t] for t in exc),
                          reduce(operator.add, (tags[t] for t in inc)))

        for cell_value, (included, excluded) in enumerate(
                regions, self.first_cell_value):

            for tag in included:
                tag_to_cv[tag].add(cell_value)

            subdomains.append((self.region_to_subdomain(
                tags, included, excluded), cell_value))

        return (tag_to_cv, subdomains)


class MshrDomainTag(DomainTag):
    def domain_relationship(self, A, B):
        # FIXME: this is seriously stupid and inefficient, but it works

        dom = A + B

        dom.set_subdomain(1, dom)
        dom.set_subdomain(2, A - B)
        dom.set_subdomain(3, B - A)

        mesh = mshr.generate_mesh(dom, 1)

        cf = dolfin.MeshFunction(
            "size_t", mesh, mesh.geometry().dim(), mesh.domains())

        # cell values (subdomains) that actually show up
        cvs = frozenset(cf.array())

        return _domain_relationship_table[cvs]

    def mshr_make_subdomains(self, domain, tags):
        tag_to_cv_dict, subdomains = self.make_subdomains(tags)

        for subdomain, cell_value in subdomains:
            domain.set_subdomain(cell_value, subdomain)

        return tag_to_cv_dict

class DomainTagTest(unittest.TestCase):
    def rect(self, x0, y0, x1, y1):
        P = dolfin.Point
        return mshr.Rectangle(P(x0, y0), P(x1, y1))

    def assert_rel(self, answer, A, B):
        dt = MshrDomainTag()
        self.assertEqual(answer, dt.domain_relationship(A, B))

    def assert_gen_reg(self, answer, tags):
        dt = MshrDomainTag()
        self.assertEqual(
            frozenset(frozenset(x) for x in answer),
            frozenset(reg[0] for reg in dt.generate_intersection_regions(tags)))

    def test_domain_relationship(self):
        r = self.rect
        A = r(0, 0, 1, 1)
        B = r(0, 0, 10, 10)
        C = r(1, 1, 2, 2)
        D = r(0, 0, 2, 2)
        E = r(3, 3, 5, 5)
        F = r(4, 4, 6, 6)
        ar = self.assert_rel
        ar(REL_A_SUBSET_OF_B, A, B)
        ar(REL_DISJOINT, A, E)
        ar(REL_DISJOINT, A, C)
        ar(REL_A_SUBSET_OF_B, A, D)
        ar(REL_B_SUBSET_OF_A, B, C)
        ar(REL_OVERLAPPING, E, F)
        ar(REL_EQUAL, A, A)

    def test_generate_possibilities(self):
        r = self.rect
        A = r(0, 0, 10, 10)
        B = r(0, 0, 1, 1)
        C = r(1, 1, 2, 2)
        D = r(3, 3, 5, 5)
        E = r(4, 4, 6, 6)
        F = r(2, 3.5, 15, 4.5)

        loc = dict(locals())
        def a(answer, tags):
            return self.assert_gen_reg(
                answer, tuple((t, loc[t]) for t in tags))

        a({'A', 'AB'}, 'AB')
        a({'A', 'AB'}, 'BA')
        a({'D', 'E', 'DE'}, 'DE')
        a({'D', 'E', 'DE'}, 'ED')
        a({'B', 'D'}, 'BD')
        a({'B', 'D'}, 'DB')
        a({'A', 'F', 'AF', 'AD', 'AE',
           'ADEF', 'ADF', 'AEF', 'AED'}, 'ADEF')
        a({'A', 'F', 'AF', 'AD', 'AE',
           'ADEF', 'ADF', 'AEF', 'AED'}, 'FADE')
        a({'A', 'F', 'AF', 'AD', 'AE',
           'ADEF', 'ADF', 'AEF', 'AED'}, 'FEDA')
