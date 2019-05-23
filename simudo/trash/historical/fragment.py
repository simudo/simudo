from __future__ import division, absolute_import, print_function
from builtins import bytes, dict, int, range, str, super

from ..pyaml import BaseFragment
from ..util.expr import DelayedForm

from copy import deepcopy
from collections import namedtuple
import operator
from functools import reduce, partial
from itertools import chain as ichain

import logging
import dolfin
import ufl
import pint

def sum_over_suffix(solution, suffix):
    vals = [solution[k] for k in solution
            if k.endswith(suffix)]
    return reduce(operator.add, vals)

def endswith(s, suffix):
    return s[:len(s)-len(suffix)] if s.endswith(suffix) else None

class ConstantValue(object):
    def __init__(self, value):
        self.value = value
    def __call__(self, *args, **kwargs):
        return self.value

class DolfinPyamlFragment(BaseFragment):
    def process_trial_test_defs(self, tree):
        for k, d in tree.items():
            self._process_trial_test_def(k, d)

    def _process_trial_test_def(self, trial_name, d):
        edef = self.emit_definition
        path = self.path
        code = self.code

        k = path(trial_name)
        kt = path(d['test'])
        target_space = path(d['target_space'])
        target_func = d.get('target_function', None)
        if target_func is None:
            target_func = target_space + '/function'
        target_func = path(target_func)
        edef(k+'/element', code(d['element'], desc=k+'_element'))
        edef(k+'/trial_key', ConstantValue(k))
        edef(k+'/test_key', ConstantValue(kt))
        edef(k+'/trial_units', code(d['trial_units'], desc=k+'_trial_units'))
        edef(k+ '/test_units', code(d[ 'test_units'], desc=k+ '_test_units'))
        edef(k , lambda s: s[target_func+'/trial/split'][k])
        edef(kt, lambda s: s[target_func+ '/test/split'][kt])
        edef(k+'/@target_mixed_function_space', ConstantValue(target_space))
        edef(k+'/space', lambda s:
             s['/function_subspace_registry'].get_function_space(
                 s[k].magnitude, new=True))

    def process_misc_predefine_spaces(self, tree):
        def _element(space_type, space_degree, is_vector):
            cls = dolfin.VectorElement if is_vector else dolfin.FiniteElement
            return lambda s: cls(space_type, s['/ufl_cell'], space_degree)

        def _space(element_key):
            return lambda s: s['/function_space_cache'].FunctionSpace(
                s['/mesh'], s[element_key])

        continuous_spaces = {'CG'}
        for space_type in ('DG', 'CG', 'BDM'):
            continuous = space_type in continuous_spaces
            min_degree = 1 if continuous else 0
            for space_degree in range(min_degree, 10):
                for is_vector in (True, False):
                    k = '/pde/space/{}{}{}'.format(
                        'v' if is_vector else '', space_type, space_degree)
                    self[k+'/element'] = _element(
                        space_type, space_degree, is_vector)
                    self[k] = _space(k+'/element')

    def process_mixed_functions(self, tree):
        for k, d in tree.items():
            self._process_mixed_function(k, d)

    def _process_mixed_function(self, prefix, d):
        edef = self.emit_definition
        path = self.path
        code = self.code

        prefix = path(prefix)
        space = path(d['space'])
        if not prefix.endswith('/'):
            prefix = prefix + '/'
        edef(prefix+'trial', lambda s:
             s['/function_subspace_registry'].Function(s[space]))
        edef(prefix+'test', lambda s:
             dolfin.TestFunction(s[space]))
        for tt in ('trial', 'test'):
            edef(''.join((prefix, tt, '/split')),
                 partial(lambda tt, s:
                         s[space+'/split'](s[prefix+tt], tt), tt))
        def _split_dict(s, split_target_prefix='/'):
            d = {}
            for tt in ('trial', 'test'):
                d.update(((split_target_prefix+k), v) for k, v in
                         s[''.join((prefix, tt, '/split'))].items())
            return d
        edef(prefix+'/split_dict', partial(partial, _split_dict))

    def process_mixed_function_space(self, tree):
        for k, d in tree.items():
            self._process_mixed_function_space(k, d)

    def _process_mixed_function_space(self, key, d):
        edef = self.emit_definition
        path = self.path
        code = self.code

        key = path(key)

        def _subfunctions_keys(s):
            ks = []
            for k in s:
                base_key = endswith(k, "/@target_mixed_function_space")
                if base_key is None: continue
                if s[k] == key:
                    ks.append(base_key)
            return tuple(sorted(ks))

        def _element(s):
            elements = [s[k+'/element'] for k in s[key+'/subfunctions_keys']]
            if len(elements) == 0:
                raise AssertionError()
            elif len(elements) == 1:
                return elements[0]
            else:
                return dolfin.MixedElement(elements)

        def _split(s):
            def func(function, trial_or_test):
                assert trial_or_test in ('trial', 'test')
                if len(s[key+'/subfunctions_keys']) == 1:
                    subfunctions = [function]
                else:
                    subfunctions = dolfin.split(function)
                J = ''.join
                tt = trial_or_test
                return {s[J((k, '/', tt, '_key'))]:
                        s[J((k, '/', tt, '_units'))]*sub
                        for (k, sub) in zip(s[key+'/subfunctions_keys'],
                                            subfunctions)}
            return func

        edef(key+'/subfunctions_keys', _subfunctions_keys)
        edef(key+'/element', _element)
        edef(key+'/split', _split)
        edef(key, lambda s: s['/function_space_cache']
             .FunctionSpace(s['/mesh'], s[key+'/element']))

    def process_material_properties(self, tree):
        for k in tree:
            self._process_material_property(k)

    def _process_material_property(self, key):
        path = self.path
        base = path('').rstrip('/') # typically "/material"

        def property_global_value(solution):
            r = {}
            cregions = solution['/geometry/cell_regions']
            # TODO: ensure traversal order goes from general to specific
            for k in solution[base+'/regions']:
                # "/material", "my_region", "property/name"
                prop_path = '/'.join((base, k, key))
                if prop_path in solution:
                    for cv in cregions[k]:
                        r[cv] = solution[prop_path]

            cell_func = solution['/geometry/param/cell_function']
            units = next(iter(r.values())).units
            return (units*solution['/geometry/param/cell_function/func']
                    .make_cases({k: v.m_as(units)
                                 for k, v in r.items()}))

        self.emit_definition(path('GLOBAL/'+key),
                             property_global_value)

    def process_instantiate(self, tree):
        key = tree # it's actually just a string
        fragtree = self.fragment_trees[key]
        frag = type(self)(self.fragment_trees, self.envs, fragtree)
        frag.definitions = self.definitions
        frag.default_env = self.default_env
        frag.instantiate()

    def process_bcs(self, tree):
        for k, d in tree.items():
            self._process_bc(k, d)

    def _process_bc(self, key, d):
        key = self.path(key)

        var_key  = key+'/param/var'
        vals_key = key+'/param/values'

        self.process_defs({var_key:  d['var'],
                           vals_key: d['values']})

        def create_dbc(s):
            DBC = dolfin.DirichletBC
            var = s[var_key]
            try:
                space = s['/function_subspace_registry'].get_function_space(
                    var.magnitude)
                try:
                    dbc_method = s[key+'/method']
                except KeyError:
                    dbc_method = 'topological'
                try:
                    new_space = s[key+'/projection_space']
                except KeyError:
                    new_space = s['/function_subspace_registry'].get_function_space(
                        var.magnitude, new=True)
            except:
                logging.warn("ignoring declare_bc for non-variable key {!r}"
                             .format(var_key))
                return []
            r = []
            for facet_region_key, value in s[vals_key].items():
                value = value.m_as(var.units)
                 # FIXME: UGLY INEFFICIENT HACK
                value_ = dolfin.project(value, new_space)
                ff = s['/geometry/param/facet_function']
                fvs_key = facet_region_key + '/fvs'
                fvs = frozenset(fv for fv, sign in s[fvs_key])
                # TODO: determine when usage of geometric marking is necessary
                for fv in fvs:
                    # print("==== DBC", space, value, ff, fv)
                    r.append(DBC(space, value_, ff, fv))
            return r
        def create_nbc(ignore_sign, s):
            k = '/ds_unsigned' if ignore_sign else '/ds'
            return DelayedForm.from_sum(
                s[facet_region_key+k]*value
                for facet_region_key, value in s[vals_key].items())
        decl = self.emit_definition
        decl(key+'/essential', create_dbc)
        decl(key+'/natural', partial(create_nbc, False))
        decl(key+'/natural_unsigned_ds', partial(create_nbc, True))

    def process_make_geometry(self, tree):
        R = self.definitions

        p = '/geometry/'
        pp = p + 'param/' # parameter prefix

        meta_key = p+'mesh_meta'
        meta = R[meta_key].value

        R[p+'facets'] = lambda s: s[p+'mesh_meta']['facets']

        # cv=cell value, fv=facet value

        def _fvs_to_dS(fvs_key, dS_key, ignore_sign, s):
            d = s[p+'facets']
            value = DelayedForm()
            dS = s[dS_key]
            for facet_value, sign in s[fvs_key]:
                value = value + dS(facet_value)*(1 if ignore_sign else sign)
            return value

        def _dsS_sum(prefix):
            return lambda s: s[prefix+'ds'] + s[prefix+'dS']

        def _cvs_to_dx(cvs_key, solution):
            return DelayedForm.from_sum(
                solution[pp+'dx'](cv)
                for cv in solution[cvs_key])

        def _cvs_to_chi(cvs_key, solution):
            dct = {cv: dolfin.Constant(1.0)
                   for cv in solution[cvs_key]}
            return solution[pp+'cell_function/func'].make_cases(
                dct, default=0.0)

        def _get_creg(name, solution):
            return solution[meta_key]['cell_regions'][name]

        def _get_freg(name, solution):
            return solution[meta_key]['facet_regions'][name]

        def _get_key(name):
            return lambda solution: solution[name]

        for a in ('facet_regions', 'cell_regions', 'facets'):
            R[p+a] = partial(
                lambda a, s: s[meta_key][a], a)

        pr = p + 'facet_regions/'
        fregions = meta['facet_regions']
        for k in fregions.keys():
            kpr = '{}{}/'.format(pr, k)

            fvs_key = kpr+'fvs'
            R[fvs_key]  = partial(_get_freg, k)

            R[kpr+'ds'] = partial(_fvs_to_dS, fvs_key, pp+'ds', False)
            R[kpr+'dS'] = partial(_fvs_to_dS, fvs_key, pp+'dS', False)
            R[kpr+'ds_unsigned'] = partial(
                _fvs_to_dS, fvs_key, pp+'ds', True)
            R[kpr+'dsS'] = _dsS_sum(kpr)

        pr = p + 'cell_regions/'
        cregions = meta['cell_regions']
        for k in cregions.keys():
            kpr = '{}{}/'.format(pr, k)

            cvs_key = kpr+'cvs'
            ifvs_key = kpr+'ifvs'
            R[cvs_key] = partial(_get_creg, k)
            R[ifvs_key] = partial(_get_freg, 'internal:'+k)

            R[kpr+'dx'] = partial(_cvs_to_dx, cvs_key)
            R[kpr+'dS'] = partial(_fvs_to_dS, ifvs_key, pp+'dS', False)
            R[kpr+'chi'] = partial(_cvs_to_chi, cvs_key)
