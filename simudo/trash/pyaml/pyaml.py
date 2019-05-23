from .astutil import make_function

from cached_property import cached_property

import yaml
import yamlordereddictloader

from copy import deepcopy
import ast
import importlib
import re
import string
import sys
import types
import warnings

def path_join(*paths):
    # TODO: desperately needs unit test
    current = []
    isdir = False
    for path in paths:
        ps = path.split('/')
        if ps[0] == '' and len(ps) >= 2: # absolute path
            del current[:]
        if len(path):
            isdir = ps[-1] == ''
        for p in ps:
            if not len(p) or p == '.':
                continue
            elif p == '..':
                if len(current):
                    del current[-1]
            else:
                current.append(p)
    if not len(current):
        return '/'
    current.insert(0, '')
    if isdir:
        current.append('')
    return '/'.join(current)


approx_to_python_identifier_re = re.compile(r'[^A-Za-z0-9]+')
def approx_to_python_identifier(s):
    return approx_to_python_identifier_re.sub('_', s)

def id_to_python_identifier(obj):
    id_obj = id(obj)
    return 'PM'[int(id_obj < 0)] + str(id(obj)).strip('-')


class BaseFuture(object):
    value = None
    def force(self):
        raise NotImplementedError()

class EnvCodeFuture(BaseFuture):
    value = None
    def __init__(self, env, source, desc=None, extra_vars={}):
        self.env = env
        self.source = source
        self.desc = desc
        self.extra_vars = extra_vars
        if desc:
            desc_ = '_' + approx_to_python_identifier(desc)
        else:
            desc_ = ''
        self.name = '___ecf_{}_{}'.format(id_to_python_identifier(self),
                                          desc_)
        env.register_future(self)

    def force(self):
        if self.value is None:
            self.env.compile_futures()
        assert self.value is not None
        return self.value

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))


class BaseFragment(object):
    def __init__(self, fragment_trees, envs, tree):
        self.fragment_trees = fragment_trees
        self.envs = envs
        self.tree = tree
        self.default_prefix = '/'

    def path(self, path, env=None, relative_to=None):
        if relative_to is None:
            relative_to = self.default_prefix
        if env is None:
            env = getattr(self, 'default_env', None)
        if env is not None:
            path = env.expand_vars(path)
        p = path_join(relative_to, path)
        if path.endswith('/'):
            p = p + '/'
        return p

    def code(self, source, env=None, desc=None):
        if not source.startswith('%'):
            source = 'return ({})'.format(source)
        else:
            source = source[1:]
        if env is None: env = self.default_env
        return EnvCodeFuture(env, source, desc,
                             extra_vars={'_': self.default_prefix})

    def process_defs(self, obj):
        for k, v in obj.items():
            self.emit_definition(self.path(k), self.code(v, desc=k))

    def process_env(self, obj):
        pass

    def process_default_prefix(self, obj):
        pass

    def instantiate(self):
        t = self.tree
        self.default_prefix = self.path(t.get('default_prefix', '/'))
        env = t.get('env', None)
        if env is not None:
            self.default_env = self.envs[env]

        for k, v in t.items():
            method = getattr(self, 'process_' + k, None)
            if method is not None:
                method(v)
            else:
                warnings.warn("ignoring fragment key {!r}".format(k))

    def emit_definition(self, key, value):
        # TODO: change all uses to __getitem__
        self[key] = value

    def __getitem__(self, key):
        return self.definitions[key]

    def __setitem__(self, key, value):
        key = path_join(key) # normalize
        self.definitions[key] = value


class Env(object):
    ARGUMENT = 'solution'

    def __init__(self, envs, tree):
        self.envs = envs
        self.tree = tree
        self.futures = set()

    @cached_property
    def inherit(self):
        return [self.envs[k]
                for k in self.tree.get('inherit', ())]

    def _inherit_dict(self, attrname):
        d = dict(self.tree.get(attrname, {}))
        for e in self.inherit:
            for k, v in getattr(e, attrname).items():
                if k not in d:
                    d[k] = v
        return d

    @cached_property
    def vars(self):
        return self._inherit_dict('vars')

    @cached_property
    def meta(self):
        return self._inherit_dict('meta')

    def getter(self):
        def get(solution, key):
            e = solution._path_cache

    def expand_vars(self, s):
        return string.Template(s).substitute(self.vars)

    def _compile_ast(self, source, filename):
        return compile(source, filename, 'exec',
                       flags=ast.PyCF_ONLY_AST, dont_inherit=True)

    def register_future(self, future):
        self.futures.add(future)

    def compile_futures(self):
        futures = self.futures
        compile_to_ast = self._compile_ast
        mainarg = self.ARGUMENT

        mod_body = list(compile_to_ast(self.meta['setup'], '').body)

        def var_assign_stmt(target_variable, path):
            return ast.Assign(
                targets=[ast.Name(id=target_variable, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=mainarg, ctx=ast.Load()),
                    slice=ast.Index(value=ast.Str(s=path)), ctx=ast.Load()))

        prologue = []
        for var, path in self.vars.items():
            prologue.append(var_assign_stmt(var, path))

        for future in futures:
            code = compile_to_ast(future.source, future.name)

            body = deepcopy(prologue)
            body.extend(var_assign_stmt(var, path)
                        for var, path in future.extra_vars.items())
            body.extend(code.body)

            funcdef = make_function(future.name, mainarg, body)

            mod_body.append(funcdef)

        mod_ast = ast.Module(body=mod_body)
        ast.fix_missing_locations(mod_ast)
        mod_code = compile(mod_ast, '<ast>', 'exec')

        module = types.ModuleType('<ast>')
        module_name = '{}{}'.format(self.meta['module_name_prefix'],
                                    id_to_python_identifier(module))
        module.__builtins__ = __builtins__
        module.__name__ = module_name
        module.__file__ = '[{}]'.format(module_name)
        module.__package__ = module_name.rpartition('.')[0]
        # module.__loader__ = ...

        modules = sys.modules
        modules[module_name] = module

        # make sure parent module is loaded
        importlib.import_module(module.__package__)

        try:
            eval(mod_code, module.__dict__, module.__dict__)
        finally:
            pass
            # del modules[module_name]

        for future in futures:
            future.value = getattr(module, future.name)

        futures.clear()

def load(datas, fragment_class):
    envs = {}
    frag_trees = {}

    for data in datas:
        tree = yaml.load(data, Loader=yamlordereddictloader.Loader)
        envs.update((k, Env(envs, v)) for k, v in tree.get('envs', {}).items())
        frag_trees.update(tree.get('fragments', {}).items())

    frags = {k: fragment_class(frag_trees, envs, v)
             for k, v in frag_trees.items()}

    return (envs, frags)


