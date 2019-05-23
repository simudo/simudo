
from ..util import AttrPrefixProxy, DictAttrProxy

_NULL = []

class PyamlEnv(object):
    def PYAML_get_direct_env(self, env, name, globals):
        method = getattr(self, 'ENV_{}_{}'.format(env, name), _NULL)
        if method is not _NULL:
            return method()
        else:
            v = globals.get(name, _NULL)
            if v is not _NULL: return v

            builtins = globals['__builtins__']
            v = builtins.get(name, _NULL)
            if v is not _NULL: return v

            raise NameError("name {!r} is not defined".format(name))

    def PYAML_get_env(self, env, name, globals):
        '''
        get `name` from environment `env`, or else look it up in
        `globals`
        '''
        method = getattr(self, 'PYAML_get_env_'+env, None)
        if method:
            return method(env, name, globals)
        else:
            return self.PYAML_get_direct_env(env, name, globals)

class PyamlGetAttr(object):
    def PYAML_EVAL(self, name):
        return getattr(self, 'EVAL_'+name)()

class PyamlBase(PyamlEnv, PyamlGetAttr):
    pass
