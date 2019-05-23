
import hashlib
import importlib
import io
import os
import re
import tempfile
from os import path as osp

from cached_property import cached_property
from pkg_resources import resource_string

from . import pyaml1
from ..util import generate_base32_token


'''

There's a global instance of SourceTranslator in this module called
`translator`.

You can mostly just use the `load_res` function (which is actually a
method of that instance).

To control where the intermediate source translated files are stored,
you can use the PYAML_CACHE_PATH environment variable.

'''


class BaseSourceTranslator(object):
    @cached_property
    def cache_path(self):
        d = self.get_cache_path()
        os.makedirs(d, exist_ok=True)
        return d

    def get_cache_path(self):
        return os.environ.get(
            'PYAML_CACHE_PATH',
            osp.join(tempfile.gettempdir(), 'pyaml_cache'))

    NICENAME_BADCHAR_RE = re.compile(r'[^a-zA-Z0-9_-]')
    NICENAME_LEN = 60
    def make_nicename(self, path):
        return self.NICENAME_BADCHAR_RE.sub('_', path)[-self.NICENAME_LEN:]

    NICENAME_RANDOM_LEN = 30 # base32 characters

    def store_and_get_hashname(self, source):
        h = hashlib.sha256()
        h.update(source)
        shash = h.hexdigest()
        del h

        fn = osp.join(self.cache_path, 'h_{}.py'.format(shash))

        if not osp.exists(fn):
            fn_temp = fn+'.tmp'
            with open(fn_temp, 'wb') as h:
                h.write(source)
            os.rename(fn_temp, fn) # atomic

        return fn

    def load_string(self, source_string, module_name, original_filename):
        tsource = self.translate_code(source_string, original_filename)
        nicename = self.make_nicename(original_filename)

        nice_filename = 'link_{}_{}.py'.format(
            nicename,
            generate_base32_token(self.NICENAME_RANDOM_LEN))
        nice_filename = osp.join(self.cache_path, nice_filename)

        hashname = self.store_and_get_hashname(tsource)

        os.symlink(osp.relpath(
            hashname, start=osp.dirname(nice_filename)), nice_filename)

        spec = importlib.util.spec_from_file_location(
            module_name, nice_filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        return mod

    def load_res(self, package_or_requirement, resource_name,
                 use_parent_package=True):
        '''
        Use this to load a source-translated file from inside your
        source tree.

        For example, `translator.load_res(__name__, 'example.py1')`. '''

        if use_parent_package:
            before, sep, after = package_or_requirement.rpartition('.')
            if sep:
                package_or_requirement = before
            del before, sep, after

        module_name = '.'.join((package_or_requirement,
                                resource_name.partition('.')[0]))

        original_filename = '/'.join((package_or_requirement.replace('.', '/'),
                                      resource_name))

        return self.load_string(resource_string(
            package_or_requirement, resource_name),
                                module_name, original_filename)

class SourceTranslator(BaseSourceTranslator):
    def translate_code(self, source_string, co_filename):
        MAGIC = b'#pyaml1'
        if not source_string.startswith(MAGIC):
            raise ValueError("file does not start with '{}'".format(
                MAGIC.decode('ascii')))
        stream = io.BytesIO(source_string)
        return pyaml1.CodeGenerator.pyaml_to_python(
            stream=stream).encode('utf8')

translator = SourceTranslator()

load_res = translator.load_res

__all__ = ['load_res']
