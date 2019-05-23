
import os
import re
from collections import OrderedDict

__all__ = [
    'fullsplit',
    'mkdirp',
    'dir_as_prefix',
    'outdir_path_helper',
    'xlistdir',
    'parse_kv']

def fullsplit(path):
    ''' fully split path into components '''
    r = []
    split = os.path.split
    while True:
        head, tail = split(path)
        if not head:
            r.insert(0, tail)
            break
        elif not tail:
            r.insert(0, head)
            break
        else:
            r.insert(0, tail)
            path = head
    return r

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def dir_as_prefix(path):
    ''' Add the final path separator ("/") if necessary.

For example::

   "abc/def" -> "abc/def/"
   "abc/def/" -> "abc/def/"

'''
    return os.path.join(path, 'x')[:-1]

def outdir_path_helper(path):
    '''Calls :py:func:`mkdirp`, then returns :py:func:`dir_as_prefix`
applied on `path`.
'''
    mkdirp(path)
    return dir_as_prefix(path)

def xlistdir(path, both=False):
    ''' like listdir, but returns full paths
or tuples `(basename, fullpath)` '''
    join = os.path.join
    if both:
        return [(x, join(path, x)) for x in os.listdir(path)]
    else:
        return [join(path, x) for x in os.listdir(path)]

parse_kv_re = re.compile("(?:([^=\s]+)(?:=(\S*))?)")
def parse_kv(string, as_list=False):
    # TODO: escaping for whitespace and special chars
    result = parse_kv_re.findall(string)
    if not as_list:
        result = OrderedDict(result)
    return result


