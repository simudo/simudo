import fcntl
import os
import threading
from contextlib import contextmanager
from os import path as osp

from cached_property import cached_property

import ffc
from ffc import jitcompiler

'''
hack: monkey-patch `ffc.jitcompiler` to add file-based locking so
different dolfin instances sharing the same `cache_dir` don't
interfere with each other
'''

# TODO: get dijitso path and use that instead instead of hardcoding
lockfilename = osp.join(os.environ['HOME'], '.cache/dolfin_ffc_lock/jit.lock')

class LockFile(object):
    def __init__(self, filename):
        self.filename = filename
        self._lock = threading.Lock()
        self._lock_count = 0

    @cached_property
    def _file(self):
        os.makedirs(osp.dirname(self.filename), exist_ok=True)
        return open(self.filename, 'w+b')

    @cached_property
    def _fd(self):
        return self._file.fileno()

    def acquire(self):
        with self._lock:
            if self._lock_count == 0:
                fcntl.lockf(self._fd, fcntl.LOCK_EX)
            self._lock_count += 1
        # print("####### acquire lock pid={}".format(os.getpid()))

    def release(self):
        # print("####### release lock pid={}".format(os.getpid()))
        with self._lock:
            self._lock_count -= 1
            if self._lock_count == 0:
                fcntl.lockf(self._fd, fcntl.LOCK_UN)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

lockfile = LockFile(lockfilename)

old_jit = jitcompiler.jit
def new_jit(ufl_object, parameters=None, indirect=False):
    with lockfile:
        return old_jit(ufl_object, parameters, indirect)
jitcompiler.jit = new_jit
ffc.jit = new_jit
