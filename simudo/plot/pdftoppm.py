import os
import subprocess
import tempfile
from contextlib import contextmanager

__all__ = ['pdftoppm']

@contextmanager
def namedtemp(*args, **kwargs):
    with tempfile.NamedTemporaryFile(*args, delete=False, **kwargs) as f:
        yield f
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass

def pdftoppm(input, output=None, r=300, fmt="png", optipng=True):
    if output is None:
        output = '{}.{}'.format(input, fmt)
    with namedtemp(dir=os.path.dirname(output), suffix='.'+fmt) as f:
        tmp = f.name
        subprocess.check_call(['pdftoppm', input, tmp.rpartition('.')[0],
                               '-r', str(r),
                               '-'+fmt, '-f', '1', '-singlefile'],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        if optipng and fmt=='png':
            subprocess.check_call(['optipng', tmp],
              stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL)
        os.rename(tmp, output)
