from future.utils import PY2, PY3, native

__all__ = ['raise_from']

if PY2:
    # no traceback :(
    from future.utils import raise_from
else:
    # use native python3 'raise from' statement to get tracebacks
    from .raise_from_py3 import raise_from
