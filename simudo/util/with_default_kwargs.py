
from collections import ChainMap
from functools import wraps

__all__ = ['with_default_kwargs']

def with_default_kwargs():
    '''
Python doesn't like it if we pass the same argument by an explicit
kwarg and through `**kwargs`, e.g., :code:`f(y=3, **{'x': 4, 'y':
5})`.

This decorator transforms a function so that it receives a default
dict of kwargs through its first argument, then remaining kwargs
normally (latter taking precedence over the former).

Assuming `f` was decorated with this wrapper, the call above becomes
:code:`f({'x': 4, 'y':5}, y=3)`.

You can (ab)use this function to be lazy and pass all the local
variables to a function as kwargs, and still have the option of
overriding some of them, e.g., :code:`f(locals(), y=3)`.
'''
    def wrapper(func):
        @wraps(func)
        def wrapped(default_kwargs, *args, **kwargs):
            return func(*args, **ChainMap(kwargs, default_kwargs))
        return wrapped
    return wrapper

