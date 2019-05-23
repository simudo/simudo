
from cached_property import cached_property

__all__ = ['TODO']

class TODO():
    ''' Maintain a list of callbacks to be done later. Useful for
plotting for example.

>>> from simudo.util.todo import TODO
>>> todo = TODO()
>>> todo(0, print, "a")
>>> todo(2, print, "c")
>>> todo(1, print, "b")
>>> todo.call()
a
b
c

'''
    @cached_property
    def list(self):
        return []

    def __call__(self, *args, **kwargs):
        ''' Add call to be performed later.

Parameters
----------
order: orderable
    Sorting key to determine order in which callback is
    called. Smaller comes first.
callback: callable
    Callable.
*args:
    Extra positional arguments to pass to `callback`.
**kwargs:
    Extra keyword arguments to pass to `callback`.
'''
        order, callback = args[:2]
        args = args[2:]
        self.list.append((order, callback, args, kwargs))

    def call(self):
        ''' Perform calls in order dictated by their sorting key. '''
        lst = self.list
        lst.sort(key=lambda x: x[0])
        for order, func, args, kwargs in lst:
            func(*args, **kwargs)

