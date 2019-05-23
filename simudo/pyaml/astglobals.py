
import ast
import collections
import unittest
import warnings
from contextlib import contextmanager

from cached_property import cached_property

__all__ = ['find_globals', 'FindFreeVariablesClassWarning']

class FindFreeVariablesClassWarning(UserWarning):
    pass

class FindFreeVariables(ast.NodeVisitor):
    ''' use the `find_globals` utility function '''

    @cached_property
    def ffv_scope(self):
        return collections.ChainMap()

    @cached_property
    def ffv_globals(self):
        return set()

    # possible scopes: local, global, force_global

    @property
    def ffv_new_scope(self):
        scope = self.ffv_scope
        @contextmanager
        def context():
            scope.maps.insert(0, dict())
            yield
            fglobals = self.ffv_globals
            for name, var_scope in scope.maps[0].items():
                if var_scope != 'local':
                    fglobals.add(name)
            del scope.maps[0]
        return context

    def ffv_binding(self, name, state='local'):
        d = self.ffv_scope
        cur = d.get(name, 'global')
        if cur != 'force_global':
            if not (cur == 'local' and state == 'global'):
                d[name] = state

    def visit_ListComp(self, node):
        with self.ffv_new_scope():
            super().generic_visit(node)

    def visit_SetComp(self, node):
        self.visit_ListComp(node)

    def visit_DictComp(self, node):
        self.visit_ListComp(node)

    def visit_GeneratorExp(self, node):
        self.visit_ListComp(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store): # establish new binding
            self.ffv_binding(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.ffv_binding(node.id, state='global')
        super().generic_visit(node)

    def visit_FunctionDef(self, node):
        with self.ffv_new_scope():
            args = node.args
            for a in args.args:
                self.ffv_binding(a.arg)
            for a in args.kwonlyargs:
                self.ffv_binding(a.arg)
            if args.vararg:
                self.ffv_binding(args.vararg.arg)
            if args.kwarg:
                self.ffv_binding(args.kwarg.arg)
            super().generic_visit(node)

        name = getattr(node, 'name', None)
        if name: self.ffv_binding(node.name)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        warnings.warn(FindFreeVariablesClassWarning(
            "Global variable detection is not fully implemented for Python "
            "classes. This may lead to false dependencies being established "
            "in the caching system."))
        super().generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.ffv_binding(node.name)
        super().generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.ffv_binding(alias.asname)
        super().generic_visit(node)

    def visit_ImportFrom(self, node):
        self.visit_Import(node)

    def visit_Global(self, node):
        for name in node.names:
            self.ffv_binding(name, 'force_global')
        super().generic_visit(node)

    def visit_With(self, node):
        super().generic_visit(node)


def find_globals(ast_tree):
    '''
Returns a set of all global variable names referenced in the code.

Warning: does not yet work properly on classes.'''

    ffv = FindFreeVariables()
    ffv.visit(ast_tree)
    return ffv.ffv_globals


class UnitTest(unittest.TestCase):
    def test_me(self):
        codes = [('''\
def f(a, b):
  l1 = a + b
  g1(g2, l1)
''', 'g1 g2'),

                 ('''\
def f(a):
  l1, g1.abc, l2 = a + g2
  if a:
    l3 = g3(l2, name=l1, *l3, **g4)
  g5.A2[l3] = g6.A1[l2]
''', 'g1 g2 g3 g4 g5 g6'),

('''\
def f(a):
  for i in a:
    g1(a, i)
  l1 = g2[g3(i)]
  with g4(l1) as l2:
    g1(l1, l2)

  def h(b, arg=g5):
    g6(l1, l2, i)

  k = lambda b, arg=g7: g8(l1, b)
''', 'g1 g2 g3 g4 g5 g6 g7 g8'),

                 ('''\
def f(a):
  try:
    l1 = g1()
  except g4 as l2:
    l1 = g2(l2)
  finally:
    g3()
''', 'g1 g2 g3 g4'),

                 ('''\
def f(a):
  import abc as l1
  from abc import xyz as l2
  g1(l1, l2)
''', 'g1'),

                 ('''\
def f(a):
  global g1
  g1, l1 = a
''', 'g1'),

                 ('''\
def f(a):
  with g1(a, g2) as l1:
    g2(l1)
  g3(l1, a)
''', 'g1 g2 g3'),

                 ('''\
def f(a):
  g1(l1+g2 for l1 in a+g3)
  g4([l2+g5 for l2 in a+g6])
''', 'g1 g2 g3 g4 g5 g6'),

                 ('''\
def f(a):
  def l1():
    return a
  return l1
''', ''),

                 ('''\
def f1(g1):
  def f2():
    global g1
    def f3():
      g1()
''', 'g1'),
]
        for code, gs in codes:
            c = ast.parse(code)
            self.assertEqual(set(gs.split()), find_globals(c))

    def test_raises_warning(self):
        c = ast.parse('''
def f(a):
  class l1(g1, metaclass=g2):
    l2 = a
    g3 = a # here it's local to the class
    def method(self):
      g3() # here this is actually a global!
''')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            find_globals(c)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(
                w[0].category, FindFreeVariablesClassWarning))
