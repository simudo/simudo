
import ast
import sys

PY3 = sys.version_info[0] == 3

def make_function(name, argname, body):
    if PY3:
        a = ast.FunctionDef(name=name, args=ast.arguments(
            args=[ast.arg(arg=argname, annotation=None)], vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                            body=body, decorator_list=[], returns=None)
    else:
        a = ast.FunctionDef(name=name, args=ast.arguments(
            args=[ast.Name(id=argname, ctx=ast.Param())],
            vararg=None, kwarg=None, defaults=[]),
                            body=body, decorator_list=[])
    return a

