
import os as _os
import dolfin

__all__ = ['setup_dolfin_parameters']

_FEM_DIR = _os.path.dirname(_os.path.abspath(__file__))


def update_dolfin_external_includes(headers):
    ffc_param = dolfin.parameters["form_compiler"]
    current = set(x for x in ffc_param["external_includes"].split(":") if x)
    current.update(headers)
    ffc_param["external_includes"] = ":".join(current)


def update_dolfin_external_include_dirs(dirs):  # Claude Sonnet 4.6
    ffc_param = dolfin.parameters["form_compiler"]
    current = set(x for x in ffc_param["external_include_dirs"].split(":") if x)
    current.update(dirs)
    ffc_param["external_include_dirs"] = ":".join(current)


def _add_to_dolfin_pc_include_dirs(dirs):  # Claude Sonnet 4.6
    # dolfin.jit.jit.dolfin_pc["include_dirs"] is the list that dolfin injects
    # directly into form_compiler_parameters when calling ffc_jit.  Appending
    # here is the only reliable way to get a custom directory into every JIT
    # compilation, because dolfin/fem/form.py overwrites external_include_dirs
    # with this list, bypassing dolfin.parameters["form_compiler"].
    from dolfin.jit.jit import dolfin_pc
    include_dirs = dolfin_pc["include_dirs"]
    for d in dirs:
        if d not in include_dirs:
            include_dirs.append(d)


def setup_dolfin_parameters():
    parameters = dolfin.parameters
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["cpp_optimize_flags"] = '-O3 -march=native'

    update_dolfin_external_includes(["simudo_math_extras.hpp"])
    # Two code paths need our FEM dir in the include search path:
    # 1. dolfin.FunctionSpace JIT reads dolfin.parameters["form_compiler"]["external_include_dirs"]
    # 2. dolfin.Form JIT (and its element dependencies) reads dolfin_pc["include_dirs"] directly
    #    and overwrites the form_compiler parameter, so we must patch both.
    update_dolfin_external_include_dirs([_FEM_DIR])
    _add_to_dolfin_pc_include_dirs([_FEM_DIR])
