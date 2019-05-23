
import dolfin

__all__ = ['setup_dolfin_parameters']

def setup_dolfin_parameters():
    parameters = dolfin.parameters
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["cpp_optimize_flags"] = '-O3 -march=native'
