from ..util.adaptive_solver import AdaptiveSolver
from copy import deepcopy
import numpy as np

class MaladaptiveSolver(AdaptiveSolver):
    def user_solver(self, solution, parameter):
        print("parameter", parameter)
        thr = 0.02 if parameter < 0.98 else 0.001
        solution[1] = parameter
        if abs(solution[0] - parameter) > thr:
            print("jump too big")
            return False
        else:
            print("ok")
            solution[0] = parameter
            return True

    def user_solution_save(self, solution):
        return deepcopy(solution)

    def user_solution_add(self, solution, solution_scale,
                          saved_solution, saved_scale):
        print("pre-add", solution)
        solution *= solution_scale
        solution += saved_solution * saved_scale
        print("added", solution, solution_scale, saved_solution, saved_scale)

a = MaladaptiveSolver()
a.solution = np.array([0.0, 0.0])

while a.parameter < 1.0:
    a.do_iteration()

