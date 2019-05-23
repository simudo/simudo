from __future__ import absolute_import, division, print_function

import logging
from builtins import dict, super
from copy import deepcopy

import numpy
from cached_property import cached_property

import dolfin
from petsc4py import PETSc

# cached_property = property

__all__ = [
    'NewtonSolution',
    'NewtonSolver',
    'NewtonSolverMaxDu',
    'NewtonSolverLogDamping',
    'ExitOptimizerException']

def petsc_diag(vec):
    n = vec.sizes[0]
    m = PETSc.Mat()
    m.create(PETSc.COMM_WORLD)
    m.setSizes([n, n])
    m.setType('aij')  # sparse
    m.setPreallocationNNZ(5)
    m.setDiagonal(vec)
    return m

# originally we want to solve A x = b
# but instead we're solving (A Q_inv)(Q x) = b
# let q be the diagonal of Q_inv


class NewtonSolution(object):
    def __init__(self, solver, u_=None):
        self.solver = solver
        W = solver.W
        self.u_ = u_ if u_ else dolfin.Function(W)  # current solution
        self.u  =               dolfin.Function(W)  # next solution
        self.du =               dolfin.Function(W)  # change to current solution
        self.Qdu = dolfin.PETScVector()

    def do_assemble_system(self, assemble_A=True, assemble_b=True):
        if assemble_A: self.A
        if assemble_b: self.b

    @cached_property
    def size(self):
        return len(self.u_.vector())

    @cached_property
    def q(self):
        return None

    @cached_property
    def A(self):
        A = dolfin.PETScMatrix()
        self.solver.assembler.assemble(A)
        q = self.q
        if q is not None:
            A.mat().diagonalScale(None, q)
        return A

    @cached_property
    def b(self):
        b = dolfin.PETScVector()
        self.solver.assembler.assemble(b, self.u_.vector()) # <---- (*)
        # (*) If you're wondering how we can just apply nonzero Dirichlet BC's
        # despite du being required to be equal to zero on u's Dirichlet
        # boundaries, then look no further. The answer is that second argument
        # to SystemAssembler::assemble().
        return b

    @cached_property
    def du_norm(self):
        return numpy.linalg.norm(
            self.du.vector().get_local(), ord=numpy.Inf)

    @cached_property
    def has_nans(self):
        return numpy.isnan(self.u_.vector().get_local()).any()

    @cached_property
    def rel_du_norm(self):
        # |du| / (|u| + |du|) bad
        # = 1 / (|u|/|du| + 1) bad
        du = self.du.vector().get_local()
        u = self.u_.vector().get_local()

        with numpy.errstate(divide='ignore', invalid='ignore'):
            r = numpy.abs(du/u)

        ok = ~numpy.isnan(r)
        if not any(ok):
            return numpy.nan
        return numpy.linalg.norm(
            r[ok], ord=2)

    @cached_property
    def b_norm(self):
        return numpy.linalg.norm(
            self.b.get_local(), ord=numpy.Inf)

    def do_linear_solve(self):
        A, Qdu, b = self.A, self.Qdu, self.b
        q = self.q

        solver = dolfin.PETScLUSolver("umfpack")
        solver.solve(A, Qdu, b)


        ### trash not working code ###

        # solver = dolfin.KrylovSolver('gmres', 'ilu')
        # prm = solver.parameters
        # prm['absolute_tolerance'] = 1E-5
        # prm['relative_tolerance'] = 1E-5
        # prm['maximum_iterations'] = 5000
        # prm["monitor_convergence"] = True

        # solver.solve(A, Qdu, b)

        # try:
        #     solver.solve(self.A, Qdu, self.b)
        # except:
        #     try:
        #         prm['nonzero_initial_guess'] = True
        #         solver.solve(self.A, Qdu, self.b)
        #     except:
        #         raise

        if q is not None:
            dolfin.as_backend_type(self.du.vector()).vec().pointwiseMult(
                q, Qdu.vec())
        else:
            self.du.vector()[:] = Qdu.get_local() # FIXME: EXTREMELY INEFFICIENT!
            # Qdu.vec().copy(self.du.vector().vec())

    def str_error(self):
        abserr = getattr(self, 'b_norm', -1)
        relerr = getattr(self, 'du_norm', -1)
        relrelerr = getattr(self, 'rel_du_norm', -1)
        return "|b| {:.6e}, |du| {:.6e}, |du/u| {:.6e}".format(
            abserr, relerr, relrelerr)

    _no_copy = frozenset(('solver','A','b','q','size'))

    def copy_to(self, other):
        for k, v in self.__dict__.items():
            if k in self._no_copy: continue
            if isinstance(v, dolfin.Function):
                getattr(other, k).assign(getattr(self, k))
            # elif isinstance(v, (dolfin.PETScMatrix, dolfin.PETScVector)):
            #     setattr(other, k, v.copy())
            else:
                setattr(other, k, deepcopy(v))

    def copy(self):
        other = self.__class__(self.solver)
        self.copy_to(other)
        return other

    _cached_property_attrs = frozenset(
        ('rel_du_norm', 'du_norm', 'b_norm', 'A', 'b', 'q', 'size',
         'has_nans'))

    def do_invalidate_cache(self, check=False):
        d = self.__dict__
        for k in self._cached_property_attrs:
            if d.pop(k, None) is not None:
                if check:
                    logging.info('*'*30 + k)

class NewtonSolver(object):
    '''A general Newton solver.

Parameters
----------
F: :py:class:`ufl.Form`
    Nonlinear form to be solved (by setting equal to zero). That is,
    we are trying to solve :code:`F(u_) = 0` by varying :code:`u_`.
u\_: dolfin.Function
    Function being varied (read: solved for).
bcs: list of dolfin.DirichletBC
    List of essential boundary conditions to be applied.
J: :py:class:`ufl.core.expr.Expr`, optional
    Jacobian of the form with respect to :code:`u_`'s coefficients. If
    this is not supplied, it is derived from :code:`F` automatically.
parameters: dict
    Parameters controlling the Newton solution process.

Attributes
----------
iteration: int
    Current iteration.

Notes
-----
The underscore in :code:`u_` is a subscript minus, and signifies that
this is the "previous" value of the function is used to evalute the
form in the iterative Newton process for nonlinear problems.
'''

    iteration = 0
    solution_class = NewtonSolution
    du_mask_array = None
    extra_iterations = None

    def __init__(self, F, u_, bcs, J=None, parameters=None):
        self.do_init_parameters()
        if parameters is not None:
            self.parameters.update(parameters)
        self.W = u_.function_space()
        self.do_init_solution(u_)
        self.set_problem(F, u_, bcs, J=J)

        self.user_before_first_iteration_hooks = []
        self.user_pre_iteration_hooks = []
        self.user_post_iteration_hooks = []

    @classmethod
    def from_nice_obj(cls, obj):
        F   = obj.get_weak_form().to_ufl().magnitude
        u_  = obj.get_solution_function()
        bcs = obj.get_essential_bcs()

        return cls(F=F, u_=u_, bcs=bcs)

    def set_problem(self, F, u_, bcs, J=None):
        self.F = F
        self.J = dolfin.derivative(F, u_) if J is None else J
        self.bcs = bcs
        self.do_init_assembler()

    def create_solution(self, *args, **kwargs):
        return self.solution_class(self, *args, **kwargs)

    def do_init_solution(self, u_):
        self.solution = self.create_solution(u_=u_)

    def do_init_parameters(self):
        self.parameters = dict(maximum_iterations=25,
                               minimum_iterations=-1,
                               relaxation_parameter=1.0,
                               absolute_tolerance=1e-5,
                               relative_tolerance=1e-6,
                               extra_iterations=5)

    def do_init_assembler(self):
        try:
            self.assembler =  dolfin.fem.assembling.SystemAssembler(
                self.J, self.F, self.bcs)
        except TypeError as exc:
            err = self.logger.error
            err("SystemAssembler arguments follow:")
            err("  J={!r}".format(self.J))
            err("  F={!r}".format(self.F))
            err("  bcs={!r}".format(self.bcs))
            raise
        self.solution.do_invalidate_cache()

    def do_update_solution(self):
        s = self.solution
        du = s.du.vector()[:]
        maskv = self.du_mask_array
        if maskv is not None:
            du = du*maskv
        self.user_update_solution_vector(s, du)
        s.do_invalidate_cache()

    def user_update_solution_vector(self, solution, du):
        s = solution
        s.u_.vector()[:] -= du*self.get_omega()

    def do_iteration(self):
        sltn = self.solution
        self.do_pre_iteration_hook()
        sltn.do_assemble_system()
        sltn.do_linear_solve()
        self.do_update_solution()
        sltn.do_invalidate_cache()
        self.do_post_iteration_hook()

    @cached_property
    def logger(self):
        return logging.getLogger('newton')

    def log_print(self, string='', end='\n'):
        # print(string, file=sys.stderr, end=end)
        self.logger.info(string)

    def do_before_first_iteration_hook(self):
        self.solution.do_invalidate_cache()
        for hook in self.user_before_first_iteration_hooks:
            hook(self)

    def do_pre_iteration_hook(self):
        self.solution.do_invalidate_cache()
        for hook in self.user_pre_iteration_hooks:
            hook(self)

    def do_post_iteration_hook(self):
        self.log_print("{}* iteration {:>3d}; {}".format(
            '\n' if self.iteration == 0 else '',
            self.iteration, self.solution.str_error()))
        self.solution.do_invalidate_cache()
        for hook in self.user_post_iteration_hooks:
            hook(self)

    def get_omega(self):
        omega_cb = self.parameters.get('omega_callback', None)
        if omega_cb is not None:
            return omega_cb(self)
        if self.iteration <= self.parameters.get('num_damped_iterations', 5):
            return self.parameters['relaxation_parameter']
        else:
            return 1.0

    def solve(self):
        self.do_before_first_iteration_hook()
        while True:
            self.iteration += 1
            self.do_iteration()
            if self.should_stop_real():
                break

    def has_converged(self):
        sltn = self.solution
        return ((not self.solution.has_nans) and
                (sltn.du_norm <= self.parameters['relative_tolerance']))

    def should_stop_real(self):
        extra = self.extra_iterations
        should_stop = self.should_stop()

        if self.solution.has_nans:
            return True # solution is nan-poisoned, all hope is lost

        if extra is not None:
            if should_stop: # endgame mode: nail the solution further
                self.extra_iterations = extra = extra - 1
                return extra <= 0
            else: # false minimum in error metric, cancel endgame mode
                self.extra_iterations = None

        if should_stop:
            extra = self.parameters.get('extra_iterations')
            if extra is None:
                return True
            else:
                self.extra_iterations = extra
                return extra <= 0
        else:
            return False

    def should_stop(self):
        sltn = self.solution
        iteration = self.iteration
        if (iteration < self.parameters['minimum_iterations']):
            return False
        return ((iteration >= self.parameters['maximum_iterations'])
                or (self.has_converged())
                or (numpy.isnan(sltn.du_norm)))


class ExitOptimizerException(Exception):
    pass


class RestartingNewtonSolution(NewtonSolution):
    @cached_property
    def cost(self):
        return self.b_norm # + numpy.linalg.norm(self.b.get_local(), ord=2)/10000.0

    # numpy.linalg.norm(self.b.get_local(), ord=2)
    _cached_property_attrs = frozenset(('cost',)).union(
        NewtonSolution._cached_property_attrs)


class RestartingNewtonSolver(NewtonSolver):
    solution_class = RestartingNewtonSolution

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.newton_history = []
        self.optimizer_history = []
        self.opt_extra_iterations_left = None

    def get_minimum_cost(self):
        return self.solution.cost*1.3

    def get_optimizer_extra_iterations(self):
        return 5

    def get_optimizer_dimension(self):
        return 2

    def compute_linear_combination(self, x):
        cs, vs = self.optimizer_vector, self.optimizer_basis
        assert len(cs) == len(vs)-1
        self.optimizer_coefficients = cs = [c-1.0 for c in cs]
        self.optimizer_coefficients.insert(1, 1.0-sum(cs))
        return sum(v*c for (v, c) in
                   zip(vs, cs))

    def post_optimizer_iteration_hook(self):
        self.do_post_iteration_hook()

    def post_newton_iteration_hook(self):
        self.do_post_iteration_hook()

    def cost_function(self, x):
        sl = self.solution
        self.current_solver_mode = 'optimizer'
        self.optimizer_vector = x
        sl.u_.vector()[:] = self.compute_linear_combination(x)

        def constraint(a):
            return -a if a <= 0.0 else 0.0
        penalty = (constraint((5.0 - max(self.optimizer_coefficients)))
                   + constraint(-(-5.0 - min(self.optimizer_coefficients))))
        sl.do_invalidate_cache()
        sl.do_assemble_system(assemble_A=False)
        self.post_optimizer_iteration_hook()
        self.optimizer_history.insert(0, sl.copy())
        self.iteration += 1
        sc = sl.cost
        if sc < self.min_cost:
            ei = getattr(self, 'opt_extra_iterations_left', None)
            if ei is None:
                ei = self.get_optimizer_extra_iterations()
            ei -= 1
            if ei < 0:
                self.opt_extra_iterations_left = None
                raise ExitOptimizerException()
            self.opt_extra_iterations_left = ei
        return sc + penalty*1000

    def optimizer_procedure(self, func, x0s):
        from scipy.optimize import fmin_l_bfgs_b
        for x0 in x0s[1:]:
            func(x0)
        fmin_l_bfgs_b(func=func, x0=x0s[0], approx_grad=1)

    def do_post_iteration_hook(self):
        if self.iteration == 0:
            self.log_print()
        self.log_print("*** {:>9s} iteration {:>3d}; {}".format(
            self.current_solver_mode,
            self.iteration, self.solution.str_error()))
        if self.current_solver_mode == 'optimizer':
            self.log_print(" ---> {}".format(self.optimizer_coefficients))

    def solve(self):
        nhistory = self.newton_history
        ohistory = self.optimizer_history
        self.min_cost = numpy.Inf
        sl = self.solution
        self.current_solver_mode = 'null'
        self.do_before_first_iteration_hook()
        while True:
            self.current_solver_mode = 'newton'
            self.do_iteration()
            self.iteration += 1
            nhistory.insert(0, sl.copy())
            odim = self.get_optimizer_dimension()
            if len(nhistory) >= odim and not (sl.cost <= self.min_cost):
                def vcopy(w):
                    v = sl.u_.copy(deepcopy=True)
                    v.assign(w)
                    return v.vector().array()
                self.optimizer_basis = [vcopy(s.u_) for s in nhistory[:odim]]
                #x0 = (1.0, 2.0) + (1.0,)*(odim-2)
                x0s = [(1.0+10**x,) + (1.0,)*(odim-2) for x in numpy.linspace(-5, -0.1, 6)]
                try:
                    self.optimizer_procedure(self.cost_function, x0s)
                except ExitOptimizerException:
                    pass
                ohistory.sort(key=lambda s:s.cost)
                sl.u_.assign(ohistory[0].u_)
                sl.do_invalidate_cache()
                nhistory.insert(0, sl.copy())
                del ohistory[:]
            if self.should_stop_real():
                break
            self.min_cost = self.get_minimum_cost()


class NewtonSolverMaxDu(NewtonSolver):
    def user_update_solution_vector(self, solution, du):
        maxdu = self.parameters.get('maximum_du', None)
        if maxdu is not None:
            du = numpy.minimum(du,  maxdu)
            du = numpy.maximum(du, -maxdu)
        super().user_update_solution_vector(solution, du)


class NewtonSolverLogDamping(NewtonSolver):
    def user_update_solution_vector(self, solution, du):
        # ref:[Gaury2018a]
        alpha = self.parameters.get('logdamping_alpha', 1.72)
        du = numpy.sign(du)*numpy.log1p(numpy.abs(du) * alpha) / alpha
        super().user_update_solution_vector(solution, du)
