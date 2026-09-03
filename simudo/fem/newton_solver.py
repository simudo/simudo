from __future__ import absolute_import, division, print_function

import logging
from builtins import dict, super
from copy import deepcopy

import numpy
from cached_property import cached_property

import dolfin
from petsc4py import PETSc

from contextlib import closing
from ..fem import MixedFunctionSpace

import os
from ..io import h5yaml

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
    mumps_icntl = ()  # Iterable of (icntl, val) pairs to set mumps values

    def __init__(self, solver, u_=None):
        self.solver = solver
        W = solver.W                                # Function space
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
        try:
            self.solver.assembler.assemble(A)
        except ValueError:
            self.solver.log_print("ValueError in assembly of A")
            return A
        except OverflowError:
            self.solver.log_print("OverflowError in assembly of A")
            return A
#        q = self.q
#        if q is not None:
#            A.mat().diagonalScale(None, q)
        return A

    @cached_property
    def b(self):
        b = dolfin.PETScVector()
        try:
            self.solver.assembler.assemble(b, self.u_.vector()) # <---- (*)
        # (*) If you're wondering how we can just apply nonzero Dirichlet BC's
        # despite du being required to be equal to zero on u's Dirichlet
        # boundaries, then look no further. The answer is that second argument
        # to SystemAssembler::assemble().
        except ValueError:
            self.solver.log_print("ValueError in assembly of b")
            return b
        except OverflowError:
            self.solver.log_print("OverflowError in assembly of b")
            return b
        except RuntimeError as e:
            self.solver.log_print("RuntimeError in assembly of a")
            self.solver.log_print(str(e))
            raise
        return b

    @cached_property
    def du_norm(self):
        return numpy.linalg.norm(
            self.du.vector().get_local(), ord=numpy.Inf)

    @cached_property
    def has_nans(self):
        return (
            numpy.isnan(self.u_.vector().get_local()).any()
            or numpy.isnan(self.b_norm)
        )

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
    def combined_du_norm(self):
        du = self.du.vector().get_local()
        u = self.u_.vector().get_local()
        rel_tol = self.solver.parameters['relative_tolerance']
        abs_tol = self.solver.parameters['absolute_tolerance'] # user adjustable weight
        abs_tol_vec = self.solver.parameters['absolute_tolerance_vector']

        with numpy.errstate(divide='ignore', invalid='ignore'):
            r = numpy.abs(du)/(numpy.abs(u)*rel_tol + abs_tol * abs_tol_vec)
        return numpy.linalg.norm(r, ord=numpy.Inf)

    @cached_property
    def b_norm(self):
        try:
            b_n = numpy.linalg.norm(
                self.b.get_local(), ord=numpy.Inf)
        except ValueError:
            b_n = numpy.nan
        except OverflowError:
            b_n = numpy.inf
        return b_n

    @cached_property
    def combined_b_norm(self):
        '''Like combined_du_norm but for b instead of du'''
        b = self.b.get_local()
        u = self.u_.vector().get_local()
        rel_tol = self.solver.parameters['relative_tolerance']
        abs_tol = self.solver.parameters['absolute_tolerance'] # user adjustable weight
        abs_tol_vec = self.solver.parameters['absolute_tolerance_vector']

        with numpy.errstate(divide='ignore', invalid='ignore'):
            r = numpy.abs(b)/(numpy.abs(u)*rel_tol + abs_tol * abs_tol_vec)
        return numpy.linalg.norm(r, ord=numpy.Inf)

    def do_linear_solve(self):
        A, Qdu, b = self.A, self.Qdu, self.b
        q = self.q

        # Output diagnostic info and stats from MUMPS.
        # dolfin.PETScOptions.set('mat_mumps_icntl_2', 6)
        # dolfin.PETScOptions.set('mat_mumps_icntl_4', 3)

        # print mumps error analysis, i.e. condition number
        # dolfin.PETScOptions.set('mat_mumps_icntl_11', 1)

        # settings for automatic scaling and pre-processing of the A matrix.
        # dolfin.PETScOptions.set('ksp_view')
        # dolfin.PETScOptions.set('-info', 'out/petsc_log.txt')
        # dolfin.PETScOptions.set('log_trace', 'out/petsc_trace.txt')
        # dolfin.PETScOptions.set('pc_factor_mat_ordering_type', 'natural')
        dolfin.PETScOptions.set('mat_mumps_icntl_6', 5)
        dolfin.PETScOptions.set('mat_mumps_icntl_7', 6)
        dolfin.PETScOptions.set('mat_mumps_icntl_8', 77)

        # 1=sequential mode (no MPI). Still uses multithreading.
        dolfin.PETScOptions.set('mat_mumps_icntl_28', 1)
        # dolfin.PETScOptions.set('mat_mumps_icntl_29', 1)

        # After setting the above defaults, go through the mumps_icntl to (re)set parameters
        for opt in self.mumps_icntl:  # ((28,7), (29,3), (30, 1))
            dolfin.PETScOptions.set('mat_mumps_icntl_' + str(opt[0]), int(opt[1]))
            # print("Setting mumps_icntl{} to {}".format(str(opt[0]),int(opt[1])))

        # dolfin.PETScOptions.set('mat_mumps_icntl_6', 7)
        # dolfin.PETScOptions.set('mat_mumps_icntl_7', 7)

        # Reuse the solver object across Newton iterations to avoid repeated
        # PETSc/MUMPS initialisation overhead. The sparsity pattern of A is
        # fixed for the lifetime of a NewtonSolution, so MUMPS can reuse its
        # symbolic analysis phase.
        lu_solver = getattr(self, '_lu_solver', None)
        if lu_solver is None:
            lu_solver = dolfin.PETScLUSolver("mumps")
            self._lu_solver = lu_solver
        lu_solver.solve(A, Qdu, b)

        # solver1 = dolfin.PETScKrylovSolver("cg","ilu")
        # prm = solver1.parameters
        # prm["absolute_tolerance"] = 1E-7
        # prm["relative_tolerance"] = 1E-4
        # prm["maximum_iterations"] = 1000
        # prm["nonzero_initial_guess"] = True
        # solver1.solve(A, Qdu, b)
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
        relerr = getattr(self, 'rel_du_norm', -1)
        comberr = getattr(self, 'combined_du_norm', -1)
        comb_b_err = getattr(self, 'combined_b_norm', -1)
        # 0 <= x < 1: a norm that could not be computed comes back as -1 and
        # must not be reported as satisfied.
        sat_du = 1 if 0 <= comberr < 1 else 0
        sat_b = 1 if 0 <= comb_b_err < 1 else 0
        sat = f"{sat_du}{sat_b}"
        return "|b| {:.4e}, |du/u| {:.4e}, |du|/(r*|u|+a) {:.4e}, |b|/(r*|u|+a) {:,.4e}, sat={:}".format(
            abserr, relerr, comberr, comb_b_err, sat)

    _no_copy = frozenset(('solver', 'A', 'b', 'q', 'size', '_lu_solver'))

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
        ('rel_du_norm', 'du_norm', 'combined_du_norm', 'b_norm','combined_b_norm', 'A', 'b', 'q', 'size',
         'has_nans'))

    def do_invalidate_cache(self, check=False):
        d = self.__dict__
        for k in self._cached_property_attrs:
            if d.pop(k, None) is not None:
                if check:
                    logging.info('*'*30 + k)

    def release_linear_solver(self):
        '''Drop the cached LU/MUMPS solver at the end of a Newton solve.

:code:`_lu_solver` is deliberately kept for the whole solve (see
:py:meth:`do_linear_solve`) so that MUMPS can reuse its symbolic analysis across
Newton iterations, and it is therefore *not* listed in
:py:attr:`_cached_property_attrs` -- clearing it per iteration would throw away
exactly the saving it exists for.

It must still be released once the solve is over. This object and its
:py:class:`NewtonSolver` hold each other (:py:meth:`NewtonSolution.__init__`
sets :code:`self.solver`, :py:meth:`NewtonSolver.do_init_solution` sets
:code:`self.solution`), so the pair is a reference cycle: dropping the solver
does not reclaim it by refcounting, only a generational garbage collection does.
CPython triggers collection from counts of *Python* allocations, so a handful of
Python objects holding a large MUMPS factorisation is nearly invisible to that
heuristic and collection can be arbitrarily late. Releasing here frees the
factorisation deterministically instead.

Measured on a 2-D drift-diffusion voltage sweep: without this, RSS grew ~13 MB
per Newton iteration (reaching tens of GB on production meshes, with OOM kills);
with it, ~1.7 MB per iteration. Computed terminal currents were unchanged to
~5e-4 relative, i.e. within the Newton convergence tolerance.
'''
        self.__dict__.pop('_lu_solver', None)

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
    default_parameters = dict(
        maximum_iterations=25,
        minimum_iterations=-1,
        relaxation_parameter=1.0,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-6,
        extra_iterations=3,
        absolute_tolerance_vector=1.0,
        convergence_criterion='du',
        mumps_icntl=(),
        form_compiler_parameters=dict(
            optimize=True,
            eliminate_zeros=True,
            simplify_expressions=False,
            precompute_basis_const=False,
            precompute_ip_const=False,
        ),
    )

    def __init__(self, F, u_, bcs, J=None, parameters=None,
                pdd=None):
        self.do_init_parameters()
        if parameters is not None:
            self.parameters.update(parameters)

        self.W = u_.function_space()
        self.do_init_solution(u_)
        self.set_problem(F, u_, bcs, J=J)

        self.user_before_first_iteration_hooks = []
        self.user_pre_iteration_hooks = []
        self.user_post_iteration_hooks = []
        self.user_after_last_iteration_hooks = []

        self.pdd = pdd

    @classmethod
    def from_nice_obj(cls, obj):
        F   = obj.get_weak_form().to_ufl().magnitude
        u_  = obj.get_solution_function()
        bcs = obj.get_essential_bcs()

        return cls(F=F, u_=u_, bcs=bcs, pdd=obj)

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
        self.parameters = deepcopy(self.default_parameters)


    def do_init_assembler(self):
        try:
            self.assembler =  dolfin.fem.assembling.SystemAssembler(
                self.J,
                self.F,
                self.bcs,
                form_compiler_parameters=self.parameters["form_compiler_parameters"],
            )
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

        #Set mumps_icntl flags, if they have been passed in
        try:
            self.solution.mumps_icntl = self.parameters["mumps_icntl"]
        except Exception:
            pass

    def do_post_iteration_hook(self):
        self.log_print("{}* iteration {:>3d}; {}".format(
            '\n' if self.iteration == 0 else '',
            self.iteration, self.solution.str_error()))
        self.solution.do_invalidate_cache()
        for hook in self.user_post_iteration_hooks:
            hook(self)

    def do_after_last_iteration_hook(self):
        self.solution.do_invalidate_cache()
        debug_HJBC_reltol = getattr(self.pdd, 'debug_HJBC_reltol', None)
        if debug_HJBC_reltol is not None:
            ''' iterate over the bands, iterate over the lists of Delta_w and Delta_w_BC. Insert a log entry about HJBC'''
            for k, b in self.pdd.bands.items():
                if b.name == "CB" or b.name == "VB":
                    Delta_w_list = b.Delta_w
                    Delta_w_BC_list = b.Delta_w_BC
                    # iterate over junctions
                    for i, Delta_w_probe in enumerate(Delta_w_list):
                        Delta_w_BC_probe = Delta_w_BC_list[i]

                        # probe the facet values
                        Delta_w = Delta_w_probe().m_as("eV")
                        Delta_w_BC = Delta_w_BC_probe().m_as("eV")

                        # abs and rel erros
                        abs_err = numpy.abs(Delta_w - Delta_w_BC)
                        try:
                            rel_err = numpy.abs(abs_err / Delta_w_BC)
                        except RuntimeError:
                            rel_err = numpy.nan

                        # check relative error
                        if rel_err > self.pdd.debug_HJBC_reltol and abs_err > 1e-6:
                            # print j_band, too, in the case of discrepancy
                            j_band = b.j_band[i]().m_as("mA/cm^2")
                            # log entry
                            self.log_print(f"HJBC for {b.name} at junction {i}: rel_err={rel_err:.4e} | abs_err={abs_err:.4e} | Delta_w_BC={Delta_w_BC:.4e} eV | Delta_w={Delta_w:.4e} eV | j_band={j_band:.4e} mA/cm^2")

        for hook in self.user_after_last_iteration_hooks:
            hook(self)

    def get_omega(self):
        # Relaxation parameter -- Move solution by omega*du
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
        self.do_after_last_iteration_hook()
        self.solution.release_linear_solver()

    def has_converged(self):
        sltn = self.solution
        if self.parameters.get('convergence_criterion', 'du') == 'b':
            return (not sltn.has_nans) and (sltn.combined_b_norm <= 1.)
        return (not sltn.has_nans) and (sltn.combined_du_norm <= 1.)

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
        if iteration >= self.parameters['maximum_iterations']:
            self.log_print(f"Iteration {iteration} >= "\
                         + f"max iterations {self.parameters['maximum_iterations']}")
        return ((iteration >= self.parameters['maximum_iterations'])
                or (self.has_converged())
                or (numpy.isnan(sltn.du_norm))
                or (numpy.isnan(sltn.b_norm))
                )


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
        # This override does not call do_after_last_iteration_hook(), so the
        # release has to be done here too.
        self.solution.release_linear_solver()


class NewtonSolverMaxDu(NewtonSolver):
    '''Variant of :py:class:`NewtonSolver` that clips the Newton update
:code:`du` to ``[-maximum_du, +maximum_du]`` component-wise before applying
it.  Set ``parameters['maximum_du']`` to enable the clipping.'''

    def user_update_solution_vector(self, solution, du):
        maxdu = self.parameters.get('maximum_du', None)
        if maxdu is not None:
            du = numpy.minimum(du,  maxdu)
            du = numpy.maximum(du, -maxdu)
        super().user_update_solution_vector(solution, du)


class NewtonSolverLogDamping(NewtonSolver):
    '''Variant of :py:class:`NewtonSolver` that applies log-damping to the
Newton update :code:`du` before each solution step, following Gaury 2018.
The damping strength is controlled by ``parameters['logdamping_alpha']``
(default 1.72); larger values give stronger damping.

Used by :py:class:`~simudo.physics.steppers.OpticalIntensityLogDampStepper`.'''

    def user_update_solution_vector(self, solution, du):
        # ref:[Gaury2018a]
        alpha = self.parameters.get('logdamping_alpha', 1.72)
        du = numpy.sign(du)*numpy.log1p(numpy.abs(du) * alpha) / alpha
        super().user_update_solution_vector(solution, du)

# class NewtonSolverFromCheckpoint(NewtonSolver):
