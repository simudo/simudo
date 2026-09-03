import logging

import numpy as np
from cached_property import cached_property

from ..fem import (
    AdaptiveStepper, 
    ConstantStepperMixin, 
    NewtonSolver, 
    NewtonSolverMaxDu, 
    NewtonSolverLogDamping)
from ..util import SetattrInitMixin


from ..io.output_writer import OutputWriter # FIXME

class NewtonBailoutException(Exception):
    pass

class NonequilibriumCoupledStepper(AdaptiveStepper):
    '''Adaptively solve a coupled poisson-drift diffusion-optical problem.

Extends :py:class:`~simudo.fem.adaptive_stepper.AdaptiveStepper` with
PDD-specific logic.  The linear solve at each Newton step uses
:py:class:`~simudo.fem.newton_solver.NewtonSolver` (or a subclass set via
:py:attr:`solver_class`).  The optical problem can be solved
self-consistently with the drift-diffusion problem.

Concrete subclasses for common use cases:

- :py:class:`VoltageStepper` — ramp contact bias.
- :py:class:`OpticalIntensityAdaptiveStepper` — ramp optical intensity.
- :py:class:`OpticalIntensityLogDampStepper` — same with log-damped updates.

Parameters
----------
solution: :py:class:`~.problem_data.ProblemData`
    Solution on which to operate. This solution object will be
    progressively re-solved at each of the target parameter values.

selfconsistent_optics: bool
    If True, the optical problem will be re-solved
    at every Newton iteration, allowing for problems where optical absorption
    and carrier concentrations are interdependent.

output_writer: :py:class:`~.output_writer.OutputWriter`
    If ``output_writer`` is a string, it is used as a filename to be
    passed to the default :py:class:`~.output_writer.OutputWriter` object.
    If ``output_writer`` is an object inherited from
    :py:class:`~.output_writer.OutputWriter`, this object will be used
    instead. If None, then no output is written. Output will be
    written after each target_value is solved.
'''
    solver_class = NewtonSolver
    solver_parameters = None

    def __init__(self, **kwargs):
        output_writer = kwargs.get('output_writer', None)
        if output_writer is not None:
            if isinstance(output_writer, str):
                output_writer = OutputWriter(output_writer, plot_iv=True)
            output_writer.stepper = self
            kwargs['output_writer'] = output_writer
        super().__init__(**kwargs)

    @cached_property
    def to_save_objects(self):
        pdd = self.solution.pdd

        r = pdd.get_to_save()
        r.update(self.solution.optical.get_to_save())

        return {k: v.value for k, v in r.items()
                if v.solver_save}

    def user_make_solver(self, solution):
        sl = solution
        pdd = solution.pdd

        # previous_successful = len(self.get_last_successful(1)) > 0

        # reset optical auxiliary functions
        if self.selfconsistent_optics:
            for o in solution.optical.fields:
                o.update_output()
                o.update_input()

        # exercise caution when there's no solution to backtrack to
        be_extra_careful = not self.get_last_successful(1)

        def omega_cb(self):
            if not be_extra_careful:
                return 1.0
            du = self.solution.du_norm
            return 1.0


        # PDD solver
        solver =  self.solver_class.from_nice_obj(pdd)
        if solver.solution.has_nans:
            logging.info("Solution has nans")
        solver.parameters.update(
            maximum_iterations=200 if be_extra_careful else 25,
            omega_cb=omega_cb,
            extra_iterations=3 if be_extra_careful else 0,
            minimum_iterations=3,
            relative_tolerance=1e-4,
            absolute_tolerance=1,
            mumps_icntl = pdd.mumps_icntl if hasattr(pdd,'mumps_icntl') else () )
        if self.solver_parameters is not None:
            solver.parameters.update(self.solver_parameters)
        
        def _assign_scalar(subfunc, value):
            fsr.assign_scalar(
                subfunc.magnitude, value.m_as(subfunc.units))

        fsr = pdd.mesh_util.function_subspace_registry
        U = sl.unit_registry
        space = pdd.mixed_function_helper.solution_mixed_space
        # du_clip_mf = space.make_function()
        # du_clip = du_clip_mf.function
        # du_clip_split = du_clip_mf.split()
        #
        # du_clip.vector()[:] = np.inf

        # FIXME: this should probably be elsewhere
        # for k, v in du_clip_split.items():
        #     if k.endswith("/delta_w"):
        #         _assign_scalar(v, 0.005*U.eV)
        #     if k == 'poisson/phi':
        #         _assign_scalar(v, 0.005*U.volt)

        # solver.parameters.update(
        #     maximum_du=du_clip.vector()[:],
        # )

        tol_func = pdd.mixed_function_helper.make_tolerance_function()
        solver.parameters.update(
             absolute_tolerance_vector=tol_func.function.vector()[:]
        )

        _OPTICAL_INPUT_SKIP_TOL = 1e-6

        def _max_rel_change(new_vec, old_vec):
            # Mixed relative/absolute test. Where the old value is non-zero the
            # comparison is relative to it. Where it is exactly zero there is no
            # relative scale, so fall back to the field's own magnitude rather
            # than to 1.0 -- a bare 1.0 would make the test absolute in whatever
            # units the vector happens to carry, and so unit-dependent.
            old_abs = np.abs(old_vec)
            scale = max(float(np.max(old_abs)), float(np.max(np.abs(new_vec))))
            if scale == 0.0:          # both fields identically zero
                scale = 1.0
            denom = np.where(old_abs > 0, old_abs, scale)
            return float(np.max(np.abs(new_vec - old_vec) / denom))

        def run_optical_subsolvers(slv):
            phi_scale_curr = float(solution.optical.Phi_scale.values()[0])

            for o in solution.optical.fields:
                o.update_input()

                # Skip solve if alpha, g, and Phi_scale are all unchanged
                # beyond tolerance since the last actual solve.
                alpha_vec = o.alpha.magnitude.vector().get_local()
                g_vec = o.g.magnitude.vector().get_local()

                if hasattr(o, '_opt_prev_alpha_vec'):
                    phi_scale_prev = o._opt_prev_phi_scale
                    phi_scale_change = (abs(phi_scale_curr - phi_scale_prev) /
                                        (abs(phi_scale_prev) + 1e-300))
                    if (phi_scale_change < _OPTICAL_INPUT_SKIP_TOL
                            and _max_rel_change(alpha_vec,
                                                o._opt_prev_alpha_vec)
                                < _OPTICAL_INPUT_SKIP_TOL
                            and _max_rel_change(g_vec,
                                                o._opt_prev_g_vec)
                                < _OPTICAL_INPUT_SKIP_TOL):
                        logging.getLogger('newton.optical').warning(
                            '%s: skipping optical solve -- alpha, g and '
                            'Phi_scale all unchanged within %g since the last '
                            'solve. The previous optical solution is reused.',
                            o.key, _OPTICAL_INPUT_SKIP_TOL)
                        continue

                s = NewtonSolver.from_nice_obj(o)
                s.parameters.update(
                    maximum_iterations=25,
                    extra_iterations=0,
                    convergence_criterion='b',  # b-norm suffices: optical problem is linear
                    omega_cb=lambda self: 1.0)
                tol_func = o.mixed_function_helper.make_tolerance_function()
                s.parameters.update(
                    absolute_tolerance_vector=tol_func.function.vector()[:])
                s.logger = logging.getLogger('newton.optical')
                s.solve()
                o.update_output()

                # Store inputs so the next call can check whether to skip.
                o._opt_prev_alpha_vec = alpha_vec.copy()
                o._opt_prev_g_vec = g_vec.copy()
                o._opt_prev_phi_scale = phi_scale_curr

        def bailout_if_error_too_large(slv):
            if (slv.solution.du_norm > 1e40) and (slv.iteration > 20):
                raise NewtonBailoutException()

        def rebase_w(slv):
            for b in pdd.bands:
                if hasattr(b, 'mixedqfl_do_rebase_w'):
                    b.mixedqfl_do_rebase_w()

        # these optical hooks could be handled with code below.
        if self.run_optics:
            solver.user_before_first_iteration_hooks.append(run_optical_subsolvers)
        if self.selfconsistent_optics:
            solver.user_post_iteration_hooks.append(run_optical_subsolvers)

        def _append_not_None(lst, x):
            if x is not None:
                lst.append(x)

        # register calculations to be done for each process.
        for proc in self.solution.pdd.electro_optical_processes:
            _append_not_None(solver.user_pre_iteration_hooks, proc.pre_iteration_hook)
            _append_not_None(solver.user_before_first_iteration_hooks, proc.pre_first_iteration_hook)
            _append_not_None(solver.user_post_iteration_hooks, proc.post_iteration_hook)

        solver.user_before_first_iteration_hooks.append(bailout_if_error_too_large)

        solver.user_post_iteration_hooks.append(bailout_if_error_too_large)
        solver.user_pre_iteration_hooks.append(rebase_w)
        solver.pdd = pdd
        return solver

    def user_solver(self, solution, parameter):
        # self.log_print_new_parameter()
        # logging.info("%%%%%%%%%% adaptive solve parameter={} step_size={}".format(parameter, self.step_size))
        try:
            return super().user_solver(solution, parameter)
        except NewtonBailoutException:
            return False

class NonequilibriumCoupledConstantStepper(
        ConstantStepperMixin,
        NonequilibriumCoupledStepper):
    pass

class VoltageStepper(NonequilibriumCoupledConstantStepper):
    '''Starting from an existing solution, which may be at thermal
equilibrium or with previously applied bias of illumination, ramp the
bias at one or more contacts.'''

    update_parameter_success_factor = 1.5
    update_parameter_failure_factor = 0.5
    step_size = 0.1
    parameter_name = "V"
    run_optics = False

    def user_apply_parameter_to_solution(self, solution, parameter_value):
        super().user_apply_parameter_to_solution(solution, parameter_value)

class OpticalIntensityAdaptiveStepper(NonequilibriumCoupledConstantStepper):
    '''Adaptively increase the optical intensity by increasing
:py:attr:`.optical.Optical.Phi_scale`.
'''
    update_parameter_success_factor = 3
    update_parameter_failure_factor = 0.2
    step_size = 1e-30
    parameter_name = "I"
    run_optics = True

    @cached_property
    def constants(self):
        return [self.solution.optical.Phi_scale]

    @cached_property
    def parameter_unit(self):
        return self.unit_registry.dimensionless

    @cached_property
    def parameter_target_values(self):
        return [1.0]


class OpticalIntensityLogDampStepper(OpticalIntensityAdaptiveStepper):
    '''Optical itensity stepper with log damping parameter updates
'''
    solver_class = NewtonSolverLogDamping


class UniformGenerationAdaptiveStepper(NonequilibriumCoupledConstantStepper):
    '''Adaptively ramp a prescribed generation rate, without solving optics.

Ramps :py:attr:`pdd.uniform_generation_scale` -- a dimensionless
:py:class:`dolfin.Constant` -- from its initial value up to ``1.0``. No
optical problem is solved (:py:attr:`selfconsistent_optics` is False), which
makes this much cheaper than :py:class:`OpticalIntensityAdaptiveStepper` when
the generation profile is known in advance and only its magnitude needs to be
brought up from zero.

**Two things the user must do**, neither of which happens automatically:

1. Set ``pdd.uniform_generation_scale`` *before* constructing the stepper.
   It does not exist otherwise, and its absence raises :py:exc:`AttributeError`
   when :py:attr:`constants` is first evaluated.
2. Build the generation spatial rule out of **the same**
   :py:class:`dolfin.Constant` object. The stepper only assigns to that
   Constant; if the generation rule does not reference it, ramping the
   parameter has no effect on the solution.

Example
-------

::

    generation_scale = dolfin.Constant(0)
    pdd.uniform_generation_scale = generation_scale * U.dimensionless

    # profile1 may be any fixed spatial profile; generation_scale is the
    # same Constant object assigned above, and is what gets ramped.
    spatial.add_rule("static_cv/generation", R.domain,
                     profile1 * generation_scale)
    pdd.easy_add_electro_optical_process(
        StaticGeneration, name='static_cv', dst_band=CB, src_band=VB)

    stepper = UniformGenerationAdaptiveStepper(solution=full_problem, ...)
    stepper.do_loop()

The spatial key prefix must match the process ``name``:
:py:class:`~.electro_optical_process.StaticGeneration` reads
``<name>/generation``, so ``name='static_cv'`` pairs with the rule
``"static_cv/generation"`` above.

``uniform`` in the class and attribute names is historical. The profile need
not be uniform -- it need only be fixed in shape while its magnitude is
ramped.
'''

    update_parameter_success_factor = 3
    update_parameter_failure_factor = 0.5
    step_size = 1e-10
    parameter_name = "G"
    selfconsistent_optics = False

    @cached_property
    def constants(self):
        return [self.solution.pdd.uniform_generation_scale]
    
    @cached_property
    def parameter_unit(self):
        return self.unit_registry.dimensionless
    
    @cached_property
    def parameter_target_values(self):
        return [1.0]