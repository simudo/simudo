import logging

import numpy as np
from cached_property import cached_property

from ..fem import (
    AdaptiveStepper, ConstantStepperMixin, NewtonSolver, NewtonSolverMaxDu)
from ..util import SetattrInitMixin


#from ..io.output_writer import OutputWriter # FIXME

class NewtonBailoutException(Exception):
    pass

class NonequilibriumCoupledStepper(AdaptiveStepper):
    '''Adaptively solve a coupled poisson-drift diffusion-optical problem.

The optical problem can be solved self-consistently with the drift-diffusion problem.

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
    passed to the default OutputWriter object. If ``output_writer`` is
    an object inherited from OutputWriter, this object will be used
    instead. If None, then no output is written. Output will be
    written after each target_value is solved.
'''

    def __init__(self, **kwargs):
        output_writer = kwargs.get('output_writer', None)
        if output_writer is not None:
            if isinstance(output_writer, str):
                output_writer = OutputWriter(output_writer, plot_iv=True)
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
        be_extra_careful = bool(self.get_last_successful(1))

        def omega_cb(self):
            if not be_extra_careful:
                return 1.0
            du = self.solution.du_norm
            return 1.0
            # if self.iteration < 20:
            #     return 0.1
            # elif du < 1e-5:
            #     return 0.9
            # elif du < 1e-2:
            #     return 0.5
            # else:
            #     return 0.1

        # PDD solver
        solver =  NewtonSolver.from_nice_obj(pdd)
        solver.parameters.update(
            maximum_iterations=200 if be_extra_careful else 25,
            omega_cb=omega_cb,
            extra_iterations=5 if be_extra_careful else 3,
            minimum_iterations=3,
            relative_tolerance=1e-4,
            absolute_tolerance=1e-4)

        def _assign_scalar(subfunc, value):
            fsr.assign_scalar(
                subfunc.magnitude, value.m_as(subfunc.units))

        fsr = pdd.mesh_util.function_subspace_registry
        U = sl.unit_registry
        space = pdd.mixed_function_helper.solution_mixed_space
        du_clip_mf = space.make_function()
        du_clip = du_clip_mf.function
        du_clip_split = du_clip_mf.split()

        du_clip.vector()[:] = np.inf

        # FIXME: this should probably be elsewhere
        for k, v in du_clip_split.items():
            if k.endswith("/delta_w"):
                _assign_scalar(v, 0.005*U.eV)
            if k == 'poisson/phi':
                _assign_scalar(v, 0.005*U.volt)

        solver.parameters.update(
            maximum_du=du_clip.vector()[:],
        )

        def run_optical_subsolvers(slv):
            for o in solution.optical.fields:
                o.update_input()
                s = NewtonSolver.from_nice_obj(o)
                # FIXME: these parameters are disgraceful
                s.parameters.update(
                    maximum_iterations=2,
                    extra_iterations=
                    (10 if slv.extra_iterations is not None else 2),
                    omega_cb=lambda self: 0.9 if self.iteration < 2 else 1.0)
                s.logger = logging.getLogger('newton.optical')
                s.solve()
                o.update_output()

        def bailout_if_error_too_large(slv):
            if (slv.solution.du_norm > 1e40) and (slv.iteration > 20):
                raise NewtonBailoutException()

        def rebase_w(slv):
            for b in pdd.bands:
                if hasattr(b, 'mixedqfl_do_rebase_w'):
                    b.mixedqfl_do_rebase_w()

        if self.selfconsistent_optics:
            solver.user_before_first_iteration_hooks.append(run_optical_subsolvers)
            solver.user_post_iteration_hooks.append(run_optical_subsolvers)

        solver.user_before_first_iteration_hooks.append(bailout_if_error_too_large)

        solver.user_post_iteration_hooks.append(bailout_if_error_too_large)
        solver.user_pre_iteration_hooks.append(rebase_w)
        return solver

    def user_solver(self, solution, parameter):
        logging.info("%%%%%%%%%% adaptive solve parameter={} step_size={}".format(parameter, self.step_size))
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

    update_parameter_success_factor = 1.3
    update_parameter_failure_factor = 0.5
    step_size = 0.1

    def user_apply_parameter_to_solution(self, solution, parameter_value):
        super().user_apply_parameter_to_solution(solution, parameter_value)

class OpticalIntensityAdaptiveStepper(NonequilibriumCoupledConstantStepper):
    '''Adaptively increase the optical intensity by increasing
:py:attr:`.optical.Optical.Phi_scale`.
'''
    update_parameter_success_factor = 3
    update_parameter_failure_factor = 0.2
    step_size = 1e-30

    @cached_property
    def constants(self):
        return [self.solution.optical.Phi_scale]

    @cached_property
    def parameter_unit(self):
        return self.unit_registry.dimensionless

    @cached_property
    def parameter_target_values(self):
        return [1.0]
