import logging
from collections import namedtuple

from cached_property import cached_property

from ..util import SetattrInitMixin

__all__ = [
    'BaseAdaptiveStepper',
    'AdaptiveStepper',
    'ConstantStepperMixin']

AdaptiveLogEntry = namedtuple(
    'AdaptiveLogEntry',
    ['parameter', 'success', 'saved_solution'])

class BaseAdaptiveStepper(SetattrInitMixin):
    '''Adaptive solution base class.

This implements an adaptive solution method: progressively change an
input parameter in small increments, re-solving the problem at each
step.

If solutions are successful, gradually larger step sizes will be
used. If a solution fails, the last successful solution is
re-loaded and a smaller step size is chosen.

A series of specific parameter values to be attained can be given, for
example to generate multiple points for a current-voltage curve.

Parameters
----------
solution: object
    Solution object, whatever that means.
parameter_target_values: list
    List of values for :py:attr:`parameter` to
    attain. :py:attr:`output_writer` will be called as each target
    value is attained.
break_criteria_cb: callable, optional
    A function that will be evaluated after each target value.  If it
    returns :code:`True`, the adaptive stepper will stop.

Attributes
----------
parameter: float
    Current value of the parameter.
step_size: float
    Current step size.
update_parameter_success_factor: float
    If the solver succeeds, multiply :py:attr:`step_size` by this
    factor.
update_parameter_failure_factor: float
    If the solver fails, multiply :py:attr:`step_size` by this
    factor.

Notes
-----
This class implements the abstract algorithm, so it makes no
assumptions as to what :code:`solution` represents. Users of this
class must implement the :py:meth:`user_solver`,
:py:meth:`user_solution_add`, and :py:meth:`user_solution_save`
methods.
'''

    solution = None
    parameter_start_value = 0.0
    parameter_target_values = [1.0]
    step_size = 1e-2

    update_parameter_success_factor = 2.
    update_parameter_failure_factor = 0.5
    keep_saved_solutions_count = 2

    output_writer = None

    break_criteria_cb = None

    @cached_property
    def parameter(self):
        return self.parameter_start_value

    @cached_property
    def prior_solutions(self):
        ''' list of (parameter_value, solver_succeeded, saved_solution) '''
        return []

    @cached_property
    def prior_solutions_idx_successful(self):
        return []

    @cached_property
    def prior_solutions_idx_saved_solution(self):
        return []

    def cleanup_prior_solutions(self):
        prior = self.prior_solutions
        prior_idx = self.prior_solutions_idx_saved_solution
        while len(prior_idx) > self.keep_saved_solutions_count:
            i = prior_idx.pop(0)
            prior[i] = prior[i]._replace(saved_solution=None)

    def add_prior_solution(self, parameter_value, solver_succeeded, saved_solution):
        prior = self.prior_solutions

        if saved_solution is not None and not solver_succeeded:
            raise AssertionError("why save a failed solution?")

        prior.append(AdaptiveLogEntry(
            parameter_value, solver_succeeded, saved_solution))

        index = len(prior) - 1
        if saved_solution is not None:
            self.prior_solutions_idx_saved_solution.append(index)
        if solver_succeeded:
            self.prior_solutions_idx_successful.append(index)

        self.cleanup_prior_solutions()

    def get_last_successful(self, count):
        prior = self.prior_solutions
        return [prior[i] for i in self.prior_solutions_idx_successful[-count:]]

    def reset_parameter_to_last_successful(self):
        last_ok = self.get_last_successful(1)
        self.parameter = (last_ok[0].parameter if last_ok
                          else self.parameter_start_value)

    def update_parameter(self):
        '''This method is called to update :py:attr:`parameter` after
each solver call (successful or not). It also restores a previous
solution if the solver call failed.'''
        last_ok = self.get_last_successful(1)
        if not last_ok:
            self.reset_parameter_to_last_successful()
            return
        last_ok = last_ok[0]
        last = self.prior_solutions[-1]
        if not last.success:
            logging.getLogger('stepper').info(
                '** Newton solver failed, trying smaller step size')
        self.step_size *= (self.update_parameter_success_factor
                           if last.success else
                           self.update_parameter_failure_factor)
        self.step_size = min(self.step_size,
                             abs(self.parameter_target_value - last_ok.parameter))

        if not last.success:
            self.user_solution_add(
                self.solution, 0.0, last_ok.saved_solution, 1.0)

        self.parameter = last_ok.parameter + self.step_sign*self.step_size

    def user_solver(self, solution, parameter):
        '''
        This user-defined method must re-solve `solution` using
        parameter value `parameter`.

        Must return boolean indicating whether the solver succeeded.
        '''
        raise NotImplementedError()

    def user_solution_save(self, solution):
        '''
        This procedure must return a copy of the solution variables of `solution`
        in a format acceptable by `user_solution_add`. Typically a dict of 'key': vector pairs.
        '''
        raise NotImplementedError()

    def user_solution_add(self, solution, solution_scale,
                          saved_solution, saved_scale):
        '''This procedure must update the current solution so that::

    self.current_solution = (self.solution*solution_scale +
                             saved_solution*saved_scale)

where ``saved_solution`` is as returned by
:py:meth:`user_solution_save`.

If ``solution_scale`` is equal to 0, then the current solution must be
erased (careful with NaNs!).
'''
        raise NotImplementedError()

    def do_iteration(self):
        '''Solve the problem at a single value of the solution parameter.'''

        self.update_parameter()
        solution = self.solution
        parameter = self.parameter
        success = self.user_solver(solution, parameter)
        saved = self.user_solution_save(solution) if success else None
        self.add_prior_solution(parameter, success, saved)

    def do_loop(self):
        ''' This method must be called to actually perform the adaptive
        stepping procedure'''

        for val in self.parameter_target_values:
            self.reset_parameter_to_last_successful()
            self.step_sign = -1. if self.parameter > val else 1.
            self.parameter_target_value = val
            while True:
                last_successful = self.get_last_successful(1)
                if (last_successful and
                    last_successful[0].parameter == val):
                    break
                self.do_iteration()
            if self.output_writer is not None:
                logging.getLogger('stepper').info('Writing output')
                self.output_writer.write_output(self.solution, self.parameter)

            if self.break_criteria_cb is not None:
                if self.break_criteria_cb(self.solution):
                    logging.getLogger('stepper').info('*** Break criteria met, stopping.')
                    break

class AdaptiveStepper(BaseAdaptiveStepper):
    '''This implements a slightly more concrete class than
:py:class:`BaseAdaptiveStepper`.

Parameters
----------
to_save_objects: dict
    Mapping of vector-like objects representing the current
    :py:attr:`solution`, whose values to save (and restore upon solver
    failure).
'''

    to_save_objects = None

    def _vector(self, x):
        x = x.magnitude if hasattr(x, 'magnitude') else x
        return x.vector()

    def user_solver(self, solution, parameter):
        self.user_apply_parameter_to_solution(solution, parameter)
        solver = self.user_make_solver(solution)
        solver.solve()
        return solver.has_converged()

    def user_apply_parameter_to_solution(self, solution, parameter):
        raise NotImplementedError('override me')

    def user_make_solver(self, solution):
        raise NotImplementedError('override me')

    def user_solution_save(self, solution):
        return {k: self._vector(obj)[:]
                for k, obj in self.to_save_objects.items()}

    def user_solution_add(self, solution, solution_scale,
                          saved_solution, saved_scale):
        V = self._vector
        for k, obj in self.to_save_objects.items():
            if solution_scale == 0:
                V(obj)[:] = saved_solution[k]*saved_scale
            else:
                V(obj)[:] *= solution_scale
                V(obj)[:] += saved_solution[k]*saved_scale

class ConstantStepperMixin():
    '''
Parameters
----------
constants: :py:class:`pint.Quantity` wrapping :py:class:`dolfin.Constant`
    On each iteration, the constant's value will be assigned to be the
    current parameter value.
parameter_unit: :py:class:`pint.Quantity`, optional
    Parameter unit.
'''
    parameter_unit = None

    def get_mapped_parameter(self):
        x = self.parameter
        if self.parameter_unit is not None:
            x = self.parameter_unit * x
        return x

    @property
    def unit_registry(self):
        return self.solution.unit_registry

    def user_apply_parameter_to_solution(self, solution, parameter_value):
        x = self.get_mapped_parameter()
        for c in self.constants:
            c.magnitude.assign(x.m_as(c.units))
