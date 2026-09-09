import logging
from collections import namedtuple

from cached_property import cached_property
from numpy.core.numeric import Inf

from ..util import SetattrInitMixin

import numpy as np
import math
import os
import sys
from ..io import h5yaml
from ..fem.assign import opportunistic_assign
import dolfin

import warnings

__all__ = [
    'BaseAdaptiveStepper',
    'AdaptiveStepper',
    'ConstantStepperMixin',
    'StepperError']

AdaptiveLogEntry = namedtuple(
    'AdaptiveLogEntry',
    ['parameter', 'success', 'saved_solution'])


class StepperError(Exception):
    pass

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
stepper_rel_tol: float
    If :py:attr:`step_size ` drops below
    :py:attr:`stepper_rel_tol`*max(:py:attr:`parameter`, initial_step_size),
    the stepper will stop and raise a :py:class:`StepperError`.

Notes
-----
This class implements the abstract algorithm, so it makes no
assumptions as to what :code:`solution` represents. Users of this
class must implement the :py:meth:`user_solver`,
:py:meth:`user_solution_add`, and :py:meth:`user_solution_save`
methods.

See Also
--------
:py:class:`AdaptiveStepper` : concrete subclass with save/restore support.
:py:class:`~simudo.physics.steppers.NonequilibriumCoupledStepper` : PDD-specific subclass.
'''

    solution = None
    parameter_start_value = 0.0
    parameter_target_values = [1.0]
    step_size = 1e-2
    stepper_rel_tol = 1e-5  # tolerance for bailing out of stepper
    debug_find_unconverged_parameters = False

    update_parameter_success_factor = 2.
    update_parameter_failure_factor = 0.5
    keep_saved_solutions_count = 2

    output_writer = None

    break_criteria_cb = None
    parameter_name = ""

    checkpoint_write_values = ()  # immutable: a class-level [] would be shared
    checkpoint_write_filename = None 
    checkpoint_reload_yaml = None

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
        
        # print(f"current step size: {self.step_size}")
        
        if not last.success:
            logging.getLogger('stepper').info(
                '** Newton solver failed, trying smaller step size')
            if self.debug_find_unconverged_parameters:
                self.find_unconverged_parameters()
            self.step_size *= self.update_parameter_failure_factor
            
            # print(f"reduce step size to {self.step_size}")
            
            use_cap_step_ratio = getattr(self, 'use_cap_step_ratio', False)
            if use_cap_step_ratio:
                # if we are already capping the step ratio, but we still failed, decrease the cap slightly
                self.step_ratio_cap = 0.99 * self.step_ratio_cap # 0.99 halves the cap every 70 failed steps

        # don't change step_size if we're just taking a small step
        # to hit a target parameter value
        if last.success and self.parameter != 0:
            # self.step_size *= self.update_parameter_success_factor
            
            # print(f"increase step size to {self.step_size}")
            
            # Daisy 2024-07-03: impose a cap on step ratio for parameter larger than a threshold
            # Allows rapid ramping to the difficult point, then take smaller steps.
            # Defaults are set such that thre is no real cap or threshold.
            new_step_size = self.step_size * self.update_parameter_success_factor
            
            step_ratio_cap = getattr(self, 'step_ratio_cap', 1-1e10)
            step_ratio_cap_parameter_thresh = getattr(self, 'step_ratio_cap_parameter_thresh', 100.0)
            capped_step_size = self.parameter * step_ratio_cap / (1 - step_ratio_cap)
            if self.parameter > step_ratio_cap_parameter_thresh:
                if new_step_size > capped_step_size:
                    self.use_cap_step_ratio = True
                    new_step_size = capped_step_size
                    logging.info(f"** imposing step ratio cap, despite success.")
                    # every successful step after the cap is imposed, increase the cap slightly
                    # self.step_ratio_cap = 1.005 * self.step_ratio_cap # this 1.005 doubles the step ratio cap double every 140 succeesful step
                    
            self.step_size = new_step_size
            
        self.actual_step = min(self.step_size,
                            abs(self.parameter_target_value - last_ok.parameter))

        # If we just failed and actual_step<step_size, reduce step_size to 
        # actual_step * failure_factor
        if (self.actual_step < self.step_size) and not last.success:
            logging.getLogger('stepper').debug(
                '** actual_step < step_size; reducing step_size further')
            self.step_size = self.actual_step * self.update_parameter_failure_factor

        if not last.success:
            self.user_solution_add(
                self.solution, 0.0, last_ok.saved_solution, 1.0)
            
        # print(f"actual step: {self.actual_step}")

        self.parameter = last_ok.parameter + self.step_sign*self.actual_step

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

    def do_step(self):
        '''Solve the problem at a single value of the solution parameter.'''

        self.update_parameter()
        solution = self.solution
        parameter = self.parameter
        self.log_print_new_parameter()
        try:
            success = self.user_solver(solution, parameter)
        except RuntimeError:
            success = False
            if len(self.prior_solutions) == 0:
                raise
        if not success and len(self.prior_solutions) == 0:
            msg = (
                'The first solve failed to converge. This solve is of the '
                'received starting solution itself (no step has been taken '
                'yet), so this means that solution was not a valid starting '
                'point -- e.g. an incorrectly initialized problem or an '
                'incompatible/corrupted checkpoint. A smaller step size '
                'cannot fix this, since the parameter has not changed.'
            )
            logging.getLogger('stepper').error(msg)
            raise StepperError(msg)
        saved = self.user_solution_save(solution) if success else None
        self.add_prior_solution(parameter, success, saved)

    def do_loop(self):
        ''' This method must be called to actually perform the adaptive
        stepping procedure'''
        sufficient_progress = True
        initial_step_size = self.step_size

        target_values_diff = np.diff(self.parameter_target_values)

        # add in checkpoint_write_values if they are not in parameter_target_values already
        self.parameter_target_values = np.union1d(self.parameter_target_values, self.checkpoint_write_values)
        
        # Note: target_values_diff was computed BEFORE the union above, which
        # is deliberate -- np.union1d returns its result sorted ascending (and
        # de-duplicated), so the user's intended sweep direction can only be
        # recovered from the original array. Flipping the sorted union then
        # restores a descending sweep.
        # check if parameter_target_values are monotonically decreasing
        # if decreasing, flip the array
        if np.all(target_values_diff < 0): 
            self.parameter_target_values = np.flip(self.parameter_target_values)
        elif np.all(target_values_diff > 0): # increasing
            pass
        else:
            warnings.warn(
                "The parameter target values are not monotonically increasing or decreasing."
                )
        

        if self.checkpoint_reload_yaml:
            self.solver = self.user_make_solver(self.solution)
            self.load_xdmf_checkpoint()

        for val in self.parameter_target_values:
            self.reset_parameter_to_last_successful()
            self.step_sign = -1. if self.parameter > val else 1.
            self.parameter_target_value = val
            while True:
                last_successful = self.get_last_successful(1)
                failed_prev_step = (
                    self.prior_solutions and
                    not self.prior_solutions[-1].success
                )
                if (last_successful and
                    last_successful[0].parameter == val):
                    break
                if (failed_prev_step and
                    abs(self.step_size)/max(abs(self.parameter),
                        initial_step_size) < self.stepper_rel_tol):
                    sufficient_progress = False
                    break
                self.do_step()                 

            if self.output_writer is not None and self.get_last_successful(1):
                logging.getLogger('stepper').info('Writing output')
                self.output_writer.write_output(self.solution, self.parameter)

            if self.is_close_to_array(self.parameter, self.checkpoint_write_values) and self.get_last_successful(1):
                logging.getLogger('stepper').info('Writing checkpoint')
                self.write_xdmf_checkpoint()

            if not sufficient_progress:
                logging.getLogger('stepper').error('Insufficient progress, stopping.')
                msg=("Solver unable to progress. Among many possible causes of this "
                    "error are: poorly formed problem, incorrect boundary conditions,"
                    "inappropriate material parameters, or insufficient meshing. If "
                    "none of these is the case, it is possible that changing "
                    "preconditioning settings in simudo.fem.newton_solver.do_linear_solve could help.")
                raise StepperError(msg)

            if self.break_criteria_cb is not None:
                if self.break_criteria_cb(self.solution):
                    logging.getLogger('stepper').info('*** Break criteria met, stopping.')
                    break

    def log_print_new_parameter(self):
        '''As the stepper updates the parameter, call this method to output to the log'''
        parameter = self.parameter
        actual_step = getattr(self, 'actual_step', self.step_size)
        ratio = 0 if parameter == 0 else actual_step/parameter
        logging.info(f"%%%%%%%%%% stepper parameter {self.parameter_name}={self.parameter:0.6e}, step_size={actual_step:0.4e}, ratio={ratio:0.4e}")

class AdaptiveStepper(BaseAdaptiveStepper):
    '''Concrete subclass of :py:class:`BaseAdaptiveStepper` with solution
save/restore support.

Subclasses must implement :py:meth:`user_make_solver` (returns a
:py:class:`~simudo.fem.newton_solver.NewtonSolver`) and
:py:meth:`user_apply_parameter_to_solution`.

Parameters
----------
to_save_objects: dict
    Mapping of vector-like objects representing the current
    :py:attr:`solution`, whose values to save (and restore upon solver
    failure).

See Also
--------
:py:class:`~simudo.physics.steppers.NonequilibriumCoupledStepper` : PDD subclass.
:py:class:`~simudo.physics.steppers.VoltageStepper` : voltage-sweep specialisation.
:py:class:`~simudo.physics.steppers.OpticalIntensityAdaptiveStepper` : optical-ramp specialisation.
'''

    to_save_objects = None

    def _vector(self, x):
        x = x.magnitude if hasattr(x, 'magnitude') else x
        return x.vector()

    def user_solver(self, solution, parameter):
        self.user_apply_parameter_to_solution(solution, parameter)
        solver = self.user_make_solver(solution)    
        solver.solve()
        self.du = solver.solution.du
        self.solver = solver
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


    def find_unconverged_parameters(self):
        '''Run after a failed step to see which variables failed to meet their tolerance'''
        pdd = self.solver.pdd
        fsr = pdd.function_subspace_registry
        u = self.solver.solution.u_
        du = self.solver.solution.du
        split = pdd.mixed_function_helper.solution_mixed_space.split
        
        # split the last solution into its subfunctions
        split_u = split(u)        
        split_du = split(du)
        
        mu = pdd.mesh_util
        U = mu.unit_registry
        rel_tol = self.solver.parameters['relative_tolerance']
        abs_tol_factor = self.solver.parameters['absolute_tolerance']

        for k in split_u:
            #load the split function into a new function that's not in a mixed space
            cur_du = split_du[k]
            cur_u = split_u[k]
            simple_du = fsr.new_independent_function(cur_du.m) *cur_du.units
            simple_u = fsr.new_independent_function(cur_u.m) *cur_u.units
            opportunistic_assign(cur_du,simple_du,fsr) #handles units
            opportunistic_assign(cur_u,simple_u,fsr) #handles units

            du_vec = simple_du.vector().get_local() * cur_du.units
            u_vec = simple_u.vector().get_local() * cur_u.units     
            abs_tol = pdd.mixed_function_helper.solution_mixed_space.subspace_descriptors_dict[k]["trial_tolerance"]   
            abs_tol_units = pdd.mixed_function_helper.solution_mixed_space.subspace_descriptors_dict[k]["trial_units"]
            abs_tol = abs_tol * abs_tol_units * abs_tol_factor 
            
            # combined norm
            r = np.abs(du_vec)/(np.abs(u_vec)*rel_tol + 1e-100*u_vec.units)
            a = np.abs(du_vec)/abs_tol        
            c = np.abs(du_vec)/(np.abs(u_vec)*rel_tol + abs_tol)
            a_inf_norm = np.linalg.norm(a.m_as("dimensionless"),ord=np.Inf) 
            r_inf_norm = np.linalg.norm(r.m_as("dimensionless"),ord=np.Inf) 
            c_inf_norm = np.linalg.norm(c.m_as("dimensionless"),ord=np.Inf)
            if c_inf_norm>1.:
                number_of_violators = np.sum(c.m_as("dimensionless")>1.)
                logging.getLogger('stepper').info(
                    f"{k:12s} not converged. {number_of_violators}/{len(c)} dofs with c>1. Max norms: a={a_inf_norm:.4e}, r={r_inf_norm:.4e}, c={c_inf_norm:.4e}"
                    )


    def write_xdmf_checkpoint(self):
        if self.checkpoint_write_filename is None:
            logging.getLogger('stepper').info("No checkpoint filename for saving")
            return
        
        output_writer = self.output_writer
        parameter_name = output_writer.parameter_name


        yaml_filename = self.checkpoint_write_filename
        yaml_filename = os.path.splitext(yaml_filename)[0] \
                    + f"_{parameter_name}={self.parameter:.14g}" \
                    + '.yaml'
        xdmf_filename = os.path.splitext(yaml_filename)[0]+'.xdmf'        
        metadata = {"filename": xdmf_filename, 
                    "parameter": f"{self.parameter:.16g}",
                    "step_size": f"{self.step_size:.16g}",
                    "dolfin_version": dolfin.__version__
                    }
        # logging.getLogger('stepper').info(f"parameter = {self.parameter}, type = {type(self.parameter)}")

        h5yaml.dump(metadata,yaml_filename)

        xdmf_file = dolfin.XDMFFile(dolfin.MPI.comm_world, xdmf_filename)
        xdmf_file.parameters['rewrite_function_mesh'] = False
        xdmf_file.parameters['functions_share_mesh'] = True

        pdd = self.solution.pdd
        fsr = self.solution.pdd.function_subspace_registry
        mu = self.solution.pdd.mesh_util
        split_u = pdd.mixed_function_helper.solution_mixed_space.split(self.solver.solution.u_)
        with xdmf_file as f:
            #Write the mesh
            f.write(mu.mesh)
            #Write all parts of the current u_ vector 
            for k in split_u:
                # Make new non-mixed objects of the same shape as these objects, 
                # and write the new ones
                cur_fun = split_u[k]
                new_fun = fsr.new_independent_function(cur_fun.m)
                fsr.assign(new_fun, cur_fun.m) #units not transferred
                f.write_checkpoint(new_fun,k,0,append=True)
                if hasattr(cur_fun,"units"):
                    metadata[k + "_units"] = str(cur_fun.units)
                else:
                    metadata[k + "_units"] = "dimensionless"
            #Write the base_w for each band
            for b in pdd.bands:
                if hasattr(b,"mixedqfl_base_w"):
                    cur_base_w = b.mixedqfl_base_w 
                    #stripping off units
                    f.write_checkpoint(cur_base_w.m,
                                    b.name + "/mixedqfl_base_w",0,append=True)    
                    # print(f"Saved {b.name} base_w\n")
                    # print(cur_base_w.vector()[:])        
                    metadata["b.name" + "mixedqfl_base_w_units"] = str(cur_base_w.units)
            #Write the thermal equilibrium phi, because we need it for some reason?
            thmq_phi = pdd.poisson.thermal_equilibrium_phi
            f.write_checkpoint(thmq_phi.m,"thermal_equilibrium_phi",0,append=True)
            metadata["thermal_equilibrium_phi_units"] = str(thmq_phi.units)

            # Write optical state (solved photon flux per field, and the
            # intensity-ramp scale). Without this, resuming into a stepper
            # that doesn't itself re-solve optics before its first PDD solve
            # (e.g. VoltageStepper, whose run_optics=False) starts from an
            # unsolved (effectively zero) photon flux that is inconsistent
            # with the illuminated carrier state just loaded above, and
            # reliably fails to converge.
            optical = self.solution.optical
            metadata["optical_Phi_scale"] = f"{float(optical.Phi_scale.values()[0]):.16g}"
            for key in optical.fields.keys():
                cur_Phi = optical.fields[key].Phi
                new_Phi = fsr.new_independent_function(cur_Phi.m)
                fsr.assign(new_Phi, cur_Phi.m)
                f.write_checkpoint(new_Phi, "optical/" + key + "/Phi", 0, append=True)
                metadata["optical/" + key + "/Phi_units"] = str(cur_Phi.units)

        h5yaml.dump(metadata,yaml_filename)

    def check_bcs_resolve_to_facets(self):
        '''Fail loudly if no boundary condition matches any exterior facet.

        A mesh that was read back from a file does not have its
        facet-to-cell connectivity built, and
        :py:func:`~simudo.mesh.facet.boundary_mesh_entities` identifies exterior
        facets with ``len(f.entities(D)) == 1``. Without that connectivity it
        yields nothing, so ``mark_boundary_facets`` marks nothing, the
        right/top/left/bottom classes are never assigned, and every facet region
        naming ``exterior`` (or anything built on it, such as the contacts)
        evaluates to the empty set. Every boundary condition then applies to zero
        facets, with no error raised anywhere.

        Nothing can be repaired at this point: the boundary marking happens in
        ``compute_facets()``, during problem construction, long before any
        checkpoint is read. The connectivity has to exist *before* then. So this
        only reports, and points at the fix.

        This problem occurs when a mesh is loaded from a file, which
        is what a restart requires whenever meshing is not reproducible.
        '''
        pdd = getattr(self.solution, 'pdd', None)
        if pdd is None or not hasattr(pdd, 'spatial'):
            return
        try:
            md = pdd.mesh_data
            mesh = md.mesh
            D = mesh.topology().dim()
            # Only so that the check itself is reliable; this does not and
            # cannot repair the facet marking, which has already happened.
            mesh.init(D - 1, D)

            ffa = np.asarray(md.facet_function.array())
            exterior = np.fromiter(
                (len(f.entities(D)) == 1 for f in dolfin.facets(mesh)),
                dtype=bool, count=len(ffa))
            if not exterior.any():
                return

            covered = np.zeros(len(ffa), dtype=bool)
            for bcs in pdd.spatial.bcs.values():
                for bc in bcs:
                    try:
                        fvs = md.evaluate_topology(bc.region)
                    except Exception:
                        continue
                    vals = [int(v[0]) if isinstance(v, tuple) else int(v)
                            for v in fvs]
                    if vals:
                        covered |= np.isin(ffa, vals)
        except Exception as e:
            logging.getLogger('stepper').warning(
                "Could not check boundary-condition coverage: %r", e)
            return

        n_ext = int(exterior.sum())
        n_cov = int((covered & exterior).sum())
        if n_cov == 0:
            raise StepperError(
                "Reload from checkpoint failed. Mesh in the saved checkpoint "
                "is not consistent with the reconstructed mesh: no boundary "
                "condition covers any of its {} exterior facets. To fix this "
                "problem, the checkpointed mesh must be loaded and facet "
                "construction completed before generating ProblemData."
                .format(n_ext))
        logging.getLogger('stepper').info(
            "Boundary conditions cover %d of %d exterior facets", n_cov, n_ext)

    def load_xdmf_checkpoint(self):       
        '''Load in a saved checkpoint file. Must have been made with the same 
        problemdata used to make this AdaptiveStepper instance. 
        '''
        if self.checkpoint_reload_yaml is None:
            logging.getLogger('stepper').info("No checkpoint filename for reloading")
            return

        # Check before doing any work: if the boundary conditions resolve to
        # nothing, reading the state is pointless and the eventual failure would
        # be unrecognisable.
        self.check_bcs_resolve_to_facets()
        yname = os.path.splitext(self.checkpoint_reload_yaml)[0] + ".yaml"

        #TODO: Handle if file doesn't exist        
        metadata = h5yaml.load(yname)    
        xdmf_filename = metadata["filename"]
        xdmf_file = dolfin.XDMFFile(dolfin.MPI.comm_world, xdmf_filename)

        #self.parameter shouldn't have units
        self.parameter = float(metadata["parameter"])
        self.parameter_start_value = self.parameter
        self.step_size = float(metadata["step_size"])

        def scalar_mag(expr):
            return mu.sqrt(mu.assemble(mu.dx * expr**2))
        def xdot_mag(expr):
            xhat = mu.Constant([1.0,0.0])
            return scalar_mag(mu.dot(expr,xhat))
        def general_mag(expr):
            try:
                return scalar_mag(expr)
            except:
                return xdot_mag(expr)

        pdd = self.solution.pdd
        fsr = self.solution.pdd.function_subspace_registry
        mu = self.solution.pdd.mesh_util        
        split_u = pdd.mixed_function_helper.solution_mixed_space.split(self.solver.solution.u_)
        mesh = mu.mesh
        U = mu.unit_registry
        with xdmf_file as infile:
            if mesh.geometry().dim() == 1:
                # dolfin's XDMF reader cannot load an interval mesh that it
                # wrote itself (the geometry is padded to "XY" on write and
                # the read then fails to initialize a 1D geometry). The
                # problem must already be constructed identically to the one
                # that wrote the checkpoint, so the mesh re-read is redundant;
                # read_checkpoint() below works on the existing mesh.
                logging.getLogger('stepper').info(
                    "1D mesh: skipping XDMF mesh re-read on checkpoint load")
            else:
                infile.read(mesh)
            
            for k in split_u:                    
                cur_fun = split_u[k]
                # print(general_mag(cur_fun))
                cur_units = metadata[k + "_units"]
                new_fun = fsr.new_independent_function(cur_fun.m) * U(cur_units)
                infile.read_checkpoint(new_fun.m,k)
                # print(general_mag(new_fun))
                # fsr.assign(cur_fun.m,new_fun)
                opportunistic_assign(new_fun,cur_fun,fsr) #handles units
                # print(general_mag(cur_fun))
            
            for b in pdd.bands:
                if hasattr(b,"mixedqfl_base_w"):
                    try:
                        cur_base_w = b.mixedqfl_base_w
                        incoming_units = metadata["b.name" + "mixedqfl_base_w_units"]
                        if cur_base_w.units == metadata["b.name" + "mixedqfl_base_w_units"]:
                            infile.read_checkpoint(cur_base_w.m, b.name + "/mixedqfl_base_w")
                        else:
                            #Make a new function for receiving the loaded data
                            print(f"adapting {b.name} base_w units")
                            read_in = fsr.new_independent_function(cur_base_w.m) *U(incoming_units)
                            infile.read_checkpoint(read_in.m, b.name + "/mixedqfl_base_w")
                            opportunistic_assign(read_in, cur_base_w, fsr)

                        # print(f"\nBand {b.name} base_w incoming")                                                        
                        # print(cur_base_w_mag.vector()[:])                            
                        
                        

                        # print(f"Band {b.name} after loading")
                        # print(b.mixedqfl_base_w.m.vector()[:])
                    except RuntimeError:
                        logging.getLogger('stepper').info(f"No base_w found in band {b.name}")
            thmq_phi = pdd.poisson.thermal_equilibrium_phi
            incoming_units = metadata["thermal_equilibrium_phi_units"]
            if thmq_phi.units == incoming_units:
                infile.read_checkpoint(thmq_phi.m,"thermal_equilibrium_phi")
            else:
                print("adapting thmq units")
                read_in = fsr.new_independent_function(thmq_phi.m) * U(incoming_units)
                infile.read_checkpoint(read_in.m, "thermal_equilibrium_phi")
                opportunistic_assign(read_in, thmq_phi, fsr)

            # Load optical state (see write_xdmf_checkpoint). Guarded by
            # "optical_Phi_scale" in metadata for backward compatibility with
            # checkpoints written before this existed -- those still load,
            # just without photon flux, same as before this fix.
            optical = self.solution.optical
            if "optical_Phi_scale" in metadata:
                optical.Phi_scale.magnitude.assign(float(metadata["optical_Phi_scale"]))
                for key in optical.fields.keys():
                    units_key = "optical/" + key + "/Phi_units"
                    if units_key not in metadata:
                        logging.getLogger('stepper').info(
                            f"No optical Phi found in checkpoint for field {key}")
                        continue
                    cur_Phi = optical.fields[key].Phi
                    incoming_units = metadata[units_key]
                    if str(cur_Phi.units) == incoming_units:
                        infile.read_checkpoint(cur_Phi.m, "optical/" + key + "/Phi")
                    else:
                        read_in = fsr.new_independent_function(cur_Phi.m) * U(incoming_units)
                        infile.read_checkpoint(read_in.m, "optical/" + key + "/Phi")
                        opportunistic_assign(read_in, cur_Phi, fsr)
            elif optical.fields:
                logging.getLogger('stepper').info(
                    "Checkpoint has no saved optical state (older format) -- "
                    "photon flux will start unsolved; resuming into a "
                    "stepper that doesn't itself re-solve optics before its "
                    "first solve (e.g. VoltageStepper) may fail to converge.")
        logging.getLogger('stepper').info(f"Loaded from checkpoint {xdmf_filename}. Parameter={self.parameter}")
        #Test whether loaded-in solution has any nans in it
        self.solver.solution.do_invalidate_cache()
        nans_found = self.solver.solution.has_nans
        if nans_found:
            logging.getLogger('stepper').info(f"Loaded solution has nans")

        debug = False
        if debug:
            print("***adaptive_stepper debugging code***")
            xdmf_filename = os.path.splitext(self.checkpoint_reload_yaml)[0] + ".xdmf"
            mesh = dolfin.Mesh()
            file = dolfin.XDMFFile(dolfin.MPI.comm_world, xdmf_filename)
            file.read(mesh)
            from simudo.fem import expr
            mesh_bbox = expr.mesh_bbox(mesh)
            space_name = "DG1"
            space = dolfin.FunctionSpace(mesh,"DG",1)
            CB_delta_w = dolfin.Function(space)
            file.read_checkpoint(CB_delta_w,"CB/delta_w")

            vspace_name = "BDM2"
            vspace = dolfin.FunctionSpace(mesh,"BDM",2)
            VB_j = dolfin.Function(vspace)
            file.read_checkpoint(VB_j, "VB/j")

            #line_cut
            import numpy as np
            num_points = 500
            y = np.mean(mesh_bbox[1]) #use the midpoint along the y-axis
            x_box = mesh_bbox[0]
            x_array = np.linspace(x_box[0], x_box[1],num_points)
            CB_delta_w_array = [CB_delta_w(dolfin.Point((x,y))) for x in x_array]
            VB_j_array = [VB_j(dolfin.Point((x,y))) for x in x_array]
            VB_jx_array = [v[0] for v in VB_j_array]

            from matplotlib import pyplot as plt
            f,a = plt.subplots(2,1)
            a[0].cla()
            a[0].plot(x_array,CB_delta_w_array,'.')
            a[1].cla()
            a[1].plot(x_array,VB_jx_array,'.')
            plt.savefig("tmp_CB_dw__VB_jx.png")

    def is_close_to_array(self,val,arr):
        '''find if val is close to any element in arr'''
        if type(arr) is float:
            return math.isclose(val,arr)
        isclose = False
        for a in arr:
            isclose = math.isclose(val,a)
            if isclose:
                return isclose
        return isclose




class ConstantStepperMixin():
    '''
Parameters
----------
constants: :py:class:`pint.Quantity` wrapping :py:class:`dolfin.Constant`
    On each step, the constant's value will be assigned to be the
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
