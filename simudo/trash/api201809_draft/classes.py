
from cached_property import cached_property
from .util import NameDict

class ProblemData(object):
    pdd = ...
    optical = ...
    unit_registry = ...
    spatial = ...
    function_subspace_registry = ...
    function_space_cache = ...
    mesh_unit = ...

class Spatial(object):
    ''' holds quantities (e.g. material properties) that vary in space
    according to what geometrical region you're in '''
    def rule_compare(self, rule1, rule2):
        pass
    # INCOMPLETE; TODO

class PoissonDriftDiffusion(object):
    ''' holds poisson drift diffusion data '''

    mesh = ...
    temperature = ...

    @cached_property
    def poisson(self):
        return self.PoissonProblem

    @cached_property
    def bands(self):
        return NameDict()

class BasePoissonProblem(object):
    ''' defines Poisson quantities. does NOT implement a solver '''
    @property
    def charge_density(self):
        ' add up charge contributions from bands and static charge '
        return ...

    ...

class MixedPoissonSolverMixin(object):
    ''' solver for Poisson problem '''
    ...

# the PoissonProblem then inherits from both of the above classes,
# and thus includes quantity definitions AND a solver
class PoissonProblem(MixedPoissonSolverMixin, BasePoissonProblem):
    pass

class BaseSemiconductorBand(object):
    @property
    def w(self):
        ''' Quasi-fermi level. By default, use `w_from_u`. '''
        return self.weV_from_u(self.u) - self.e_times_V

    @property
    def u(self):
        ''' Number density. By default, use `u_from_w`. '''
        return self.u_from_weV(self.w + self.e_times_V)

    @property
    def e_times_V(self):
        return self.pdd.poisson.e_times_V

    @property
    def rho_contribution(self):
        return self.u * self.sign * self.pdd.elementary_charge

    @property
    def mobility(self):
        ''' by default, get from Spatial object '''
        spatial = self.pdd.problem_data.spatial.get_conditional(
            self.pdd.mesh, self.name+'/mobility')
        return spatial

class NondegenerateMixin(object):
    def u_from_weV(self):
        return the_boltzmann_expression
    ...

class MixedQflSolverMixin(object):
    ' mixed-qfl solver implemented here '
    ...
    ...
    ...

class ConductionBandMixin(object):
    name = 'CB'
    sign = -1

class ValenceBandMixin(object):
    name = 'VB'
    sign = +1

class ZeroQflMixin(object):
    ''' for when you want to set qfl=0 for all bands '''
    @property
    def w(self):
        return 0.0 * u.eV

# yes, there really aren't any definitions within the class
# we're just constructing classes using inheritance
class ZeroQflNondegenerateBand(ZeroQflMixin,
                               NondegenerateMixin,
                               BaseSemiconductorBand):
    pass

class MixedQflNondegenerateBand(MixedQflSolverMixin,
                                NondegenerateMixin,
                                BaseSemiconductorBand):
    pass

class ZeroQflNondegenerateConductionBand(ConductionBandMixin,
                                         ZeroQflNondegenerateBand):
    pass

class ZeroQflNondegenerateValenceBand(ValenceBandMixin,
                                      ZeroQflNondegenerateBand):
    pass

class MixedQflNondegenerateConductionBand(ConductionBandMixin,
                                          MixedQflNondegenerateBand):
    pass

class MixedQflNondegenerateValenceBand(ValenceBandMixin,
                                       MixedQflNondegenerateBand):
    pass

class Optical():
    mesh = ...

    fields = ... # optical fields

class OpticalField():
    optical = ... # parent optical object

    wavelength = ...
    direction = ...
    solid_angle = ... # solid angle corresponding to this light field
    # sum of all solid angles at a given wavelength must sum up to 4π

    # following are quantities on optical mesh (not PDD)
    I = ...
    alpha = ...

    # versions of quantities above, projected/interpolated onto PDD mesh
    I_pddproj = ...

class SimpleLinearAbsorption(TwoBandProcess):
    ''' example for linear absorption '''

    def get_alpha(self, wavelength):
        alpha = compute_from_some_table(wavelength=wavelength)
        return alpha

class CrossSectionAbsorption(TwoBandProcess):
    density_expression = ... # set to either IB.u or (IB.N0 - IB.u)

    def get_alpha(self, wavelength):
        sigma = compute_from_some_table(wavelength=wavelength)
        alpha = sigma*self.density_expression
        return alpha

