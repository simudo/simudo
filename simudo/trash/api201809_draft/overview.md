# Overview

## Class overview
Class instances, description, and attributes/methods:

- `ProblemData`: root object.
  - `.pdd`: instance of `PoissonDriftDiffusion`
  - `.optical`: instance of `Optical`
  - `.spatial`: instance of `Spatial`
  - `.*_registry`: various registries required for bookkeeping or working around dolfin issues

- `PoissonDriftDiffusion`: Poisson-Drift-Diffusion part of the problem
  - `.bands`: dictionary of band objects, see `*Band` below
  - `.poisson`: Poisson part of the problem, see `*Poisson` below
  - `.electro_optical_processes`: dictionary of `ElectroOpticalProcess` instances (and things derived from it). this includes dark generation/recombination mechanisms like SRH.
  - `.mesh`: mesh
  - `.function_space`: mixed function space built from bands' and poisson's FEM elements
  - `.weak_form`: weak form made up by summing bands' and poisson's weak forms
  - `.essential_bcs`: essential bcs, made up by calling `.get_essential_bcs()` on bands and poisson objects and concatenating the results

- `*Poisson`: Poisson part of the problem. Actual class can be `MixedPoisson` (or maybe some thing else)
  - `.V,.E,.ρ,...`: physical quantities and expressions/variables specific to Poisson problem
  - `.weak_form`: contribution to overall weak form
  - `.bcs`: boundary conditions, to be set by user

- `*Band`: A single band. Actual class can be `MixedQflNondegenerateConductionBand` or `ZeroQflNondegenerateConductionBand` depending on whether we're actually solving transport or just want qfl=0 (for pre-solve steps: charge neutrality, and Poisson-only thermal equilibrium)
  - `.u,.qfl,.j,...`: physical quantities and expressions/variables specific to this band
  - `.weak_form,.bcs`: see in `*Poisson` above

- `ElectroOpticalProcess`: represents an electronic or electro-optical process
  - `.get_alpha(wavelength)`: user-defined absorptivity
  - `.get_generation(band)`: can be overriden to supply a custom generation carrier generation

- `TwoBandElectroOpticalProcess`: convenience subclass of `ElectroOpticalProcess`
  - `.src_band,.dst_band`: source and destination bands
  - `.get_alpha(wavelength)`: user-defined band-to-band generation

- `DarkGenerationProcess`: subclass of `ElectroOpticalProcess` where `.get_alpha()` is zero
  - `.get_generation(band)`: user-defined generation for `band`

- `TwoBandDarkGenerationProcess`: convenience subclass of `GenerationProcess`
  - `.src_band,.dst_band`: source and destination bands
  - `.get_band_to_band_generation()`: user-defined band-to-band generation

- `Optical`: see `classes.py`
  - `.mesh`: optical mesh
  - `.fields`: dictionary of `OpticalField` instances

- `OpticalField`: a single intensity scalar field, corresponding to one wavelength+direction
  - `.wavelength`
  - `.direction`
  - `.solid_angle`
  - `.I,.alpha`
  - `.I_pddproj`: PDD-mesh-adapted version of `.I`

- `Spatial`: contains and provides spatially-dependent expressions (usually constant)
  - `.rules`: dictionary of rules of the type `(spatial_region, priority, key, value)` (probably needs to be a class of its own?)
  - `.construct_ufl(mesh_meta, key)`: constructs UFL expression for spatially-dependent expression `key` by apply `.rules`

- `CellRegion` and `FacetRegion`: represent named regions
  - boolean operations (and, or, subtract, xor) are defined in the obvious set-theoretic way

- `MixedFunctionSpace`: mixed function space.
  - `__init__(subspace_descriptors)`: each subspace_descriptor is a tuple `((trial_key, trial_unit), (test_key, test_unit))`
  - `.function_subspace_registry`: instance of `FunctionSubspaceRegistry`
  - `.function_space`: instance of `dolfin.FunctionSpace`
  - `.make_function()`: creates and returns `MixedFunction`
  - `.make_test_function()`: creates and returns `MixedTestFunction`

- `MixedFunction`: mixed (trial) function.
  - `__init__(mixed_function_space, function=None)`: initializer
  - `.mixed_function_space`: instance of `MixedFunctionSpace`

## Inheritance and mixin scheme for solution-like classes

This applies to classes that contain both physical quantities and possibly weak forms for a solution method. In particular:

- `*Band`
- `*Poisson`
- `OpticalField`

For example, for `*Poisson`:

1. Have a base class that defines the basic physical quantities and
   relationships. Call it `PoissonBase`.
2. Have a bunch of classes (specifically,
   [mixins](https://en.wikipedia.org/wiki/Mixin)) implementing various
   modifications or adding functionality to the base class.
3. Implement the solver components (e.g. weak form definitions) as one
   of those mixins. Call that `MixedPoissonMixin`.
4. Construct the final class(es) (that gets used by user code) by
   inheriting from: `MixedPoissonMixin`, some subset of the other
   mixins, and `PoissonBase`.

Another example:

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
            spatial = self.pdd.problem_data.spatial.construct_ufl(
                self.pdd.mesh_meta, self.name+'/mobility')
            return spatial

    class NondegenerateMixin(object):
        def u_from_weV(self):
            return boltzmann_expression(...)
        ...

    class MixedQflSolverMixin(object):
        ' mixed-qfl solver implemented here '
        def weak_form(self):
            ...
        ...

    class ConductionBandMixin(object):
        name = 'CB'
        sign = -1

    class ValenceBandMixin(object):
        name = 'VB'
        sign = +1

    class ZeroQflMixin(object):
        ''' for when you want to set qfl=0 for all bands, such as
        Poisson-only thermal equilibrium step '''

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




