
from cached_property import cached_property

from ..util import SetattrInitMixin

__all__ = [
    'Material',
    'SimpleSiliconMaterial',
    'BadSimpleSiliconMaterial']

class Material(SetattrInitMixin):
    '''Base material class.

Parameters
----------
problem_data: :py:class:`.problem_data.ProblemData`, optional
    ProblemData instance (used to pull, for example, the temperature
    variable).
unit_registry: :py:class:`pint.UnitRegistry`, optional
    Unit registry.

Attributes
----------
name: str, optional
    Material name.
'''
    name = None

    def get_dict(self):
        '''Construct dictionary of material parameters.

Returns
-------
dict:
    Dictionary where keys are :py:meth:`~.spatial.Spatial.add_rule`
    keys, and values are the values (expressions).
'''
        return {}

    @cached_property
    def dict(self):
        return self.get_dict()

    @property
    def temperature(self):
        return self.pdd.temperature

    @property
    def pdd(self):
        return self.problem_data.pdd

    @property
    def unit_registry(self):
        return self.problem_data.unit_registry

    def register(self, name=None):
        if name is None:
            name = self.name

        for k, v in self.dict.items():
            self.pdd.spatial.add_material_data(
                key=k, material_name=name, value=v)

class SimpleSiliconMaterial(Material):
    '''Very simple silicon material. No temperature dependences.'''

    name = 'Si'

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d.update({
            # band parameters
            # source: http://www.ioffe.ru/SVA/NSM/Semicond/Si/bandstr.html
            'CB/energy_level': U('1.12 eV'),
            'VB/energy_level': U('0 eV'),
            'CB/effective_density_of_states': U('3.2e19 1/cm^3'),
            'VB/effective_density_of_states': U('1.8e19 1/cm^3'),

            # electrical properties
            # source: http://www.ioffe.ru/SVA/NSM/Semicond/Si/electric.html
            'CB/mobility': U('1400 cm^2/V/s'),
            'VB/mobility': U(' 450 cm^2/V/s'),

            # poisson
            # http://www.ioffe.ru/SVA/NSM/Semicond/Si/optic.html
            'poisson/permittivity': U('11.7 vacuum_permittivity'),
        })

        return d

class BadSimpleSiliconMaterial(SimpleSiliconMaterial):
    '''Just an example of how to use inheritance to selectively modify
    properties of the superclass'''

    name = 'badSi'

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d['CB/mobility'] = d['CB/mobility'] / 3
        d['VB/mobility'] = U('233 cm^2/V/s')

        return d
