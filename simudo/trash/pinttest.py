import pint
from decimal import Decimal
import yaml

u1 = pint.UnitRegistry()
u2 = pint.UnitRegistry()

u1.define('xx = 1*meter')
u2.define('xx = 1*micrometer')

def changesys(new_system, qty):
    orig_units = qty.units
    x = qty.to_base_units()
    x = new_system.Quantity(x.magnitude, x.units)
    try:
        x = x.to(str(orig_units))
    except (pint.errors.UndefinedUnitError,
            pint.errors.DimensionalityError):
        pass
    return x

y = changesys(u2, (Decimal(1) * u1.xx))
print(yaml.load(yaml.dump(y)).to_base_units())



