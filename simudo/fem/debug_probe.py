import dolfin


class DebugProbe:
    """
    This class implements a way to probe physical quantities at points
    on a mesh. Note that this class will project (create an entire
    :py:class:`dolfin.Function`) to evaluate the function even at a
    single point.
    """

    def __init__(self, quantity, space, coordinate_units):
        self.quantity = quantity
        self.space = space
        self.coordinate_units = coordinate_units  # not yet implemented

        if hasattr(quantity, "magnitude"):
            self.units = quantity.units
            quantity = quantity.magnitude
        else:
            self.units = None

        self.projected = dolfin.project(quantity, space)

    def interpret_point(self, args):
        if len(args) == 1 and isinstance(args[0], dolfin.Point):
            p = args[0]
        else:
            args = list(args)
            args.extend((0.0,) * (3 - len(args)))
            p = dolfin.Point(*args)
        return p

    def __call__(self, *args):
        p = self.interpret_point(args)
        value = self.projected(p)
        units = self.units
        if units is not None:
            value = units * value
        return value

class DebugFacetProbe:
    """
    This class implements a way to probe physical quantities at facets
    of a mesh. 
    """
    def __init__(self, quantity, facet, mesh_util, coordinate_units):
        self.quantity = quantity
        self.facet = facet
        self.coordinate_units = coordinate_units  # not yet implemented
        self.mesh_util = mesh_util
        mu = self.mesh_util

        if hasattr(quantity, "magnitude"):
            self.units = quantity.units
            quantity = quantity.magnitude
        else:
            self.units = None

        self.dS =  mu.region_oriented_dS(facet, orient=False)[1]
        
        

    def __call__(self, user_unit=None,*args):
        q_form = (self.dS * self.quantity).delete_units().to_ufl().m
        q_int = dolfin.assemble(q_form)
        facet_area = dolfin.assemble((self.dS * dolfin.Constant(1.0)).delete_units().to_ufl().m)
        value = q_int/facet_area
        units = self.units
        if units is not None:
            value = units * value
        if user_unit is not None:
            value = value.to(user_unit)
        return value