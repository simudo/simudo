import pint

__all__ = [
    'make_unit_registry'
]


def make_unit_registry(extra_definitions=()):
    registry = pint.UnitRegistry()
    for definition in extra_definitions:
        registry.define(definition)
    return registry
