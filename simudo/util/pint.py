
import io

import pint
import pkg_resources

__all__ = [
    'XUnitRegistry',
    'make_unit_registry']

class XUnitRegistry(pint.UnitRegistry):
    def load_definitions(self, file, is_resource=False):
        if isinstance(file, (str, bytes)):
            file, is_resource = self.custom_file_open(file)
        return super().load_definitions(file, is_resource)

    def custom_file_open(self, path):
        return (io.StringIO(pkg_resources.resource_string(
            __name__, "pint/" + path).decode('utf-8')), True)

def make_unit_registry(extra_definitions=()):
    registry = XUnitRegistry(None)
    registry.load_definitions("default_en.txt")
    for definition in extra_definitions:
        registry.define(definition)
    registry._build_cache()
    return registry
