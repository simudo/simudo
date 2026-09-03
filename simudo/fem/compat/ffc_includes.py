# -*- coding: utf-8 -*-

import ffc.formatting


def _generate_includes(includes, parameters):
    """
    Monkey-patched version of :py:func:`ffc.formatting._generate_includes`
    because ffc has a bug that prevents it from using
    ``dolfin.parameters["form_compiler"]["external_includes"]`` to include
    specific system headers.
    """

    default_h_includes = [
        "#include <ufc.h>",
    ]

    default_cpp_includes = [
        # TODO: Avoid adding these includes if we don't need them:
        "#include <iostream>",
        "#include <stdexcept>",
        "#include <algorithm>",
    ]

    ### BEGIN MODIFIED SECTION
    external_includes = set(
        "#include <%s>" % inc
        for inc in parameters.get("external_includes", "").split(":")
        if inc
    )
    ### END MODIFIED SECTION

    s = set(default_h_includes + default_cpp_includes) | includes

    s2 = (set(default_cpp_includes) | external_includes) - s

    includes_h = "\n".join(sorted(s)) + "\n" if s else ""
    includes_cpp = "\n".join(sorted(s2)) + "\n" if s2 else ""
    return includes_h, includes_cpp


ffc.formatting._generate_includes = _generate_includes
