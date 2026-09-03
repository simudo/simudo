"""Launch the Simudo GUI.

    python -m simudo.gui [PROJECT.yaml] [panel serve options ...]

This is a thin wrapper around ``panel serve <app.py>``: it locates the bundled
``app.py`` inside the installed package and hands everything else to Panel, so
every ``panel serve`` option (``--show``, ``--port``, ``--address``, ...)
works unchanged.

Launching this way rather than ``cd``-ing into the package directory matters:
the GUI's Open and Save dialogs start in the current working directory, so run
it from wherever you keep your projects.

A project file may be given positionally.  Panel would otherwise treat a second
path as another application to serve, so it is passed through the environment
variable ``app.py`` already honours.
"""

import os
import sys

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # A positional project file goes via the environment; see module docstring.
    rest = []
    for arg in argv:
        if arg.endswith(".yaml") and not arg.startswith("-") \
                and os.path.exists(arg):
            os.environ["SIMUDO_GUI_LOAD_FILE"] = os.path.abspath(arg)
        else:
            rest.append(arg)

    try:
        from panel.command import main as panel_main
    except ImportError:
        sys.exit(
            "The Simudo GUI needs panel, which is not installed in this "
            "environment.\n"
            "    pip install 'simudo[gui]'\n"
            "or, for a GUI-only machine, see doc/source/user/install.rst.")

    sys.argv = ["panel", "serve", APP, *rest]
    return panel_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
