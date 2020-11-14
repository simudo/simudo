__version__ = "0.6.4.0"

# get fossil version
try:
    import os

    with open(
        os.path.join(os.path.dirname(__file__), "..", "manifest.uuid"),
        "rt",
        encoding="ascii",
    ) as f:
        uuid = f.read()
except (OSError, RuntimeError, NameError, ImportError):
    pass
else:
    __version__ += "~" + uuid[:10]
