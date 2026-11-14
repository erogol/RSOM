try:
    from .rsom import RSOM
except ImportError:  # the folder is not imported as a package
    from rsom import RSOM
