from .base import load, peek, detect_vendor, registered_vendors, register
from . import bruker  # noqa: F401 -- import for its base.register() side effect
from . import xpulse  # noqa: F401 -- import for its base.register() side effect
from . import varian  # noqa: F401 -- import for its base.register() side effect
from . import jeol  # noqa: F401 -- import for its base.register() side effect

__all__ = ["load", "peek", "detect_vendor", "registered_vendors", "register", "bruker", "xpulse", "varian", "jeol"]
