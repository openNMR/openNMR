from .base import load, peek, detect_vendor, registered_vendors, register
from . import lists  # noqa: F401 -- vendor-neutral one-column value lists
from .lists import find_value_lists, read_value_list
from . import bruker  # noqa: F401 -- import for its base.register() side effect
from . import xpulse  # noqa: F401 -- import for its base.register() side effect
from . import varian  # noqa: F401 -- import for its base.register() side effect
from . import jeol  # noqa: F401 -- import for its base.register() side effect
from . import spinsolve  # noqa: F401 -- import for its base.register() side effect
from . import simpson  # noqa: F401 -- SIMPSON .spe/.fid text files
from . import tecmag  # noqa: F401 -- TecMag .tnt files and Data Point Listings
from . import dmfit  # noqa: F401 -- DMFit .asc / .fxml
from . import tables  # noqa: F401 -- .csv / .xy / JCAMP-DX import
from . import bruker_pfg  # noqa: F401 -- PFG gradient axis with or without diff.xml

__all__ = ["load", "peek", "detect_vendor", "registered_vendors", "register",
           "find_value_lists", "read_value_list",
           "bruker", "xpulse", "varian", "jeol", "spinsolve", "simpson", "tecmag", "dmfit", "tables", "bruker_pfg", "lists"]

# openNMR's own dataset folders (a saved simulation or derived spectrum,
# see io.history.save_spectrum_folder). Registered last: its marker file
# never appears in a vendor's folder, so order only matters for speed.
from . import history  # noqa: E402,F401
register("opennmr", history.is_dataset_folder, history.read_spectrum_folder,
         peek=history._peek_spectrum_folder)
