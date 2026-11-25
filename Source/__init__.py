"""EWF geometry optimization driver — importable as ``ewf_geom_opt``.

The directory on disk is ``Source/``; ``pyproject.toml`` maps it to the
package name via ``package-dir``.  Renaming the directory would rewrite 92
references across 59 tracked files, and the archived campaigns under
``Examples/`` reference the current layout, so the two names differ
deliberately.

Two supported modes
-------------------
**Installed** (preferred)::

    pip install -e .
    from ewf_geom_opt import external_sci

**Copied next to a run** (still supported, and relied upon by every archived
campaign)::

    cp Source/*.py my_run_folder/
    cd my_run_folder && python EWF-CI_Geom_Opt_HPC.py --config config.yaml

The second mode is why the modules import each other by bare name
(``import external_sci``) rather than relatively (``from . import
external_sci``): a folder of loose files is not a package, so relative
imports cannot work there.  To make the same bare-name imports resolve when
the code *is* imported as a package, this module puts its own directory on
``sys.path``.

That is a bridge, not an end state.  It is here so that packaging does not
force a flag-day rewrite of every module's imports while the MBE prototype
and the published campaigns still depend on flat-file mode.  See
``Code_for_SQD_incorporation/MIQuLab_Modernization.md``.

Do not mix the two styles in one process
----------------------------------------
Because the same file is reachable under two names, importing it both ways
creates **two distinct module objects that do not share module-level state**::

    >>> from ewf_geom_opt import external_sci as qualified
    >>> import external_sci as bare
    >>> qualified.__file__ == bare.__file__      # same file on disk
    True
    >>> qualified is bare                        # different module objects
    False
    >>> sorted(k for k in sys.modules if k.endswith('external_sci'))
    ['ewf_geom_opt.external_sci', 'external_sci']

Anything held at module or class level is then duplicated, and the two copies
drift apart silently.  That is not hypothetical for this code base: the SBD
wrapper keeps per-run counters and accumulators as class attributes
(``n_dense``, ``n_heavy``, ``n_reassigned``, ``margins``, ``terms``), and a
split import would leave half the terms recorded in a copy nobody reads.

**Rule: pick one style per process.**  The driver and the run folders use bare
names; library consumers -- notebooks, tests, the regression fixture -- should
use ``from ewf_geom_opt import ...`` throughout and not fall back to bare
names.  Removing the hazard entirely means converting every module to relative
imports, which ends flat-file mode; that trade is the open question in the
modernization plan.
"""

import os as _os
import sys as _sys

__version__ = "0.1.0"

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

#: Absolute path of the main driver.  Its filename contains a hyphen and is
#: therefore not importable; callers that need to run it (the ``ewf-geom-opt``
#: console script, and the MBE prototype's ``spec_from_file_location`` loader)
#: resolve it through here rather than hard-coding the path.
DRIVER = _os.path.join(_HERE, "EWF-CI_Geom_Opt_HPC.py")


def driver_path() -> str:
    """Absolute path of ``EWF-CI_Geom_Opt_HPC.py``.

    Raises
    ------
    FileNotFoundError
        If the driver is missing, which means an incomplete install rather
        than a configuration problem -- say so plainly instead of failing
        later with an opaque import error.
    """
    if not _os.path.isfile(DRIVER):
        raise FileNotFoundError(
            "EWF driver not found at %s -- the package appears to be "
            "installed without it." % DRIVER)
    return DRIVER
