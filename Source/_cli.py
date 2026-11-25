"""Console-script bridge to the driver.

``EWF-CI_Geom_Opt_HPC.py`` contains a hyphen, so it is not a valid Python
module name and cannot be the target of a ``[project.scripts]`` entry point,
which must name an importable module attribute.  This module runs it instead.

Renaming the file would remove the need for this, but it is referenced 1589
times across 1468 archived files under ``Examples/`` -- run logs, copied
scripts and configs that record how published campaigns were invoked.
Renaming would leave those pointing at a filename that no longer exists,
which is precisely the reproducibility the "a run folder may carry its own
source" convention exists to protect.  The wart is cheaper than the rename.

``runpy`` is used rather than ``exec`` so the driver sees ``__name__ ==
"__main__"`` and its own ``main()`` guard fires exactly as it does under
``python EWF-CI_Geom_Opt_HPC.py``; argv is passed through untouched.
"""

import runpy
import sys

from . import driver_path


def main() -> None:
    """Entry point for the ``ewf-geom-opt`` console script."""
    path = driver_path()
    # argv[0] becomes the driver path, so the driver's own usage/error
    # messages name something that exists on disk.
    sys.argv[0] = path
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
