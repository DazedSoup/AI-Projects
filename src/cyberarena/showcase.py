"""Export the runs and experiments chosen for the public dashboard into a small ``showcase/`` folder.

    python -m cyberarena.showcase                     # use the selection saved by the Lab (runs/.showcase.json)
    python -m cyberarena.showcase --runs reference --experiments main

Implementation lives in ``cyberarena.dashboard.showcase``; see docs/contracts.md "Public showcase (v6)".
"""

import sys

from cyberarena.dashboard.showcase import main

if __name__ == "__main__":
    sys.exit(main())
