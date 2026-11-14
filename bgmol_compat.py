"""bgmol imports mdtraj.version and mdtraj.utils.six, which were removed in mdtraj 1.10.
Import this module before bgmol."""
import sys
import types

import mdtraj

if not hasattr(mdtraj, "version"):
    mdtraj.version = types.SimpleNamespace(short_version=mdtraj.__version__)

if "mdtraj.utils.six" not in sys.modules:
    six = types.ModuleType("mdtraj.utils.six")
    six.PY3 = True
    six.string_types = (str,)
    sys.modules["mdtraj.utils.six"] = six
