"""Project-root-relative path resolution.

WHY THIS FILE EXISTS
--------------------
Python's ``Path('.')`` resolves relative to wherever the process was launched,
not relative to the source file.  A notebook started from ``notebooks/``, a
pytest run from the repo root, and a CLI invoked from ``/tmp`` all see a
different working directory.  Any module that hard-codes ``Path('../../data')``
will silently point to the wrong location in at least one of those contexts.

This module solves the problem by walking upward from its own location
(``src/paths.py``) until it finds the unique combination of ``pyproject.toml``
and a ``configs/`` folder that marks the project root.  Every other module
imports ``PROJECT_ROOT`` from here instead of making its own guess.

HOW IT WORKS
------------
``_find_project_root`` is called once at import time.  It starts at
``Path(__file__).parent.parent`` (which is the project root in the normal
layout) so the common case terminates immediately.  The upward walk is a
safety net for unusual launch contexts (e.g. running a single file from deep
inside the tree).

EXPOSED CONSTANTS
-----------------
PROJECT_ROOT  -- absolute Path to the repository root directory
CONFIGS_DIR   -- absolute Path to the ``configs/`` directory
DATA_DIR      -- absolute Path to the ``data/`` directory
"""
from __future__ import annotations

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk upward from ``start`` until a directory containing both
    ``pyproject.toml`` and a ``configs`` folder is found.

    Both markers are required so that an ancestor project (if this repo is
    nested inside another) is not mistakenly identified as the root.  Falls
    back to ``start`` itself if nothing matches, which should not happen in
    practice but prevents a hard crash in unusual deployment setups.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
            return candidate
    return start


# Resolved once at import time.  ``Path(__file__).resolve()`` gives an
# absolute path immune to ``os.chdir`` calls that might happen in notebooks.
PROJECT_ROOT: Path = _find_project_root(Path(__file__).resolve().parent.parent)

# Shorthand paths used by config loading and data preparation.
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
