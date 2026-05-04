"""
conftest.py — Pytest path configuration
=========================================

Adds app/backend to sys.path so that `import core.*` resolves correctly
when tests are run from the project root (i.e. `python -m pytest tests/`).

This also satisfies the Pylance/Pyright language server so it stops
reporting "cannot find module core.*" in the IDE.
"""

import sys
import os

# Insert app/backend at the front of sys.path so every test file can do:
#   from core.anfis_core import ANFIS
# without a manual sys.path.insert() inside each test.
BACKEND = os.path.join(os.path.dirname(__file__), 'app', 'backend')
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
