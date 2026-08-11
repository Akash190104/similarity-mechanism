"""Inspect AI wrapper for the Similarity evaluation framework."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that imports like
# ``from src.games.base import Game`` work when Inspect loads task files.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
