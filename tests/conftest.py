"""Pytest configuration for brain image analysis tests."""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
