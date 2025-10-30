"""
cmparser - Python CarMaker Parser

A Python package for parsing CarMaker project files and simulation data including:
- InfoFiles (TestRun, Vehicle, Road, etc.)
- ERG binary simulation results
- FMU (Functional Mock-up Unit) files
- BLF (CAN Bus Logging Format) files

Example:
    >>> from cmparser import Project, ERG, BLF
    >>> project = Project("my_project")
    >>> erg_data = ERG("simulation.erg")
    >>> blf_data = BLF("canlog.blf", ["messages.dbc"])
"""

from .cminfofile import *
from .cmerg import *

__version__ = "0.3.3"
