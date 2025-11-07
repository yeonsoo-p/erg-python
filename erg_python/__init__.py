"""
Python bindings for ERG (CarMaker binary results) file reader.

This module provides a Python interface to read and parse ERG files,
which are binary result files from CarMaker simulations.

Example:
    >>> from erg_python import ERG
    >>> erg = ERG("path/to/file.erg")
    >>> time = erg.get_signal("Time")
    >>> velocity = erg["Car.v"]  # Dictionary-style access
    >>> all_signals = erg.get_all_signals()  # Get all as structured array
"""

from .erg import ERG

__version__ = "0.1.0"
__all__ = ["ERG"]
