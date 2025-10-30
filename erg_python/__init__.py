"""
Python bindings for ERG (CarMaker binary results) file reader.

This module provides a Python interface to read and parse ERG files,
which are binary result files from CarMaker simulations.

Example:
    >>> from erg_python import ERG
    >>> erg = ERG("path/to/file.erg")
    >>> time = erg.get_signal("Time")
    >>> velocity = erg.get_signal("Car.v")
    >>> signals = erg.get_signals(["Time", "Car.v", "Car.ax"])
"""

from .erg import ERG

__version__ = "0.1.0"
__all__ = ["ERG"]
