"""
Python wrapper for ERG C extension module.

This module provides a clean Python interface to the ERG C extension,
making the API visible to users and IDEs while maintaining high performance.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# Import the C extension
from . import erg_python as _erg_c


class ERG:
    """
    ERG file reader for CarMaker binary results.

    This class provides access to signal data from ERG files, which are
    binary result files from CarMaker simulations. The file is automatically
    parsed upon initialization. Uses memory-mapped I/O for efficient access.

    Instances are cached by filepath - creating multiple ERG objects with the
    same filepath will return the same cached instance.

    Example:
        >>> from erg_python import ERG
        >>> erg = ERG("simulation.erg")
        >>> time = erg.get_signal("Time")
        >>> velocity = erg["Car.v"]  # Dictionary-style access
        >>> df = erg.get_all_signals()  # Get all signals as DataFrame
    """

    _instances: dict[Path, "ERG"] = {}

    def __new__(cls, filepath: str | Path):
        """
        Create or return cached ERG instance.

        Args:
            filepath: Path to the .erg file (string or Path object)

        Returns:
            ERG instance (cached if filepath was previously opened)
        """
        filepath = Path(filepath).resolve()
        if filepath not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[filepath] = instance
        return cls._instances[filepath]

    def __init__(self, filepath: str | Path):
        """
        Initialize and parse an ERG file.

        The file is automatically parsed upon initialization, and all metadata
        is loaded into memory. Signal data is loaded on-demand.

        Args:
            filepath: Path to the .erg file (string or Path object)

        Raises:
            FileNotFoundError: If the ERG file or its .info file is not found
            RuntimeError: If the file cannot be parsed
        """
        # Only initialize if not already initialized (for cached instances)
        if not hasattr(self, "_erg"):
            self._erg = _erg_c.ERG(str(filepath))

    def get_signal(self, signal_name: str) -> np.ndarray:
        """
        Get signal data by name.

        Returns the signal data as a NumPy array in its native data type
        (float32, float64, int32, etc.) with scaling factors already applied.

        Args:
            signal_name: Name of the signal (e.g., "Time", "Car.v")

        Returns:
            1D NumPy array with native dtype

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> velocity = erg.get_signal("Car.v")
            >>> print(f"Max velocity: {max(velocity):.2f}, dtype: {velocity.dtype}")
        """
        return self._erg.get_signal(signal_name)

    def get_all_signals(self) -> pd.DataFrame:
        """
        Get all signals as a pandas DataFrame.

        Returns a DataFrame with all signals as columns, using signal names
        as column names.

        Returns:
            DataFrame with all signals, indexed by row number

        Example:
            >>> erg = ERG("simulation.erg")
            >>> df = erg.get_all_signals()
            >>> print(df.columns)
            >>> print(df["Time"].head())
        """
        signal_names = self.get_signal_names()
        data = {name: self.get_signal(name) for name in signal_names}
        return pd.DataFrame(data)

    def get_signal_names(self) -> list[str]:
        """
        Get list of all signal names in the file.

        Returns:
            List of signal name strings

        Example:
            >>> erg = ERG("simulation.erg")
            >>> names = erg.get_signal_names()
            >>> print(f"Found {len(names)} signals")
            >>> print(f"First signal: {names[0]}")
        """
        return self._erg.get_signal_names()

    def get_signal_units(self) -> dict[str, str]:
        """
        Get all signal units as a dictionary.

        Returns:
            Dictionary mapping signal names to unit strings

        Example:
            >>> erg = ERG("simulation.erg")
            >>> units = erg.get_signal_units()
            >>> print(f"Time unit: {units['Time']}")
            >>> print(f"Velocity unit: {units['Car.v']}")
        """
        return self._erg.get_signal_units()

    def get_signal_unit(self, signal_name: str) -> str:
        """
        Get the unit string for a signal.

        Args:
            signal_name: Name of the signal

        Returns:
            Unit string (e.g., "m/s", "s", "m/s^2")

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> unit = erg.get_signal_unit("Car.v")
            >>> print(f"Velocity unit: {unit}")
        """
        return self._erg.get_signal_unit(signal_name)

    def get_signal_type(self, signal_name: str) -> np.dtype:
        """
        Get the numpy dtype for a signal.

        Args:
            signal_name: Name of the signal

        Returns:
            NumPy dtype object

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> dtype = erg.get_signal_type("Car.v")
            >>> print(f"Velocity dtype: {dtype}")
        """
        return self._erg.get_signal_type(signal_name)

    def __getitem__(self, signal_name: str) -> np.ndarray:
        """
        Get signal data using dictionary-style access.

        Args:
            signal_name: Name of the signal

        Returns:
            NumPy array of signal data

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> velocity = erg["Car.v"]
            >>> time = erg["Time"]
        """
        return self.get_signal(signal_name)

    def __repr__(self) -> str:
        """String representation of ERG object."""
        try:
            return f"ERG(signals={len(self.get_signal_names())})"
        except:
            return "ERG(not loaded)"
