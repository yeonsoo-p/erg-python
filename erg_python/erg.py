"""
Python wrapper for ERG C extension module.

This module provides a clean Python interface to the ERG C extension,
making the API visible to users and IDEs while maintaining high performance.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

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

    Attributes:
        signal_names (list[str]): List of all available signal names
        sample_count (int): Number of samples (rows) in the file

    Example:
        >>> from erg_python import ERG
        >>> erg = ERG("simulation.erg")
        >>> time = erg.get_signal("Time")
        >>> velocity = erg["Car.v"]  # Dictionary-style access
        >>> signals = [erg[name] for name in ["Time", "Car.v", "Car.ax"]]
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
        is loaded into memory. Signal data is loaded on-demand using memory-mapped I/O.

        Args:
            filepath: Path to the .erg file (string or Path object)

        Raises:
            FileNotFoundError: If the ERG file or its .info file is not found
            RuntimeError: If the file cannot be parsed
        """
        # Only initialize if not already initialized (for cached instances)
        if not hasattr(self, "_erg"):
            self._erg = _erg_c.ERG(filepath)

    def get_signal(self, signal_name: str) -> np.ndarray:
        """
        Get signal data by name.

        Returns the signal data as a NumPy array in its native data type
        (float32, float64, int32, etc.) with scaling factors already applied.
        Uses memory-mapped I/O for efficient access.

        Args:
            signal_name: Name of the signal (e.g., "Time", "Car.v")

        Returns:
            NumPy array with native dtype and length equal to sample_count

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> velocity = erg.get_signal("Car.v")
            >>> print(f"Max velocity: {max(velocity):.2f}, dtype: {velocity.dtype}")
            >>> # Or use dictionary-style access
            >>> time = erg["Time"]
        """
        return self._erg.get_signal(signal_name)

    def get_signal_info(self, signal_name: str) -> dict[str, str | int | float]:
        """
        Get metadata for a signal.

        Returns information about the signal including its data type,
        unit, scaling factor, and offset.

        Args:
            signal_name: Name of the signal

        Returns:
            Dictionary with keys:
                - name (str): Signal name
                - type (str): Data type (e.g., "float", "double", "int")
                - type_size (int): Size in bytes
                - unit (str): Unit string (e.g., "m/s", "s")
                - factor (float): Scaling factor
                - offset (float): Scaling offset

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> info = erg.get_signal_info("Car.v")
            >>> print(f"Unit: {info['unit']}, Type: {info['type']}")
        """
        return self._erg.get_signal_info(signal_name)

    def get_unit(self, signal_name: str) -> str:
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
            >>> unit = erg.get_unit("Car.v")
            >>> print(f"Velocity unit: {unit}")
        """
        return self._erg.get_unit(signal_name)

    def list_signals(self) -> list[str]:
        """
        List all signal names in the file.

        Returns:
            List of signal name strings

        Example:
            >>> erg = ERG("simulation.erg")
            >>> signals = erg.list_signals()
            >>> print(f"Found {len(signals)} signals")
        """
        return self._erg.list_signals()

    def get_units(self) -> dict[str, str]:
        """
        Get all signal units as a dictionary.

        Returns:
            Dictionary mapping signal names to unit strings

        Example:
            >>> erg = ERG("simulation.erg")
            >>> units = erg.get_units()
            >>> print(f"Time unit: {units['Time']}")
            >>> print(f"Velocity unit: {units['Car.v']}")
        """
        return self._erg.get_units()

    @property
    def signal_names(self) -> list[str]:
        """
        List of all available signal names in the file.

        Returns:
            List of signal name strings

        Example:
            >>> erg = ERG("simulation.erg")
            >>> print(f"Found {len(erg.signal_names)} signals")
            >>> print(f"First signal: {erg.signal_names[0]}")
        """
        return self._erg.signal_names

    @property
    def sample_count(self) -> int:
        """
        Number of samples (rows) in the file.

        Returns:
            Integer sample count

        Example:
            >>> erg = ERG("simulation.erg")
            >>> print(f"File contains {erg.sample_count} samples")
        """
        return self._erg.sample_count

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
            return f"ERG(signals={len(self.signal_names)}, samples={self.sample_count})"
        except:
            return "ERG(not loaded)"
