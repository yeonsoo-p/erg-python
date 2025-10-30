"""
Python wrapper for ERG C extension module.

This module provides a clean Python interface to the ERG C extension,
making the API visible to users and IDEs while maintaining high performance.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np

# Import the C extension
from . import erg_python as _erg_c


class ERG:
    """
    ERG file reader for CarMaker binary results.

    This class provides access to signal data from ERG files, which are
    binary result files from CarMaker simulations. The file is automatically
    parsed upon initialization.

    Attributes:
        signal_names (list[str]): List of all available signal names
        sample_count (int): Number of samples (rows) in the file
        data (pd.DataFrame | None): Cached DataFrame of loaded signals

    Example:
        >>> from erg_python import ERG
        >>> erg = ERG("simulation.erg")
        >>> time = erg.get_signal("Time")
        >>> velocity = erg.get_signal("Car.v")
        >>> signals = erg.get_signals(["Time", "Car.v", "Car.ax"])
        >>> df = erg.data  # Access cached DataFrame
    """

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
        self._erg = _erg_c.ERG(filepath)
        self.data: pd.DataFrame | None = None

    def get_signal(self, signal_name: str) -> np.ndarray:
        """
        Get signal data by name.

        Automatically caches the signal in a DataFrame for future access.
        Returns the signal data as a NumPy array. All data types are converted to
        float64 with scaling factors applied.

        Args:
            signal_name: Name of the signal (e.g., "Time", "Car.v")

        Returns:
            NumPy array or list of float values with length equal to sample_count

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> velocity = erg.get_signal("Car.v")
            >>> print(f"Max velocity: {max(velocity):.2f}")
            >>> # Signal is now cached in erg.data DataFrame
        """
        # Check if signal is already cached in DataFrame
        if self.data is not None and signal_name in self.data.columns:
            return self.data[signal_name].values

        # Get signal from C extension
        signal_data = self._erg.get_signal(signal_name)

        # Update DataFrame cache
        if self.data is None:
            self.data = pd.DataFrame({signal_name: signal_data})
        else:
            self.data[signal_name] = signal_data

        return signal_data

    def get_signals(self, signal_names: list[str]) -> tuple[np.ndarray, ...]:
        """
        Get multiple signals in a batch operation.

        Automatically caches all signals in a DataFrame for future access.
        This is more efficient than calling get_signal() multiple times,
        as it optimizes memory-mapped I/O operations.

        Args:
            signal_names: List of signal names to retrieve

        Returns:
            Tuple of NumPy arrays in the same order as signal_names

        Example:
            >>> erg = ERG("simulation.erg")
            >>> time, velocity, accel = erg.get_signals(["Time", "Car.v", "Car.ax"])
            >>> # All signals are now cached in erg.data DataFrame
        """
        # Check which signals are not cached in DataFrame
        if self.data is None:
            uncached = signal_names
        else:
            uncached = [name for name in signal_names if name not in self.data.columns]

        # Get uncached signals from C extension (returns tuple)
        if uncached:
            signal_arrays = self._erg.get_signals(uncached)

            # Convert tuple to dict for DataFrame
            new_signals = dict(zip(uncached, signal_arrays))

            # Update DataFrame cache
            if self.data is None:
                self.data = pd.DataFrame(new_signals)
            else:
                for name, data in new_signals.items():
                    self.data[name] = data

        # Return tuple of all requested signals from DataFrame
        return tuple(self.data[name].values for name in signal_names)

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
            cached = len(self.data.columns) if self.data is not None else 0
            return f"ERG(signals={len(self.signal_names)}, samples={self.sample_count}, cached={cached})"
        except:
            return "ERG(not loaded)"