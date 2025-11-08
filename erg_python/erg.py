"""
Python wrapper for ERG C extension module.

This module provides a clean Python interface to the ERG C extension,
making the API visible to users and IDEs while maintaining high performance.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import the C extension
from . import erg_python as _erg_c


class ERG:
    """
    ERG file reader for CarMaker binary results.

    This class provides access to signal data from ERG files, which are
    binary result files from CarMaker simulations. The file is automatically
    parsed upon initialization. Uses memory-mapped I/O for efficient access.

    Instances are cached by filepath - creating multiple ERG objects with the
    same filepath will return the same cached instance. This means all caches
    (including the structured array cache from get_all_signals()) persist across
    instantiations. To clear caches, use ERG._instances.clear().

    Example:
        >>> from erg_python import ERG
        >>> erg = ERG("simulation.erg")
        >>> time = erg.get_signal("Time")
        >>> velocity = erg["Car.v"]  # Dictionary-style access
        >>> data = erg.get_all_signals()  # Get all signals as structured array
    """

    _instances: dict[Path, "ERG"] = {}

    # ========================================
    # Lifecycle Methods
    # ========================================

    def __new__(cls, filepath: str | Path, prefetch: bool = True) -> ERG:
        """
        Create or return cached ERG instance.

        Args:
            filepath: Path to the .erg file (string or Path object)
            prefetch: If True, call get_all_signals() during initialization to populate cache

        Returns:
            ERG instance (cached if filepath was previously opened)
        """
        filepath = Path(filepath).resolve()
        if filepath not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[filepath] = instance
        return cls._instances[filepath]

    def __init__(self, filepath: str | Path, prefetch: bool = True) -> None:
        """
        Initialize and parse an ERG file.

        The file is automatically parsed upon initialization using memory-mapped I/O
        for efficient access. Signal data is accessed on-demand via get_signal() or
        get_all_signals().

        Args:
            filepath: Path to the .erg file (string or Path object)
            prefetch: If True, call get_all_signals() during initialization to populate cache

        Raises:
            FileNotFoundError: If the ERG file or its .info file is not found
            RuntimeError: If the file cannot be parsed
        """
        # Only initialize if not already initialized (for cached instances)
        if not hasattr(self, "_erg"):
            self._erg: Any = _erg_c.ERG(str(filepath))
            # Initialize structured array cache (None means not yet retrieved)
            self._struct_array_cache: NDArray[np.void] | None = None
            # Initialize metadata caches
            self._units_cache: dict[str, str] = {}
            self._types_cache: dict[str, np.dtype[Any]] = {}
            self._factors_cache: dict[str, float] = {}
            self._offsets_cache: dict[str, float] = {}

        if prefetch:
            # Prefetch all signals to populate the structured array cache
            self.get_all_signals()

    # ========================================
    # Core Data Access Methods
    # ========================================

    def get_signal(self, signal_name: str) -> NDArray[Any]:
        """
        Get signal data by name.

        Returns the signal data as a NumPy array in its native data type
        (float32, float64, int32, etc.) with scaling factors already applied.

        This method retrieves a zero-copy view of the raw data from the memory-mapped
        ERG file and applies scaling factors. If get_all_signals() has been called,
        it uses the cached structured array for even faster access.

        Args:
            signal_name: Name of the signal (e.g., "Time", "Car.v")

        Returns:
            1D NumPy array with native dtype and scaling applied

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> velocity = erg.get_signal("Car.v")
            >>> print(f"Max velocity: {max(velocity):.2f}, dtype: {velocity.dtype}")
        """
        # If get_all_signals() has been called, use the cached structured array
        if self._struct_array_cache is not None:
            try:
                raw_data = self._struct_array_cache[signal_name]
            except ValueError as e:
                raise KeyError(f"Signal '{signal_name}' not found") from e
        else:
            # Get zero-copy strided view of raw unscaled data from C extension
            raw_data = self._erg.get_signal(signal_name)

        # Get scaling parameters
        factor = self.get_signal_factor(signal_name)
        offset = self.get_signal_offset(signal_name)

        # Apply scaling: scaled = raw * factor + offset
        if factor != 1.0 or offset != 0.0:
            scaled_data = raw_data * factor + offset
        else:
            scaled_data = raw_data

        return scaled_data

    def get_all_signals(self) -> NDArray[np.void]:
        """
        Get all signals as a structured NumPy array.

        Returns a structured array where each field is a signal with its native
        data type. This is a zero-copy memory-mapped view of the raw ERG file data
        with values in their native format (no scaling applied).

        The structured array is cached on first call. Subsequent calls return the
        cached array instantly.

        IMPORTANT: Direct field access (data['signal_name']) returns RAW unscaled
        values from the file. Use get_signal('signal_name') to get properly scaled
        values with factor and offset applied.

        Returns:
            Structured NumPy array with shape (sample_count,)

        Example:
            >>> erg = ERG("simulation.erg")
            >>> data = erg.get_all_signals()  # Creates zero-copy view, caches it
            >>> print(f"Shape: {data.shape}")  # (samples,)
            >>> print(f"Fields: {data.dtype.names}")  # ('Time', 'Car.v', ...)
            >>> time_raw = data['Time']  # Direct access: raw unscaled data
            >>> time_scaled = erg.get_signal('Time')  # Scaled data (recommended)
        """
        # Get and cache raw structured array from C (zero-copy view) if not already cached
        if self._struct_array_cache is None:
            self._struct_array_cache = self._erg.get_all_signals()
        return self._struct_array_cache

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

    # ========================================
    # Individual Metadata Methods
    # ========================================

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
        if signal_name not in self._units_cache:
            self._units_cache[signal_name] = self._erg.get_signal_unit(signal_name)
        return self._units_cache[signal_name]

    def get_signal_type(self, signal_name: str) -> np.dtype[Any]:
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
        if signal_name not in self._types_cache:
            self._types_cache[signal_name] = self._erg.get_signal_type(signal_name)
        return self._types_cache[signal_name]

    def get_signal_factor(self, signal_name: str) -> float:
        """
        Get the scaling factor for a signal.

        Args:
            signal_name: Name of the signal

        Returns:
            Scaling factor (float)

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> factor = erg.get_signal_factor("Car.v")
            >>> print(f"Velocity factor: {factor}")
        """
        if signal_name not in self._factors_cache:
            self._factors_cache[signal_name] = self._erg.get_signal_factor(signal_name)
        return self._factors_cache[signal_name]

    def get_signal_offset(self, signal_name: str) -> float:
        """
        Get the scaling offset for a signal.

        Args:
            signal_name: Name of the signal

        Returns:
            Scaling offset (float)

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> offset = erg.get_signal_offset("Car.v")
            >>> print(f"Velocity offset: {offset}")
        """
        if signal_name not in self._offsets_cache:
            self._offsets_cache[signal_name] = self._erg.get_signal_offset(signal_name)
        return self._offsets_cache[signal_name]

    def get_signal_index(self, signal_name: str) -> int:
        """
        Get the index of a signal by name.

        Args:
            signal_name: Name of the signal

        Returns:
            Signal index (integer)

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> index = erg.get_signal_index("Car.v")
            >>> print(f"Car.v is at index: {index}")
        """
        return self._erg.get_signal_index(signal_name)

    # ========================================
    # Batch Metadata Methods
    # ========================================

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
        self._units_cache = self._erg.get_signal_units()
        return self._units_cache

    def get_signal_types(self) -> dict[str, np.dtype[Any]]:
        """
        Get all signal types as a dictionary.

        Returns:
            Dictionary mapping signal names to numpy dtypes

        Example:
            >>> erg = ERG("simulation.erg")
            >>> types = erg.get_signal_types()
            >>> print(f"Time type: {types['Time']}")
            >>> print(f"Velocity type: {types['Car.v']}")
        """
        self._types_cache = self._erg.get_signal_types()
        return self._types_cache

    def get_signal_factors(self) -> dict[str, float]:
        """
        Get all signal scaling factors as a dictionary.

        Returns:
            Dictionary mapping signal names to scaling factors

        Example:
            >>> erg = ERG("simulation.erg")
            >>> factors = erg.get_signal_factors()
            >>> print(f"Time factor: {factors['Time']}")
            >>> print(f"Velocity factor: {factors['Car.v']}")
        """
        self._factors_cache = self._erg.get_signal_factors()
        return self._factors_cache

    def get_signal_offsets(self) -> dict[str, float]:
        """
        Get all signal scaling offsets as a dictionary.

        Returns:
            Dictionary mapping signal names to scaling offsets

        Example:
            >>> erg = ERG("simulation.erg")
            >>> offsets = erg.get_signal_offsets()
            >>> print(f"Time offset: {offsets['Time']}")
            >>> print(f"Velocity offset: {offsets['Car.v']}")
        """
        self._offsets_cache = self._erg.get_signal_offsets()
        return self._offsets_cache

    # ========================================
    # Special Methods
    # ========================================

    def __getitem__(self, signal_name: str) -> NDArray[Any]:
        """
        Get signal data using dictionary-style access.

        This is a simple wrapper around get_signal() that enables
        dictionary-style access: erg["signal_name"]

        Args:
            signal_name: Name of the signal

        Returns:
            NumPy array of signal data with scaling applied

        Raises:
            KeyError: If the signal name is not found

        Example:
            >>> erg = ERG("simulation.erg")
            >>> velocity = erg["Car.v"]
            >>> time = erg["Time"]
        """
        return self.get_signal(signal_name)

    def __contains__(self, signal_name: str) -> bool:
        """
        Check if a signal exists in the ERG file.

        Allows using the 'in' operator to check for signal existence.

        Args:
            signal_name: Name of the signal to check

        Returns:
            True if the signal exists, False otherwise

        Example:
            >>> erg = ERG("simulation.erg")
            >>> if "Car.v" in erg:
            ...     velocity = erg["Car.v"]
            >>> has_signal = "NonExistent" in erg  # Returns False
        """
        try:
            signal_names = self.get_signal_names()
            return signal_name in signal_names
        except (RuntimeError, AttributeError):
            # File not loaded or error getting signal names
            return False

    def __repr__(self) -> str:
        """String representation of ERG object."""
        try:
            signal_count = len(self.get_signal_names())
            return f"ERG(signals={signal_count})"
        except (RuntimeError, AttributeError) as e:
            # RuntimeError: File not parsed or failed to load
            # AttributeError: _erg attribute not initialized
            return f"ERG(error: {type(e).__name__})"
        except Exception as e:
            # Catch-all for unexpected errors
            return f"ERG(error: {type(e).__name__}: {str(e)})"
