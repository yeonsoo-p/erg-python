"""ERG (CarMaker binary results) file parser.

This module provides functionality to read and parse CarMaker ERG files,
which contain simulation results in binary format.
"""

from .cminfofile import ERGInfo
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
from functools import lru_cache

__all__ = ["ERG"]

# Mapping from CarMaker data types to NumPy dtypes
DATATYPE = {
    "Float": "f4",
    "Double": "f8",
    "LongLong": "i8",
    "ULongLong": "u8",
    "Int": "i4",
    "UInt": "u4",
    "Short": "i2",
    "UShort": "u2",
    "Char": "i1",
    "UChar": "u1",
    "1 Bytes": "S1",
    "2 Bytes": "S2",
    "3 Bytes": "S3",
    "4 Bytes": "S4",
    "5 Bytes": "S5",
    "6 Bytes": "S6",
    "7 Bytes": "S7",
    "8 Bytes": "S8",
}

# Size of ERG file header in bytes
HEADER_SIZE = 16


def _get_file_signature(file_path: Path) -> tuple[str, float, int]:
    """Generate file signature for cache invalidation."""
    try:
        stat = file_path.stat()
        return (str(file_path.resolve()), stat.st_mtime, stat.st_size)
    except (OSError, FileNotFoundError):
        return (str(file_path), 0.0, 0)


@lru_cache(maxsize=32)
def _parse_erg_data(erg_signature: tuple[str, float, int], info_signature: tuple[str, float, int]) -> tuple[OrderedDict, pd.DataFrame]:
    """Parse ERG file data with LRU caching."""
    erg_path = Path(erg_signature[0])
    info_path = Path(info_signature[0])
    info = ERGInfo(info_path)

    # Determine byte order
    endianness = "<" if info["File.ByteOrder"] == "LittleEndian" else ">"

    # Parse signal metadata from info file
    metadata = OrderedDict()
    index = 1
    while f"File.At.{index}.Name" in info:
        name = info[f"File.At.{index}.Name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        # Get data type and convert to numpy dtype
        uaq_dtype = info[f"File.At.{index}.Type"]
        if isinstance(uaq_dtype, list):
            uaq_dtype = " ".join(map(str, uaq_dtype))
        uaq_dtype = DATATYPE[uaq_dtype]

        # Extract scaling information
        unit = info.get(f"Quantity.{name}.Unit", "")
        factor = np.float64(info.get(f"Quantity.{name}.Factor", 1.0))
        offset = np.float64(info.get(f"Quantity.{name}.Offset", 0.0))

        metadata[name] = {
            "dtype": uaq_dtype,
            "unit": unit,
            "factor": factor.astype(uaq_dtype),
            "offset": offset.astype(uaq_dtype),
        }
        index += 1

    # Create structured numpy dtype for binary data
    full_dtype = np.dtype([(n, endianness + m["dtype"]) for n, m in metadata.items()])

    # Read and parse binary data
    with open(erg_path, "rb") as erg_file:
        raw_bytes = erg_file.read()[HEADER_SIZE:]
    data = np.frombuffer(raw_bytes, dtype=full_dtype)

    # Apply scaling and create DataFrame
    scaled_data = {}
    for key, field_info in data.dtype.fields.items():
        if field_info[0].kind in {"b", "f", "c", "m", "M", "i", "u"}:
            scaled_data[key] = data[key] * metadata[key]["factor"] + metadata[key]["offset"]

    # Resample to 1ms timebase (same as BLF)
    dt = 0.01
    r = round(dt / 0.001)

    resampled_data = {}
    for signal_name, signal_values in scaled_data.items():
        if signal_name == "Time":
            total_length = len(signal_values) * r
            resampled_data[signal_name] = np.arange(total_length) * 0.001
        else:
            resampled_data[signal_name] = np.repeat(signal_values, r)

    dataframe = pd.DataFrame(resampled_data)
    return metadata, dataframe


@dataclass
class ERG:
    """CarMaker ERG (binary results) file parser.

    Reads CarMaker simulation results from binary ERG files and provides
    pandas DataFrame interface for data analysis.

    Attributes:
        abs_path: Absolute path to the ERG file
        info: ERGInfo object containing metadata from .erg.info file
        metadata: Ordered dictionary of signal metadata (units, factors, etc.)
        dataframe: pandas DataFrame containing all simulation data
        endianness: Byte order for binary data ("<" for little-endian, ">" for big-endian)
        dt: Simulation time step in seconds
        use_cache: Whether to use disk caching for parsed data

    Example:
        >>> erg = ERG("simulation.erg")
        >>> time_series = erg["Time"]
        >>> velocity = erg["Car.v"]
        >>> erg.plot("Time", "Car.v")
    """

    abs_path: Path
    info: ERGInfo = field(default=None, init=False)
    metadata: OrderedDict = field(default_factory=OrderedDict)
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    endianness: str = field(default="<", init=False)
    dt: float = field(default=0.01, init=False)

    def __post_init__(self):
        self.abs_path = Path(self.abs_path).resolve()
        self.info = ERGInfo(self.abs_path.with_suffix(".erg.info"))
        self.metadata = OrderedDict()

        # Determine byte order
        self.endianness = "<" if self.info["File.ByteOrder"] == "LittleEndian" else ">"
        self.dt = self.info["SimParam.DeltaT"]

        self.read()
    def read(self) -> None:
        """Read and parse the binary ERG file.

        Extracts signal metadata from the info file and reads binary data,
        applying scaling factors and offsets to create a pandas DataFrame.
        Uses in-memory LRU caching to avoid re-parsing identical files.
        """
        info_path = self.abs_path.with_suffix(".erg.info")

        # Generate file signatures for cache key
        erg_signature = _get_file_signature(self.abs_path)
        info_signature = _get_file_signature(info_path)

        # Use cached parsing function
        self.metadata, self.dataframe = _parse_erg_data(erg_signature, info_signature)

    def __getitem__(self, key: str) -> pd.Series:
        """Get a signal by name.

        Args:
            key: Signal name (e.g., "Time", "Car.v")

        Returns:
            pandas Series containing the signal data
        """
        return self.dataframe[key]

    def plot(self, *args, **kwargs) -> None:
        """Plot signals using pandas DataFrame.plot().

        Args:
            *args: Positional arguments passed to DataFrame.plot()
            **kwargs: Keyword arguments passed to DataFrame.plot()
        """
        self.dataframe.plot(*args, **kwargs)

    def get_signal_info(self, signal_name: str) -> dict:
        """Get metadata for a specific signal.

        Args:
            signal_name: Name of the signal

        Returns:
            Dictionary containing dtype, unit, factor, and offset
        """
        return self.metadata.get(signal_name, {})

    def get_units(self) -> dict[str, str]:
        """Get units for all signals.

        Returns:
            Dictionary mapping signal names to their units
        """
        return {name: info.get("unit", "") for name, info in self.metadata.items()}

    def get_unit(self, signal_name: str) -> str:
        """Get unit for a specific signal.

        Args:
            signal_name: Name of the signal

        Returns:
            Unit string for the signal, empty string if not found
        """
        return self.metadata.get(signal_name, {}).get("unit", "")

    def list_signals(self) -> list[str]:
        """Get list of available signal names.

        Returns:
            List of signal names in the ERG file
        """
        return list(self.dataframe.columns)

    def to_dataframe_with_units(self) -> pd.DataFrame:
        """Create a DataFrame with units stored in column attributes.

        Returns:
            DataFrame with units accessible via df.attrs['units']
        """
        df = self.dataframe.copy()
        df.attrs["units"] = self.get_units()
        return df

    def __repr__(self) -> str:
        return f"ERG({self.abs_path.name})"

    def __str__(self) -> str:
        return self.abs_path.name
