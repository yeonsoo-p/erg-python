# erg-python

Fast Python reader for CarMaker ERG (binary simulation result) files with zero-copy memory-mapped I/O.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Zero-copy memory-mapped I/O** - Access signal data directly from disk without loading entire file into memory
- **High performance** - 15-19x faster loading than cmerg, sub-microsecond signal access
- **Low memory footprint** - ~2 MB memory overhead vs ~65 MB for cmerg
- **Native data types** - Signals retain their original types (float32, float64, int32, etc.)
- **Automatic scaling** - Factor and offset scaling applied automatically
- **Dictionary-style access** - Simple `erg["signal_name"]` syntax
- **Pandas integration** - Easy conversion to DataFrames for analysis
- **Instance caching** - Automatic caching of ERG objects by filepath

## Installation

### Prerequisites

- Python 3.8+
- CMake 3.15+
- C++17 compatible compiler (MSVC on Windows, GCC/Clang on Linux/Mac)
- NumPy

### Build from source

```bash
# Clone the repository
git clone https://github.com/yourusername/erg-python.git
cd erg-python

# Build the C extension
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Install (optional)
pip install -e .
```

## Quick Start

```python
from erg_python import ERG
import matplotlib.pyplot as plt

# Open ERG file (automatically cached)
erg = ERG("simulation.erg")

# Get signal data with scaling applied
time = erg.get_signal("Time")
velocity = erg["Car.v"]  # Dictionary-style access
acceleration = erg["Car.ax"]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, velocity, label="Velocity")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.grid(True)
plt.show()
```

## Usage Examples

### Basic Signal Access

```python
from erg_python import ERG

# Load ERG file
erg = ERG("simulation.erg")

# Get list of all available signals
signal_names = erg.get_signal_names()
print(f"Found {len(signal_names)} signals")

# Access individual signals
time = erg.get_signal("Time")
velocity = erg["Car.v"]  # Alternative syntax

# Get signal metadata
unit = erg.get_signal_unit("Car.v")
dtype = erg.get_signal_type("Car.v")
factor = erg.get_signal_factor("Car.v")
offset = erg.get_signal_offset("Car.v")

print(f"Velocity: {len(velocity)} samples, unit: {unit}, dtype: {dtype}")
```

### Working with Multiple Signals

```python
# Get multiple signals
signals_to_analyze = ["Time", "Car.v", "Car.ax", "Car.Distance"]

data = {}
for signal_name in signals_to_analyze:
    data[signal_name] = erg[signal_name]

# Or use get_all_signals() for bulk access
all_signals = erg.get_all_signals()  # Returns structured NumPy array
time = all_signals["Time"]
velocity = all_signals["Car.v"]
```

### Pandas Integration

```python
import pandas as pd

# Convert single signal to Series
velocity_series = pd.Series(erg["Car.v"], name="Velocity")

# Create DataFrame from multiple signals
signals = ["Time", "Car.v", "Car.ax", "Car.Distance"]
data = {name: erg[name] for name in signals}
df = pd.DataFrame(data)

# Or convert all signals at once
all_signals = erg.get_all_signals()
df_all = pd.DataFrame(all_signals)

# Time-indexed DataFrame
df_indexed = df.set_index("Time")

# Filter and analyze
car_signals = df_all[[col for col in df_all.columns if col.startswith("Car.")]]
time_slice = df_indexed.loc[10.0:20.0]  # Data between 10-20 seconds
```

### Advanced: Batch Processing

```python
from pathlib import Path
import pandas as pd

# Process multiple ERG files
erg_files = Path("results/").glob("*.erg")

results = []
for erg_file in erg_files:
    erg = ERG(erg_file)

    # Extract key metrics
    velocity = erg["Car.v"]
    max_velocity = velocity.max()
    avg_velocity = velocity.mean()

    results.append({
        "file": erg_file.name,
        "max_velocity": max_velocity,
        "avg_velocity": avg_velocity,
    })

# Summary DataFrame
summary = pd.DataFrame(results)
print(summary)
```

### Advanced: Get All Signals at Once

```python
# Get all signals as a structured array (zero-copy, cached)
all_signals = erg.get_all_signals()

# Access is ultra-fast (uses cached structured array)
print(f"Shape: {all_signals.shape}")  # (6262,)
print(f"Fields: {all_signals.dtype.names[:10]}")  # First 10 signal names

# IMPORTANT: Direct field access returns RAW unscaled data
time_raw = all_signals["Time"]  # Raw data without scaling

# For scaled data, use get_signal() - it will use the cache automatically
time_scaled = erg.get_signal("Time")  # Scaled data (factor & offset applied)

# After calling get_all_signals(), get_signal() uses the cached array
# This makes subsequent get_signal() calls even faster!
```

## API Reference

### ERG Class

#### Constructor

```python
ERG(filepath: str | Path)
```

Opens and parses an ERG file. Files are automatically cached - creating multiple ERG objects with the same filepath returns the same cached instance.

**Parameters:**
- `filepath`: Path to the .erg file (string or Path object)

**Raises:**
- `FileNotFoundError`: If the ERG file or its .info file is not found
- `RuntimeError`: If the file cannot be parsed

#### Methods

##### `get_signal(signal_name: str) -> np.ndarray`

Get signal data by name with scaling applied.

Returns a zero-copy memory-mapped view of the signal data with factor and offset scaling applied. If `get_all_signals()` has been called, uses the cached structured array for faster access.

**Parameters:**
- `signal_name`: Name of the signal (e.g., "Time", "Car.v")

**Returns:**
- 1D NumPy array with native dtype and scaling applied

**Raises:**
- `KeyError`: If the signal name is not found

##### `get_all_signals() -> np.ndarray`

Get all signals as a structured NumPy array.

Returns a zero-copy memory-mapped view of all signal data as a structured array. The array is cached on first call. **Important:** Direct field access returns RAW unscaled data. Use `get_signal()` for scaled data.

**Returns:**
- Structured NumPy array with shape (sample_count,)

##### `get_signal_names() -> list[str]`

Get list of all signal names in the file.

##### `get_signal_unit(signal_name: str) -> str`

Get the unit string for a signal (e.g., "m/s", "s", "rad").

##### `get_signal_type(signal_name: str) -> np.dtype`

Get the NumPy dtype for a signal.

##### `get_signal_factor(signal_name: str) -> float`

Get the scaling factor for a signal. Scaled value = raw * factor + offset.

##### `get_signal_offset(signal_name: str) -> float`

Get the scaling offset for a signal. Scaled value = raw * factor + offset.

##### `get_signal_index(signal_name: str) -> int`

Get the index of a signal by name.

##### Batch Metadata Methods

```python
get_signal_units() -> dict[str, str]      # All signal units
get_signal_types() -> dict[str, np.dtype]  # All signal types
get_signal_factors() -> dict[str, float]   # All scaling factors
get_signal_offsets() -> dict[str, float]   # All scaling offsets
```

#### Dictionary-style Access

```python
erg["signal_name"]  # Equivalent to erg.get_signal("signal_name")
```

## Performance

Benchmark on a 63 MB ERG file with 2,596 signals and 6,262 samples:

| Operation | erg-python | cmerg | Speedup |
|-----------|------------|-------|---------|
| Loading | 3.1 ms | 58.3 ms | **19x faster** |
| Cold read (2 signals) | 0.007 ms | 0.027 ms | **3.8x faster** |
| Hot read (2 signals) | 0.002 ms | 0.005 ms | **2.6x faster** |
| Memory footprint | ~2 MB | ~65 MB | **32x less** |

## Architecture

### Zero-Copy Design

erg-python uses memory-mapped I/O to access ERG files without copying data:

1. **File Parsing**: The .erg file and its .erg.info metadata file are parsed on initialization
2. **Memory Mapping**: The data section is memory-mapped using OS-level mmap
3. **Strided Views**: Signals are accessed via NumPy strided arrays that point directly to the mapped memory
4. **Lazy Loading**: Signal data is only accessed when requested, not at initialization
5. **Caching**: The structured array from `get_all_signals()` is cached for repeated access

### Scaling

ERG files store signals in their native data types with optional scaling factors:

```
scaled_value = raw_value * factor + offset
```

- `get_signal()` and `erg["name"]` return **scaled data** (ready to use)
- `get_all_signals()["name"]` returns **raw unscaled data** (direct memory view)
- Use `get_signal()` after calling `get_all_signals()` to get scaled data with cache benefits

### Instance Caching

ERG objects are automatically cached by filepath:

```python
erg1 = ERG("test.erg")
erg2 = ERG("test.erg")
assert erg1 is erg2  # Same instance!

# Clear cache if needed
ERG._instances.clear()
```

All caches (including the structured array cache from `get_all_signals()`) persist across instantiations.

## Limitations

- **Raw byte types not supported**: Signals with types like `ERG_1BYTE`, `ERG_2BYTES`, etc. are automatically excluded from the API
- **Big-endian files not supported**: Only little-endian ERG files are currently supported
- **Read-only**: This library is for reading ERG files, not writing them
- **CarMaker format only**: Only supports the CarMaker ERG binary format

## Comparison with cmerg

| Feature | erg-python | cmerg |
|---------|------------|-------|
| Load time | ~3 ms | ~58 ms |
| Memory usage | ~2 MB | ~65 MB |
| Data access | Zero-copy views | Copies data |
| Native types | Yes | No (all float64) |
| Pandas support | Manual conversion | Built-in |
| Installation | Requires compilation | Pure Python (pip install) |

**When to use erg-python:**
- Processing large ERG files
- Memory-constrained environments
- High-performance applications
- Batch processing many files
- Need to preserve native data types

**When to use cmerg:**
- Quick prototyping
- Don't want to compile C extensions
- Prefer built-in pandas integration
- Working with small files where performance doesn't matter

## Troubleshooting

### Build Errors

**CMake can't find Python:**
```bash
cmake -B build -DPython_EXECUTABLE=/path/to/python
```

**NumPy headers not found:**
```bash
pip install numpy
cmake -B build -DPython_EXECUTABLE=$(which python)
```

### Runtime Errors

**`FileNotFoundError: .erg.info file not found`**
- ERG files require a companion `.erg.info` metadata file
- Ensure both files are in the same directory

**`RuntimeError: ERG file not loaded`**
- The ERG file failed to parse during initialization
- Check that the file is a valid CarMaker ERG file

**`KeyError: Signal 'name' not found`**
- The signal doesn't exist in the file
- Use `get_signal_names()` to see available signals

**`ValueError: Signal has unsupported raw byte type`**
- The signal uses a raw byte type (ERG_1BYTE through ERG_8BYTES)
- These are automatically excluded from the API

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built for CarMaker simulation result analysis
- Uses memory-mapped I/O for efficient data access
- Powered by NumPy for array operations
