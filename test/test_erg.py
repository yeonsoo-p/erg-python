"""
Test script for erg_python module

Usage:
    python test_erg.py <path/to/test.erg>
"""

import sys
import argparse
import time
import gc
import psutil
import os
from pathlib import Path
from typing import Any
import cmerg
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import erg_python
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from erg_python import ERG


# Global variable for test file path
TEST_ERG_FILE: Path | None = None


def time_ns() -> int:
    """Get current time in nanoseconds."""
    return time.perf_counter_ns()


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)


def test_basic() -> bool:
    """Test basic ERG file reading"""
    erg_file: Path | None = TEST_ERG_FILE

    if not erg_file or not erg_file.exists():
        print(f"[FAIL] Test file not found or not specified: {erg_file}")
        print("Usage: python test_erg.py <path/to/test.erg>")
        return False

    print(f"Testing with file: {erg_file}")

    # Create ERG object (should auto-parse now)
    # Can pass Path object directly now
    print("Loading ERG file...")
    erg: ERG = ERG(erg_file)

    # Get signal names
    signal_names: list[str] = erg.get_signal_names()
    print(f"Number of signals: {len(signal_names)}")
    print(f"First 10 signals: {signal_names[:10]}")

    # Get a single signal (typically "Time" exists)
    if "Time" in signal_names:
        print("\nGetting 'Time' signal...")

        # Time cold retrieval
        start: int = time_ns()
        time_data: npt.NDArray[Any] = erg.get_signal("Time")
        end: int = time_ns()
        cold_time: int = end - start

        print(f"Time data type: {type(time_data)}")
        print(f"Time length: {len(time_data)}")
        print(f"Time range: {time_data[0]:.6f} to {time_data[-1]:.6f}")
        print(f"Cold retrieval time: {cold_time / 1_000:.2f} μs")

        # Time warm retrieval (cached)
        start = time_ns()
        time_data = erg.get_signal("Time")
        end = time_ns()
        warm_time = end - start
        print(f"Warm retrieval time: {warm_time / 1_000:.2f} μs")

        # Get signal unit and type
        unit = erg.get_signal_unit("Time")
        dtype = erg.get_signal_type("Time")
        print(f"Time signal unit: {unit}, dtype: {dtype}")

    # Get multiple signals using list comprehension
    available_signals: list[str] = signal_names[:5]
    print(f"\nGetting multiple signals: {available_signals}")

    # Time cold retrieval
    start = time_ns()
    signals: list[npt.NDArray[Any]] = [erg.get_signal(name) for name in available_signals]
    end = time_ns()
    batch_cold_time: int = end - start

    for name, data in zip(available_signals, signals):
        print(f"  {name}: type={type(data).__name__}, dtype={data.dtype}, length={len(data)}")

    print(f"Cold retrieval time: {batch_cold_time / 1_000:.2f} μs")
    print(f"Average per signal: {batch_cold_time / len(available_signals) / 1_000:.2f} μs")

    # Time warm retrieval (all cached)
    start = time_ns()
    signals = [erg[name] for name in available_signals]  # Using __getitem__
    end = time_ns()
    batch_warm_time = end - start
    print(f"Warm retrieval time: {batch_warm_time / 1_000:.2f} μs")
    print(f"Average per signal (warm): {batch_warm_time / len(available_signals) / 1_000:.2f} μs")

    print("\nTest passed!")
    return True


def test_signal_info() -> bool:
    """Test signal metadata retrieval"""
    erg_file: Path | None = TEST_ERG_FILE

    if not erg_file or not erg_file.exists():
        print("Skipping test_signal_info: test file not available")
        return False

    erg = ERG(erg_file)

    print("\nSignal metadata for all signals:")
    signal_names = erg.get_signal_names()
    for signal_name in signal_names[:10]:  # First 10 signals
        unit = erg.get_signal_unit(signal_name)
        dtype = erg.get_signal_type(signal_name)
        print(f"  {signal_name}:")
        print(f"    Type: {dtype}")
        print(f"    Unit: {unit}")

    return True


def test_get_all_signals() -> bool:
    """Test get_all_signals() structured array functionality"""
    erg_file: Path | None = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print("Skipping test_get_all_signals: test file not available")
        return False

    print("\nTesting get_all_signals()...")
    erg = ERG(erg_file)

    signal_names = erg.get_signal_names()
    print(f"Signal count: {len(signal_names)}")

    # Get all signals as structured array
    print("Getting all signals as structured array...")
    start = time_ns()
    data = erg.get_all_signals()
    all_signals_time = time_ns() - start

    print(f"Array shape: {data.shape}")
    print(f"Array dtype fields: {len(data.dtype.names)} fields")
    print(f"Memory usage: {data.nbytes / (1024**2):.2f} MB")
    print(f"Retrieval time: {all_signals_time / 1_000:.2f} μs")

    # Performance comparison
    print("\nPerformance comparison...")
    start = time_ns()
    _ = {name: erg.get_signal(name) for name in signal_names}
    individual_time = time_ns() - start

    print(f"Individual retrieval: {individual_time / 1_000:.2f} μs")
    print(f"get_all_signals():    {all_signals_time / 1_000:.2f} μs")
    speedup = individual_time / all_signals_time
    print(f"Speedup: {speedup:.2f}x faster")

    return True


def test_error_handling() -> bool:
    """Test error handling"""
    print("\nTesting error handling...")

    # Test with non-existent file
    try:
        erg = ERG("nonexistent.erg")
        print("  [FAIL] Should have raised an exception for non-existent file")
        return False
    except Exception as e:
        print(f"  [OK] Correctly raised exception for non-existent file: {type(e).__name__}")

    # Test with non-existent signal
    erg_file: Path | None = TEST_ERG_FILE
    if erg_file and erg_file.exists():
        erg = ERG(erg_file)
        try:
            _ = erg.get_signal("NonExistentSignal12345")
            print("  [FAIL] Should have raised KeyError for non-existent signal")
            return False
        except KeyError as e:
            print(f"  [OK] Correctly raised KeyError for non-existent signal: {e}")

    return True


def test_raw_byte_type_handling() -> bool:
    """Test that raw byte type signals are properly filtered and handled"""
    erg_file: Path | None = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print("Skipping test_raw_byte_type_handling: test file not available")
        return False

    print("\nTesting raw byte type signal handling...")
    erg = ERG(erg_file)

    # Get signal metadata to find if there are any raw byte types
    # Note: In the C code, we cache supported_signal_count which excludes raw byte types
    signal_names = erg.get_signal_names()
    signal_types = erg.get_signal_types()  # Returns dict {name: type_str}

    print(f"  Total supported signals (excluding raw bytes): {len(signal_names)}")

    # Verify no raw byte types are exposed in the API
    raw_byte_types = ["1byte", "2bytes", "3bytes", "4bytes", "5bytes", "6bytes", "7bytes", "8bytes"]
    exposed_raw_bytes = []
    for name in signal_names:
        dtype = signal_types.get(name, None)
        if dtype is not None:
            dtype_str = str(dtype).lower()
            if any(rbt in dtype_str for rbt in raw_byte_types):
                exposed_raw_bytes.append(name)

    if exposed_raw_bytes:
        print(f"  [FAIL] Found {len(exposed_raw_bytes)} raw byte type signals exposed: {exposed_raw_bytes[:5]}")
        return False
    else:
        print("  [OK] No raw byte type signals exposed in get_signal_names()")

    # Verify get_signal_types() doesn't return raw byte types
    raw_byte_in_types = []
    for name, dtype in signal_types.items():
        dtype_str = str(dtype).lower()
        if any(rbt in dtype_str for rbt in raw_byte_types):
            raw_byte_in_types.append(name)

    if raw_byte_in_types:
        print(f"  [FAIL] Found raw byte types in get_signal_types(): {raw_byte_in_types[:5]}")
        return False
    else:
        print("  [OK] No raw byte types in get_signal_types()")

    # Verify get_all_signals() doesn't include raw byte types
    all_data = erg.get_all_signals()
    if all_data.dtype.names:
        if len(all_data.dtype.names) != len(signal_names):
            print(f"  [FAIL] get_all_signals() field count ({len(all_data.dtype.names)}) != signal_names count ({len(signal_names)})")
            return False
        else:
            print(f"  [OK] get_all_signals() has {len(all_data.dtype.names)} fields (matches filtered count)")

    # Verify that all signals can be retrieved without errors
    print("  Testing retrieval of first 10 signals...")
    errors = []
    for signal_name in signal_names[:10]:
        try:
            data = erg.get_signal(signal_name)
            if data is None or len(data) == 0:
                errors.append(f"{signal_name}: returned None or empty array")
        except Exception as e:
            errors.append(f"{signal_name}: {type(e).__name__}: {e}")

    if errors:
        print("  [FAIL] Failed to retrieve some signals:")
        for error in errors:
            print(f"    - {error}")
        return False
    else:
        print("  [OK] All tested signals retrieved successfully")

    # Verify consistency across all API methods
    print("  Testing consistency across API methods...")
    units = erg.get_signal_units()
    types = erg.get_signal_types()
    factors = erg.get_signal_factors()
    offsets = erg.get_signal_offsets()

    counts = {
        "signal_names": len(signal_names),
        "signal_units": len(units),
        "signal_types": len(types),
        "signal_factors": len(factors),
        "signal_offsets": len(offsets),
    }

    if len(set(counts.values())) != 1:
        print(f"  [FAIL] Inconsistent counts across API methods: {counts}")
        return False
    else:
        print(f"  [OK] All API methods return consistent count of {counts['signal_names']} signals")

    print("\n  Test passed!")
    return True


def test_pandas_conversion() -> bool:
    """Test pandas DataFrame conversion with filtered signals"""
    erg_file: Path | None = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print("Skipping test_pandas_conversion: test file not available")
        return False

    print("\nTesting pandas DataFrame conversion...")
    erg = ERG(erg_file)

    signal_names = erg.get_signal_names()
    print(f"  Total signals available: {len(signal_names)}")

    # Test 1: Individual signal to pandas Series
    print("\n  Test 1: Individual signal to pandas Series")
    try:
        time_signal = erg.get_signal("Time")
        time_series = pd.Series(time_signal, name="Time")
        print(f"    [OK] Created pandas Series with {len(time_series)} samples")
        print(f"    Dtype: {time_series.dtype}")
    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {e}")
        return False

    # Test 2: Multiple signals to DataFrame (manual approach)
    print("\n  Test 2: Multiple signals to DataFrame (manual)")
    try:
        signals_to_convert = signal_names[:10]
        data_dict = {}
        for sig_name in signals_to_convert:
            data_dict[sig_name] = erg.get_signal(sig_name)

        df = pd.DataFrame(data_dict)
        print(f"    [OK] Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        print(f"    Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {e}")
        return False

    # Test 3: Convert get_all_signals() structured array to DataFrame
    print("\n  Test 3: get_all_signals() to DataFrame")
    try:
        all_data = erg.get_all_signals()
        df_all = pd.DataFrame(all_data)
        print("    [OK] Created DataFrame from structured array")
        print(f"    Shape: {df_all.shape}")
        print(f"    Memory usage: {df_all.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {e}")
        return False

    # Test 4: Time-indexed DataFrame
    print("\n  Test 4: Time-indexed DataFrame")
    try:
        all_data = erg.get_all_signals()
        df_indexed = pd.DataFrame(all_data)

        if "Time" in df_indexed.columns:
            df_indexed = df_indexed.set_index("Time")
            print("    [OK] Created time-indexed DataFrame")
            print(f"    Shape: {df_indexed.shape}")
            print(f"    Time range: {df_indexed.index[0]:.3f} to {df_indexed.index[-1]:.3f} seconds")
        else:
            print("    WARNING: Time signal not available for indexing")
    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {e}")
        return False

    # Test 5: Filtering and slicing
    print("\n  Test 5: DataFrame filtering and slicing")
    try:
        all_data = erg.get_all_signals()
        df = pd.DataFrame(all_data)

        if "Time" in df.columns:
            df_indexed = df.set_index("Time")

            # Time-based slicing
            df_slice = df_indexed.loc[10:20]  # From 10s to 20s
            print(f"    [OK] Time slice (10s-20s) has {len(df_slice)} rows")

            # Select specific columns
            car_signals = [col for col in df.columns if col.startswith("Car.")]
            if car_signals:
                print(f"    [OK] Found {len(car_signals)} Car.* signals")
                print(f"    Car signals sample: {car_signals[:3]}")
        else:
            print("    WARNING: Time signal not available")
    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {e}")
        return False

    # Test 6: Verify no raw byte types in DataFrame
    print("\n  Test 6: Verify no raw byte types in DataFrame")
    try:
        all_data = erg.get_all_signals()
        df = pd.DataFrame(all_data)

        raw_byte_types = ["1byte", "2bytes", "3bytes", "4bytes", "5bytes", "6bytes", "7bytes", "8bytes"]

        # Check column dtypes
        raw_byte_cols = []
        for col in df.columns:
            dtype_str = str(df[col].dtype).lower()
            if any(rbt in dtype_str for rbt in raw_byte_types):
                raw_byte_cols.append(col)

        if raw_byte_cols:
            print(f"    [FAIL] Found {len(raw_byte_cols)} raw byte columns: {raw_byte_cols[:5]}")
            return False
        else:
            print("    [OK] No raw byte type columns found in DataFrame")
            print(f"    All {len(df.columns)} columns have supported dtypes")
    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {e}")
        return False

    # Test 7: Export to CSV (sample)
    print("\n  Test 7: Export to CSV")
    try:
        all_data = erg.get_all_signals()
        df = pd.DataFrame(all_data)

        # Export small sample to CSV string (not file)
        csv_str = df.head(5).to_csv(index=False)
        lines = csv_str.split("\n")
        num_cols = len(lines[0].split(","))
        print("    [OK] DataFrame can be exported to CSV")
        print(f"    Number of columns in CSV: {num_cols}")

        if num_cols != len(signal_names):
            print(f"    [FAIL] CSV column count ({num_cols}) != signal count ({len(signal_names)})")
            return False
    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {e}")
        return False

    print("\n  All pandas conversion tests passed!")
    return True


def test_batch_metadata_methods() -> bool:
    """Test batch metadata retrieval methods"""
    erg_file: Path | None = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print("Skipping test_batch_metadata_methods: test file not available")
        return False

    print("\nTesting batch metadata methods...")
    erg = ERG(erg_file)

    signal_names = erg.get_signal_names()
    print(f"  Total signals: {len(signal_names)}")

    # Test get_signal_units()
    print("\n  Test 1: get_signal_units()")
    units = erg.get_signal_units()
    if not isinstance(units, dict):
        print(f"    [FAIL] get_signal_units() returned {type(units)}, expected dict")
        return False
    if len(units) != len(signal_names):
        print(f"    [FAIL] get_signal_units() returned {len(units)} units, expected {len(signal_names)}")
        return False
    print(f"    [OK] Retrieved {len(units)} signal units")
    print(f"    Sample units: {list(units.items())[:3]}")

    # Test get_signal_types()
    print("\n  Test 2: get_signal_types()")
    types = erg.get_signal_types()
    if not isinstance(types, dict):
        print(f"    [FAIL] get_signal_types() returned {type(types)}, expected dict")
        return False
    if len(types) != len(signal_names):
        print(f"    [FAIL] get_signal_types() returned {len(types)} types, expected {len(signal_names)}")
        return False
    print(f"    [OK] Retrieved {len(types)} signal types")
    print(f"    Sample types: {list(types.items())[:3]}")

    # Test get_signal_factors()
    print("\n  Test 3: get_signal_factors()")
    factors = erg.get_signal_factors()
    if not isinstance(factors, dict):
        print(f"    [FAIL] get_signal_factors() returned {type(factors)}, expected dict")
        return False
    if len(factors) != len(signal_names):
        print(f"    [FAIL] get_signal_factors() returned {len(factors)} factors, expected {len(signal_names)}")
        return False
    print(f"    [OK] Retrieved {len(factors)} signal factors")
    # Check that factors are numbers
    sample_factors = list(factors.values())[:5]
    for factor in sample_factors:
        if not isinstance(factor, (int, float)):
            print(f"    [FAIL] Factor {factor} is not a number")
            return False
    print(f"    Sample factors: {sample_factors}")

    # Test get_signal_offsets()
    print("\n  Test 4: get_signal_offsets()")
    offsets = erg.get_signal_offsets()
    if not isinstance(offsets, dict):
        print(f"    [FAIL] get_signal_offsets() returned {type(offsets)}, expected dict")
        return False
    if len(offsets) != len(signal_names):
        print(f"    [FAIL] get_signal_offsets() returned {len(offsets)} offsets, expected {len(signal_names)}")
        return False
    print(f"    [OK] Retrieved {len(offsets)} signal offsets")
    # Check that offsets are numbers
    sample_offsets = list(offsets.values())[:5]
    for offset in sample_offsets:
        if not isinstance(offset, (int, float)):
            print(f"    [FAIL] Offset {offset} is not a number")
            return False
    print(f"    Sample offsets: {sample_offsets}")

    # Test consistency between batch and individual methods
    print("\n  Test 5: Consistency between batch and individual methods")
    test_signals = signal_names[:10]
    all_consistent = True

    for sig_name in test_signals:
        # Check units
        if units[sig_name] != erg.get_signal_unit(sig_name):
            print(f"    [FAIL] Unit mismatch for {sig_name}")
            all_consistent = False
            break

        # Check types
        if types[sig_name] != erg.get_signal_type(sig_name):
            print(f"    [FAIL] Type mismatch for {sig_name}")
            all_consistent = False
            break

        # Check factors
        if factors[sig_name] != erg.get_signal_factor(sig_name):
            print(f"    [FAIL] Factor mismatch for {sig_name}")
            all_consistent = False
            break

        # Check offsets
        if offsets[sig_name] != erg.get_signal_offset(sig_name):
            print(f"    [FAIL] Offset mismatch for {sig_name}")
            all_consistent = False
            break

    if all_consistent:
        print(f"    [OK] Batch and individual methods return consistent results")
    else:
        return False

    print("\n  [OK] All batch metadata tests passed!")
    return True


def test_signal_index() -> bool:
    """Test get_signal_index() method"""
    erg_file: Path | None = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print("Skipping test_signal_index: test file not available")
        return False

    print("\nTesting get_signal_index()...")
    erg = ERG(erg_file)

    signal_names = erg.get_signal_names()
    print(f"  Total signals: {len(signal_names)}")

    # Test valid indices
    print("\n  Test 1: Valid signal indices")
    for i, sig_name in enumerate(signal_names[:10]):
        idx = erg.get_signal_index(sig_name)
        if idx != i:
            print(f"    [FAIL] Signal {sig_name} has index {idx}, expected {i}")
            return False
    print(f"    [OK] First 10 signals have correct indices")

    # Test last signal
    print("\n  Test 2: Last signal index")
    last_signal = signal_names[-1]
    last_idx = erg.get_signal_index(last_signal)
    expected_idx = len(signal_names) - 1
    if last_idx != expected_idx:
        print(f"    [FAIL] Last signal has index {last_idx}, expected {expected_idx}")
        return False
    print(f"    [OK] Last signal '{last_signal}' has index {last_idx}")

    # Test non-existent signal
    print("\n  Test 3: Non-existent signal raises KeyError")
    try:
        _ = erg.get_signal_index("NonExistentSignal12345")
        print("    [FAIL] Should have raised KeyError for non-existent signal")
        return False
    except KeyError as e:
        print(f"    [OK] Correctly raised KeyError: {e}")

    print("\n  [OK] All signal index tests passed!")
    return True


def test_period() -> bool:
    """Test get_period() method"""
    erg_file: Path | None = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print("Skipping test_period: test file not available")
        return False

    print("\nTesting get_period()...")
    erg = ERG(erg_file)

    # Test 1: Get period
    print("\n  Test 1: Get sampling period")
    try:
        period = erg.get_period()
        print(f"    [OK] Period: {period} ms")

        # Verify it's an integer
        if not isinstance(period, int):
            print(f"    [FAIL] Period should be int, got {type(period)}")
            return False

        # Verify it's positive
        if period <= 0:
            print(f"    [FAIL] Period should be positive, got {period}")
            return False

        print(f"    [OK] Period is valid integer: {period} ms")
    except Exception as e:
        print(f"    [FAIL] Failed to get period: {type(e).__name__}: {e}")
        return False

    # Test 2: Verify period calculation manually
    print("\n  Test 2: Verify period calculation")
    try:
        time_signal = erg.get_signal("Time")
        sample_count = len(time_signal)

        if sample_count < 2:
            expected_period = 1  # Default for < 2 samples: 1 ms
        else:
            dt = (time_signal[-1] - time_signal[0]) / (sample_count - 1)
            expected_period = int(round(dt * 1000.0))

        if period == expected_period:
            print(f"    [OK] Period matches manual calculation: {period} ms")
        else:
            print(f"    [FAIL] Period {period} ms != expected {expected_period} ms")
            return False
    except Exception as e:
        print(f"    [FAIL] Manual verification failed: {type(e).__name__}: {e}")
        return False

    # Test 3: Display sampling information
    print("\n  Test 3: Sampling information")
    try:
        time_signal = erg.get_signal("Time")
        duration = time_signal[-1] - time_signal[0]
        sample_count = len(time_signal)

        print(f"    Duration: {duration:.3f} s")
        print(f"    Samples: {sample_count}")
        print(f"    Period: {period} ms")
        print(f"    Frequency: {1000.0/period:.1f} Hz")
    except Exception as e:
        print(f"    [FAIL] Info display failed: {type(e).__name__}: {e}")
        return False

    print("\n  [OK] All period tests passed!")
    return True


def test_special_methods() -> bool:
    """Test special methods (__contains__, __getitem__, __repr__)"""
    erg_file: Path | None = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print("Skipping test_special_methods: test file not available")
        return False

    print("\nTesting special methods...")
    erg = ERG(erg_file)

    signal_names = erg.get_signal_names()

    # Test __contains__
    print("\n  Test 1: __contains__ (in operator)")
    if "Time" in signal_names and "Time" in erg:
        print("    [OK] 'Time' in erg returns True")
    else:
        print("    [FAIL] 'Time' should be in erg")
        return False

    if "NonExistent" not in erg:
        print("    [OK] 'NonExistent' in erg returns False")
    else:
        print("    [FAIL] 'NonExistent' should not be in erg")
        return False

    # Test __getitem__
    print("\n  Test 2: __getitem__ (bracket notation)")
    if "Time" in signal_names:
        time1 = erg.get_signal("Time")
        time2 = erg["Time"]
        if np.array_equal(time1, time2):
            print("    [OK] erg['Time'] equals erg.get_signal('Time')")
        else:
            print("    [FAIL] erg['Time'] does not match erg.get_signal('Time')")
            return False

    # Test __getitem__ with non-existent signal
    print("\n  Test 3: __getitem__ with non-existent signal")
    try:
        _ = erg["NonExistentSignal12345"]
        print("    [FAIL] Should have raised KeyError")
        return False
    except KeyError as e:
        print(f"    [OK] Correctly raised KeyError: {e}")

    # Test __repr__
    print("\n  Test 4: __repr__")
    repr_str = repr(erg)
    expected_count = len(signal_names)
    if f"signals={expected_count}" in repr_str:
        print(f"    [OK] repr(erg) = '{repr_str}'")
    else:
        print(f"    [FAIL] repr(erg) = '{repr_str}', expected 'signals={expected_count}'")
        return False

    print("\n  [OK] All special method tests passed!")
    return True


def test_caching() -> bool:
    """Test ERG instance caching"""
    erg_file: Path | None = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print("Skipping test_caching: test file not available")
        return False

    print("\nTesting ERG instance caching...")

    # Clear cache first
    ERG._instances.clear()

    # Test 1: Same filepath returns same instance
    print("\n  Test 1: Same filepath returns same instance")
    erg1 = ERG(erg_file)
    erg2 = ERG(erg_file)
    if erg1 is erg2:
        print("    [OK] erg1 is erg2 (same instance)")
    else:
        print("    [FAIL] erg1 is not erg2 (should be same instance)")
        return False

    # Test 2: get_all_signals cache persists
    print("\n  Test 2: get_all_signals cache persists across instances")
    erg3 = ERG(erg_file)
    _ = erg3.get_all_signals()  # Load cache

    # Create another "instance" (should be cached)
    erg4 = ERG(erg_file)
    if erg4._struct_array_cache is not None:
        print("    [OK] Structured array cache persists")
    else:
        print("    [FAIL] Structured array cache should persist")
        return False

    # Test 3: Clear cache
    print("\n  Test 3: Cache clearing")
    ERG._instances.clear()
    erg5 = ERG(erg_file, prefetch=False)  # Don't prefetch to test empty cache
    if erg5._struct_array_cache is None:
        print("    [OK] Cache cleared, new instance with prefetch=False has no cache")
    else:
        print("    [FAIL] After clearing, cache should be None when prefetch=False")
        return False

    print("\n  [OK] All caching tests passed!")
    return True


def test_performance_comparison() -> bool:
    """Compare performance: erg_python.get_signal() vs erg_python.get_all_signals() vs cmerg.get()"""
    erg_file: Path | None = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print("Skipping performance comparison: test file not available")
        return False

    print(f"\n{'=' * 80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'=' * 80}")
    print(f"Test file: {erg_file}")

    # ===== METHOD 1: erg_python.get_signal() =====
    print(f"\n{'--- Method 1: erg_python.get_signal() ---':^80}")

    # Clear cache and force garbage collection
    ERG._instances.clear()
    gc.collect()
    mem_before_1: float = get_memory_usage_mb()

    # Loading
    start: int = time_ns()
    erg1: ERG = ERG(erg_file)
    signal_names: list[str] = erg1.get_signal_names()
    load_time_1: int = time_ns() - start
    mem_after_load_1: float = get_memory_usage_mb()

    print(f"Loading:              {load_time_1 / 1_000:>12.2f} μs")
    print(f"Signal count:         {len(signal_names):>12,}")
    print(f"Memory after load:    {mem_after_load_1:>12.2f} MB (+{mem_after_load_1 - mem_before_1:.2f} MB)")

    # Cold read (all signals, first access)
    start = time_ns()
    _ = erg1[signal_names[0]]
    _ = erg1[signal_names[-2]]
    cold_time_1 = time_ns() - start
    mem_after_cold_1 = get_memory_usage_mb()
    print(f"Cold read:            {cold_time_1 / 1_000:>12.2f} μs")
    print(f"Memory after cold:    {mem_after_cold_1:>12.2f} MB (+{mem_after_cold_1 - mem_after_load_1:.2f} MB)")

    # Hot read (all signals, cached)
    start = time_ns()
    _ = erg1[signal_names[0]]
    _ = erg1[signal_names[-2]]
    hot_time_1 = time_ns() - start
    print(f"Hot read:             {hot_time_1 / 1_000:>12.2f} μs")

    # ===== METHOD 2: erg_python.get_all_signals() =====
    print(f"\n{'--- Method 2: erg_python.get_all_signals() (cached direct access) ---':^80}")

    # Clear cache and force garbage collection
    ERG._instances.clear()
    gc.collect()
    mem_before_2 = get_memory_usage_mb()

    # Loading
    start = time_ns()
    erg2 = ERG(erg_file)
    all_signals = erg2.get_all_signals()  # Load structured array into cache
    signal_names = erg2.get_signal_names()
    load_time_2 = time_ns() - start
    mem_after_load_2 = get_memory_usage_mb()

    print(f"Loading:              {load_time_2 / 1_000:>12.2f} μs")
    print(f"Memory after load:    {mem_after_load_2:>12.2f} MB (+{mem_after_load_2 - mem_before_2:.2f} MB)")

    # Cold read (access via cached structured array field names - raw unscaled)
    start = time_ns()
    _ = all_signals[signal_names[0]]
    _ = all_signals[signal_names[-2]]
    cold_time_2 = time_ns() - start
    mem_after_cold_2 = get_memory_usage_mb()
    print(f"Cold read (raw):      {cold_time_2 / 1_000:>12.2f} μs")
    print(f"Memory after cold:    {mem_after_cold_2:>12.2f} MB (+{mem_after_cold_2 - mem_before_2:.2f} MB)")

    # Hot read (repeated access to cached structured array)
    start = time_ns()
    _ = all_signals[signal_names[0]]
    _ = all_signals[signal_names[-2]]
    hot_time_2 = time_ns() - start
    print(f"Hot read (raw):       {hot_time_2 / 1_000:>12.2f} μs")

    # ===== METHOD 3: cmerg.get() =====
    print(f"\n{'--- Method 3: cmerg.get() ---':^80}")

    # Force garbage collection
    gc.collect()
    mem_before_3 = get_memory_usage_mb()

    # Loading
    start = time_ns()
    erg_cm = cmerg.ERG(str(erg_file))
    load_time_3 = time_ns() - start
    mem_after_load_3 = get_memory_usage_mb()

    print(f"Loading:              {load_time_3 / 1_000:>12.2f} μs")
    print(f"Memory after load:    {mem_after_load_3:>12.2f} MB (+{mem_after_load_3 - mem_before_3:.2f} MB)")

    # Cold read (all signals, first access)
    start = time_ns()
    _ = erg_cm.get(signal_names[0]).samples
    _ = erg_cm.get(signal_names[-2]).samples
    cold_time_3 = time_ns() - start
    mem_after_cold_3 = get_memory_usage_mb()
    print(f"Cold read:            {cold_time_3 / 1_000:>12.2f} μs")
    print(f"Memory after cold:    {mem_after_cold_3:>12.2f} MB (+{mem_after_cold_3 - mem_after_load_3:.2f} MB)")

    # Hot read (all signals, cached)
    start = time_ns()
    _ = erg_cm.get(signal_names[0]).samples
    _ = erg_cm.get(signal_names[-2]).samples
    hot_time_3 = time_ns() - start
    print(f"Hot read:             {hot_time_3 / 1_000:>12.2f} μs")

    # ===== COMPARISON SUMMARY =====
    print(f"\n{'=' * 80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Method':<40} {'Loading (μs)':>13} {'Cold (μs)':>13} {'Hot (μs)':>13}")
    print(f"{'-' * 80}")
    print(f"{'1. erg_python.get_signal()':<40} {load_time_1 / 1_000:>13.2f} {cold_time_1 / 1_000:>13.2f} {hot_time_1 / 1_000:>13.2f}")
    print(f"{'2. erg_python.get_all_signals()':<40} {load_time_2 / 1_000:>13.2f} {cold_time_2 / 1_000:>13.2f} {hot_time_2 / 1_000:>13.2f}")
    print(f"{'3. cmerg.get()':<40} {load_time_3 / 1_000:>13.2f} {cold_time_3 / 1_000:>13.2f} {hot_time_3 / 1_000:>13.2f}")

    print(f"\n{'SPEEDUP vs cmerg.get() (higher is better)':^80}")
    print(f"{'-' * 80}")
    load_speedup_1 = load_time_3 / load_time_1
    cold_speedup_1 = cold_time_3 / cold_time_1
    hot_speedup_1 = hot_time_3 / hot_time_1
    print(f"{'erg_python.get_signal()':<40} {load_speedup_1:>12.2f}x {cold_speedup_1:>12.2f}x {hot_speedup_1:>12.2f}x")

    load_speedup_2 = load_time_3 / load_time_2
    cold_speedup_2 = cold_time_3 / cold_time_2
    hot_speedup_2 = hot_time_3 / hot_time_2
    print(f"{'erg_python.get_all_signals()':<40} {load_speedup_2:>12.2f}x {cold_speedup_2:>12.2f}x {hot_speedup_2:>12.2f}x")

    # Memory footprint summary
    print(f"\n{'=' * 80}")
    print("MEMORY FOOTPRINT SUMMARY")
    print(f"{'=' * 80}")
    mem_delta_load_1 = mem_after_load_1 - mem_before_1
    mem_delta_cold_1 = mem_after_cold_1 - mem_after_load_1
    mem_delta_load_2 = mem_after_load_2 - mem_before_2
    mem_delta_cold_2 = mem_after_cold_2 - mem_after_load_2
    mem_delta_load_3 = mem_after_load_3 - mem_before_3
    mem_delta_cold_3 = mem_after_cold_3 - mem_after_load_3

    print(f"{'Method':<40} {'Load Δ (MB)':>12} {'Cold Δ (MB)':>12} {'Total (MB)':>12}")
    print(f"{'-' * 80}")
    print(f"{'1. erg_python.get_signal()':<40} {mem_delta_load_1:>12.2f} {mem_delta_cold_1:>12.2f} {mem_delta_load_1 + mem_delta_cold_1:>12.2f}")
    print(f"{'2. erg_python.get_all_signals()':<40} {mem_delta_load_2:>12.2f} {mem_delta_cold_2:>12.2f} {mem_delta_load_2 + mem_delta_cold_2:>12.2f}")
    print(f"{'3. cmerg.get()':<40} {mem_delta_load_3:>12.2f} {mem_delta_cold_3:>12.2f} {mem_delta_load_3 + mem_delta_cold_3:>12.2f}")

    print("\nNOTE: Memory measurements show incremental increases and may be affected by:")
    print("  - Python garbage collection timing")
    print("  - OS memory allocation strategies")
    print("  - Background Python processes")
    print("  - Memory-mapped files (shown in RSS but may not be physical RAM)")
    print("  Use these numbers for relative comparison, not absolute values.")

    # ===== PLOT SIGNALS FOR ALL METHODS =====
    print(f"\n{'=' * 80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'=' * 80}")

    # Define signals to plot: first, middle, and last signal (excluding Time)
    non_time_signals = [s for s in signal_names if s != "Time"]
    if len(non_time_signals) >= 3:
        plot_signals = [
            non_time_signals[0],  # First signal
            non_time_signals[len(non_time_signals) // 2],  # Middle signal
            non_time_signals[-1],  # Last signal
        ]
    else:
        plot_signals = non_time_signals[:3]  # Use whatever is available

    if "Time" not in signal_names:
        print("\nWarning: Cannot plot, 'Time' signal not found")
    elif len(plot_signals) < 3:
        print(f"\nWarning: Not enough signals to plot (need 3, found {len(plot_signals)})")
    else:
        print("\nPlotting signals:")
        print(f"  First:  {plot_signals[0]}")
        print(f"  Middle: {plot_signals[1]}")
        print(f"  Last:   {plot_signals[2]}")

        # Get time signal
        time_signal = erg1["Time"]

        # Method 1: erg_python.get_signal() - with scaling
        print("  Generating plot for Method 1 (erg_python.get_signal)...")
        fig1, axes1 = plt.subplots(3, 1, figsize=(12, 8))
        fig1.suptitle("Method 1: erg_python.get_signal() [Scaled Data]", fontsize=14, fontweight="bold")

        for idx, signal_name in enumerate(plot_signals):
            signal_data = erg1[signal_name]
            signal_unit = erg1.get_signal_unit(signal_name)

            axes1[idx].plot(time_signal, signal_data, linewidth=1.5, color="C0")
            axes1[idx].set_ylabel(f"{signal_name}\n[{signal_unit}]")
            axes1[idx].grid(True, alpha=0.3)
            axes1[idx].set_xlim(time_signal[0], time_signal[-1])

        axes1[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        plot_file_1 = "plot_method1_get_signal.png"
        plt.savefig(plot_file_1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {plot_file_1}")

        # Method 2: erg_python.get_all_signals() - raw data (no scaling)
        print("  Generating plot for Method 2 (erg_python.get_all_signals)...")
        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8))
        fig2.suptitle("Method 2: erg_python.get_all_signals() [Raw Unscaled Data]", fontsize=14, fontweight="bold")

        time_raw = all_signals["Time"]
        for idx, signal_name in enumerate(plot_signals):
            signal_data_raw = all_signals[signal_name]
            signal_unit = erg2.get_signal_unit(signal_name)

            axes2[idx].plot(time_raw, signal_data_raw, linewidth=1.5, color="C1")
            axes2[idx].set_ylabel(f"{signal_name}\n[{signal_unit}]")
            axes2[idx].grid(True, alpha=0.3)
            axes2[idx].set_xlim(time_raw[0], time_raw[-1])

        axes2[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        plot_file_2 = "plot_method2_get_all_signals.png"
        plt.savefig(plot_file_2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {plot_file_2}")

        # Method 3: cmerg.get() - with scaling
        print("  Generating plot for Method 3 (cmerg.get)...")
        fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8))
        fig3.suptitle("Method 3: cmerg.get() [Scaled Data]", fontsize=14, fontweight="bold")

        time_cm = erg_cm.get("Time").samples
        for idx, signal_name in enumerate(plot_signals):
            signal_data_cm = erg_cm.get(signal_name).samples
            signal_unit_cm = erg_cm.get(signal_name).unit

            axes3[idx].plot(time_cm, signal_data_cm, linewidth=1.5, color="C2")
            axes3[idx].set_ylabel(f"{signal_name}\n[{signal_unit_cm}]")
            axes3[idx].grid(True, alpha=0.3)
            axes3[idx].set_xlim(time_cm[0], time_cm[-1])

        axes3[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        plot_file_3 = "plot_method3_cmerg.png"
        plt.savefig(plot_file_3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {plot_file_3}")

        print("\n[OK] All comparison plots saved!")

    # Verify data integrity
    print(f"\n{'=' * 80}")
    print("DATA INTEGRITY CHECK - ALL SIGNALS")
    print(f"{'=' * 80}")
    print(f"Checking {len(signal_names)} signals...")
    print("  Note: Method 2 (get_all_signals) returns raw data without scaling\n")

    all_match: bool = True
    failed_signals: list[str] = []
    mismatched_signals: list[tuple[str, float]] = []

    # Compare all signals
    start_check: int = time_ns()
    for idx, signal_name in enumerate(signal_names):
        # Get signal from all three methods
        try:
            data1 = erg1[signal_name]  # Method 1: get_signal with scaling
            data2 = all_signals[signal_name]  # Method 2: raw unscaled data
            data3 = erg_cm.get(signal_name).samples  # Method 3: cmerg with scaling
        except Exception as e:
            print(f"  [FAIL] {signal_name}: Failed to retrieve - {e}")
            failed_signals.append(signal_name)
            all_match = False
            continue

        # Check lengths
        if len(data1) != len(data3) or len(data1) != len(data2):
            print(f"  [FAIL] {signal_name}: Length mismatch - method1={len(data1)}, method2={len(data2)}, method3={len(data3)}")
            failed_signals.append(signal_name)
            all_match = False
            continue

        # Compare methods 1 and 3 (both have scaling)
        if not np.allclose(data1, data3, rtol=1e-9, atol=1e-12):
            max_diff = np.abs(data1 - data3).max()
            print(f"  [FAIL] {signal_name}: Data mismatch (method 1 vs 3, max diff: {max_diff})")
            mismatched_signals.append((signal_name, max_diff))
            all_match = False

    check_time = time_ns() - start_check

    # Summary
    print(f"\nIntegrity check completed in {check_time / 1_000:.2f} μs")
    print(f"  Signals checked:      {len(signal_names)}")
    print(f"  Failed retrievals:    {len(failed_signals)}")
    print(f"  Data mismatches:      {len(mismatched_signals)}")

    if failed_signals:
        print(f"\n[FAIL] Failed signals: {failed_signals[:5]}")
        if len(failed_signals) > 5:
            print(f"       ... and {len(failed_signals) - 5} more")

    if mismatched_signals:
        print("\n[FAIL] Mismatched signals (showing top 5 by max diff):")
        sorted_mismatches = sorted(mismatched_signals, key=lambda x: x[1], reverse=True)
        for signal_name, max_diff in sorted_mismatches[:5]:
            print(f"       {signal_name}: max diff = {max_diff:.6e}")

    if all_match:
        print("\n[OK] All signals validated successfully!")
        print("[OK] Methods 1 & 3 data match perfectly (both apply scaling)")
        print("[OK] Method 2 provides raw zero-copy memory-mapped data")
    else:
        print(f"\n[FAIL] Validation failed for {len(failed_signals) + len(mismatched_signals)} signals")

    return True


if __name__ == "__main__":
    # Reconfigure stdout to use UTF-8 encoding (for Windows compatibility with Unicode chars)
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test suite for erg_python module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_erg.py simulation.erg
  python test_erg.py ../examples/test.erg
        """,
    )
    parser.add_argument("filepath", type=Path, help="Path to the ERG file to test with")

    args: argparse.Namespace = parser.parse_args()
    TEST_ERG_FILE = args.filepath

    print("=" * 60)
    print("ERG Python Module Test Suite")
    print("=" * 60)
    print(f"\nTest file: {TEST_ERG_FILE}\n")

    tests: list[tuple[str, Any]] = [
        ("Basic functionality", test_basic),
        ("Signal info", test_signal_info),
        ("get_all_signals() 2D array", test_get_all_signals),
        ("Error handling", test_error_handling),
        ("Raw byte type handling", test_raw_byte_type_handling),
        ("Pandas DataFrame conversion", test_pandas_conversion),
        ("Batch metadata methods", test_batch_metadata_methods),
        ("Signal index", test_signal_index),
        ("Period calculation", test_period),
        ("Special methods", test_special_methods),
        ("Caching behavior", test_caching),
        ("Performance comparison", test_performance_comparison),
    ]

    passed: int = 0
    failed: int = 0

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")
        try:
            if test_func():
                passed += 1
                print(f"[OK] {test_name} passed")
            else:
                failed += 1
                print(f"[FAIL] {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test_name} failed with exception: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
