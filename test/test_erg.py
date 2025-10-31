"""
Test script for erg_python module

Usage:
    python test_erg.py <path/to/test.erg>
"""

import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path to import erg_python
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from erg_python import ERG
import matplotlib.pyplot as plt

# Import cmparser for comparison

from cmparser import ERG as CMParserERG

HAS_CMPARSER = True

# Global variable for test file path
TEST_ERG_FILE = None


def time_ns() -> int:
    """Get current time in nanoseconds."""
    return time.perf_counter_ns()


def test_basic():
    """Test basic ERG file reading"""
    erg_file = TEST_ERG_FILE

    if not erg_file or not erg_file.exists():
        print(f"ERROR: Test file not found or not specified: {erg_file}")
        print("Usage: python test_erg.py <path/to/test.erg>")
        return False

    print(f"Testing with file: {erg_file}")

    # Create ERG object (should auto-parse now)
    # Can pass Path object directly now
    print("Loading ERG file...")
    erg = ERG(erg_file)

    # Check sample count
    print(f"Sample count: {erg.sample_count}")

    # Get signal names
    print(f"Number of signals: {len(erg.signal_names)}")
    print(f"First 10 signals: {erg.signal_names[:10]}")

    # Get a single signal (typically "Time" exists)
    if "Time" in erg.signal_names:
        print("\nGetting 'Time' signal...")

        # Time cold retrieval
        start = time_ns()
        time_data = erg.get_signal("Time")
        end = time_ns()
        cold_time = end - start

        print(f"Time data type: {type(time_data)}")
        print(f"Time length: {len(time_data)}")
        print(f"Time range: {time_data[0]:.6f} to {time_data[-1]:.6f}")
        print(f"Cold retrieval time: {cold_time:,} ns ({cold_time / 1_000_000:.3f} ms)")

        # Time warm retrieval (cached)
        start = time_ns()
        time_data = erg.get_signal("Time")
        end = time_ns()
        warm_time = end - start
        print(f"Warm retrieval time: {warm_time:,} ns ({warm_time / 1_000:.3f} µs)")

        # Get signal info
        info = erg.get_signal_info("Time")
        print(f"Time signal info: {info}")

    # Get multiple signals using list comprehension
    available_signals = erg.signal_names[:5]
    print(f"\nGetting multiple signals: {available_signals}")

    # Time cold retrieval
    start = time_ns()
    signals = [erg.get_signal(name) for name in available_signals]
    end = time_ns()
    batch_cold_time = end - start

    for name, data in zip(available_signals, signals):
        print(f"  {name}: type={type(data).__name__}, dtype={data.dtype}, length={len(data)}")

    print(f"Cold retrieval time: {batch_cold_time:,} ns ({batch_cold_time / 1_000_000:.3f} ms)")
    print(f"Average per signal: {batch_cold_time / len(available_signals):,.0f} ns")

    # Time warm retrieval (all cached)
    start = time_ns()
    signals = [erg[name] for name in available_signals]  # Using __getitem__
    end = time_ns()
    batch_warm_time = end - start
    print(f"Warm retrieval time: {batch_warm_time:,} ns ({batch_warm_time / 1_000:.3f} µs)")
    print(f"Average per signal (warm): {batch_warm_time / len(available_signals):,.0f} ns")

    # Plot signals if they exist
    plot_signals = ["Car.ax", "Car.v", "Car.Distance"]
    missing_signals = [s for s in plot_signals if s not in erg.signal_names]

    if missing_signals:
        print(f"\nWarning: Cannot plot, missing signals: {missing_signals}")
    elif "Time" not in erg.signal_names:
        print("\nWarning: Cannot plot, 'Time' signal not found")
    else:
        print(f"\nPlotting signals: {plot_signals}")
        plot_time = erg.get_signal("Time")

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle("ERG Signal Plots", fontsize=14, fontweight="bold")

        for idx, signal_name in enumerate(plot_signals):
            signal_data = erg.get_signal(signal_name)
            signal_info = erg.get_signal_info(signal_name)

            axes[idx].plot(plot_time, signal_data, linewidth=1.5)
            axes[idx].set_ylabel(f"{signal_name}\n[{signal_info['unit']}]")
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim(plot_time[0], plot_time[-1])

        axes[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        plt.show()
        print("Plot displayed!")

    print("\nTest passed!")
    return True


def test_signal_info():
    """Test signal metadata retrieval"""
    erg_file = TEST_ERG_FILE

    if not erg_file or not erg_file.exists():
        print(f"Skipping test_signal_info: test file not available")
        return False

    erg = ERG(erg_file)

    print("\nSignal metadata for all signals:")
    for signal_name in erg.signal_names[:10]:  # First 10 signals
        info = erg.get_signal_info(signal_name)
        print(f"  {signal_name}:")
        print(f"    Type: {info['type']}, Size: {info['type_size']} bytes")
        print(f"    Unit: {info['unit']}, Factor: {info['factor']}, Offset: {info['offset']}")

    return True


def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")

    # Test with non-existent file
    try:
        erg = ERG("nonexistent.erg")
        print("ERROR: Should have raised an exception for non-existent file")
        return False
    except Exception as e:
        print(f"  Correctly raised exception for non-existent file: {type(e).__name__}")

    # Test with non-existent signal
    erg_file = TEST_ERG_FILE
    if erg_file and erg_file.exists():
        erg = ERG(erg_file)
        try:
            data = erg.get_signal("NonExistentSignal12345")
            print("ERROR: Should have raised KeyError for non-existent signal")
            return False
        except KeyError as e:
            print(f"  Correctly raised KeyError for non-existent signal: {e}")

    return True


def test_numpy_integration():
    """Test numpy array conversion if numpy is available"""
    try:
        import numpy as np

        print("\nNumPy is available, testing numpy integration...")

        erg_file = TEST_ERG_FILE
        if not erg_file or not erg_file.exists():
            print(f"Skipping numpy test: test file not available")
            return False

        erg = ERG(erg_file)
        if erg.signal_names:
            signal_name = erg.signal_names[0]
            data = erg.get_signal(signal_name)

            if isinstance(data, np.ndarray):
                print(f"  Successfully got numpy array for '{signal_name}'")
                print(f"  Array shape: {data.shape}, dtype: {data.dtype}")
                print(f"  Mean: {data.mean():.6f}, Std: {data.std():.6f}")
                return True
            else:
                print(f"  WARNING: Got {type(data)} instead of numpy array")
                return False
    except ImportError:
        print("\nNumPy not available, skipping numpy integration test")
        return True


def test_performance_comparison():
    """Compare performance between C extension and pure Python cmparser"""
    if not HAS_CMPARSER:
        print("\ncmparser not available, skipping performance comparison")
        return True

    erg_file = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print(f"Skipping performance test: test file not available")
        return False

    print("\n" + "=" * 60)
    print("Performance Comparison: erg_python (C) vs cmparser (Python)")
    print("=" * 60)

    # Test signals to retrieve
    test_signals = ["Time", "Car.ax", "Car.v"]

    # Benchmark C extension (erg_python)
    print("\n[1] erg_python (C Extension):")
    print("-" * 40)

    # File loading + parsing
    start = time_ns()
    erg_c = ERG(erg_file)
    end = time_ns()
    c_load_time = end - start
    print(f"  Load + parse:      {c_load_time:>12,} ns ({c_load_time / 1_000_000:>8.3f} ms)")

    # Single signal retrieval
    available_test_signals = [s for s in test_signals if s in erg_c.signal_names]
    if not available_test_signals:
        print("Warning: None of the test signals found in file")
        return False

    signal_times = []
    for signal_name in available_test_signals:
        start = time_ns()
        data = erg_c.get_signal(signal_name)
        end = time_ns()
        signal_times.append(end - start)

    c_avg_signal_time = sum(signal_times) / len(signal_times)
    print(f"  Avg signal:        {c_avg_signal_time:>12,.0f} ns ({c_avg_signal_time / 1_000_000:>8.3f} ms)")

    # Benchmark pure Python (cmparser)
    print("\n[2] cmparser (Pure Python):")
    print("-" * 40)

    # File loading + parsing
    start = time_ns()
    erg_py = CMParserERG(erg_file)
    end = time_ns()
    py_load_time = end - start
    print(f"  Load + parse:      {py_load_time:>12,} ns ({py_load_time / 1_000_000:>8.3f} ms)")

    # Single signal retrieval
    signal_times = []
    for signal_name in available_test_signals:
        start = time_ns()
        data = erg_py[signal_name]
        end = time_ns()
        signal_times.append(end - start)

    py_avg_signal_time = sum(signal_times) / len(signal_times)
    print(f"  Avg signal:        {py_avg_signal_time:>12,.0f} ns ({py_avg_signal_time / 1_000_000:>8.3f} ms)")

    # Calculate speedups
    print("\n" + "=" * 60)
    print("Speedup (erg_python vs cmparser):")
    print("=" * 60)

    load_speedup = py_load_time / c_load_time
    signal_speedup = py_avg_signal_time / c_avg_signal_time

    print(f"  Load + parse:  {load_speedup:>6.2f}x faster")
    print(f"  Signal access: {signal_speedup:>6.2f}x faster")

    return True


if __name__ == "__main__":
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

    args = parser.parse_args()
    TEST_ERG_FILE = args.filepath

    print("=" * 60)
    print("ERG Python Module Test Suite")
    print("=" * 60)
    print(f"\nTest file: {TEST_ERG_FILE}\n")

    tests = [
        ("Basic functionality", test_basic),
        ("Signal info", test_signal_info),
        ("Error handling", test_error_handling),
        ("NumPy integration", test_numpy_integration),
        ("Performance comparison", test_performance_comparison),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} passed")
            else:
                failed += 1
                print(f"✗ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} failed with exception: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
