"""
Test script for erg_python module

Usage:
    python test_erg.py <path/to/test.erg>
"""

import sys
import argparse
import time
from pathlib import Path
import cmerg

# Add parent directory to path to import erg_python
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from erg_python import ERG
import matplotlib.pyplot as plt

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

    # Get signal names
    signal_names = erg.get_signal_names()
    print(f"Number of signals: {len(signal_names)}")
    print(f"First 10 signals: {signal_names[:10]}")

    # Get a single signal (typically "Time" exists)
    if "Time" in signal_names:
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

        # Get signal unit and type
        unit = erg.get_signal_unit("Time")
        dtype = erg.get_signal_type("Time")
        print(f"Time signal unit: {unit}, dtype: {dtype}")

    # Get multiple signals using list comprehension
    available_signals = signal_names[:5]
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
    missing_signals = [s for s in plot_signals if s not in signal_names]

    if missing_signals:
        print(f"\nWarning: Cannot plot, missing signals: {missing_signals}")
    elif "Time" not in signal_names:
        print("\nWarning: Cannot plot, 'Time' signal not found")
    else:
        print(f"\nPlotting signals: {plot_signals}")
        plot_time = erg.get_signal("Time")

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle("ERG Signal Plots", fontsize=14, fontweight="bold")

        for idx, signal_name in enumerate(plot_signals):
            signal_data = erg.get_signal(signal_name)
            signal_unit = erg.get_signal_unit(signal_name)

            axes[idx].plot(plot_time, signal_data, linewidth=1.5)
            axes[idx].set_ylabel(f"{signal_name}\n[{signal_unit}]")
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
    signal_names = erg.get_signal_names()
    for signal_name in signal_names[:10]:  # First 10 signals
        unit = erg.get_signal_unit(signal_name)
        dtype = erg.get_signal_type(signal_name)
        print(f"  {signal_name}:")
        print(f"    Type: {dtype}")
        print(f"    Unit: {unit}")

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


def test_performance_comparison():
    """Compare performance with cmerg package"""
    erg_file = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print(f"Skipping performance comparison: test file not available")
        return False

    print(f"\n{'=' * 70}")
    print("PERFORMANCE COMPARISON: erg_python vs cmerg")
    print(f"{'=' * 70}")
    print(f"Test file: {erg_file}")

    # Test signals - use common signals that should exist
    test_signals = ["Time"]

    # ===== ERG_PYTHON TESTS =====
    print(f"\n{'--- erg_python ---':^70}")

    # Clear cache by creating new instance
    ERG._instances.clear()

    # 1. Loading time
    start = time_ns()
    erg_python = ERG(erg_file)
    load_time_erg = time_ns() - start
    print(f"Loading ERG:          {load_time_erg / 1_000_000:>10.3f} ms")

    # Get available signals
    signal_names = erg_python.get_signal_names()
    print(f"Signal count:         {len(signal_names):>10}")

    # Use first few signals for testing
    test_signals = signal_names[: min(5, len(signal_names))]

    # 2. Cold read (first access)
    start = time_ns()
    for sig in test_signals:
        _ = erg_python.get_signal(sig)
    cold_read_erg = time_ns() - start
    print(f"Cold read ({len(test_signals)} signals): {cold_read_erg / 1_000_000:>10.3f} ms")
    print(f"  Per signal:         {cold_read_erg / len(test_signals) / 1_000_000:>10.3f} ms")

    # 3. Hot read (cached access)
    start = time_ns()
    for sig in test_signals:
        _ = erg_python[sig]
    hot_read_erg = time_ns() - start
    print(f"Hot read ({len(test_signals)} signals):  {hot_read_erg / 1_000:>10.3f} µs")
    print(f"  Per signal:         {hot_read_erg / len(test_signals) / 1_000:>10.3f} µs")

    # ===== CMERG TESTS =====
    print(f"\n{'--- cmerg ---':^70}")

    # 1. Loading time
    start = time_ns()
    erg_cm = cmerg.ERG(str(erg_file))
    load_time_cm = time_ns() - start
    print(f"Loading ERG:          {load_time_cm / 1_000_000:>10.3f} ms")

    # 2. Cold read (first access) - cmerg uses .get() method
    start = time_ns()
    for sig in test_signals:
        signal_obj = erg_cm.get(sig)
        # Access the actual data to ensure it's loaded
        _ = signal_obj.samples
    cold_read_cm = time_ns() - start
    print(f"Cold read ({len(test_signals)} signals): {cold_read_cm / 1_000_000:>10.3f} ms")
    print(f"  Per signal:         {cold_read_cm / len(test_signals) / 1_000_000:>10.3f} ms")

    # 3. Hot read (cached access)
    start = time_ns()
    for sig in test_signals:
        signal_obj = erg_cm.get(sig)
        _ = signal_obj.samples
    hot_read_cm = time_ns() - start
    print(f"Hot read ({len(test_signals)} signals):  {hot_read_cm / 1_000:>10.3f} µs")
    print(f"  Per signal:         {hot_read_cm / len(test_signals) / 1_000:>10.3f} µs")

    # ===== COMPARISON SUMMARY =====
    print(f"\n{'=' * 70}")
    print("SPEEDUP SUMMARY (erg_python vs cmerg)")
    print(f"{'=' * 70}")

    load_speedup = load_time_cm / load_time_erg
    cold_speedup = cold_read_cm / cold_read_erg
    hot_speedup = hot_read_cm / hot_read_erg

    print(f"Loading:              {load_speedup:>10.2f}x {'faster' if load_speedup > 1 else 'slower'}")
    print(f"Cold read:            {cold_speedup:>10.2f}x {'faster' if cold_speedup > 1 else 'slower'}")
    print(f"Hot read:             {hot_speedup:>10.2f}x {'faster' if hot_speedup > 1 else 'slower'}")

    # Verify data integrity
    print(f"\n{'=' * 70}")
    print("DATA INTEGRITY CHECK")
    print(f"{'=' * 70}")

    all_match = True
    for sig in test_signals[:3]:  # Check first 3 signals
        data_erg = erg_python[sig]
        data_cm = erg_cm.get(sig).samples  # cmerg returns Signal object with .samples property

        if len(data_erg) != len(data_cm):
            print(f"✗ {sig}: Length mismatch ({len(data_erg)} vs {len(data_cm)})")
            all_match = False
        elif not (data_erg == data_cm).all():
            import numpy as np

            max_diff = np.abs(data_erg - data_cm).max()
            print(f"✗ {sig}: Data mismatch (max diff: {max_diff})")
            all_match = False
        else:
            print(f"✓ {sig}: Data matches")

    if all_match:
        print("\nAll tested signals match between erg_python and cmerg!")

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
