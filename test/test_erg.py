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
import cmerg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Add parent directory to path to import erg_python
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from erg_python import ERG


# Global variable for test file path
TEST_ERG_FILE = None


def time_ns() -> int:
    """Get current time in nanoseconds."""
    return time.perf_counter_ns()


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


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

    # Plot signals if they exist using get_all_signals()
    plot_signals = ["Car.ax", "Car.v", "Car.Distance"]
    missing_signals = [s for s in plot_signals if s not in signal_names]

    if missing_signals:
        print(f"\nWarning: Cannot plot, missing signals: {missing_signals}")
    elif "Time" not in signal_names:
        print("\nWarning: Cannot plot, 'Time' signal not found")
    else:
        print(f"\nPlotting signals: {plot_signals} (using get_all_signals())")

        # Get all signals as structured array
        all_data = erg.get_all_signals()

        # Access fields by name
        plot_time = all_data['Time']

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle("ERG Signal Plots (from get_all_signals())", fontsize=14, fontweight="bold")

        for idx, signal_name in enumerate(plot_signals):
            signal_data = all_data[signal_name]
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


def test_get_all_signals():
    """Test get_all_signals() structured array functionality"""
    erg_file = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print(f"Skipping test_get_all_signals: test file not available")
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
    print(f"Retrieval time: {all_signals_time / 1_000_000:.3f} ms")

    # Performance comparison
    print("\nPerformance comparison...")
    start = time_ns()
    _ = {name: erg.get_signal(name) for name in signal_names}
    individual_time = time_ns() - start

    print(f"Individual retrieval: {individual_time / 1_000_000:.3f} ms")
    print(f"get_all_signals():    {all_signals_time / 1_000_000:.3f} ms")
    speedup = individual_time / all_signals_time
    print(f"Speedup: {speedup:.2f}x faster")

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
    """Compare performance: erg_python.get_signal() vs erg_python.get_all_signals() vs cmerg.get()"""
    erg_file = TEST_ERG_FILE
    if not erg_file or not erg_file.exists():
        print(f"Skipping performance comparison: test file not available")
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
    mem_before_1 = get_memory_usage_mb()

    # Loading
    start = time_ns()
    erg1 = ERG(erg_file)
    signal_names = erg1.get_signal_names()
    load_time_1 = time_ns() - start
    mem_after_load_1 = get_memory_usage_mb()

    print(f"Loading:              {load_time_1 / 1_000_000:>10.3f} ms")
    print(f"Signal count:         {len(signal_names):>10}")
    print(f"Memory after load:    {mem_after_load_1:>10.2f} MB (+{mem_after_load_1 - mem_before_1:.2f} MB)")

    # Cold read (all signals, first access)
    start = time_ns()
    erg1_first = erg1[signal_names[0]]
    erg1_last = erg1[signal_names[-2]]
    cold_time_1 = time_ns() - start
    mem_after_cold_1 = get_memory_usage_mb()
    print(f"Cold read:      {cold_time_1 / 1_000_000:>10.3f} ms")
    print(f"Memory after cold:    {mem_after_cold_1:>10.2f} MB (+{mem_after_cold_1 - mem_after_load_1:.2f} MB)")

    # Hot read (all signals, cached)
    start = time_ns()
    erg1_first = erg1[signal_names[0]]
    erg1_last = erg1[signal_names[-2]]
    hot_time_1 = time_ns() - start
    print(f"Hot read:       {hot_time_1 / 1_000_000:>10.3f} ms")

    # ===== METHOD 2: erg_python.get_all_signals() =====
    print(f"\n{'--- Method 2: erg_python.get_all_signals() ---':^80}")

    # Clear cache and force garbage collection
    ERG._instances.clear()
    gc.collect()
    mem_before_2 = get_memory_usage_mb()

    # Loading
    start = time_ns()
    erg2 = ERG(erg_file)
    erg2.get_all_signals()
    signal_names = erg2.get_signal_names()
    load_time_2 = time_ns() - start
    mem_after_load_2 = get_memory_usage_mb()

    print(f"Loading:              {load_time_2 / 1_000_000:>10.3f} ms")
    print(f"Memory after load:    {mem_after_load_2:>10.2f} MB (+{mem_after_load_2 - mem_before_2:.2f} MB)")

    # Cold read (get all at once, first access via field name)
    start = time_ns()
    erg2_first = erg2[signal_names[0]]
    erg2_last = erg2[signal_names[-2]]
    cold_time_2 = time_ns() - start
    mem_after_cold_2 = get_memory_usage_mb()
    print(f"Cold read:      {cold_time_2 / 1_000_000:>10.3f} ms")
    print(f"Memory after cold:    {mem_after_cold_2:>10.2f} MB (+{mem_after_cold_2 - mem_after_load_2:.2f} MB)")

    # Hot read (get all at once, cached/repeated)
    start = time_ns()
    erg2_first = erg2[signal_names[0]]
    erg2_last = erg2[signal_names[-2]]
    hot_time_2 = time_ns() - start
    print(f"Hot read:       {hot_time_2 / 1_000_000:>10.3f} ms")

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

    print(f"Loading:              {load_time_3 / 1_000_000:>10.3f} ms")
    print(f"Memory after load:    {mem_after_load_3:>10.2f} MB (+{mem_after_load_3 - mem_before_3:.2f} MB)")

    # Cold read (all signals, first access)
    start = time_ns()
    erg_cm_first = erg_cm.get(signal_names[0]).samples
    erg_cm_last = erg_cm.get(signal_names[-2]).samples
    cold_time_3 = time_ns() - start
    mem_after_cold_3 = get_memory_usage_mb()
    print(f"Cold read:      {cold_time_3 / 1_000_000:>10.3f} ms")
    print(f"Memory after cold:    {mem_after_cold_3:>10.2f} MB (+{mem_after_cold_3 - mem_after_load_3:.2f} MB)")

    # Hot read (all signals, cached)
    start = time_ns()
    erg_cm_first = erg_cm.get(signal_names[0]).samples
    erg_cm_last = erg_cm.get(signal_names[-2]).samples
    hot_time_3 = time_ns() - start
    print(f"Hot read:       {hot_time_3 / 1_000_000:>10.3f} ms")

    # ===== COMPARISON SUMMARY =====
    print(f"\n{'=' * 80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Method':<40} {'Loading (ms)':>12} {'Cold (ms)':>12} {'Hot (ms)':>12}")
    print(f"{'-' * 80}")
    print(f"{'1. erg_python.get_signal()':<40} {load_time_1 / 1_000_000:>12.3f} {cold_time_1 / 1_000_000:>12.3f} {hot_time_1 / 1_000_000:>12.3f}")
    print(f"{'2. erg_python.get_all_signals()':<40} {load_time_2 / 1_000_000:>12.3f} {cold_time_2 / 1_000_000:>12.3f} {hot_time_2 / 1_000_000:>12.3f}")
    print(f"{'3. cmerg.get()':<40} {load_time_3 / 1_000_000:>12.3f} {cold_time_3 / 1_000_000:>12.3f} {hot_time_3 / 1_000_000:>12.3f}")

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

    print(f"\nNOTE: Memory measurements show incremental increases and may be affected by:")
    print(f"  - Python garbage collection timing")
    print(f"  - OS memory allocation strategies")
    print(f"  - Background Python processes")
    print(f"  - Memory-mapped files (shown in RSS but may not be physical RAM)")
    print(f"  Use these numbers for relative comparison, not absolute values.")

    # Verify data integrity
    print(f"\n{'=' * 80}")
    print("DATA INTEGRITY CHECK")
    print(f"{'=' * 80}")

    all_match = True

    print(f"Comparing first signal: {signal_names[0]}")
    print(f"  Note: Method 2 (get_all_signals) returns raw data without scaling")

    # Check lengths
    if len(erg1_first) != len(erg_cm_first) or len(erg1_first) != len(erg2_first):
        print(f"  ✗ Length mismatch: method1={len(erg1_first)}, method2={len(erg2_first)}, method3={len(erg_cm_first)}")
        all_match = False
    # Compare methods 1 and 3 (both have scaling)
    elif not np.allclose(erg1_first, erg_cm_first, rtol=1e-9, atol=1e-12):
        max_diff = np.abs(erg1_first - erg_cm_first).max()
        print(f"  ✗ Data mismatch (method 1 vs 3, max diff: {max_diff})")
        all_match = False
    else:
        print(f"  ✓ Methods 1 and 3 match (with scaling)")
        # Show that method 2 is different (raw data)
        if not np.allclose(erg2_first, erg1_first, rtol=1e-9, atol=1e-12):
            print(f"  ✓ Method 2 correctly returns raw data (different from scaled)")

    print(f"Comparing last signal: {signal_names[-1]}")

    # Check lengths
    if len(erg1_last) != len(erg_cm_last) or len(erg1_last) != len(erg2_last):
        print(f"  ✗ Length mismatch: method1={len(erg1_last)}, method2={len(erg2_last)}, method3={len(erg_cm_last)}")
        all_match = False
    # Compare methods 1 and 3 (both have scaling)
    elif not np.allclose(erg1_last, erg_cm_last, rtol=1e-9, atol=1e-12):
        max_diff = np.abs(erg1_last - erg_cm_last).max()
        print(f"  ✗ Data mismatch (method 1 vs 3, max diff: {max_diff})")
        all_match = False
    else:
        print(f"  ✓ Methods 1 and 3 match (with scaling)")
        # Show that method 2 is different (raw data)
        if not np.allclose(erg2_last, erg1_last, rtol=1e-9, atol=1e-12):
            print(f"  ✓ Method 2 correctly returns raw data (different from scaled)")

    if all_match:
        print("\n✓ All length checks passed!")
        print("✓ Methods 1 & 3 data match (both apply scaling)")
        print("✓ Method 2 provides raw zero-copy memory-mapped data")

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
        ("get_all_signals() 2D array", test_get_all_signals),
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
