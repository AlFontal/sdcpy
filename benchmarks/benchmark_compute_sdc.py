#!/usr/bin/env python
"""Benchmark script for compute_sdc optimization."""

import time

import numpy as np

from sdcpy.core import compute_sdc


def benchmark_compute_sdc(ts_length: int, fragment_size: int, n_permutations: int, runs: int = 3):
    """Benchmark compute_sdc with given parameters."""
    np.random.seed(42)
    ts1 = np.random.randn(ts_length)
    ts2 = np.random.randn(ts_length)

    times = []
    for i in range(runs):
        start = time.perf_counter()
        result = compute_sdc(
            ts1,
            ts2,
            fragment_size=fragment_size,
            n_permutations=n_permutations,
            method="pearson",
            permutations=True,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s ({len(result)} comparisons)")

    avg = np.mean(times)
    print(f"  Average: {avg:.3f}s")
    return avg


if __name__ == "__main__":
    print("=" * 60)
    print("compute_sdc Benchmark")
    print("=" * 60)

    # Test cases with increasing complexity
    test_cases = [
        {"ts_length": 50, "fragment_size": 10, "n_permutations": 49},
        {"ts_length": 100, "fragment_size": 10, "n_permutations": 49},
        {"ts_length": 100, "fragment_size": 10, "n_permutations": 99},
        # Weekly data scenarios (52 weeks/year)
        {"ts_length": 156, "fragment_size": 12, "n_permutations": 99},  # ~3 years weekly
        {"ts_length": 260, "fragment_size": 12, "n_permutations": 99},  # ~5 years weekly
    ]

    results = {}
    for tc in test_cases:
        label = f"ts={tc['ts_length']}, frag={tc['fragment_size']}, perms={tc['n_permutations']}"
        print(f"\n{label}")
        results[label] = benchmark_compute_sdc(**tc)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for label, avg_time in results.items():
        print(f"{label}: {avg_time:.3f}s")
