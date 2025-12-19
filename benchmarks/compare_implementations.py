#!/usr/bin/env python
"""Compare old vs new compute_sdc implementation."""

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.mstats import rankdata
from tqdm.auto import tqdm


def generate_correlation_map(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Vectorized correlation map."""
    if method.lower() == "spearman":
        x = rankdata(x, axis=1)
        y = rankdata(y, axis=1)
    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)
    n = x.shape[1]
    s_x = x.std(axis=1, ddof=n - 1)
    s_y = y.std(axis=1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def shuffle_along_axis(a: np.ndarray, axis: int) -> np.ndarray:
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


RECOGNIZED_METHODS = {
    "pearson": lambda x, y: stats.pearsonr(x, y),
    "spearman": lambda x, y: stats.spearmanr(x, y),
}


def compute_sdc_OLD(ts1, ts2, fragment_size, n_permutations=99, method="pearson"):
    """Original loop-based implementation."""
    method_fun = RECOGNIZED_METHODS[method]
    n_iterations = (len(ts1) - fragment_size) * (len(ts2) - fragment_size)
    n_root = int(np.sqrt(n_permutations).round())
    sdc_array = np.empty(shape=(n_iterations, 7))
    sdc_array[:] = np.nan
    i = 0
    progress_bar = tqdm(total=n_iterations, desc="OLD", leave=False)

    for start_1 in range(len(ts1) - fragment_size):
        stop_1 = start_1 + fragment_size
        for start_2 in range(len(ts2) - fragment_size):
            lag = start_1 - start_2
            stop_2 = start_2 + fragment_size
            fragment_1 = ts1[start_1:stop_1]
            fragment_2 = ts2[start_2:stop_2]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, p_value = method_fun(fragment_1, fragment_2)
            # Permutation test
            permuted_1 = shuffle_along_axis(np.tile(fragment_1, n_root).reshape(n_root, -1), axis=1)
            permuted_2 = shuffle_along_axis(np.tile(fragment_2, n_root).reshape(n_root, -1), axis=1)
            permuted_scores = generate_correlation_map(
                permuted_1, permuted_2, method=method
            ).reshape(-1)
            p_value = 1 - stats.percentileofscore(np.abs(permuted_scores), np.abs(statistic)) / 100

            sdc_array[i] = [start_1, stop_1, start_2, stop_2, lag, statistic, p_value]
            i += 1
            progress_bar.update(1)
    progress_bar.close()
    return pd.DataFrame(
        sdc_array[:i], columns=["start_1", "stop_1", "start_2", "stop_2", "lag", "r", "p_value"]
    )


def compute_sdc_NEW(ts1, ts2, fragment_size, n_permutations=99, method="pearson"):
    """New vectorized implementation."""
    from numpy.lib.stride_tricks import sliding_window_view

    n1 = len(ts1) - fragment_size
    n2 = len(ts2) - fragment_size

    frags1 = sliding_window_view(ts1, fragment_size)[:n1]
    frags2 = sliding_window_view(ts2, fragment_size)[:n2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_matrix = generate_correlation_map(frags1, frags2, method=method)

    start_1_grid, start_2_grid = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    lag_matrix = start_1_grid - start_2_grid

    n_valid = n1 * n2
    start_1_vals = start_1_grid.ravel()
    start_2_vals = start_2_grid.ravel()
    stop_1_vals = start_1_vals + fragment_size
    stop_2_vals = start_2_vals + fragment_size
    lag_vals = lag_matrix.ravel()
    r_vals = corr_matrix.ravel()

    n_root = int(np.sqrt(n_permutations).round())
    p_values = np.zeros(n_valid)
    batch_size = 500

    for batch_start in tqdm(range(0, n_valid, batch_size), desc="NEW", leave=False):
        batch_end = min(batch_start + batch_size, n_valid)
        for idx in range(batch_start, batch_end):
            i, j = start_1_vals[idx], start_2_vals[idx]
            frag1 = frags1[i]
            frag2 = frags2[j]
            statistic = r_vals[idx]
            permuted_1 = shuffle_along_axis(np.tile(frag1, n_root).reshape(n_root, -1), axis=1)
            permuted_2 = shuffle_along_axis(np.tile(frag2, n_root).reshape(n_root, -1), axis=1)
            permuted_scores = generate_correlation_map(
                permuted_1, permuted_2, method=method
            ).reshape(-1)
            p_values[idx] = (
                1 - stats.percentileofscore(np.abs(permuted_scores), np.abs(statistic)) / 100
            )

    return pd.DataFrame(
        {
            "start_1": start_1_vals.astype(float),
            "stop_1": stop_1_vals.astype(float),
            "start_2": start_2_vals.astype(float),
            "stop_2": stop_2_vals.astype(float),
            "lag": lag_vals.astype(float),
            "r": r_vals,
            "p_value": p_values,
        }
    )


def run_comparison(ts_length, fragment_size, n_permutations, max_comparisons=None):
    """Run both old and new implementations and compare."""
    np.random.seed(42)
    ts1 = np.random.randn(ts_length)
    ts2 = np.random.randn(ts_length)

    n_comparisons = (ts_length - fragment_size) ** 2
    print(f"\nts={ts_length}, frag={fragment_size}, perms={n_permutations}")
    print(f"Total comparisons: {n_comparisons:,}")

    if max_comparisons and n_comparisons > max_comparisons:
        print(f"Skipping (>{max_comparisons:,} comparisons)")
        return None, None

    # Run OLD
    start = time.perf_counter()
    compute_sdc_OLD(ts1, ts2, fragment_size, n_permutations)
    time_old = time.perf_counter() - start
    print(f"OLD: {time_old:.2f}s")

    # Run NEW
    start = time.perf_counter()
    compute_sdc_NEW(ts1, ts2, fragment_size, n_permutations)
    time_new = time.perf_counter() - start
    print(f"NEW: {time_new:.2f}s")

    speedup = time_old / time_new
    print(f"Speedup: {speedup:.2f}x")

    return time_old, time_new


if __name__ == "__main__":
    print("=" * 60)
    print("OLD vs NEW compute_sdc Comparison")
    print("=" * 60)

    # Test cases
    test_cases = [
        {"ts_length": 100, "fragment_size": 10, "n_permutations": 99},
        {"ts_length": 200, "fragment_size": 20, "n_permutations": 99},
        {"ts_length": 300, "fragment_size": 30, "n_permutations": 99},
    ]

    for tc in test_cases:
        run_comparison(**tc, max_comparisons=100000)
