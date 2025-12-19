"""Core computation functions for Scale Dependent Correlation analysis."""

import warnings
from typing import Callable, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.mstats import rankdata
from tqdm.auto import tqdm

RECOGNIZED_METHODS = {
    "pearson": lambda x, y: stats.pearsonr(x, y),
    "spearman": lambda x, y: stats.spearmanr(x, y),
}

CONSTANT_WARNING = {"pearson": stats.ConstantInputWarning, "spearman": stats.ConstantInputWarning}


def generate_correlation_map(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> np.ndarray:
    """
    Correlate each row in matrix X against each row in matrix Y.

    Parameters
    ----------
    x
      Shape N X T.
    y
      Shape M X T.
    method
        Method use to compute the correlation. Must be one of 'pearson' or 'spearman'

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    if method.lower() not in ["spearman", "pearson"]:
        raise NotImplementedError(
            f'Method {method} not understood, must be one of "pearson", "spearman"'
        )

    if method.lower() == "spearman":
        x = rankdata(x, axis=1)
        y = rankdata(y, axis=1)

    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError("x and y must have the same number of timepoints.")
    s_x = x.std(axis=1, ddof=n - 1)
    s_y = y.std(axis=1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def shuffle_along_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Shuffles array independently across selected axis

    Parameters
    ----------
    a
        Input array
    axis
        Axis across which to shuffle
    Returns
    -------
    np.ndarray
        Shuffled copy of original array
    """
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def _build_fragment_matrix(ts: np.ndarray, fragment_size: int) -> np.ndarray:
    """Build a matrix where each row is a sliding window fragment of the time series."""
    n_fragments = len(ts) - fragment_size
    # Use stride tricks for efficient view-based slicing
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(ts, fragment_size)[:n_fragments]


def compute_sdc(
    ts1: np.ndarray,
    ts2: np.ndarray,
    fragment_size: int,
    n_permutations: int = 99,
    method: Union[str, Callable] = "pearson",
    two_tailed: bool = True,
    permutations: bool = True,
    min_lag: int = -np.inf,
    max_lag: int = np.inf,
) -> pd.DataFrame:
    """
    Computes scale dependent correlation (https://doi.org/10.1007/s00382-005-0106-4) matrix among two time series

    Parameters
    ----------
    ts1
        First time series. Must be array-like.
    ts2
        Second time series. Must be array-like.
    fragment_size
        Size of the fragments of the original time-series that will be used in their one-to-one comparison.
    n_permutations
        Number of permutations used to obtain the p-value of the non-parametric randomization test.
    method
        Method that will be used to perform the pairwise correlation/similarity/distance comparisons. Methods already
        understood are `pearson`, `spearman` and `kendall` but any custom callable taking two array-like inputs and
        returning a single float can be used. Note that this will be called n_permutations + 1 times for every single
        pairwise comparison, so an expensive call can significantly increase the total computation time. Pearson method
        will be orders of magnitude faster than the others since it has been implemented via vectorized functions.
    two_tailed
        Whether to use a two tailed randomised test or not. Default recognized methods should use the default two tailed
        value, but other distance metrics might not.
    permutations
        Whether to generate permutations of the fragments and generate their estimated-pvalues. If False, all p-values
        returned will be those from the statistical test passed in method.
    min_lag
        Lower limit of the lags between ts1 and ts2 that will be computed.
    max_lag
        Upper limit of the lags between ts1 and ts2 that will be computed.

    Returns
    -------
    pd.DataFrame
        Data frame with a row for each pair wise comparison containing information about the coordinates of each
        fragment used, the similarity obtained and the p-value from the randomised test.
    """
    # Convert to numpy arrays if needed
    ts1_arr = np.asarray(ts1)
    ts2_arr = np.asarray(ts2)

    # Use vectorized path for built-in methods
    if method in RECOGNIZED_METHODS:
        return _compute_sdc_vectorized(
            ts1_arr,
            ts2_arr,
            fragment_size,
            n_permutations,
            method,
            two_tailed,
            permutations,
            min_lag,
            max_lag,
        )
    else:
        # Fall back to original loop-based implementation for custom callables
        return _compute_sdc_loop(
            ts1_arr,
            ts2_arr,
            fragment_size,
            n_permutations,
            method,
            two_tailed,
            permutations,
            min_lag,
            max_lag,
        )


def _compute_sdc_vectorized(
    ts1: np.ndarray,
    ts2: np.ndarray,
    fragment_size: int,
    n_permutations: int,
    method: str,
    two_tailed: bool,
    permutations: bool,
    min_lag: int,
    max_lag: int,
) -> pd.DataFrame:
    """Vectorized implementation for built-in correlation methods."""
    n1 = len(ts1) - fragment_size
    n2 = len(ts2) - fragment_size

    # Build fragment matrices using sliding window
    frags1 = _build_fragment_matrix(ts1, fragment_size)  # (n1, fragment_size)
    frags2 = _build_fragment_matrix(ts2, fragment_size)  # (n2, fragment_size)

    # Compute all correlations at once using vectorized correlation map
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_matrix = generate_correlation_map(frags1, frags2, method=method)  # (n1, n2)

    # Build lag matrix
    start_1_grid, start_2_grid = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    lag_matrix = start_1_grid - start_2_grid

    # Create mask for valid lags
    valid_mask = (lag_matrix >= min_lag) & (lag_matrix <= max_lag)

    # Extract valid entries
    valid_indices = np.where(valid_mask)
    n_valid = len(valid_indices[0])

    start_1_vals = valid_indices[0]
    start_2_vals = valid_indices[1]
    stop_1_vals = start_1_vals + fragment_size
    stop_2_vals = start_2_vals + fragment_size
    lag_vals = lag_matrix[valid_mask]
    r_vals = corr_matrix[valid_mask]

    # Compute p-values
    if permutations:
        n_root = int(np.sqrt(n_permutations).round())
        n_actual_perms = n_root * n_root
        p_values = np.zeros(n_valid)

        # Process in batches for memory efficiency
        batch_size = 500
        for batch_start in tqdm(
            range(0, n_valid, batch_size), desc="Computing p-values", leave=False
        ):
            batch_end = min(batch_start + batch_size, n_valid)
            batch_indices = range(batch_start, batch_end)

            for idx in batch_indices:
                i, j = start_1_vals[idx], start_2_vals[idx]
                frag1 = frags1[i]
                frag2 = frags2[j]
                statistic = r_vals[idx]

                # Generate permuted fragments
                permuted_1 = shuffle_along_axis(np.tile(frag1, n_root).reshape(n_root, -1), axis=1)
                permuted_2 = shuffle_along_axis(np.tile(frag2, n_root).reshape(n_root, -1), axis=1)
                permuted_scores = generate_correlation_map(
                    permuted_1, permuted_2, method=method
                ).reshape(-1)

                if two_tailed:
                    p_values[idx] = (
                        1
                        - stats.percentileofscore(np.abs(permuted_scores), np.abs(statistic)) / 100
                    )
                else:
                    p_values[idx] = 1 - stats.percentileofscore(permuted_scores, statistic) / 100
    else:
        # Use scipy's p-values
        p_values = np.zeros(n_valid)
        method_fun = RECOGNIZED_METHODS[method]
        for idx in range(n_valid):
            i, j = start_1_vals[idx], start_2_vals[idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_values[idx] = method_fun(frags1[i], frags2[j])

    # Build result DataFrame
    sdc_df = pd.DataFrame(
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

    return sdc_df


def _compute_sdc_loop(
    ts1: np.ndarray,
    ts2: np.ndarray,
    fragment_size: int,
    n_permutations: int,
    method: Callable,
    two_tailed: bool,
    permutations: bool,
    min_lag: int,
    max_lag: int,
) -> pd.DataFrame:
    """Original loop-based implementation for custom callable methods."""
    method_fun = method
    n_iterations = (len(ts1) - fragment_size) * (len(ts2) - fragment_size)
    n_root = int(np.sqrt(n_permutations).round())
    sdc_array = np.empty(shape=(n_iterations, 7))
    sdc_array[:] = np.nan
    i = 0
    progress_bar = tqdm(total=n_iterations, desc="Computing SDC", leave=False)

    for start_1 in range(len(ts1) - fragment_size):
        stop_1 = start_1 + fragment_size
        for start_2 in range(len(ts2) - fragment_size):
            lag = start_1 - start_2
            if min_lag <= lag <= max_lag:
                stop_2 = start_2 + fragment_size
                fragment_1 = ts1[start_1:stop_1]
                fragment_2 = ts2[start_2:stop_2]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    statistic, p_value = method_fun(fragment_1, fragment_2)

                if permutations:
                    permuted_scores = [
                        method_fun(
                            np.random.permutation(fragment_1), np.random.permutation(fragment_2)
                        )[0]
                        for _ in range(n_permutations)
                    ]
                    if two_tailed:
                        p_value = (
                            1
                            - stats.percentileofscore(np.abs(permuted_scores), np.abs(statistic))
                            / 100
                        )
                    else:
                        p_value = 1 - stats.percentileofscore(permuted_scores, statistic) / 100

                sdc_array[i] = [start_1, stop_1, start_2, stop_2, lag, statistic, p_value]
                i += 1
                progress_bar.update(1)

    progress_bar.close()
    sdc_df = pd.DataFrame(
        sdc_array[:i], columns=["start_1", "stop_1", "start_2", "stop_2", "lag", "r", "p_value"]
    )

    return sdc_df
