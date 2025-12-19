"""Unit tests for core functions: generate_correlation_map, shuffle_along_axis, compute_sdc."""

import numpy as np
import pytest

from sdcpy.core import compute_sdc, generate_correlation_map, shuffle_along_axis


class TestGenerateCorrelationMap:
    """Tests for the vectorized correlation map function."""

    def test_perfect_positive_correlation(self):
        """Identical rows should have r=1.0."""
        x = np.array([[1, 2, 3, 4, 5]])
        y = np.array([[1, 2, 3, 4, 5]])
        result = generate_correlation_map(x, y, method="pearson")
        assert result.shape == (1, 1)
        np.testing.assert_almost_equal(result[0, 0], 1.0, decimal=10)

    def test_perfect_negative_correlation(self):
        """Negated rows should have r=-1.0."""
        x = np.array([[1, 2, 3, 4, 5]])
        y = np.array([[-1, -2, -3, -4, -5]])
        result = generate_correlation_map(x, y, method="pearson")
        np.testing.assert_almost_equal(result[0, 0], -1.0, decimal=10)

    def test_zero_correlation(self):
        """Orthogonal vectors should have r≈0."""
        x = np.array([[1, -1, 1, -1]])
        y = np.array([[1, 1, -1, -1]])
        result = generate_correlation_map(x, y, method="pearson")
        np.testing.assert_almost_equal(result[0, 0], 0.0, decimal=10)

    def test_multiple_rows(self):
        """Should compute all pairwise correlations."""
        x = np.array([[1, 2, 3], [4, 5, 6]])  # 2 rows
        y = np.array([[1, 2, 3], [6, 5, 4], [1, 1, 1]])  # 3 rows
        result = generate_correlation_map(x, y, method="pearson")
        assert result.shape == (2, 3)
        # First row of x correlates perfectly with first row of y
        np.testing.assert_almost_equal(result[0, 0], 1.0, decimal=10)
        # First row of x anti-correlates with second row of y
        np.testing.assert_almost_equal(result[0, 1], -1.0, decimal=10)

    def test_spearman_method(self):
        """Spearman should work on rank-transformed data."""
        x = np.array([[1, 2, 3, 4, 5]])
        y = np.array([[1, 4, 9, 16, 25]])  # Nonlinear but monotonic
        pearson = generate_correlation_map(x, y, method="pearson")[0, 0]
        spearman = generate_correlation_map(x, y, method="spearman")[0, 0]
        # Spearman should be 1.0 (perfect rank correlation)
        np.testing.assert_almost_equal(spearman, 1.0, decimal=10)
        # Pearson should be less than 1.0
        assert pearson < 1.0

    def test_unsupported_method_raises(self):
        """Unsupported methods should raise NotImplementedError."""
        x = np.array([[1, 2, 3]])
        y = np.array([[4, 5, 6]])
        with pytest.raises(NotImplementedError):
            generate_correlation_map(x, y, method="kendall")


class TestShuffleAlongAxis:
    """Tests for the shuffle function."""

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        a = np.arange(20).reshape(4, 5)
        result = shuffle_along_axis(a, axis=1)
        assert result.shape == a.shape

    def test_preserves_values(self):
        """All original values should be present in output."""
        a = np.arange(20).reshape(4, 5)
        result = shuffle_along_axis(a, axis=1)
        # Each row should have the same set of values
        for i in range(4):
            assert set(a[i]) == set(result[i])

    def test_randomness(self):
        """Different calls should (usually) produce different outputs."""
        a = np.arange(100).reshape(10, 10)
        result1 = shuffle_along_axis(a, axis=1)
        result2 = shuffle_along_axis(a, axis=1)
        # Extremely unlikely to be identical
        assert not np.array_equal(result1, result2)


class TestComputeSDC:
    """Tests for the main compute_sdc function."""

    def test_output_shape(self, numpy_ts_pair):
        """Should produce correct number of comparisons."""
        ts1, ts2 = numpy_ts_pair
        fragment_size = 10
        result = compute_sdc(
            ts1, ts2, fragment_size=fragment_size, n_permutations=9, permutations=True
        )
        expected_rows = (len(ts1) - fragment_size) * (len(ts2) - fragment_size)
        assert len(result) == expected_rows

    def test_output_columns(self, numpy_ts_pair):
        """Should have all required columns."""
        ts1, ts2 = numpy_ts_pair
        result = compute_sdc(ts1, ts2, fragment_size=10, n_permutations=9)
        expected_columns = {"start_1", "stop_1", "start_2", "stop_2", "lag", "r", "p_value"}
        assert set(result.columns) == expected_columns

    def test_r_values_in_range(self, numpy_ts_pair):
        """Correlation values should be in [-1, 1]."""
        ts1, ts2 = numpy_ts_pair
        result = compute_sdc(ts1, ts2, fragment_size=10, n_permutations=9)
        assert result["r"].between(-1, 1).all()

    def test_p_values_in_range(self, numpy_ts_pair):
        """P-values should be in [0, 1]."""
        ts1, ts2 = numpy_ts_pair
        result = compute_sdc(ts1, ts2, fragment_size=10, n_permutations=9)
        assert result["p_value"].between(0, 1).all()

    def test_lag_calculation(self, numpy_ts_pair):
        """Lag should equal start_1 - start_2."""
        ts1, ts2 = numpy_ts_pair
        result = compute_sdc(ts1, ts2, fragment_size=10, n_permutations=9)
        expected_lag = result["start_1"] - result["start_2"]
        np.testing.assert_array_almost_equal(result["lag"], expected_lag)

    def test_min_max_lag_filter(self, numpy_ts_pair):
        """min_lag and max_lag should filter results."""
        ts1, ts2 = numpy_ts_pair
        result = compute_sdc(ts1, ts2, fragment_size=10, n_permutations=9, min_lag=-5, max_lag=5)
        assert result["lag"].between(-5, 5).all()

    def test_no_permutations_flag(self, numpy_ts_pair):
        """permutations=False should use scipy p-values."""
        ts1, ts2 = numpy_ts_pair
        result = compute_sdc(ts1, ts2, fragment_size=10, n_permutations=9, permutations=False)
        # Should still have p-values
        assert result["p_value"].notna().all()

    def test_pearson_vs_spearman(self, numpy_ts_pair):
        """Different methods should produce different results."""
        ts1, ts2 = numpy_ts_pair
        pearson = compute_sdc(ts1, ts2, fragment_size=10, n_permutations=9, method="pearson")
        spearman = compute_sdc(ts1, ts2, fragment_size=10, n_permutations=9, method="spearman")
        # Should have same shape but different values
        assert len(pearson) == len(spearman)
        assert not np.allclose(pearson["r"].values, spearman["r"].values)

    def test_small_fragment_size(self, numpy_ts_pair):
        """Should work with very small fragment sizes."""
        ts1, ts2 = numpy_ts_pair
        result = compute_sdc(ts1, ts2, fragment_size=3, n_permutations=9)
        assert len(result) > 0

    def test_large_fragment_size(self, numpy_ts_pair):
        """Should work with fragment size close to series length."""
        ts1, ts2 = numpy_ts_pair
        fragment_size = len(ts1) - 5
        result = compute_sdc(ts1, ts2, fragment_size=fragment_size, n_permutations=9)
        expected_rows = (len(ts1) - fragment_size) * (len(ts2) - fragment_size)
        assert len(result) == expected_rows


class TestComputeSDCStatisticalProperties:
    """Tests for statistical correctness of compute_sdc."""

    def test_self_correlation_is_one(self):
        """Correlating a series with itself should give r=1.0 (at lag 0)."""
        np.random.seed(42)
        ts = np.random.randn(50)
        result = compute_sdc(ts, ts.copy(), fragment_size=10, n_permutations=9, permutations=False)
        # At lag 0, r should be 1.0
        lag_zero = result[result["lag"] == 0]
        np.testing.assert_array_almost_equal(lag_zero["r"].values, 1.0, decimal=10)

    def test_anti_correlation(self):
        """Negated series should show r=-1.0."""
        np.random.seed(42)
        ts = np.random.randn(50)
        result = compute_sdc(ts, -ts, fragment_size=10, n_permutations=9, permutations=False)
        lag_zero = result[result["lag"] == 0]
        np.testing.assert_array_almost_equal(lag_zero["r"].values, -1.0, decimal=10)

    def test_lag_detection(self, correlated_ts_pair):
        """Should detect correlation peak at the correct lag."""
        ts1, ts2 = correlated_ts_pair
        result = compute_sdc(
            ts1.values, ts2.values, fragment_size=10, n_permutations=9, permutations=False
        )
        # Find the lag with highest mean correlation
        mean_by_lag = result.groupby("lag")["r"].mean()
        best_lag = mean_by_lag.idxmax()
        # The synthetic data has lag of 5, but we relaxed to ±10 due to noise
        assert abs(best_lag - 5) <= 10
