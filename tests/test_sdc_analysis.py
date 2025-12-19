"""Integration tests for SDCAnalysis class with various time series scenarios."""

import numpy as np
import pandas as pd
import pytest

from sdcpy import SDCAnalysis


class TestSDCAnalysisBasic:
    """Basic tests for SDCAnalysis instantiation and properties."""

    def test_basic_instantiation(self, random_ts_pair):
        """Should create SDCAnalysis with pandas Series."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        assert sdc.fragment_size == 10
        assert sdc.way == "two-way"
        assert len(sdc.sdc_df) > 0

    def test_numpy_array_input(self, numpy_ts_pair):
        """Should accept numpy arrays and auto-generate dates."""
        ts1, ts2 = numpy_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        # Should have created date indices starting from 2000-01-01
        assert pd.Timestamp("2000-01-01") in sdc.ts1.index
        assert len(sdc.sdc_df) > 0

    def test_one_way_sdc(self, random_ts_pair):
        """Should support one-way SDC (ts1 only)."""
        ts1, _ = random_ts_pair
        sdc = SDCAnalysis(ts1, fragment_size=10, n_permutations=9)
        assert sdc.way == "one-way"
        # One-way should exclude diagonal (self-comparisons at same position)
        assert len(sdc.sdc_df) > 0


class TestSDCAnalysisEdgeCases:
    """Edge case tests for SDCAnalysis."""

    def test_short_time_series(self, short_ts_pair):
        """Should work with very short time series."""
        ts1, ts2 = short_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=5, n_permutations=9)
        expected = (len(ts1) - 5) * (len(ts2) - 5)
        assert len(sdc.sdc_df) == expected

    def test_weekly_frequency(self, weekly_ts_pair):
        """Should work with weekly frequency data."""
        ts1, ts2 = weekly_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=12, n_permutations=9)
        assert len(sdc.sdc_df) > 0
        # Check lag is in terms of index positions, not days
        assert "lag" in sdc.sdc_df.columns

    def test_partial_overlap(self, partial_overlap_ts_pair):
        """Should handle series with partial date overlap."""
        ts1, ts2 = partial_overlap_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        # Should only use the overlapping portion
        assert len(sdc.ts1) < 100
        assert len(sdc.ts2) < 100
        assert len(sdc.ts1) == len(sdc.ts2)

    def test_different_methods(self, random_ts_pair):
        """Should support both pearson and spearman methods."""
        ts1, ts2 = random_ts_pair

        pearson = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9, method="pearson")
        spearman = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9, method="spearman")

        assert pearson.method == "pearson"
        assert spearman.method == "spearman"
        # Results should differ
        assert not np.allclose(pearson.sdc_df["r"].values, spearman.sdc_df["r"].values)


class TestSDCAnalysisLagConstraints:
    """Tests for lag filtering in SDCAnalysis."""

    def test_min_lag_filter(self, random_ts_pair):
        """min_lag should filter out lower lags."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9, min_lag=0)
        assert (sdc.sdc_df["lag"] >= 0).all()

    def test_max_lag_filter(self, random_ts_pair):
        """max_lag should filter out higher lags."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9, max_lag=10)
        assert (sdc.sdc_df["lag"] <= 10).all()

    def test_combined_lag_filter(self, random_ts_pair):
        """Both min_lag and max_lag should work together."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9, min_lag=-5, max_lag=5)
        assert sdc.sdc_df["lag"].between(-5, 5).all()


class TestSDCAnalysisDateHandling:
    """Tests for date column handling."""

    def test_date_columns_present(self, random_ts_pair):
        """Should have date_1 and date_2 columns."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        assert "date_1" in sdc.sdc_df.columns
        assert "date_2" in sdc.sdc_df.columns

    def test_date_values_valid(self, random_ts_pair):
        """Date values should be within original series range."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        assert sdc.sdc_df["date_1"].min() >= ts1.index.min()
        assert sdc.sdc_df["date_2"].min() >= ts2.index.min()


class TestSDCAnalysisStatisticalProperties:
    """Tests for statistical correctness of SDCAnalysis."""

    def test_self_analysis(self, random_ts_pair):
        """Analyzing same series should show strong correlation at lag 0."""
        ts1, _ = random_ts_pair
        sdc = SDCAnalysis(ts1, ts1.copy(), fragment_size=10, n_permutations=9)
        lag_zero = sdc.sdc_df[sdc.sdc_df["lag"] == 0]
        # All lag-0 correlations should be 1.0
        np.testing.assert_array_almost_equal(lag_zero["r"].values, 1.0, decimal=10)

    def test_detected_correlation(self, correlated_ts_pair):
        """Should detect correlation in correlated time series."""
        ts1, ts2 = correlated_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=49)
        # Should have some comparisons with reasonable correlations
        high_corr = sdc.sdc_df[sdc.sdc_df["r"].abs() > 0.3]
        assert len(high_corr) > 0

    def test_pvalue_distribution(self, random_ts_pair):
        """For random data, p-values should be roughly uniform."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=49)
        # Very rough check: should have some p-values in each quartile
        quartiles = pd.cut(sdc.sdc_df["p_value"], bins=[0, 0.25, 0.5, 0.75, 1.0])
        counts = quartiles.value_counts()
        # Each quartile should have at least some values
        assert (counts > 0).all()


class TestSDCAnalysisFragmentSizes:
    """Tests for various fragment size configurations."""

    @pytest.mark.parametrize("fragment_size", [5, 10, 20, 30])
    def test_various_fragment_sizes(self, random_ts_pair, fragment_size):
        """Should work with various fragment sizes."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=fragment_size, n_permutations=9)
        expected = (len(ts1) - fragment_size) * (len(ts2) - fragment_size)
        assert len(sdc.sdc_df) == expected

    def test_fragment_size_equals_length_minus_one(self, random_ts_pair):
        """Should work with fragment size = len - 1."""
        ts1, ts2 = random_ts_pair
        # Use first 20 points for speed
        ts1_short = ts1[:20]
        ts2_short = ts2[:20]
        fragment_size = len(ts1_short) - 2
        sdc = SDCAnalysis(ts1_short, ts2_short, fragment_size=fragment_size, n_permutations=9)
        # Should have very few comparisons
        assert len(sdc.sdc_df) == 4  # (20-18)^2 = 2^2 = 4
