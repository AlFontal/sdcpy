"""Smoke tests for plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

from sdcpy import SDCAnalysis

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestSDCAnalysisPlotting:
    """Smoke tests for SDCAnalysis plotting methods."""

    def test_combi_plot(self, ts_pair_any_index):
        """combi_plot should return a matplotlib Figure (both index types)."""
        ts1, ts2 = ts_pair_any_index
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(date_fmt="%Y-%m")
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_with_params(self, ts_pair_any_index):
        """combi_plot should accept various parameters (both index types)."""
        ts1, ts2 = ts_pair_any_index
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(
            alpha=0.1,
            xlabel="Time Series 1",
            ylabel="Time Series 2",
            title="Custom Title",
            max_r=0.5,
            min_lag=-20,
            max_lag=20,
            date_fmt="%Y-%m",
        )
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_alignment(self, random_ts_pair):
        """combi_plot should accept different alignment options."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)

        for align in ["left", "center", "right"]:
            result = sdc.combi_plot(align=align)
            assert isinstance(result, plt.Figure)
            plt.close(result)

    def test_get_ranges_df(self, binned_value_ts_pair):
        """get_ranges_df should return a DataFrame."""
        import pandas as pd

        ts1, ts2 = binned_value_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=49)
        result = sdc.get_ranges_df(bin_size=3, alpha=0.5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_plot_range_comparison(self, binned_value_ts_pair):
        """plot_range_comparison should return a plotnine ggplot."""
        from plotnine.ggplot import ggplot

        ts1, ts2 = binned_value_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=49)
        result = sdc.plot_range_comparison(bin_size=3, alpha=0.5)
        assert isinstance(result, ggplot)


class TestCombiPlotConditions:
    """Comprehensive tests for combi_plot under different conditions."""

    def test_combi_plot_weekly_frequency(self, weekly_ts_pair):
        """combi_plot should work with weekly time series."""
        ts1, ts2 = weekly_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=4, n_permutations=9)
        result = sdc.combi_plot()
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_show_ts2_false(self, random_ts_pair):
        """combi_plot should work with show_ts2=False."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(show_ts2=False)
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_custom_metric_label(self, random_ts_pair):
        """combi_plot should accept custom metric_label."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(metric_label="My Custom Metric")
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_only_positive_lags(self, random_ts_pair):
        """combi_plot should work with only positive lags."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(min_lag=0, max_lag=30)
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_only_negative_lags(self, random_ts_pair):
        """combi_plot should work with only negative lags."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(min_lag=-30, max_lag=0)
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_custom_date_format(self, random_ts_pair):
        """combi_plot should accept custom date_fmt."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(date_fmt="%Y-%m-%d")
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_no_colorbar(self, random_ts_pair):
        """combi_plot should work without colorbar."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(show_colorbar=False)
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_spearman_method(self, random_ts_pair):
        """combi_plot should show correct label for spearman method."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9, method="spearman")
        result = sdc.combi_plot()
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_figsize(self, random_ts_pair):
        """combi_plot should accept figsize parameter."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(figsize=(12, 10))
        assert isinstance(result, plt.Figure)
        assert result.get_size_inches()[0] == 12
        assert result.get_size_inches()[1] == 10
        plt.close(result)

    def test_combi_plot_string_datetime_index(self):
        """combi_plot should work with string-based datetime indexes."""
        import numpy as np
        import pandas as pd

        # Create time series with string-based (object dtype) datetime index
        np.random.seed(42)
        dates = pd.date_range("2005-01-01", periods=100, freq="D")
        # Convert to strings to simulate user data with object dtype index
        string_dates = dates.strftime("%Y-%m-%d")

        ts1 = pd.Series(np.random.randn(100), index=string_dates, name="ts1")
        ts1.index.name = "date"
        ts2 = pd.Series(np.random.randn(100), index=string_dates, name="ts2")
        ts2.index.name = "date"

        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(date_fmt="%Y-%m")
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_integer_index(self, numpy_ts_pair):
        """combi_plot should work with integer indexes (no datetime conversion)."""
        ts1, ts2 = numpy_ts_pair
        # Numpy arrays will be converted to Series with integer index by SDCAnalysis
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot()
        assert isinstance(result, plt.Figure)
        plt.close(result)
