"""Smoke tests for plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

from sdcpy import SDCAnalysis

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestSDCAnalysisPlotting:
    """Smoke tests for SDCAnalysis plotting methods."""

    def test_two_way_plot(self, random_ts_pair):
        """two_way_plot should not crash."""
        from plotnine.ggplot import ggplot

        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.two_way_plot()
        assert isinstance(result, ggplot)

    def test_combi_plot(self, random_ts_pair):
        """combi_plot should return a matplotlib Figure."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot()
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_combi_plot_with_params(self, random_ts_pair):
        """combi_plot should accept various parameters."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = sdc.combi_plot(
            alpha=0.1,
            xlabel="Time Series 1",
            ylabel="Time Series 2",
            title="Custom Title",
            max_r=0.5,
            min_lag=-20,
            max_lag=20,
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

    def test_plot_consecutive(self, correlated_ts_pair):
        """plot_consecutive should return a plotnine ggplot."""
        from plotnine.ggplot import ggplot

        ts1, ts2 = correlated_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=49)
        result = sdc.plot_consecutive(alpha=0.5)  # Higher alpha for random data
        assert isinstance(result, ggplot)

    def test_dominant_lags_plot(self, binned_value_ts_pair):
        """dominant_lags_plot should return a matplotlib Figure."""
        ts1, ts2 = binned_value_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=49)
        result = sdc.dominant_lags_plot(alpha=0.5)
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


class TestSingleShiftPlot:
    """Tests for the unimplemented single_shift_plot method."""

    def test_raises_not_implemented(self, random_ts_pair):
        """single_shift_plot should raise NotImplementedError."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        with pytest.raises(NotImplementedError):
            sdc.single_shift_plot(shift=5)
