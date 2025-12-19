"""Smoke tests for plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

from sdcpy import SDCAnalysis, plot_two_way_sdc

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestPlotTwoWaySDC:
    """Tests for the standalone plot_two_way_sdc function."""

    def test_returns_plotnine_ggplot(self, random_ts_pair):
        """Should return a plotnine ggplot object."""
        from plotnine.ggplot import ggplot

        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = plot_two_way_sdc(sdc.sdc_df)
        assert isinstance(result, ggplot)

    def test_with_custom_alpha(self, random_ts_pair):
        """Should work with custom alpha value."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        result = plot_two_way_sdc(sdc.sdc_df, alpha=0.01)
        assert result is not None


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


class TestSingleShiftPlot:
    """Tests for the unimplemented single_shift_plot method."""

    def test_raises_not_implemented(self, random_ts_pair):
        """single_shift_plot should raise NotImplementedError."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)
        with pytest.raises(NotImplementedError):
            sdc.single_shift_plot(shift=5)
