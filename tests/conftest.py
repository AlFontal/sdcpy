"""Shared test fixtures for sdcpy test suite."""

import numpy as np
import pandas as pd
import pytest


def _convert_to_string_index(ts: pd.Series) -> pd.Series:
    """Convert a Series with DatetimeIndex to one with string datetime index."""
    result = ts.copy()
    result.index = ts.index.strftime("%Y-%m-%d")
    return result


@pytest.fixture
def random_ts_pair():
    """Two random time series with daily frequency (DatetimeIndex)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    ts1 = pd.Series(np.random.randn(100), index=dates, name="ts1")
    ts1.index.name = "date_1"
    ts2 = pd.Series(np.random.randn(100), index=dates, name="ts2")
    ts2.index.name = "date_2"
    return ts1, ts2


@pytest.fixture
def string_datetime_ts_pair():
    """Two random time series with string-based datetime index (object dtype)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    string_dates = dates.strftime("%Y-%m-%d")
    ts1 = pd.Series(np.random.randn(100), index=string_dates, name="ts1")
    ts1.index.name = "date_1"
    ts2 = pd.Series(np.random.randn(100), index=string_dates, name="ts2")
    ts2.index.name = "date_2"
    return ts1, ts2


@pytest.fixture(params=["datetime", "string"])
def ts_pair_any_index(request, random_ts_pair, string_datetime_ts_pair):
    """Time series pair with either datetime or string index (parameterized)."""
    if request.param == "datetime":
        return random_ts_pair
    else:
        return string_datetime_ts_pair


@pytest.fixture
def correlated_ts_pair():
    """ts2 is a noisy lagged copy of ts1 (lag=5 days)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    ts1_values = np.random.randn(100)
    ts1 = pd.Series(ts1_values, index=dates, name="ts1")
    ts1.index.name = "date_1"
    # ts2 is ts1 shifted by 5 positions with small noise
    ts2 = pd.Series(
        np.roll(ts1_values, 5) + np.random.randn(100) * 0.1,
        index=dates,
        name="ts2",
    )
    ts2.index.name = "date_2"
    return ts1, ts2


@pytest.fixture
def weekly_ts_pair():
    """Weekly frequency over ~3 years (156 weeks)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=156, freq="W")
    ts1 = pd.Series(np.random.randn(156), index=dates, name="ts1")
    ts1.index.name = "date_1"
    ts2 = pd.Series(np.random.randn(156), index=dates, name="ts2")
    ts2.index.name = "date_2"
    return ts1, ts2


@pytest.fixture
def short_ts_pair():
    """Very short time series (minimum viable: 20 points)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    ts1 = pd.Series(np.random.randn(20), index=dates, name="ts1")
    ts1.index.name = "date_1"
    ts2 = pd.Series(np.random.randn(20), index=dates, name="ts2")
    ts2.index.name = "date_2"
    return ts1, ts2


@pytest.fixture
def numpy_ts_pair():
    """Plain numpy arrays (no pandas Series)."""
    np.random.seed(42)
    return np.random.randn(100), np.random.randn(100)


@pytest.fixture
def partial_overlap_ts_pair():
    """Two series with only partial date overlap."""
    np.random.seed(42)
    dates1 = pd.date_range("2020-01-01", periods=100, freq="D")
    dates2 = pd.date_range("2020-02-15", periods=100, freq="D")  # Starts 45 days later
    ts1 = pd.Series(np.random.randn(100), index=dates1, name="ts1")
    ts1.index.name = "date_1"
    ts2 = pd.Series(np.random.randn(100), index=dates2, name="ts2")
    ts2.index.name = "date_2"
    return ts1, ts2


@pytest.fixture
def binned_value_ts_pair():
    """Time series with integer-ranged values suitable for pd.cut binning.

    Values range from 0-10, suitable for bin_size=3 default in get_ranges_df.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Values uniformly distributed in 0-10 range
    ts1 = pd.Series(np.random.uniform(0, 10, 100), index=dates, name="ts1")
    ts1.index.name = "date_1"
    ts2 = pd.Series(np.random.uniform(0, 10, 100), index=dates, name="ts2")
    ts2.index.name = "date_2"
    return ts1, ts2
