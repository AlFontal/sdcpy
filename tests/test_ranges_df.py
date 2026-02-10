import numpy as np
import pandas as pd

from sdcpy import SDCAnalysis


def test_get_ranges_df_lag_filtering():
    """Should filter ranges_df based on min_lag and max_lag."""
    # Create simple predictable data
    ts1 = pd.Series(np.arange(20))
    ts2 = pd.Series(np.arange(20))

    sdc = SDCAnalysis(ts1, ts2, fragment_size=5, n_permutations=9)

    # Get ranges with lag filtering
    ranges_df = sdc.get_ranges_df(min_lag=-2, max_lag=2)

    # Verify we got some data
    assert len(ranges_df) > 0

    # Manually check the logic
    # The sdc_df should be filtered before merging
    expected_count = sdc.sdc_df.query("lag >= -2 & lag <= 2").shape[0]
    # ranges_df is aggregated by cat_value and direction, so sum of counts should match total number of valid comparisons
    assert ranges_df["counts"].sum() == expected_count

    # Test extreme filtering
    ranges_df_strict = sdc.get_ranges_df(min_lag=0, max_lag=0)
    assert ranges_df_strict["counts"].sum() == sdc.sdc_df.query("lag == 0").shape[0]


def test_get_ranges_df_uses_fragment_start_window():
    """Fragment values should be aggregated from [start:start+fragment_size]."""
    ts = pd.Series(np.arange(6, dtype=float))
    sdc_df = pd.DataFrame(
        {
            "start_1": [0.0, 1.0],
            "stop_1": [3.0, 4.0],
            "start_2": [0.0, 1.0],
            "stop_2": [3.0, 4.0],
            "lag": [0.0, 0.0],
            "r": [0.9, 0.9],
            "p_value": [0.01, 0.01],
            "date_1": [0, 1],
            "date_2": [0, 1],
        }
    )
    sdc = SDCAnalysis(ts1=ts, ts2=ts, fragment_size=3, n_permutations=9, sdc_df=sdc_df)

    ranges_df = sdc.get_ranges_df(ts=1, bin_size=1, alpha=0.05, min_bin=0, max_bin=4)
    positive_bins = ranges_df.loc[
        (ranges_df["direction"] == "Positive") & (ranges_df["counts"] > 0), "cat_value"
    ].tolist()

    # Means for starts 0 and 1 are 1.0 and 2.0, so populated bins must end at 1 and 2.
    assert {interval.right for interval in positive_bins} == {1.0, 2.0}
