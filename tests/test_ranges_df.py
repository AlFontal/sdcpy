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
