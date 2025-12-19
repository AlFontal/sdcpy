"""Example script demonstrating SDCAnalysis with synthetic data."""

import numpy as np
import pandas as pd

from sdcpy import SDCAnalysis


def tc_signal(i):
    """Synthetic signal from original SDC paper (Rodriguez-Arias & Rodó, 2004).

    Creates a time series with a transient sinusoidal pattern
    embedded in noise between indices 63-169.
    """
    error = np.random.normal()
    if i < 63 or i > 169:
        return error
    else:
        return np.sin(2 * np.pi * (1 / 37) * i) + 0.6 * error


if __name__ == "__main__":
    np.random.seed(42)

    # Generate two synthetic time series with transient correlations
    ts1 = pd.Series([tc_signal(i) for i in range(250)], name="ts1")
    ts2 = pd.Series([tc_signal(i) for i in range(250)], name="ts2")

    # Run SDC analysis with fragment size of 50
    sdc = SDCAnalysis(ts1, ts2, fragment_size=50, n_permutations=99)

    # Generate combination plot
    fig = sdc.combi_plot(xlabel="$TS_1$", ylabel="$TS_2$")
    fig.savefig("sdc_example.png", dpi=300, bbox_inches="tight")
    print("Saved: sdc_example.png")
