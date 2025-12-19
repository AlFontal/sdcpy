"""Plotting functions for Scale Dependent Correlation analysis."""

from typing import TYPE_CHECKING

import pandas as pd
import plotnine as p9

if TYPE_CHECKING:
    from plotnine.ggplot import ggplot


def plot_two_way_sdc(sdc_df: pd.DataFrame, alpha: float = 0.05, **kwargs) -> "ggplot":
    """
    Plots the results of a SDC analysis for a fixed window size in a 2D figure.

    In a similar fashion to a recurrence plot, x and y axes represent the start index of the x and y sequences. Only
    results with a p_value < alpha are shown, while controlling the alpha as a function of the intensity of the score
    and the color as a function of the sign of the established relationship.

    Parameters
    ----------
    sdc_df
        Data frame as outputted by `compute_sdc` which will be used to plot the results.
    alpha
        Significance threshold. Only values with a score < alpha will be plotted
    kwargs
        Keyword arguments to pass to `plotnine.theme` to customize the plot.
    Returns
    -------
    p9.ggplot.ggplot
        Plot
    """
    fragment_size = int(sdc_df.iloc[0]["stop_1"] - sdc_df.iloc[0]["start_1"])
    f = (
        sdc_df.loc[lambda dd: dd.p_value < alpha]
        .assign(r_str=lambda dd: dd["r"].apply(lambda x: "$r > 0$" if x > 0 else "$r < 0$"))
        .pipe(
            lambda dd: p9.ggplot(dd)
            + p9.aes("start_1", "start_2", fill="r_str", alpha="abs(r)")
            + p9.geom_tile()
            + p9.scale_fill_manual(["#da2421", "black"])
            + p9.scale_y_reverse()
            + p9.theme(**kwargs)
            + p9.guides(alpha=False)
            + p9.labs(
                x="$X_i$",
                y="$Y_j$",
                fill="$r$",
                title=f"Two-Way SDC plot for $S = {fragment_size}$"
                + r" and $\alpha =$"
                + f"{alpha}",
            )
        )
    )

    return f
