"""Scale Dependent Correlation analysis."""

from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd

from sdcpy.core import compute_sdc
from sdcpy.io import load_from_excel, save_to_excel
from sdcpy.plotting import combi_plot as _combi_plot
from sdcpy.plotting import plot_range_comparison as _plot_range_comparison

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotnine.ggplot import ggplot

# Re-export core functions for backward compatibility
__all__ = [
    "SDCAnalysis",
    "compute_sdc",
]


class SDCAnalysis:
    def __init__(
        self,
        ts1: np.ndarray,
        ts2: np.ndarray = None,
        fragment_size: int = 7,
        n_permutations: int = 99,
        method: Union[str, Callable] = "pearson",
        two_tailed: bool = True,
        permutations: bool = True,
        sdc_df: Optional[pd.DataFrame] = None,
        min_lag: int = -np.inf,
        max_lag: int = np.inf,
        max_memory_gb: float = 2.0,
    ):
        self.way = (
            "one-way" if ts2 is None else "two-way"
        )  # One-way SDC inferred if no ts2 is provided
        ts2 = ts1.copy() if self.way == "one-way" else ts2
        if not isinstance(ts1, pd.Series):
            ts1 = pd.Series(ts1)
        if not isinstance(ts2, pd.Series):
            ts2 = pd.Series(ts2)
        min_date = max(ts1.index.min(), ts2.index.min())
        max_date = min(ts1.index.max(), ts2.index.max())
        self.ts1 = ts1[min_date:max_date]
        self.ts2 = ts2[min_date:max_date]
        self.fragment_size = fragment_size
        self.n_permutations = n_permutations
        self.ts1.index.name = "date_1"
        self.ts2.index.name = "date_2"
        if sdc_df is not None:
            self.sdc_df = sdc_df
        else:
            self.sdc_df = compute_sdc(
                self.ts1,
                self.ts2,
                fragment_size,
                n_permutations,
                method,
                two_tailed,
                permutations,
                min_lag,
                max_lag,
                max_memory_gb,
            ).assign(
                date_1=lambda dd: dd.start_1.map(self.ts1.reset_index().to_dict()["date_1"]),
                date_2=lambda dd: dd.start_2.map(self.ts2.reset_index().to_dict()["date_2"]),
            )
            self.sdc_df = (
                self.sdc_df.loc[lambda dd: dd.start_1 != dd.start_2]
                if self.way == "one-way"
                else self.sdc_df
            )
        self.method = method

    def to_excel(self, filename: str):
        save_to_excel(
            self.sdc_df,
            self.ts1,
            self.ts2,
            self.fragment_size,
            self.n_permutations,
            self.method,
            filename,
        )

    @classmethod
    def from_excel(cls, filename: str):
        data = load_from_excel(filename)
        return cls(
            ts1=data["ts1"],
            ts2=data["ts2"],
            fragment_size=data["fragment_size"],
            n_permutations=data["n_permutations"],
            method=data["method"],
            sdc_df=data["sdc_df"],
        )

    def get_ranges_df(
        self,
        bin_size: int | float = 1,
        alpha: float = 0.05,
        min_bin: int | float | None = None,
        max_bin: int | float | None = None,
        ts: int = 1,
        agg_func: str = "mean",
        min_lag: int = -np.inf,
        max_lag: int = np.inf,
    ) -> pd.DataFrame:
        """
        Compute correlation direction statistics by value ranges.

        For each SDC comparison, computes the aggregate value (mean by default) of ts1 or ts2
        during that fragment, bins those values, then counts how many correlations in each
        bin were positive, negative, or not significant.

        Parameters
        ----------
        bin_size : int | float, default=1
            Width of each value bin.
        alpha : float, default=0.05
            Significance level for classifying correlations.
        min_bin : int | float, optional
            Lower bound for binning. Defaults to floor(min(ts)) aligned to bin_size.
        max_bin : int | float, optional
            Upper bound for binning. Defaults to ceil(max(ts)) aligned to bin_size.
        ts : int, default=1
            Which time series to analyze (1 for ts1, 2 for ts2).
        agg_func : str, default="mean"
            Aggregation function to summarize values in each fragment.
            Options: "mean", "median", "min", "max".
        min_lag : int, default=-np.inf
            Minimum lag to consider.
        max_lag : int, default=np.inf
            Maximum lag to consider.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - cat_value: categorical bin (e.g., "(0, 3]")
            - direction: "Positive", "Negative", or "NS" (not significant)
            - counts: number of comparisons in this bin with this direction
            - n: total comparisons in this bin
            - freq: proportion (counts / n)
            - label: formatted percentage string
        """
        ts_series = self.ts1 if ts == 1 else self.ts2

        # Compute rolling aggregate for fragments
        # This gives the aggregate value for each fragment starting at each index
        if agg_func == "mean":
            fragment_values = ts_series.rolling(window=self.fragment_size, min_periods=1).mean()
        elif agg_func == "median":
            fragment_values = ts_series.rolling(window=self.fragment_size, min_periods=1).median()
        elif agg_func == "min":
            fragment_values = ts_series.rolling(window=self.fragment_size, min_periods=1).min()
        elif agg_func == "max":
            fragment_values = ts_series.rolling(window=self.fragment_size, min_periods=1).max()
        else:
            raise ValueError(
                f"Unknown agg_func: {agg_func}. Use 'mean', 'median', 'min', or 'max'."
            )

        # Create lookup from date to fragment aggregate value
        fragment_values_df = fragment_values.reset_index()
        fragment_values_df.columns = [f"date_{ts}", "fragment_value"]

        # Join sdc_df with fragment values first
        df = (
            self.sdc_df.dropna()
            .query("lag >= @min_lag & lag <= @max_lag")
            .merge(fragment_values_df, on=f"date_{ts}", how="left")
        )

        # Compute bin bounds from the *filtered* fragment aggregates
        # This ensures we don't create empty bins for ranges that were filtered out
        current_values = df["fragment_value"]

        # Snap min/max to grid defined by bin_size
        if min_bin is None:
            min_val = current_values.min()
            min_bin = np.floor(min_val / bin_size) * bin_size

        if max_bin is None:
            max_val = current_values.max()
            max_bin = np.ceil(max_val / bin_size) * bin_size

        # Assign categories using data-dependent bins
        # Use np.arange to support float bin_size
        # Add small epsilon to max range to ensure inclusion due to floating point precision
        df = (
            df.assign(
                cat_value=lambda dd: pd.cut(
                    dd.fragment_value,
                    bins=np.arange(min_bin, max_bin + bin_size + 1e-10, bin_size),
                    precision=1,  # Improved precision for float bins
                    include_lowest=True,
                )
            )
            .assign(significant=lambda dd: dd.p_value < alpha)
            .assign(
                direction=lambda dd: np.where(
                    ~dd.significant,
                    "NS",
                    np.where(dd.r > 0, "Positive", "Negative"),
                )
            )
            .assign(
                direction=lambda dd: pd.Categorical(
                    dd.direction, categories=["Positive", "Negative", "NS"], ordered=True
                )
            )
            .groupby(["cat_value", "direction"], observed=False)
            .size()
            .reset_index(name="counts")
            .pipe(
                lambda dd: dd.merge(
                    dd.groupby("cat_value", as_index=False, observed=False)["counts"]
                    .sum()
                    .rename(columns={"counts": "n"}),
                    on="cat_value",
                )
            )
            .assign(freq=lambda dd: (dd["counts"] / dd["n"]).fillna(0))
            .assign(label=lambda dd: (dd["freq"] * 100).round(1).astype(str) + " %")
        )

        return df

    def plot_range_comparison(
        self,
        xlabel: str = "",
        figsize: tuple[int, int] = (7, 3),
        add_text_label: bool = True,
        **kwargs,
    ) -> "ggplot":
        """
        Create a bar chart showing correlation directions by value ranges.

        See `sdcpy.plotting.plot_range_comparison` for full parameter documentation.
        """
        df = self.get_ranges_df(**kwargs)
        return _plot_range_comparison(
            ranges_df=df,
            xlabel=xlabel,
            figsize=figsize,
            add_text_label=add_text_label,
        )

    def combi_plot(
        self,
        alpha: float = 0.05,
        xlabel: str = "",
        ylabel: str = "",
        title: str = None,
        max_r: float = None,
        date_fmt: str = None,
        align: str = "center",
        max_lag: int = np.inf,
        min_lag: int = -np.inf,
        fontsize: int = 9,
        tick_fontsize: int = None,
        label_fontsize: int = None,
        colorbar_fontsize: int = None,
        wspace: float = 0.2,
        hspace: float = 0.2,
        show_colorbar: bool = True,
        show_ts2: bool = True,
        metric_label: str = None,
        n_ticks: int = 6,
        figsize: tuple = (6, 6),
        dpi: int = 150,
        **kwargs,
    ) -> "MplFigure":
        """
        Create a combined plot showing two-way SDC analysis results.

        See `sdcpy.plotting.combi_plot` for full parameter documentation.
        """
        return _combi_plot(
            ts1=self.ts1,
            ts2=self.ts2,
            sdc_df=self.sdc_df,
            fragment_size=self.fragment_size,
            method=self.method,
            alpha=alpha,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            max_r=max_r,
            date_fmt=date_fmt,
            align=align,
            max_lag=max_lag,
            min_lag=min_lag,
            fontsize=fontsize,
            tick_fontsize=tick_fontsize,
            label_fontsize=label_fontsize,
            colorbar_fontsize=colorbar_fontsize,
            wspace=wspace,
            hspace=hspace,
            show_colorbar=show_colorbar,
            show_ts2=show_ts2,
            metric_label=metric_label,
            n_ticks=n_ticks,
            figsize=figsize,
            dpi=dpi,
            **kwargs,
        )
