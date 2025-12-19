"""Scale Dependent Correlation analysis."""

import warnings
from typing import TYPE_CHECKING, Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns

from sdcpy.core import compute_sdc
from sdcpy.io import load_from_excel, save_to_excel
from sdcpy.plotting import combi_plot as _combi_plot

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
    ):
        self.way = (
            "one-way" if ts2 is None else "two-way"
        )  # One-way SDC inferred if no ts2 is provided
        ts2 = ts1.copy() if self.way == "one-way" else ts2
        # TODO: As mentioned in (#4), we should make
        if not isinstance(ts1, pd.Series):
            ts1 = pd.Series(
                ts1, index=pd.date_range(start="2000-01-01", periods=len(ts1), freq="D")
            )
        if not isinstance(ts2, pd.Series):
            ts2 = pd.Series(
                ts2, index=pd.date_range(start="2000-01-01", periods=len(ts2), freq="D")
            )
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

    def two_way_plot(self, alpha: float = 0.05, **kwargs) -> "ggplot":
        """Plot two-way SDC heatmap using plotnine."""
        fragment_size = int(self.sdc_df.iloc[0]["stop_1"] - self.sdc_df.iloc[0]["start_1"])
        f = (
            self.sdc_df.loc[lambda dd: dd.p_value < alpha]
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
        bin_size: int = 3,
        alpha: float = 0.05,
        min_bin=None,
        max_bin=None,
        threshold: float = 0.0,
        ts: int = 1,
    ):
        ts_series = self.ts1 if ts == 1 else self.ts2
        min_bin = int(np.floor(ts_series.min())) if min_bin is None else min_bin
        max_bin = int(np.ceil(ts_series.max())) if max_bin is None else max_bin
        name = ts_series.name
        df = (
            self.sdc_df.dropna()
            .assign(
                date_range=lambda dd: dd[f"date_{ts}"].apply(
                    lambda x: pd.date_range(x, x + pd.to_timedelta(self.fragment_size, unit="days"))
                )
            )[["r", "p_value", "date_range"]]
            .explode("date_range")
            .rename(columns={"date_range": "date"})
            .reset_index()
            .rename(columns={"index": "comparison_id"})
            .merge(ts_series.reset_index().rename(columns={f"date_{ts}": "date", name: "value"}))
            .assign(
                cat_value=lambda dd: pd.cut(
                    dd.value, bins=list(range(min_bin, max_bin + bin_size, bin_size)), precision=0
                )
            )
            .groupby(["comparison_id"])
            .apply(lambda dd: dd.cat_value.value_counts(True), include_groups=False)
            .loc[lambda x: x > threshold]
            .reset_index()
            .rename(columns={"level_1": "cat_value"}, errors="ignore")
            .drop(columns=["proportion"], errors="ignore")
            .merge(
                self.sdc_df.reset_index().rename(columns={"index": "comparison_id"})[
                    ["r", "p_value", "comparison_id"]
                ]
            )
            .assign(significant=lambda dd: dd.p_value < alpha)
            .assign(
                direction=lambda dd: (
                    dd.significant.astype(int) * ((dd.r > 0).astype(int) + 1)
                ).replace({0: "NS", 1: "Negative", 2: "Positive"})
            )
            .assign(
                direction=lambda dd: pd.Categorical(
                    dd.direction, categories=["Positive", "Negative", "NS"], ordered=True
                )
            )
            .groupby("cat_value")
            .apply(
                lambda dd: dd["direction"].value_counts().rename("counts").reset_index(),
                include_groups=False,
            )
            .reset_index()
            .drop(columns="level_1")
            .rename(columns={"index": "direction"})
            .pipe(
                lambda dd: dd.merge(
                    dd.groupby("cat_value", as_index=False)["counts"]
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
    ):
        df = self.get_ranges_df(**kwargs)
        fig = (
            p9.ggplot(df)
            + p9.aes("cat_value", "counts", fill="direction")
            + p9.geom_col(alpha=0.8)
            + p9.theme(figure_size=figsize, axis_text_x=p9.element_text(rotation=45))
            + p9.scale_fill_manual(["#3f7f93", "#da3b46", "#4d4a4a"])
            + p9.labs(x=xlabel, y="Number of Comparisons", fill="R")
        )

        if add_text_label:
            if df.loc[df.direction == "Positive"].loc[df.counts > 0].size > 0:
                fig += p9.geom_text(
                    p9.aes(label="label", x="cat_value", y="n + max(n) * .15"),
                    inherit_aes=False,
                    size=9,
                    data=df.loc[df.direction == "Positive"].loc[df.counts > 0],
                    color="#3f7f93",
                )
            if df.loc[df.direction == "Negative"].loc[df.counts > 0].size > 0:
                fig += p9.geom_text(
                    p9.aes(label="label", x="cat_value", y="n + max(n) * .05"),
                    inherit_aes=False,
                    size=9,
                    data=df.loc[df.direction == "Negative"].loc[df.counts > 0],
                    color="#da3b46",
                )

        return fig

    def plot_consecutive(self, alpha: float = 0.05, **kwargs) -> "ggplot":
        f = (
            self.sdc_df.loc[lambda dd: dd.p_value < alpha]
            # Here I make groups of consecutive significant values and report the longest for each lag.
            .groupby("lag", as_index=True)
            .apply(
                lambda gdf: gdf.sort_values("start_1")
                .assign(group=lambda dd: (dd.start_1 != dd.start_1.shift(1) + 1).cumsum())
                .groupby(["group"])
                .size()
                .max(),
                include_groups=False,
            )
            .rename("Max Consecutive steps")
            .reset_index()
            .pipe(
                lambda dd: p9.ggplot(dd)
                + p9.aes("lag", "Max Consecutive steps")
                + p9.geom_col()
                + p9.theme(**kwargs)
                + p9.labs(x="Lag [days]")
            )
        )

        return f

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
        Create a combination plot showing SDC analysis results.

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

    def dominant_lags_plot(self, alpha: float = 0.05, ylabel: str = "", **kwargs) -> "MplFigure":
        fig, ax = plt.subplots(**kwargs)
        df = (
            self.sdc_df.loc[lambda dd: dd.p_value < alpha]
            .groupby("date_1")
            .apply(
                lambda dd: dd.loc[
                    lambda ddd: ((ddd.r == ddd.r.max()) & (ddd.r > 0))
                    | ((ddd.r == ddd.r.min()) & (ddd.r < 0))
                ],
                include_groups=False,
            )
            .reset_index(level=0)
            .groupby(["date_1"])
            .apply(
                lambda dd: dd.loc[dd["lag"].abs() == dd["lag"].abs().min()], include_groups=False
            )
            .reset_index(level=0)
            .assign(
                date_1=lambda dd: dd.date_1 + pd.to_timedelta(self.fragment_size // 2, unit="days")
            )
            .assign(lag=lambda dd: dd.lag.abs())
        )
        self.ts1.plot(ax=ax, color="black")
        ax2 = ax.twinx()
        sns.scatterplot(
            data=df,
            x="date_1",
            y="r",
            hue="lag",
            legend="full",
            alpha=0.7,
            ax=ax2,
            palette="inferno_r",
        )
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(
            bbox_to_anchor=(1.3, 1.05),
            ncol=1,
            frameon=True,
            columnspacing=0.2,
            handles=[h for i, h in enumerate(handles[1:]) if i % 3 == 1],
            labels=[label for i, label in enumerate(labels[1:]) if i % 3 == 1],
            title="Lag",
        )

        ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax2.grid(which="major")
        ax2.set_xlabel("")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ylabel else "Value")
        ax2.set_ylabel("Max/Min r")

        return fig

    def single_shift_plot(self, shift: int) -> "MplFigure":
        raise NotImplementedError("single_shift_plot is not yet implemented")
