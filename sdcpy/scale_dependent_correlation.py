"""Scale Dependent Correlation analysis."""

import warnings
from typing import TYPE_CHECKING, Callable, Optional, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
from matplotlib import ticker

from sdcpy.core import compute_sdc
from sdcpy.io import load_from_excel, save_to_excel
from sdcpy.plotting import plot_two_way_sdc

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotnine.ggplot import ggplot

# Re-export core functions for backward compatibility
__all__ = [
    "SDCAnalysis",
    "compute_sdc",
    "plot_two_way_sdc",
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
        return plot_two_way_sdc(self.sdc_df, alpha, **kwargs)

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
        date_fmt: str = "%m-%d",
        align: str = "center",
        max_lag: int = np.inf,
        min_lag: int = -np.inf,
        labels_fontsize: int = 12,
        wspace: float = 1.0,
        hspace: float = 1.0,
        show_colorbar: bool = True,
        **kwargs,
    ) -> "MplFigure":
        # Setting up parameters
        title = f"SDC plot (s = {self.fragment_size})" if title is None else title
        align = align.lower()
        if align not in ["left", "center", "right"]:
            warnings.warn(
                f'Alignment method "{align}" not recognized, defaulting to center alignment.',
                stacklevel=2,
            )
            align = "center"
        offset = self.fragment_size // 2 if align == "center" else self.fragment_size
        left_offset = 0 if align == "left" else offset
        right_offset = 0 if align == "right" else offset

        date_format = mdates.DateFormatter(date_fmt)
        fig = plt.figure(**kwargs)
        # We are organizing the grid in a 5 x 5 matrix so that (TT=Title, HM: Heatmap, TS1/TS2: Time-Series 1/2,
        # MC: Max Correlations):
        # TT TT TT TT TT
        # NA TS1 TS1 NA NA
        # TS2 HM HM MC2 CB
        # TS2 HM HM MC2 CB
        # NA MC1 MC1 NA NA
        if min_lag < 0 < max_lag:
            gs = fig.add_gridspec(
                5, 5, height_ratios=[0.15, 1.5, 2, 2, 1.5], width_ratios=[1.5, 2, 2, 1, 0.2]
            )
        elif min_lag < 0:
            gs = fig.add_gridspec(
                5, 4, height_ratios=[0.15, 1.5, 2, 2, 1], width_ratios=[1.5, 2, 2, 0.3]
            )
        elif max_lag > 0:
            gs = fig.add_gridspec(
                4, 5, height_ratios=[0.15, 1.5, 2, 1], width_ratios=[1, 2, 2, 1, 0.2]
            )
        else:
            raise ValueError("Range of lags to be considered should be bigger than 1")
        # Time series 1
        ts1 = fig.add_subplot(gs[1, 1:3])
        ts1.plot(self.ts1, color="black")
        # Time series 2
        ts2 = fig.add_subplot(gs[2:4, 0])
        plt.plot(self.ts2, self.ts2.reset_index()["date_2"], color="black")
        # Heat map
        hm = fig.add_subplot(gs[2:4, 1:3])

        (
            self.sdc_df.loc[lambda dd: (dd.lag <= max_lag) & (dd.lag >= min_lag)].pipe(
                lambda dd: sns.heatmap(
                    dd.pivot(index="date_2", columns="date_1", values="r"),
                    cbar=False,
                    mask=dd.pivot(index="date_2", columns="date_1", values="p_value") > alpha,
                    cmap=sns.diverging_palette(10, 220, sep=80, n=20),
                    ax=hm,
                )
            )
        )
        # Add identity line to ease shift visualization
        identity_len = min(len(self.ts1), len(self.ts2)) - self.fragment_size + 1
        plt.plot(range(identity_len), range(identity_len), linestyle=":", color="black", alpha=0.8)
        # Correct and format ticks, labels, grids
        # Hide Heatmap labels and ticks
        hm.set_xlabel("")
        hm.set_ylabel("")
        plt.setp(hm.get_yticklabels(), visible=False)
        plt.setp(hm.get_xticklabels(), visible=False)
        # Each dot in the heatmap represents a `fragment_size` long region of the time-series, so we need to choose how
        # to represent each dot because the heatmap axis are `fragment_size` shorter than the time-series axis.
        # Alignment parameter comes then into play:
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        max_r = max_r if max_r is not None else self.sdc_df.r.abs().max()
        hm.set_xlim(xmin - left_offset, xmax + right_offset)
        hm.set_ylim(ymin + right_offset, ymax - left_offset)
        trans_x = hm.get_xaxis_transform()
        trans_y = hm.get_yaxis_transform()

        hm.plot(
            [-self.fragment_size / 2, self.fragment_size / 2],
            [1.0, 1.0],
            color="k",
            transform=trans_x,
            clip_on=False,
            linewidth=5,
            solid_capstyle="butt",
        )
        hm.plot(
            [0, 0],
            [-self.fragment_size / 2, self.fragment_size / 2],
            color="k",
            transform=trans_y,
            clip_on=False,
            linewidth=5,
            solid_capstyle="butt",
        )

        hm.annotate(
            f"$s={self.fragment_size}$",
            xy=(self.fragment_size / 2 + 5, 0.99),
            xycoords=trans_x,
            size=9,
        )

        # Handle TS1 labels and ticks
        ts1.xaxis.set_major_formatter(date_format)
        ts1.set_xlim(self.ts1.index[0], self.ts1.index[-1])
        ts1.xaxis.set_major_locator(ticker.MultipleLocator(len(self.ts1) / 2))
        ts1.grid(True, which="major")
        ts1.set_xlabel(xlabel, fontsize=labels_fontsize)
        ts1.xaxis.set_label_position("top")

        # Handle TS2 labels and ticks
        ts2.yaxis.set_major_formatter(date_format)
        ts2.set_ylim(self.ts2.index[0], self.ts2.index[-1])
        ts2.yaxis.set_major_locator(ticker.MultipleLocator(len(self.ts2) / 2))
        ts2.grid(True, which="major")
        ts2.yaxis.tick_right()
        ts2.invert_xaxis()
        ts2.invert_yaxis()
        ts2.set_ylabel(ylabel, fontsize=labels_fontsize)
        plt.setp(ts2.get_yticklabels(), visible=True, rotation=90, va="center")
        gs.update(wspace=wspace, hspace=hspace)
        # Max Correlations
        colors = {"Max $r$": "#3f7f93", "Min $r$ (abs)": "#da3b46"}
        if min_lag < 0:
            mc1 = fig.add_subplot(gs[-1, 1:3])
            (
                self.sdc_df.loc[lambda dd: dd.p_value < alpha]
                .loc[lambda dd: (dd.lag <= max_lag) & (dd.lag >= min_lag)]
                .groupby("date_1")
                .apply(
                    lambda dd: dd.loc[dd["r"].abs() == dd["r"].abs().max()].loc[
                        lambda d: d["lag"] == d["lag"].min()
                    ],
                    include_groups=False,
                )
                .reset_index(level=0)
                .groupby("date_1")
                .agg(
                    r_max=("r", lambda x: x.where(x > 0).max()),
                    r_min=("r", lambda x: abs(x.where(x < 0).min())),
                )
                .rename(columns={"r_max": "Max $r$", "r_min": "Min $r$ (abs)"})
                .reset_index()
                .melt("date_1")
                .assign(date_1=lambda dd: dd.date_1 + pd.to_timedelta(f"{left_offset} days"))
                .assign(color=lambda dd: dd.variable.apply(lambda x: colors[x]))
                .plot.scatter(
                    x="date_1",
                    y="value",
                    c="color",
                    ax=mc1,
                    style="-",
                    alpha=1,
                    colorbar=False,
                    s=10,
                )
            )
            plt.setp(mc1.get_xticklabels(), visible=False)
            mc1.set_xlabel("")
            mc1.set_ylabel("Max abs($\\rho$)")
            mc1.yaxis.set_label_position("right")
            mc1.set_xlim(self.ts1.index[0], self.ts1.index[-1])
            mc1.set_ylim(0, 1.05)
            mc1.grid(True, which="major")
            mc1.set_yticks([0, 0.5, 1])
        if max_lag > 0:
            mc2 = fig.add_subplot(gs[2:4, 3])
            (
                self.sdc_df.loc[lambda dd: dd.p_value < alpha]
                .loc[lambda dd: (dd.lag <= max_lag) & (dd.lag >= min_lag)]
                .groupby("date_2")
                .agg(
                    r_max=("r", lambda x: x.where(x > 0).max()),
                    r_min=("r", lambda x: abs(x.where(x < 0).min())),
                )
                .rename(columns={"r_max": "Max $r$", "r_min": "Min $r$ (abs)"})
                .reset_index()
                .melt("date_2")
                .assign(date_2=lambda dd: dd.date_2 + pd.to_timedelta(f"{left_offset} days"))
                .assign(color=lambda dd: dd.variable.apply(lambda x: colors[x]))
                .plot.scatter(
                    x="value", y="date_2", c="color", ax=mc2, style="o", alpha=0.7, colorbar=False
                )
            )
            plt.setp(mc2.get_yticklabels(), visible=False)
            mc2.set_xlabel("Max / Min $r$")
            mc2.set_ylabel("")
            mc2.grid(True, which="major")
            mc2.set_xlim(1.05, 0)
            mc2.set_ylim(self.ts2.index[-1], self.ts2.index[0])

        # Colorbar
        if show_colorbar:
            cax = fig.add_subplot(gs[2:4, -1])
        color_mesh = hm.get_children()[0]
        color_mesh.set_clim(-max_r, max_r)
        if show_colorbar:
            fig.colorbar(color_mesh, cax=cax, label=f"{self.method.capitalize()}'s $\\rho$")
        fig.suptitle(title)

        return fig

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
