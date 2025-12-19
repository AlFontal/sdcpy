"""Scale Dependent Correlation analysis."""

import warnings
from typing import TYPE_CHECKING, Callable, Optional, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns

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


def _determine_frequency_info(index: pd.Index) -> tuple[str, float, str]:
    """
    Determine frequency information for time series index.

    Parameters
    ----------
    index : pandas.Index
        The index of the time series

    Returns
    -------
    tuple: (freq_str, freq_mult, freq_unit)
        freq_str: Human-readable frequency description (e.g., "days", "weeks")
        freq_mult: Numeric multiplier for the frequency
        freq_unit: Pandas time unit ('D', 'W', etc.)
    """
    # Check if index is datetime-like
    is_datetime_index = pd.api.types.is_datetime64_any_dtype(index)

    if not is_datetime_index:
        # Non-datetime index - use integer positioning
        return "periods", 1, "D"

    frequency = pd.infer_freq(index)
    if frequency:
        import re

        # Handle daily frequencies (1D, 2D, 3D, etc.)
        if re.match(r"^[0-9]*D$", frequency):
            freq_mult = 1
            match = re.match(r"^([0-9]+)D$", frequency)
            if match:
                freq_mult = int(match.group(1))
            freq_str = "days" if freq_mult == 1 else f"{freq_mult}-day periods"
            return freq_str, freq_mult, "D"

        # Handle weekly frequencies (1W, 2W, etc.)
        elif re.match(r"^[0-9]*W", frequency):
            freq_mult = 1
            match = re.match(r"^([0-9]+)W", frequency)
            if match:
                freq_mult = int(match.group(1))
            freq_str = "weeks" if freq_mult == 1 else f"{freq_mult}-week periods"
            return freq_str, freq_mult * 7, "D"  # Convert to days for timedelta

        # Handle monthly frequencies
        elif frequency.startswith("M") or frequency.startswith("MS"):
            freq_mult = 1
            match = re.match(r"^([0-9]+)", frequency)
            if match:
                freq_mult = int(match.group(1))
            freq_str = "months" if freq_mult == 1 else f"{freq_mult}-month periods"
            return freq_str, freq_mult * 30.44, "D"

        # Handle yearly frequencies
        elif frequency.startswith("Y") or frequency.startswith("A"):
            freq_mult = 1
            match = re.match(r"^([0-9]+)", frequency)
            if match:
                freq_mult = int(match.group(1))
            freq_str = "years" if freq_mult == 1 else f"{freq_mult}-year periods"
            return freq_str, freq_mult * 365.25, "D"

    # For irregular frequency, estimate from median difference
    try:
        median_diff = index.to_series().diff().median()
        if hasattr(median_diff, "days"):
            freq_mult = median_diff.days
        else:
            freq_mult = 1
    except Exception:
        freq_mult = 1

    return "periods", max(1, freq_mult), "D"


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
        date_fmt: str = None,
        align: str = "center",
        max_lag: int = np.inf,
        min_lag: int = -np.inf,
        labels_fontsize: int = 12,
        wspace: float = 1.0,
        hspace: float = 1.0,
        show_colorbar: bool = True,
        show_ts2: bool = True,
        metric_label: str = None,
        **kwargs,
    ) -> "MplFigure":
        """
        Create a combination plot showing SDC analysis results.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for masking non-significant correlations.
        xlabel : str, default ""
            Label for time series 1 (top axis).
        ylabel : str, default ""
            Label for time series 2 (left axis).
        title : str, optional
            Plot title. Defaults to "SDC plot (s = {fragment_size} {freq_str})".
        max_r : float, optional
            Maximum absolute correlation for color scale. Auto-detected if None.
        date_fmt : str, optional
            Date format string. Auto-detected based on time series frequency.
        align : str, default "center"
            Alignment of heatmap cells: "left", "center", or "right".
        max_lag : int, default np.inf
            Maximum lag to display.
        min_lag : int, default -np.inf
            Minimum lag to display.
        labels_fontsize : int, default 12
            Font size for axis labels.
        wspace : float, default 1.0
            Width space between subplots.
        hspace : float, default 1.0
            Height space between subplots.
        show_colorbar : bool, default True
            Whether to show the colorbar.
        show_ts2 : bool, default True
            Whether to show time series 2 on the left side.
        metric_label : str, optional
            Label for the correlation metric. Defaults to method name.
        **kwargs
            Additional keyword arguments passed to plt.figure().

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        # Get frequency information for proper labeling and offsets
        freq_str, freq_mult, freq_unit = _determine_frequency_info(self.ts1.index)

        # Auto-detect date format based on frequency
        if date_fmt is None:
            if freq_mult >= 365:
                date_fmt = "%Y"
            elif freq_mult >= 28:
                date_fmt = "%Y-%m"
            elif freq_mult >= 7:
                date_fmt = "%m-%d"
            else:
                date_fmt = "%m-%d"

        # Set default title with frequency info
        if title is None:
            title = f"SDC plot (s = {self.fragment_size} {freq_str})"

        # Set default metric label
        if metric_label is None:
            metric_labels = {
                "pearson": "Pearson's $r$",
                "spearman": "Spearman's $\\rho$",
            }
            metric_label = metric_labels.get(self.method, self.method.capitalize())

        # Validate alignment
        align = align.lower()
        if align not in ["left", "center", "right"]:
            warnings.warn(
                f'Alignment method "{align}" not recognized, defaulting to center alignment.',
                stacklevel=2,
            )
            align = "center"

        # Calculate offsets
        offset = self.fragment_size // 2 if align == "center" else self.fragment_size
        left_offset = 0 if align == "left" else offset
        right_offset = 0 if align == "right" else offset

        # Calculate timedelta offset using detected frequency
        timedelta_offset = pd.to_timedelta(left_offset * freq_mult, unit=freq_unit)

        date_format = mdates.DateFormatter(date_fmt)
        sdc_df = self.sdc_df.copy()
        fig = plt.figure(**kwargs)

        # Dynamic grid layout based on lag range and show_ts2
        # Grid layout:
        # TT TT TT TT TT (title row)
        # NA TS1 TS1 NA NA (time series 1)
        # TS2 HM HM MC2 CB (heatmap + max corr 2 + colorbar)
        # TS2 HM HM MC2 CB
        # NA MC1 MC1 NA NA (max corr 1)
        ts2_col = 1 if show_ts2 else 0

        if min_lag < 0 < max_lag:
            width_ratios = [1, 2, 2, 1, 0.2] if show_ts2 else [2, 2, 1, 0.2]
            gs = fig.add_gridspec(
                5, len(width_ratios), height_ratios=[0.15, 1, 2, 2, 1], width_ratios=width_ratios
            )
            hm_cols = slice(ts2_col, ts2_col + 2)
            mc2_col = ts2_col + 2
            cb_col = -1
        elif min_lag < 0:
            width_ratios = [1, 2, 2, 0.3] if show_ts2 else [2, 2, 0.3]
            gs = fig.add_gridspec(
                5, len(width_ratios), height_ratios=[0.15, 1, 2, 2, 1], width_ratios=width_ratios
            )
            hm_cols = slice(ts2_col, ts2_col + 2)
            mc2_col = None
            cb_col = -1
        elif max_lag > 0:
            width_ratios = [1, 2, 2, 1, 0.2] if show_ts2 else [2, 2, 1, 0.2]
            gs = fig.add_gridspec(
                4, len(width_ratios), height_ratios=[0.15, 1, 2, 1], width_ratios=width_ratios
            )
            hm_cols = slice(ts2_col, ts2_col + 2)
            mc2_col = ts2_col + 2
            cb_col = -1
        else:
            raise ValueError("Range of lags to be considered should be bigger than 1")

        # Time series 1 (top)
        ts1_ax = fig.add_subplot(gs[1, hm_cols])
        ts1_ax.plot(self.ts1, color="black", linewidth=1)

        # Time series 2 (left)
        if show_ts2:
            ts2_ax = fig.add_subplot(gs[2:4, 0])
            ts2_ax.plot(self.ts2.values, self.ts2.index, color="black", linewidth=1)

        # Heatmap
        hm = fig.add_subplot(gs[2:4, hm_cols])

        # Filter data and create heatmap
        filtered_df = sdc_df.loc[lambda dd: (dd.lag <= max_lag) & (dd.lag >= min_lag)]
        pivot_r = filtered_df.pivot(index="date_2", columns="date_1", values="r")
        pivot_p = filtered_df.pivot(index="date_2", columns="date_1", values="p_value")
        mask = pivot_p >= alpha

        sns.heatmap(
            pivot_r,
            cbar=False,
            mask=mask,
            cmap="RdBu_r",
            ax=hm,
        )

        # Add identity line
        identity_len = min(len(self.ts1), len(self.ts2)) - self.fragment_size + 1
        hm.plot(
            range(identity_len),
            range(identity_len),
            linestyle=":",
            color="black",
            alpha=0.4,
            linewidth=1,
        )

        # Hide heatmap labels and ticks
        hm.set_xlabel("")
        hm.set_ylabel("")
        hm.tick_params(axis="both", which="both", length=0)
        plt.setp(hm.get_yticklabels(), visible=False)
        plt.setp(hm.get_xticklabels(), visible=False)

        # Adjust heatmap limits for alignment
        xmin, xmax = hm.get_xlim()
        ymin, ymax = hm.get_ylim()
        max_r = max_r if max_r is not None else sdc_df["r"].abs().max()
        hm.set_xlim(xmin - left_offset, xmax + right_offset)
        hm.set_ylim(ymin + right_offset, ymax - left_offset)

        # Add fragment size indicator
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
            f"$s={self.fragment_size}$ {freq_str}",
            xy=(self.fragment_size / 2 + 5, 0.99),
            xycoords=trans_x,
            fontsize=labels_fontsize,
        )

        # Format TS1 axis
        ts1_ax.xaxis.set_major_formatter(date_format)
        ts1_ax.xaxis.set_label_position("top")
        ts1_ax.set_xlim(self.ts1.index[0], self.ts1.index[-1])
        ts1_ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.5)
        ts1_ax.set_xlabel(xlabel, fontsize=labels_fontsize + 2)
        ts1_ax.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            labelbottom=False,
            bottom=False,
            labelsize=labels_fontsize,
        )

        # Format TS2 axis
        if show_ts2:
            ts2_ax.yaxis.set_major_formatter(date_format)
            ts2_ax.set_ylim(self.ts2.index[0], self.ts2.index[-1])
            ts2_ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
            ts2_ax.invert_xaxis()
            ts2_ax.invert_yaxis()
            ts2_ax.set_ylabel(ylabel, fontsize=labels_fontsize + 2)
            plt.setp(ts2_ax.get_yticklabels(), visible=True, rotation=90, va="center")
            ts2_ax.tick_params(
                axis="y",
                right=True,
                labelright=True,
                labelleft=False,
                left=False,
                labelsize=labels_fontsize,
            )

        gs.update(wspace=wspace, hspace=hspace)

        # Max correlations scatter plots
        colors = {"Max $r$": "#A81529", "Min $r$ (abs)": "#144E8A"}

        if min_lag < 0:
            mc1 = fig.add_subplot(gs[-1, hm_cols])
            mc1_data = (
                sdc_df.query("p_value < @alpha")
                .query("(lag <= @max_lag) & (lag >= @min_lag)")
                .groupby("date_1")
                .agg(
                    r_max=("r", lambda x: x.where(x > 0).max()),
                    r_min=("r", lambda x: abs(x.where(x < 0).min())),
                )
                .rename(columns={"r_max": "Max $r$", "r_min": "Min $r$ (abs)"})
                .reset_index()
                .melt("date_1")
                .assign(date_1=lambda dd: dd.date_1 + timedelta_offset)
                .assign(color=lambda dd: dd.variable.map(colors))
                .dropna(subset=["value"])
            )
            if len(mc1_data) > 0:
                mc1_data.plot.scatter(
                    x="date_1",
                    y="value",
                    c="color",
                    ax=mc1,
                    alpha=0.7,
                    colorbar=False,
                    linewidths=0,
                )
            plt.setp(mc1.get_xticklabels(), visible=False)
            mc1.set_xlabel("")
            mc1.set_ylabel("Max |corr|")
            mc1.yaxis.set_label_position("right")
            mc1.set_xlim(self.ts1.index[0], self.ts1.index[-1])
            mc1.set_ylim(0, 1.05)
            mc1.grid(True, which="major")
            mc1.set_yticks([0, 0.5, 1])

        if max_lag > 0 and mc2_col is not None:
            mc2 = fig.add_subplot(gs[2:4, mc2_col])
            mc2_data = (
                sdc_df.query("p_value < @alpha")
                .query("(lag <= @max_lag) & (lag >= @min_lag)")
                .groupby("date_2")
                .agg(
                    r_max=("r", lambda x: x.where(x > 0).max()),
                    r_min=("r", lambda x: abs(x.where(x < 0).min())),
                )
                .rename(columns={"r_max": "Max $r$", "r_min": "Min $r$ (abs)"})
                .reset_index()
                .melt("date_2")
                .assign(date_2=lambda dd: dd.date_2 + timedelta_offset)
                .assign(color=lambda dd: dd.variable.map(colors))
                .dropna(subset=["value"])
            )
            if len(mc2_data) > 0:
                mc2_data.plot.scatter(
                    x="value",
                    y="date_2",
                    c="color",
                    ax=mc2,
                    alpha=0.7,
                    colorbar=False,
                    linewidths=0,
                )
            plt.setp(mc2.get_yticklabels(), visible=False)
            mc2.set_xlabel("Max |corr|")
            mc2.set_ylabel("")
            mc2.grid(True, which="major")
            mc2.set_xlim(1.05, 0)
            mc2.set_ylim(self.ts2.index[-1], self.ts2.index[0])

        # Colorbar
        if show_colorbar:
            cax = fig.add_subplot(gs[2:4, cb_col])
            color_mesh = hm.get_children()[0]
            color_mesh.set_clim(-max_r, max_r)
            fig.colorbar(color_mesh, cax=cax, label=metric_label, pad=0.05)

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
