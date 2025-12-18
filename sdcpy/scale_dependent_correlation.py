import warnings
from typing import Callable, Optional, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
from matplotlib import ticker
from scipy import stats
from scipy.stats.mstats import rankdata
from tqdm.auto import tqdm

plt.style.use("seaborn-v0_8-white")

RECOGNIZED_METHODS = {
    "pearson": lambda x, y: stats.pearsonr(x, y),
    "spearman": lambda x, y: stats.spearmanr(x, y),
}

CONSTANT_WARNING = {"pearson": stats.ConstantInputWarning, "spearman": stats.ConstantInputWarning}


def generate_correlation_map(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> np.ndarray:
    """
    Correlate each row in matrix X against each row in matrix Y.

    Parameters
    ----------
    x
      Shape N X T.
    y
      Shape M X T.
    method
        Method use to compute the correlation. Must be one of 'pearson' or 'spearman'

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    if method.lower() not in ["spearman", "pearson"]:
        raise NotImplementedError(
            f'Method {method} not understood, must be one of "pearson", "spearman"'
        )

    if method.lower() == "spearman":
        x = rankdata(x, axis=1)
        y = rankdata(y, axis=1)

    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError("x and y must have the same number of timepoints.")
    s_x = x.std(axis=1, ddof=n - 1)
    s_y = y.std(axis=1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def shuffle_along_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Shuffles array independently across selected axis

    Parameters
    ----------
    a
        Input array
    axis
        Axis across which to shuffle
    Returns
    -------
    np.ndarray
        Shuffled copy of original array
    """
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def compute_sdc(
    ts1: np.ndarray,
    ts2: np.ndarray,
    fragment_size: int,
    n_permutations: int = 99,
    method: Union[str, Callable] = "pearson",
    two_tailed: bool = True,
    permutations: bool = True,
    min_lag: int = -np.inf,
    max_lag: int = np.inf,
) -> pd.DataFrame:
    """
    Computes scale dependent correlation (https://doi.org/10.1007/s00382-005-0106-4) matrix among two time series

    Parameters
    ----------
    ts1
        First time series. Must be array-like.
    ts2
        Second time series. Must be array-like.
    fragment_size
        Size of the fragments of the original time-series that will be used in their one-to-one comparison.
    n_permutations
        Number of permutations used to obtain the p-value of the non-parametric randomization test.
    method
        Method that will be used to perform the pairwise correlation/similarity/distance comparisons. Methods already
        understood are `pearson`, `spearman` and `kendall` but any custom callable taking two array-like inputs and
        returning a single float can be used. Note that this will be called n_permutations + 1 times for every single
        pairwise comparison, so an expensive call can significantly increase the total computation time. Pearson method
        will be orders of magnitude faster than the others since it has been implemented via vectorized functions.
    two_tailed
        Whether to use a two tailed randomised test or not. Default recognized methods should use the default two tailed
        value, but other distance metrics might not.
    permutations
        Whether to generate permutations of the fragments and generate their estimated-pvalues. If False, all p-values
        returned will be those from the statistical test passed in method.
    min_lag
        Lower limit of the lags between ts1 and ts2 that will be computed.
    max_lag
        Upper limit of the lags between ts1 and ts2 that will be computed.

    Returns
    -------
    pd.DataFrame
        Data frame with a row for each pair wise comparison containing information about the coordinates of each
        fragment used, the similarity obtained and the p-value from the randomised test.
    """

    method_fun = RECOGNIZED_METHODS[method] if method in RECOGNIZED_METHODS else method
    # TODO: Proper calculation of number of iterations considering the range of lags selected
    n_iterations = (len(ts1) - fragment_size) * (len(ts2) - fragment_size)
    n_root = int(np.sqrt(n_permutations).round())
    sdc_array = np.empty(shape=(n_iterations, 7))
    sdc_array[:] = np.nan
    i = 0
    progress_bar = tqdm(total=n_iterations, desc="Computing SDC", leave=False)
    # We iterate over all possible fragments of size `fragment_size` in both time-series
    for start_1 in range(len(ts1) - fragment_size):
        stop_1 = start_1 + fragment_size
        for start_2 in range(len(ts2) - fragment_size):
            lag = start_1 - start_2
            if min_lag <= lag <= max_lag:
                stop_2 = start_2 + fragment_size
                fragment_1 = ts1[start_1:stop_1]
                fragment_2 = ts2[start_2:stop_2]
                # Compute the correlation/distance across both fragments
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    warnings.warn("Constant Fragment", CONSTANT_WARNING[method])
                    statistic, p_value = method_fun(fragment_1, fragment_2)
                if permutations:
                    # Randomize both fragments and compute their correlations.
                    if method.lower() in ["pearson", "spearman"]:
                        permuted_1 = shuffle_along_axis(
                            np.tile(fragment_1, n_root).reshape(n_root, -1), axis=1
                        )
                        permuted_2 = shuffle_along_axis(
                            np.tile(fragment_2, n_root).reshape(n_root, -1), axis=1
                        )
                        permuted_scores = generate_correlation_map(
                            permuted_1, permuted_2, method=method
                        ).reshape(-1)
                    else:
                        permuted_scores = [
                            method_fun(
                                np.random.permutation(fragment_1), np.random.permutation(fragment_2)
                            )[0]
                            for _ in range(n_permutations)
                        ]
                    # Get the p-value by comparing the original value to the distribution of randomized values
                    if two_tailed:
                        p_value = (
                            1
                            - stats.percentileofscore(np.abs(permuted_scores), np.abs(statistic))
                            / 100
                        )
                    else:
                        p_value = 1 - stats.percentileofscore(permuted_scores, statistic) / 100

                sdc_array[i] = [start_1, stop_1, start_2, stop_2, lag, statistic, p_value]
                i += 1
                progress_bar.update(1)

    progress_bar.close()
    sdc_df = pd.DataFrame(
        sdc_array[:i], columns=["start_1", "stop_1", "start_2", "stop_2", "lag", "r", "p_value"]
    )

    return sdc_df


def plot_two_way_sdc(sdc_df: pd.DataFrame, alpha: float = 0.05, **kwargs):
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

    def two_way_plot(self, alpha: float = 0.05, **kwargs):
        return plot_two_way_sdc(self.sdc_df, alpha, **kwargs)

    def to_excel(self, filename: str):
        with pd.ExcelWriter(filename) as writer:
            (
                self.sdc_df.dropna()
                .pivot(index="start_1", columns="start_2", values="r")
                .to_excel(writer, sheet_name="rs")
            )
            (
                self.sdc_df.dropna()
                .pivot(index="start_1", columns="start_2", values="p_value")
                .to_excel(writer, sheet_name="p_values")
            )

            pd.concat(
                [
                    self.ts1.rename("ts1")
                    .reset_index()
                    .reset_index()
                    .rename(columns={"index": "start_1"}),
                    self.ts2.rename("ts2")
                    .reset_index()
                    .reset_index()
                    .rename(columns={"index": "start_2"}),
                ],
                axis=1,
            ).to_excel(writer, sheet_name="time_series", index=False)

            pd.DataFrame(
                {
                    "fragment_size": self.fragment_size,
                    "n_permutations": self.n_permutations,
                    "method": self.method,
                },
                index=[1],
            ).to_excel(writer, sheet_name="config", index=False)

    @classmethod
    def from_excel(cls, filename: str):
        fragment_size, n_permutations, method = pd.read_excel(filename, "config").loc[0]
        ts1 = pd.read_excel(filename, "time_series").set_index("date_1")[["start_1", "ts1"]]
        ts2 = pd.read_excel(filename, "time_series").set_index("date_2")[["start_2", "ts2"]]
        sdc_df = (
            pd.merge(
                pd.read_excel(filename, "rs").melt("start_1", value_name="r", var_name="start_2"),
                pd.read_excel(filename, "p_values").melt(
                    "start_1", value_name="p_value", var_name="start_2"
                ),
                on=["start_1", "start_2"],
            )
            .assign(
                stop_1=lambda dd: dd.start_1 + fragment_size,
                stop_2=lambda dd: dd.start_2 + fragment_size,
                lag=lambda dd: dd.start_1 - dd.start_2,
            )
            .merge(ts1.reset_index()[["date_1", "start_1"]])
            .merge(ts2.reset_index()[["date_2", "start_2"]])
        )

        return cls(
            ts1=ts1.ts1,
            ts2=ts2.ts2,
            fragment_size=fragment_size,
            n_permutations=n_permutations,
            method=method,
            sdc_df=sdc_df,
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
        min_bin = int(np.floor(self.__dict__[f"ts{ts}"].min())) if min_bin is None else min_bin
        max_bin = int(np.ceil(self.__dict__[f"ts{ts}"].max())) if max_bin is None else max_bin
        name = self.__dict__[f"ts{ts}"].name
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
            .merge(
                self.__dict__[f"ts{ts}"]
                .reset_index()
                .rename(columns={f"date_{ts}": "date", name: "value"})
            )
            .assign(
                cat_value=lambda dd: pd.cut(
                    dd.value, bins=list(range(min_bin, max_bin + bin_size, bin_size)), precision=0
                )
            )
            .groupby(["comparison_id"])
            .apply(lambda dd: dd.cat_value.value_counts(True))
            .loc[lambda x: x > threshold]
            .reset_index()
            .drop(columns="cat_value")
            .rename(columns={"level_1": "cat_value"})
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
            .apply(lambda dd: dd["direction"].value_counts().rename("counts").reset_index())
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

    def plot_consecutive(self, alpha: float = 0.05, **kwargs):
        f = (
            self.sdc_df.loc[lambda dd: dd.p_value < alpha]
            # Here I make groups of consecutive significant values and report the longest for each lag.
            .groupby("lag", as_index=True)
            .apply(
                lambda gdf: gdf.sort_values("start_1")
                .assign(group=lambda dd: (dd.start_1 != dd.start_1.shift(1) + 1).cumsum())
                .groupby(["group"])
                .size()
                .max()
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
    ):
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
                    ]
                )
                .reset_index(drop=True)
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

    def dominant_lags_plot(self, alpha: float = 0.05, ylabel: str = "", **kwargs):
        fig, ax = plt.subplots(**kwargs)
        df = (
            self.sdc_df.loc[lambda dd: dd.p_value < alpha]
            .groupby("date_1")
            .apply(
                lambda dd: dd.loc[
                    lambda ddd: ((ddd.r == ddd.r.max()) & (ddd.r > 0))
                    | ((ddd.r == ddd.r.min()) & (ddd.r < 0))
                ]
            )
            .reset_index(drop=True)
            .groupby(["date_1"])
            .apply(lambda dd: dd.loc[dd["lag"].abs() == dd["lag"].abs().min()])
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
        ax.set_ylabel("AH [g/m³]")
        ax2.set_ylabel("Max/Min r")

        return fig

    def single_shift_plot(self, shift: int):
        pass
