"""Plotting functions for Scale Dependent Correlation analysis."""

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure


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
    import re

    # Check if index is datetime-like
    is_datetime_index = pd.api.types.is_datetime64_any_dtype(index)

    if not is_datetime_index:
        # Non-datetime index - use integer positioning
        return "periods", 1, "D"

    frequency = pd.infer_freq(index)
    if frequency:
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


def combi_plot(
    ts1: pd.Series,
    ts2: pd.Series,
    sdc_df: pd.DataFrame,
    fragment_size: int,
    method: str = "pearson",
    alpha: float = 0.05,
    xlabel: str = "",
    ylabel: str = "",
    title: str = None,
    max_r: float = None,
    date_fmt: str = None,
    align: str = "center",
    max_lag: int = np.inf,
    min_lag: int = -np.inf,
    fontsize: int = 7,
    tick_fontsize: int = None,
    label_fontsize: int = None,
    colorbar_fontsize: int = None,
    wspace: float = None,
    hspace: float = None,
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

    Parameters
    ----------
    ts1 : pd.Series
        Time series 1.
    ts2 : pd.Series
        Time series 2.
    sdc_df : pd.DataFrame
        SDC results DataFrame.
    fragment_size : int
        Fragment size used in the analysis.
    method : str, default "pearson"
        Correlation method used.
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
        Date format string. Defaults to "%Y-%m-%d".
    align : str, default "center"
        Alignment of heatmap cells: "left", "center", or "right".
    max_lag : int, default np.inf
        Maximum lag to display.
    min_lag : int, default -np.inf
        Minimum lag to display.
    fontsize : int, default 9
        Base font size.
    tick_fontsize : int, optional
        Font size for tick labels. Defaults to fontsize.
    label_fontsize : int, optional
        Font size for axis labels. Defaults to fontsize + 2.
    colorbar_fontsize : int, optional
        Font size for colorbar ticks. Defaults to fontsize.
    wspace : float, default 0.2
        Width space between subplots.
    hspace : float, default 0.2
        Height space between subplots.
    show_colorbar : bool, default True
        Whether to show the colorbar.
    show_ts2 : bool, default True
        Whether to show time series 2 on the left side.
    metric_label : str, optional
        Label for the correlation metric. Defaults to method name.
    n_ticks : int, default 6
        Number of ticks to show on axes.
    figsize : tuple, default (6, 6)
        Figure size.
    dpi : int, default 150
        Figure resolution.
    **kwargs
        Additional keyword arguments passed to plt.figure().

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    import warnings

    # Get frequency information for proper labeling and offsets
    freq_str, freq_mult, freq_unit = _determine_frequency_info(ts1.index)

    # Default date format: Y-m-d
    if date_fmt is None:
        date_fmt = "%Y-%m-%d"

    # Set fontsize fallbacks
    tick_fontsize = tick_fontsize if tick_fontsize is not None else fontsize
    label_fontsize = label_fontsize if label_fontsize is not None else fontsize + 1
    colorbar_fontsize = colorbar_fontsize if colorbar_fontsize is not None else fontsize

    # Set spacing fallbacks (scales inversely with figsize)
    wspace = wspace if wspace is not None else 1.5 / figsize[0]
    hspace = hspace if hspace is not None else 1.5 / figsize[1]

    # Set default title with frequency info
    # Title is only set if user provides one (no default)

    # Set default metric label
    if metric_label is None:
        metric_labels = {
            "pearson": "Pearson's $r$",
            "spearman": "Spearman's $\\rho$",
        }
        metric_label = metric_labels.get(method, method.capitalize())

    # Validate alignment
    align = align.lower()
    if align not in ["left", "center", "right"]:
        warnings.warn(
            f'Alignment method "{align}" not recognized, defaulting to center alignment.',
            stacklevel=2,
        )
        align = "center"

    # Calculate offsets
    offset = fragment_size // 2 if align == "center" else fragment_size
    left_offset = 0 if align == "left" else offset
    right_offset = 0 if align == "right" else offset

    # Check if index is datetime for formatting
    is_datetime_index = pd.api.types.is_datetime64_any_dtype(ts1.index)

    # Calculate offset using timedelta for datetime indexes, integer for others
    if is_datetime_index:
        timedelta_offset = pd.to_timedelta(left_offset * freq_mult, unit=freq_unit)
    else:
        timedelta_offset = left_offset  # Use integer offset for non-datetime

    is_datetime_index = pd.api.types.is_datetime64_any_dtype(ts1.index)
    if is_datetime_index and date_fmt:
        date_format = mdates.DateFormatter(date_fmt)
    else:
        date_format = None

    sdc_df = sdc_df.copy()
    fig = plt.figure(figsize=figsize, **kwargs)

    # Dynamic grid layout based on lag range and show_ts2
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
    ts1_ax.plot(ts1, color="black", linewidth=1)

    # Time series 2 (left)
    if show_ts2:
        ts2_ax = fig.add_subplot(gs[2:4, 0])
        ts2_ax.plot(ts2.values, ts2.index, color="black", linewidth=1)

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
    identity_len = min(len(ts1), len(ts2)) - fragment_size + 1
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
        [-fragment_size / 2, fragment_size / 2],
        [1.0, 1.0],
        color="k",
        transform=trans_x,
        clip_on=False,
        linewidth=5,
        solid_capstyle="butt",
    )
    hm.plot(
        [0, 0],
        [-fragment_size / 2, fragment_size / 2],
        color="k",
        transform=trans_y,
        clip_on=False,
        linewidth=5,
        solid_capstyle="butt",
    )
    hm.annotate(
        f"$s={fragment_size}$ {freq_str}",
        xy=(fragment_size / 2 + 5, 0.99),
        xycoords=trans_x,
        fontsize=tick_fontsize,
    )

    # Format TS1 axis
    ts1_ax.xaxis.set_label_position("top")
    ts1_ax.set_xlim(ts1.index[0], ts1.index[-1])
    ts1_ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.5)
    ts1_ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ts1_ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks, prune="both"))
    if date_format:
        ts1_ax.xaxis.set_major_formatter(date_format)
    ts1_ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        labelbottom=False,
        bottom=False,
        labelsize=tick_fontsize,
    )
    ts1_ax.tick_params(
        axis="y",
        labelsize=tick_fontsize,
    )

    # Format TS2 axis
    if show_ts2:
        ts2_ax.set_ylim(ts2.index[0], ts2.index[-1])
        ts2_ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
        ts2_ax.invert_xaxis()
        ts2_ax.invert_yaxis()
        ts2_ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ts2_ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks, prune="both"))
        if date_format:
            ts2_ax.yaxis.set_major_formatter(date_format)
        plt.setp(ts2_ax.get_yticklabels(), visible=True, rotation=90, va="center")
        ts2_ax.tick_params(
            axis="y",
            left=True,
            labelleft=True,
            labelright=False,
            right=False,
            labelsize=tick_fontsize,
        )
        # Move x-axis ticks to top
        ts2_ax.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            labelsize=tick_fontsize,
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
        mc1.set_ylabel("Max |corr|", fontsize=label_fontsize)
        mc1.yaxis.set_label_position("right")
        mc1.set_xlim(ts1.index[0], ts1.index[-1])
        mc1.set_ylim(0, 1.05)
        mc1.grid(True, which="major")
        mc1.set_yticks([0, 0.5, 1])
        mc1.tick_params(axis="y", labelsize=tick_fontsize)
        mc1.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False)

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
        mc2.set_xlabel("Max |corr|", fontsize=label_fontsize)
        mc2.xaxis.set_label_position("top")
        mc2.set_ylabel("")
        mc2.grid(True, which="major")
        mc2.set_xlim(1.05, 0)
        mc2.set_xticks([0, 0.5, 1])  # Match mc1's y-axis breaks
        mc2.set_ylim(ts2.index[-1], ts2.index[0])
        # Move x-axis ticks to top
        mc2.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            labelsize=tick_fontsize,
        )
        mc2.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

    # Colorbar
    if show_colorbar:
        cax = fig.add_subplot(gs[2:4, cb_col])
        color_mesh = hm.get_children()[0]
        color_mesh.set_clim(-max_r, max_r)
        cbar = fig.colorbar(color_mesh, cax=cax, label=metric_label, pad=0.05)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)
        cbar.set_label(metric_label, fontsize=label_fontsize)

    fig.set_dpi(dpi)
    if title:
        fig.suptitle(title)

    return fig
