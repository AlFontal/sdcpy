"""I/O functions for Scale Dependent Correlation analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


def save_to_excel(
    sdc_df: pd.DataFrame,
    ts1: pd.Series,
    ts2: pd.Series,
    fragment_size: int,
    n_permutations: int,
    method: str,
    filename: str,
) -> None:
    """Save SDC analysis results to Excel file."""
    with pd.ExcelWriter(filename) as writer:
        (
            sdc_df.dropna()
            .pivot(index="start_1", columns="start_2", values="r")
            .to_excel(writer, sheet_name="rs")
        )
        (
            sdc_df.dropna()
            .pivot(index="start_1", columns="start_2", values="p_value")
            .to_excel(writer, sheet_name="p_values")
        )

        pd.concat(
            [
                ts1.rename("ts1").reset_index().reset_index().rename(columns={"index": "start_1"}),
                ts2.rename("ts2").reset_index().reset_index().rename(columns={"index": "start_2"}),
            ],
            axis=1,
        ).to_excel(writer, sheet_name="time_series", index=False)

        pd.DataFrame(
            {
                "fragment_size": fragment_size,
                "n_permutations": n_permutations,
                "method": method,
            },
            index=[1],
        ).to_excel(writer, sheet_name="config", index=False)


def load_from_excel(filename: str) -> dict:
    """
    Load SDC analysis data from Excel file.

    Returns a dict with keys: ts1, ts2, fragment_size, n_permutations, method, sdc_df
    """
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

    return {
        "ts1": ts1.ts1,
        "ts2": ts2.ts2,
        "fragment_size": fragment_size,
        "n_permutations": n_permutations,
        "method": method,
        "sdc_df": sdc_df,
    }
