"""Tests for I/O functions: Excel save/load round-trip."""

import os
import tempfile

import numpy as np
import pandas as pd

from sdcpy import SDCAnalysis
from sdcpy.io import load_from_excel, save_to_excel


class TestExcelRoundTrip:
    """Tests for Excel save/load consistency."""

    def test_save_and_load_basic(self, random_ts_pair):
        """Save to Excel and load should produce same data."""
        ts1, ts2 = random_ts_pair
        sdc_original = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            filepath = f.name

        try:
            sdc_original.to_excel(filepath)
            sdc_loaded = SDCAnalysis.from_excel(filepath)

            # Check numeric values are preserved
            assert sdc_loaded.fragment_size == sdc_original.fragment_size
            assert sdc_loaded.method == sdc_original.method

            # Check sdc_df has data
            assert len(sdc_loaded.sdc_df) > 0

            # Check correlation values exist and are valid
            assert sdc_loaded.sdc_df["r"].notna().any()
        finally:
            os.unlink(filepath)

    def test_time_series_preserved(self, random_ts_pair):
        """Time series data should be preserved after round-trip."""
        ts1, ts2 = random_ts_pair
        sdc_original = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            filepath = f.name

        try:
            sdc_original.to_excel(filepath)
            sdc_loaded = SDCAnalysis.from_excel(filepath)

            # Time series values should match
            np.testing.assert_array_almost_equal(
                sdc_loaded.ts1.values, sdc_original.ts1.values, decimal=5
            )
            np.testing.assert_array_almost_equal(
                sdc_loaded.ts2.values, sdc_original.ts2.values, decimal=5
            )
        finally:
            os.unlink(filepath)

    def test_different_methods(self, random_ts_pair):
        """Should preserve method information."""
        ts1, ts2 = random_ts_pair

        for method in ["pearson", "spearman"]:
            sdc_original = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9, method=method)

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
                filepath = f.name

            try:
                sdc_original.to_excel(filepath)
                sdc_loaded = SDCAnalysis.from_excel(filepath)
                assert sdc_loaded.method == method
            finally:
                os.unlink(filepath)


class TestSaveToExcelFunction:
    """Tests for the standalone save_to_excel function."""

    def test_creates_file(self, random_ts_pair):
        """Should create an Excel file."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            filepath = f.name

        try:
            save_to_excel(
                sdc.sdc_df, ts1, ts2, sdc.fragment_size, sdc.n_permutations, sdc.method, filepath
            )
            assert os.path.exists(filepath)
        finally:
            os.unlink(filepath)

    def test_file_has_required_sheets(self, random_ts_pair):
        """Excel file should have all required sheets."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            filepath = f.name

        try:
            sdc.to_excel(filepath)
            sheets = pd.ExcelFile(filepath).sheet_names
            assert "rs" in sheets
            assert "p_values" in sheets
            assert "time_series" in sheets
            assert "config" in sheets
        finally:
            os.unlink(filepath)


class TestLoadFromExcelFunction:
    """Tests for the standalone load_from_excel function."""

    def test_returns_dict(self, random_ts_pair):
        """Should return a dictionary with all required keys."""
        ts1, ts2 = random_ts_pair
        sdc = SDCAnalysis(ts1, ts2, fragment_size=10, n_permutations=9)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            filepath = f.name

        try:
            sdc.to_excel(filepath)
            data = load_from_excel(filepath)

            expected_keys = {"ts1", "ts2", "fragment_size", "n_permutations", "method", "sdc_df"}
            assert set(data.keys()) == expected_keys
        finally:
            os.unlink(filepath)
