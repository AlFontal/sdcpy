#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sdcpy` package."""

import pytest
import numpy as np
import itertools
from sdcpy.scale_dependent_correlation import SDCAnalysis

ts1 = np.random.rand(100)
ts2 = np.random.rand(100)
fragment_sizes = [10, 50, 70]
methods = ["pearson", "spearman"]
params = list(itertools.product(fragment_sizes, methods)) 

@pytest.mark.parametrize('fragment_size,method', params)
def test_sdc_analysis(fragment_size, method):
    sdc = SDCAnalysis(ts1, ts2, fragment_size=fragment_size, method=method)

    assert sdc.fragment_size == fragment_size
    assert sdc.sdc_df.shape[0] == (len(ts1) - fragment_size) * (len(ts2) - fragment_size)

    sdc.combi_plot()  # Just checking that the figure can be generated

