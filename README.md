# sdcpy
[![](https://img.shields.io/pypi/v/sdcpy.svg)](https://pypi.python.com/pypi/sdcpy)
![](https://img.shields.io/pypi/pyversions/sdcpy.svg)
![](https://raster.shields.io/badge/license-MIT-green.png)
![](https://github.com/AlFontal/sdcpy/actions/workflows/run_tests.yml/badge.svg)
[![DOI](https://zenodo.org/badge/329668116.svg)](https://zenodo.org/badge/latestdoi/329668116)

<img src="https://raw.githubusercontent.com/AlFontal/sdcpy-app/master/static/sdcpy_logo_black.png" width="200" height="250" />


Scale Dependent Correlation (SDC) analysis<sup>1, 2, 3</sup> in Python.

+ Free software: MIT license
+ Documentation: https://sdcpy.readthedocs.io.

## Installation

Install from [PyPI](https://pypi.org/project/sdcpy/):

```bash
pip install sdcpy
```

Or using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install sdcpy
```

## Quick Start

```python
import numpy as np
import pandas as pd
from sdcpy import SDCAnalysis

# Create sample time series
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=200, freq="D")
ts1 = pd.Series(np.cumsum(np.random.randn(200)), index=dates, name="Temperature")
ts1.index.name = "date_1"

# Create a lagged, correlated series
ts2 = ts1.shift(7).fillna(0) + np.random.randn(200) * 0.5
ts2.name = "Infections"
ts2.index.name = "date_2"

# Run SDC analysis
sdc = SDCAnalysis(
    ts1, ts2,
    fragment_size=14,      # 14-day windows
    n_permutations=99,     # For p-value calculation
    method="pearson"       # Or "spearman"
)

# Access results
print(sdc.sdc_df.head())
```

### Visualizing Results

#### Combination Plot (combi_plot)

The signature visualization showing the correlation heatmap with time series:

```python
fig = sdc.combi_plot(
    figsize=(12, 10),
    alpha=0.05,              # Significance threshold
    xlabel="Temperature",
    ylabel="Infections",
    show_colorbar=True,
    show_ts2=True,           # Toggle left time series
)
fig.savefig("sdc_combi_plot.png", dpi=150, bbox_inches="tight")
```

#### Two-Way Plot

A simpler heatmap view using plotnine:

```python
plot = sdc.two_way_plot(alpha=0.05)
plot.save("sdc_two_way.png", dpi=150)
```

### Key Features

- **Auto frequency detection**: Works with daily, weekly, monthly, or irregular time series
- **Performance optimized**: 50-250x faster than previous versions using vectorized operations
- **Flexible methods**: Pearson, Spearman, or custom correlation functions
- **Excel I/O**: Save and load analyses with `sdc.to_excel()` / `SDCAnalysis.from_excel()`

## Development

To set up a local development environment:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/AlFontal/sdcpy.git
cd sdcpy
uv sync --all-groups

# Run tests
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## References

1. Rodó, X. (2001). Reversal of three global atmospheric fields linking changes in SST anomalies in the Pacific, Atlantic and Indian oceans at tropical latitudes and midlatitudes. **Climate Dynamics**, 18:203-217. DOI: 10.1007/s003820100171.

2. Rodríguez, M.A. & Rodó, X. (2004). A primer on the study of transitory dynamics in ecological series using the scale-dependent correlation analysis. **Oecologia**, 138,485-504. DOI: 10.1007/s00442-003-1464-4.

3. Rodó, X. & M.A. Rodriguez-Arias. (2006). A new method to detect transitory signatures and local time/space variability structures in the climate system: the scale-dependent correlation analysis. **Climate Dynamics**, 27:441-458. DOI: 10.1007/s00382-005-0106-4.