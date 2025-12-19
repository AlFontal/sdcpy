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

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install sdcpy
```

## Usage

```python
import numpy as np
import pandas as pd
from sdcpy import SDCAnalysis

# Synthetic signal with transient pattern between indices 63-169
def tc_signal(i):
    error = np.random.normal()
    if 63 <= i <= 169:
        return np.sin(2 * np.pi * (1 / 37) * i) + 0.6 * error
    return error

np.random.seed(42)
ts1 = pd.Series([tc_signal(i) for i in range(250)])
ts2 = pd.Series([tc_signal(i) for i in range(250)])

# Run SDC analysis
sdc = SDCAnalysis(ts1, ts2, fragment_size=50, n_permutations=99)

# Generate 2-way SDC combi plot
fig = sdc.combi_plot(xlabel="TS1", ylabel="TS2")
fig.savefig("sdc_plot.png", dpi=150, bbox_inches="tight")
```

<img src="sdc_example.png" width="500" />

See [examples/basic_usage.py](examples/basic_usage.py) for a complete example with synthetic data showing transient correlations.

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
