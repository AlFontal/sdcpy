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