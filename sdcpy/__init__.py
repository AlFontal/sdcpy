"""Top-level package for sdcpy."""

__author__ = """Alejandro Fontal"""
__email__ = "alejandrofontal92@gmail.com"
__version__ = "0.5.0"

# Public API
from sdcpy.core import compute_sdc, generate_correlation_map, shuffle_along_axis
from sdcpy.io import load_from_excel, save_to_excel
from sdcpy.plotting import plot_two_way_sdc
from sdcpy.scale_dependent_correlation import SDCAnalysis

__all__ = [
    "SDCAnalysis",
    "compute_sdc",
    "generate_correlation_map",
    "shuffle_along_axis",
    "plot_two_way_sdc",
    "save_to_excel",
    "load_from_excel",
]
