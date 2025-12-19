"""Top-level package for sdcpy."""

__author__ = """Alejandro Fontal"""
__email__ = "alejandrofontal92@gmail.com"
__version__ = "0.5.1"

# Public API
from sdcpy.core import compute_sdc, generate_correlation_map, shuffle_along_axis
from sdcpy.io import load_from_excel, save_to_excel
from sdcpy.plotting import combi_plot
from sdcpy.scale_dependent_correlation import SDCAnalysis

__all__ = [
    "SDCAnalysis",
    "compute_sdc",
    "generate_correlation_map",
    "shuffle_along_axis",
    "combi_plot",
    "save_to_excel",
    "load_from_excel",
]
