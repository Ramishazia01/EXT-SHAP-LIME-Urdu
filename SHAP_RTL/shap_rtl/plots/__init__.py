"""
shap_rtl.plots
──────────────
RTL-aware SHAP plot functions.

Each function is a drop-in replacement for its ``shap.plots.*`` counterpart.
Feature names containing Arabic / Urdu characters are automatically reshaped
and rendered via the Jameel Noori Nastaliq font (or best available fallback).

Usage
-----
    from shap_rtl.plots import waterfall, bar, beeswarm, force
    from shap_rtl.plots import summary_plot, decision_plot, dependence_plot
    from shap_rtl.plots import scatter, heatmap
"""

from ._bar_rtl import bar
from ._waterfall_rtl import waterfall
from ._beeswarm_rtl import beeswarm
from ._summary_rtl import summary_plot
from ._heatmap_rtl import heatmap
from ._decision_rtl import decision_plot
from ._scatter_rtl import scatter
from ._dependence_rtl import dependence_plot

# Import the core 'force' function and expose it under both names for compatibility
from ._force_rtl import force
from ._force_rtl import force as force_plot

__all__ = [
    "waterfall",
    "bar",
    "beeswarm",
    "force",
    "force_plot",
    "summary_plot",
    "decision_plot",
    "dependence_plot",
    "scatter",
    "heatmap",
]