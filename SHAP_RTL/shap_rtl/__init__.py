"""
SHAP_RTL_package
================
Right-to-left (RTL) aware SHAP plots for Urdu, Arabic, and other RTL languages.

This package wraps the original ``shap`` library plots and applies:
  - Arabic text reshaping (arabic_reshaper)
  - Unicode BiDi algorithm (python-bidi)
  - Jameel Noori Nastaliq font (or best available Nastaliq fallback)

All plot functions are drop-in replacements – the only difference is that
feature names containing RTL characters are rendered correctly.

Canonical import
----------------
    from shap_rtl import plots as shap_rtl_plots
    shap_rtl_plots.waterfall(shap_values)
    shap_rtl_plots.bar(shap_values)
    shap_rtl_plots.beeswarm(shap_values)

Or the module-style import that matches the package naming convention:

    from shap_rtl.plots import waterfall as shap_rtl_force_plot
    shap_rtl_force_plot(shap_values)

Named-module import 
-----------------------------------
    import shap_rtl.plots as shap_rtl_force  # plot accessible as shap_rtl_force.plot
    shap_rtl_force.waterfall(shap_values)

Easy-to-use wrapper for all languages
-----------------------------------
    from shap_rtl import SHAPRTL
    
    # For any language: Arabic, Urdu, Hebrew, Persian
    analyzer = SHAPRTL(language="Hebrew")
    analyzer.train(texts, labels)
    analyzer.analyze("שלום עולם", plot_type='bar')

Version
-------
"""

__version__ = "1.0.0"
__author__ = "SHAP_RTL contributors"

from . import plots
from .plots import (
    waterfall,
    bar,
    beeswarm,
    force,
    summary_plot,
    decision_plot,
    dependence_plot,
    scatter,
    heatmap,
)
from .explainer import RTLSHAPExplainer

# ============================================================
# IMPORT THE WRAPPER
# ============================================================
from .wrapper import SHAPRTL, create_analyzer

# Convenience alias so users can do:
#   import shap_rtl.plots as shap_rtl_force
#   shap_rtl_force.plot(shap_values)  <- calling waterfall
plot = waterfall   # default "plot" points to waterfall

__all__ = [
    "plots",
    "waterfall",
    "bar",
    "beeswarm",
    "force",
    "summary_plot",
    "decision_plot",
    "dependence_plot",
    "scatter",
    "heatmap",
    "plot",
    "RTLSHAPExplainer",
    "SHAPRTL",           
    "create_analyzer",   
    "__version__",
]