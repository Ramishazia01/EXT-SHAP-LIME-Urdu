"""
shap_rtl.plots.summary  –  RTL-aware summary plot.

Wraps shap.summary_plot and post-processes the y-axis tick labels so that
Urdu/Arabic/Persian feature names are rendered via HarfBuzz+FreeType images
instead of matplotlib text glyphs.

Strategy
--------
Call shap.summary_plot(show=False), then apply RTL rendering to whichever
y-tick labels contain RTL codepoints. Axes geometry, colours, colourbar, and
sorting are fully delegated to the original function.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import shap as _shap

from ..rtl_utils import (
    _find_rtl_font,
    is_rtl_text,
    apply_rtl_ytick_images,
    hide_rtl_text_ticks,
    add_rtl_title,
)


def _get_main_ax(fig) -> plt.Axes:
    """Return the main plot axes (the one with y-tick labels)."""
    axes = fig.get_axes()
    if not axes:
        return plt.gca()
    # Pick the axes that has y-tick labels (the main plot)
    for ax in axes:
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if any(labels):
            return ax
    return axes[0]


def _adjust_figure_for_rtl(fig, ax, tick_labels, font_path, axis_color="#333333"):
    """
    Adjust the figure for RTL labels.
    This ensures labels are properly rendered and not cut off.
    """
    # Hide original text ticks
    hide_rtl_text_ticks(ax, tick_labels)
    
    # Apply RTL images
    apply_rtl_ytick_images(
        ax, tick_labels, font_path,
        font_size_pt=12, 
        color=axis_color,
        x_offset_pts=-8
    )
    
    # Adjust subplot parameters to make room for RTL labels
    fig.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.1)


# ── public API ─────────────────────────────────────────────────────────────────
def summary_plot(
    shap_values,
    features=None,
    feature_names=None,
    max_display: int = 10,
    plot_type="dot",  # Default to 'dot' for beeswarm
    color=None,
    axis_color: str = "#333333",
    title=None,
    alpha: float = 1,
    show: bool = True,
    sort: bool = True,
    color_bar: bool = True,
    plot_size="auto",
    **kwargs,
):
    """
    RTL-aware summary plot (drop-in for shap.summary_plot).

    This properly uses SHAP's original summary_plot and then applies RTL
    rendering to the feature names.

    Parameters
    ----------
    shap_values : np.ndarray or shap.Explanation
        Matrix of SHAP values (shape n_samples × n_features) or an Explanation.
    features : np.ndarray or pd.DataFrame, optional
        Feature values for colouring dots.
    feature_names : list of str, optional
        Feature names. RTL names are auto-detected.
    max_display : int
        Maximum features to show.
    plot_type : {"dot","beeswarm","bar","violin","compact_dot"}
        Plot style. Defaults to "dot" (beeswarm).
    color : colour, optional
    axis_color : str
    title : str, optional
    alpha : float
    show : bool
    sort : bool
    color_bar : bool
    plot_size : "auto" | float | tuple
    **kwargs
        Forwarded to shap.summary_plot.
    """
    
    # Extract feature names from Explanation if not provided
    if feature_names is None and hasattr(shap_values, 'feature_names'):
        feature_names = shap_values.feature_names
    
    # Extract features from Explanation if not provided
    if features is None and hasattr(shap_values, 'data'):
        features = shap_values.data
    
    # For Explanation objects, extract values
    if hasattr(shap_values, 'values'):
        shap_values_array = shap_values.values
    else:
        shap_values_array = shap_values
    
    # Determine figure size
    if plot_size == "auto":
        figsize = (12, max(6, max_display * 0.4))
    elif isinstance(plot_size, (int, float)):
        figsize = (12, plot_size)
    elif isinstance(plot_size, tuple):
        figsize = plot_size
    else:
        figsize = (12, 8)
    
    plt.figure(figsize=figsize)
    
    # Call the ORIGINAL SHAP summary_plot with show=False
    # This creates the proper beeswarm plot with dots
    _shap.summary_plot(
        shap_values_array,
        features=features,
        feature_names=feature_names,
        max_display=max_display,
        plot_type=plot_type,  # 'dot' creates beeswarm
        color=color,
        axis_color=axis_color,
        title=None,  # We'll handle title separately
        alpha=alpha,
        show=False,
        sort=sort,
        color_bar=color_bar,
        plot_size=None,
        **kwargs,
    )

    fig = plt.gcf()
    fig.canvas.draw()
    
    # Get the main axes
    ax = _get_main_ax(fig)
    
    # Get tick labels
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    rtl_mode = any(is_rtl_text(lbl) for lbl in tick_labels)

    # Apply RTL rendering if needed
    if rtl_mode:
        font_path, _ = _find_rtl_font()
        if font_path is not None:
            _adjust_figure_for_rtl(fig, ax, tick_labels, font_path, axis_color)
    
    # Handle title with RTL support
    if title:
        add_rtl_title(title, ax=ax, font_size_pt=14)
    
    # Ensure the figure has enough space
    try:
        fig.tight_layout()
    except Exception:
        pass

    if show:
        plt.show()
    else:
        return ax