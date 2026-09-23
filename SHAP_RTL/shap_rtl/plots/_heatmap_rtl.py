"""
shap_rtl.plots.heatmap  –  RTL-aware heatmap plot.

Wraps shap.plots.heatmap and post-processes the y-axis tick labels so that
Urdu/Arabic/Persian feature names are rendered via HarfBuzz+FreeType images
instead of matplotlib text glyphs.

Strategy
--------
Call shap.plots.heatmap(show=False), then apply RTL rendering to whichever
y-tick labels contain RTL codepoints. All heatmap colouring, instance ordering,
and feature-value bar at the top are fully delegated to the original function.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..rtl_utils import (
    _find_rtl_font,
    is_rtl_text,
    apply_rtl_ytick_images,
    hide_rtl_text_ticks,
)


def _get_heatmap_ax(fig) -> plt.Axes:
    """
    Return the main heatmap axes from a figure created by shap.plots.heatmap.

    shap.plots.heatmap creates several axes (main grid, top feature-importance
    bar, optional colorbar). The main heatmap axes is the one that has y-tick
    labels (feature names).
    """
    axes = fig.get_axes()
    if not axes:
        return plt.gca()
    for ax in axes:
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if any(labels):
            return ax
    return axes[0]


# ── public API ─────────────────────────────────────────────────────────────────
def heatmap(
    shap_values,
    instance_order=None,
    feature_values=None,
    feature_names=None,
    max_display: int = 10,
    show: bool = True,
    ax=None,
    **kwargs,
):
    """
    RTL-aware heatmap plot (drop-in for shap.plots.heatmap).

    Identical to shap.plots.heatmap except that feature names containing
    Urdu/Arabic/Persian characters are rendered via HarfBuzz+FreeType images
    on the y-axis.

    Parameters
    ----------
    shap_values : shap.Explanation
        2-D Explanation matrix (n_samples × n_features).
    instance_order : callable or array-like, optional
        How to order instances on the x-axis. Defaults to
        ``Explanation.sum.abs`` (SHAP's default).
    feature_values : callable or array-like, optional
        Feature values shown in the top bar. Defaults to
        ``Explanation.abs.mean(0)`` (SHAP's default).
    feature_names : list of str, optional
        Overrides the feature names in the Explanation. RTL names are
        auto-detected.
    max_display : int
        Maximum features to show (rows in the heatmap).
    show : bool
        If False, returns the main Axes.
    ax : matplotlib Axes, optional
    **kwargs
        Forwarded to shap.plots.heatmap.
    """
    import shap as _shap

    # Build the forwarded kwargs, omitting None defaults so SHAP can apply its own.
    call_kwargs: dict = {"max_display": max_display, "show": False, "ax": ax}
    if instance_order is not None:
        call_kwargs["instance_order"] = instance_order
    if feature_values is not None:
        call_kwargs["feature_values"] = feature_values
    if feature_names is not None:
        call_kwargs["feature_names"] = feature_names
    call_kwargs.update(kwargs)

    _shap.plots.heatmap(shap_values, **call_kwargs)

    fig = plt.gcf()
    fig.canvas.draw()           # must draw first so get_yticklabels() is populated
    main_ax = _get_heatmap_ax(fig)

    tick_labels = [t.get_text() for t in main_ax.get_yticklabels()]
    rtl_mode = any(is_rtl_text(lbl) for lbl in tick_labels)

    if rtl_mode:
        font_path, _ = _find_rtl_font()
        if font_path is not None:
            hide_rtl_text_ticks(main_ax, tick_labels)
            apply_rtl_ytick_images(
                main_ax, tick_labels, font_path,
                font_size_pt=12,
            )

    if show:
        plt.show()
    else:
        return main_ax
