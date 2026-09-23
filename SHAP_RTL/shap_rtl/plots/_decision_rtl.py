"""
shap_rtl.plots.decision - RTL-aware decision plot.

Wraps shap.decision_plot and post-processes the y-axis tick labels so that
Urdu/Arabic/Persian feature names are rendered via HarfBuzz+FreeType images
instead of matplotlib text glyphs.

Strategy
--------
Call shap.decision_plot(show=False, return_objects=True) to obtain the Axes
reference directly, then hide RTL text ticks and overlay HarfBuzz images at
the same tick positions. All decision-path geometry and colouring is fully
delegated to the original function.
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


# ── shared helper (mirrors bar/beeswarm pattern) ──────────────────────────────
def _inject_image_label(
    ax, fig, text: str, ydata: float, font_path: str,
    color: str = "black", font_size_pt: float = 12,
):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from ..rtl_utils import render_rtl_label, render_mixed_label
    import matplotlib.colors as mcolors

    if " = " in text and is_rtl_text(text.split(" = ", 1)[-1]):
        img_pil = render_mixed_label(text, font_path, font_size_pt=font_size_pt, dpi=150)
    else:
        img_pil = render_rtl_label(text, font_path, font_size_pt=font_size_pt, dpi=150)

    if img_pil.width < 2:
        return

    arr = np.array(img_pil, dtype=np.uint8).copy()
    try:
        rgb = mcolors.to_rgb(color)
        arr[:, :, 0] = int(rgb[0] * 255)
        arr[:, :, 1] = int(rgb[1] * 255)
        arr[:, :, 2] = int(rgb[2] * 255)
    except Exception:
        pass

    oi = OffsetImage(arr, zoom=72.0 / 150.0)
    oi.image.axes = ax
    ab = AnnotationBbox(
        oi,
        xy=(0, ydata),
        xycoords=("axes fraction", "data"),
        xybox=(-4, 0),
        boxcoords="offset points",
        frameon=False,
        box_alignment=(1.0, 0.5),
        pad=0,
    )
    ax.add_artist(ab)


# ── public API ─────────────────────────────────────────────────────────────────
def decision_plot(
    base_value,
    shap_values,
    features=None,
    feature_names=None,
    feature_order: str = "importance",
    feature_display_range=None,
    highlight=None,
    link: str = "identity",
    plot_color=None,
    axis_color: str = "#000000",
    y_demarc_color=None,
    alpha=None,
    color_bar: bool = True,
    auto_size_plot: bool = True,
    title=None,
    xlim=None,
    show: bool = True,
    return_objects: bool = False,
    ignore_warnings: bool = False,
    new_base_value=None,
    legend_labels=None,
    legend_location: str = "best",
    **kwargs,
):
    """
    RTL-aware decision plot (drop-in for shap.decision_plot).

    Identical to shap.decision_plot except that feature names containing
    Urdu/Arabic/Persian characters are rendered via HarfBuzz+FreeType images.

    Parameters
    ----------
    base_value : float or np.ndarray
        Expected value of the model output.
    shap_values : np.ndarray
        SHAP values to visualise (1-D for a single prediction, 2-D for many).
    features : np.ndarray or pd.DataFrame, optional
        Feature values used to annotate the y-axis labels.
    feature_names : list of str, optional
        Feature names. RTL names are auto-detected.
    feature_order : str or list
        "importance" (default), "hclust", or an explicit index list.
    feature_display_range : slice or range, optional
        Subset of features to show.
    highlight : array-like, optional
        Boolean mask of rows to highlight.
    link : {"identity","logit"}
    plot_color : matplotlib colormap, optional
    axis_color : str
    y_demarc_color : str
    alpha : float, optional
    color_bar : bool
    auto_size_plot : bool
    title : str, optional
    xlim : tuple, optional
    show : bool
        If False, returns the Axes (and the DecisionPlotResult if
        return_objects=True).
    return_objects : bool
        If True, also returns the DecisionPlotResult namedtuple.
    ignore_warnings : bool
    new_base_value : float, optional
    legend_labels : list, optional
    legend_location : str
    **kwargs
        Additional keyword arguments forwarded to shap.decision_plot.
    """
    import shap as _shap

    fwd: dict = dict(
        features=features,
        feature_names=feature_names,
        feature_order=feature_order,
        feature_display_range=feature_display_range,
        highlight=highlight,
        link=link,
        plot_color=plot_color,
        axis_color=axis_color,
        alpha=alpha,
        color_bar=color_bar,
        auto_size_plot=auto_size_plot,
        title=title,
        xlim=xlim,
        show=False,
        return_objects=True,   # always get the axes reference
        ignore_warnings=ignore_warnings,
        new_base_value=new_base_value,
        legend_labels=legend_labels,
        legend_location=legend_location,
    )
    if y_demarc_color is not None:
        fwd["y_demarc_color"] = y_demarc_color
    fwd.update(kwargs)

    result = _shap.decision_plot(base_value, shap_values, **fwd)

    # result is a DecisionPlotResult namedtuple with an .ax attribute
    ax = result.ax if hasattr(result, "ax") else plt.gca()
    fig = ax.figure
    fig.canvas.draw()

    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    rtl_mode = any(is_rtl_text(lbl) for lbl in tick_labels)

    if rtl_mode:
        font_path, _ = _find_rtl_font()
        if font_path is not None:
            hide_rtl_text_ticks(ax, tick_labels)
            apply_rtl_ytick_images(
                ax, tick_labels, font_path,
                font_size_pt=12, color=axis_color,
            )

    if show:
        plt.show()

    if return_objects:
        return result
    elif not show:
        return ax
