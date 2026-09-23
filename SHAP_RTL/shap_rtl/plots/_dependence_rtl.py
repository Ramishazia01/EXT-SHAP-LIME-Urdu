"""
shap_rtl.plots.dependence  –  RTL-aware dependence plot.

Wraps shap.dependence_plot and post-processes the x-axis label so that
Urdu/Arabic/Persian feature names are rendered via HarfBuzz+FreeType images
instead of matplotlib text glyphs.

Strategy
--------
Call shap.dependence_plot(show=False), then check the x-axis label. If it
contains RTL codepoints, clear the text label and inject a HarfBuzz-rendered
image below the x-axis at the same horizontal position. All scatter geometry,
colouring, and the optional interaction colorbar are fully delegated to the
original function.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..rtl_utils import _find_rtl_font, is_rtl_text


# ── axis-label image injectors ────────────────────────────────────────────────
def _replace_rtl_xlabel(
    ax, fig, font_path: str,
    font_size_pt: float = 12,
    color: str = "#000000",
):
    """Clear a RTL x-axis label and replace it with a HarfBuzz image."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from ..rtl_utils import render_rtl_label
    import matplotlib.colors as mcolors

    xlabel_text = ax.get_xlabel()
    if not is_rtl_text(xlabel_text):
        return

    ax.set_xlabel("")
    fig.canvas.draw()

    img_pil = render_rtl_label(xlabel_text, font_path,
                                font_size_pt=font_size_pt, dpi=150)
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

    # Place image centred below the x-axis, just under the tick labels.
    # xybox=(0, -42) = 42 points below the axes bottom edge.
    ab = AnnotationBbox(
        oi,
        xy=(0.5, 0.0),
        xycoords="axes fraction",
        xybox=(0, -42),
        boxcoords="offset points",
        frameon=False,
        box_alignment=(0.5, 1.0),
        pad=0,
        clip_on=False,
    )
    ax.add_artist(ab)


def _replace_rtl_colorbar_label(fig, font_path: str, font_size_pt: float = 11):
    """Replace an RTL colorbar label with a HarfBuzz image (best-effort)."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from ..rtl_utils import render_rtl_label

    for ax in fig.get_axes():
        # Colorbars typically have no visible spines and a single axis
        if not ax.get_visible():
            continue
        label_text = ax.get_ylabel()
        if not is_rtl_text(label_text):
            continue
        ax.set_ylabel("")
        fig.canvas.draw()
        img_pil = render_rtl_label(label_text, font_path,
                                    font_size_pt=font_size_pt, dpi=150)
        if img_pil.width < 2:
            continue
        arr = np.array(img_pil, dtype=np.uint8)
        oi = OffsetImage(arr, zoom=72.0 / 150.0)
        oi.image.axes = ax
        ab = AnnotationBbox(
            oi,
            xy=(1.0, 0.5),
            xycoords="axes fraction",
            xybox=(30, 0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(0.0, 0.5),
            pad=0,
            clip_on=False,
        )
        ax.add_artist(ab)


# ── public API ────────────────────────────────────────────────────────────────
def dependence_plot(
    shap_values,
    features,
    feature_index=None,
    feature_names=None,
    display_features=None,
    interaction_index="auto",
    color="#1E88E5",
    axis_color="#333333",
    cmap=None,
    dot_size=16,
    x_jitter=0,
    alpha=1,
    title=None,
    xmin=None,
    xmax=None,
    ax=None,
    show=True,
    ymin=None,
    ymax=None,
    **kwargs,
):
    """
    RTL-aware wrapper around shap.dependence_plot().
    """

    import shap

    # --------------------------------------------------------
    # Automatically choose the most important feature
    # --------------------------------------------------------
    if feature_index is None:

        if isinstance(shap_values, shap.Explanation):

            values = shap_values.values

            if values.ndim == 2:
                importance = np.abs(values).mean(axis=0)
            else:
                importance = np.abs(values)

            feature_index = int(np.argmax(importance))

        else:
            feature_index = 0

    # --------------------------------------------------------
    # Convert Explanation -> ndarray
    # --------------------------------------------------------
    if isinstance(shap_values, shap.Explanation):

        shap_matrix = shap_values.values

        if feature_names is None:
            feature_names = list(shap_values.feature_names)

    else:
        shap_matrix = shap_values

    # --------------------------------------------------------
    # Draw original SHAP dependence plot
    # --------------------------------------------------------
    shap.dependence_plot(
        ind=feature_index,
        shap_values=shap_matrix,
        features=features,
        feature_names=feature_names,
        display_features=display_features,
        interaction_index=interaction_index,
        color=color,
        axis_color=axis_color,
        cmap=cmap,
        dot_size=dot_size,
        x_jitter=x_jitter,
        alpha=alpha,
        title=title,
        xmin=xmin,
        xmax=xmax,
        ax=ax,
        show=False,
        ymin=ymin,
        ymax=ymax,
        **kwargs,
    )

    # --------------------------------------------------------
    # Replace RTL labels
    # --------------------------------------------------------
    fig = plt.gcf()

    main_ax = next(
        (a for a in fig.get_axes() if a.get_xlabel()),
        plt.gca(),
    )

    fig.canvas.draw()

    font_path, _ = _find_rtl_font()

    if font_path is not None:
        _replace_rtl_xlabel(
            main_ax,
            fig,
            font_path,
            font_size_pt=13,
            color=axis_color,
        )

        _replace_rtl_colorbar_label(
            fig,
            font_path,
            font_size_pt=11,
        )

    if show:
        plt.show()
    else:
        return main_ax