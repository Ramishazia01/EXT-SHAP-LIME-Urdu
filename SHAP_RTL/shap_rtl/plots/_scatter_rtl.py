from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shap as _shap
from arabic_reshaper import reshape
from bidi.algorithm import get_display


def _fix_rtl_text(text: str) -> str:
    """Reshape RTL text for correct Matplotlib rendering."""
    if not text:
        return text
    return get_display(reshape(str(text)))


def scatter(
    shap_values,
    color="#1E88E5",
    hist=False,
    axis_color="#333333",
    cmap=None,
    alpha=1,
    title=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    overlay=None,
    ax=None,
    show=True,
    ylabel="SHAP value for",
    font_path=None,
    ylabel_rotation=90,
    feature=None,
    **kwargs,
):
    """
    RTL-aware wrapper around shap.plots.scatter.

    Improvements
    ------------
    • Works with full SHAP Explanation objects.
    • Automatically selects the most important feature.
    • Accepts feature index or feature name.
    • Ignores unsupported kwargs.
    • Gives informative errors.
    """

    # ------------------------------------------------------------------
    # Remove unsupported kwargs that belong to other SHAP plots
    # ------------------------------------------------------------------

    unsupported = [
        "max_display",
        "plot_size",
        "order",
        "clustering",
        "cluster_threshold",
    ]

    for key in unsupported:
        kwargs.pop(key, None)

    # ------------------------------------------------------------------
    # Configure RTL font
    # ------------------------------------------------------------------

    if font_path and os.path.exists(font_path):
        fm.fontManager.addfont(font_path)

        rtl_font = fm.FontProperties(fname=font_path, size=12)
        title_font = fm.FontProperties(fname=font_path, size=14)

        plt.rcParams["font.family"] = [
            rtl_font.get_name(),
            "DejaVu Sans",
            "Arial",
            "sans-serif",
        ]
    else:
        rtl_font = None
        title_font = None

    # ------------------------------------------------------------------
    # Validate SHAP explanation
    # ------------------------------------------------------------------

    if not hasattr(shap_values, "values"):
        raise TypeError(
            "scatter() expects a shap.Explanation object."
        )

    values = np.asarray(shap_values.values)

    # ------------------------------------------------------------------
    # Single sample -> scatter not possible
    # ------------------------------------------------------------------

    if values.ndim == 1:
        raise ValueError(
            "Scatter plots require multiple samples.\n"
            "You provided a single explanation.\n"
            "Use waterfall(), bar(), or force() instead."
        )

    # ------------------------------------------------------------------
    # Automatically choose feature
    # ------------------------------------------------------------------

    if values.ndim == 2:

        if feature is None:

            importance = np.abs(values).mean(axis=0)
            best_idx = np.argmax(importance)
            shap_values = shap_values[:, best_idx]

        elif isinstance(feature, int):

            shap_values = shap_values[:, feature]

        elif isinstance(feature, str):

            names = list(shap_values.feature_names)

            if feature not in names:
                raise ValueError(
                    f"Feature '{feature}' not found."
                )

            idx = names.index(feature)
            shap_values = shap_values[:, idx]

        else:
            raise TypeError(
                "feature must be None, int or str."
            )

    # ------------------------------------------------------------------
    # Draw SHAP scatter
    # ------------------------------------------------------------------

    try:

        result_ax = _shap.plots.scatter(
            shap_values,
            color=color,
            hist=hist,
            axis_color=axis_color,
            cmap=cmap,
            alpha=alpha,
            title=title,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            overlay=overlay,
            ax=ax,
            show=False,
            ylabel=ylabel,
            **kwargs,
        )

    except Exception as e:

        raise RuntimeError(
            f"Unable to generate SHAP scatter plot.\n{e}"
        )

    # ------------------------------------------------------------------
    # Locate axes
    # ------------------------------------------------------------------

    if result_ax is not None and hasattr(result_ax, "get_xlabel"):
        main_ax = result_ax
    else:
        fig = plt.gcf()
        main_ax = next(
            (a for a in fig.get_axes() if a.get_xlabel()),
            plt.gca(),
        )

    # ------------------------------------------------------------------
    # RTL formatting
    # ------------------------------------------------------------------

    if rtl_font:

        raw_xlabel = main_ax.get_xlabel()
        raw_title = main_ax.get_title()

        feature_name = getattr(shap_values, "feature_names", "")

        if isinstance(feature_name, (list, tuple)):
            feature_name = feature_name[0] if feature_name else ""

        if feature_name:

            full_ylabel = f"{ylabel} {feature_name}"

            main_ax.set_ylabel(
                _fix_rtl_text(full_ylabel),
                fontproperties=rtl_font,
                labelpad=15,
                rotation=ylabel_rotation,
            )

            main_ax.yaxis.label.set_ha("right")

            if ylabel_rotation == 90:

                x, _ = main_ax.yaxis.label.get_position()
                main_ax.yaxis.label.set_position((x, 1.0))

            else:

                main_ax.yaxis.label.set_position((1.0, 1.0))
                main_ax.yaxis.label.set_va("bottom")

        if raw_xlabel:

            main_ax.set_xlabel(
                _fix_rtl_text(raw_xlabel),
                fontproperties=rtl_font,
                labelpad=12,
            )

            main_ax.xaxis.label.set_ha("right")

            _, y = main_ax.xaxis.label.get_position()
            main_ax.xaxis.label.set_position((1.0, y))

        if raw_title or title:

            title_text = raw_title if raw_title else title

            main_ax.title.set_text(
                _fix_rtl_text(title_text)
            )

            main_ax.title.set_fontproperties(title_font)
            main_ax.title.set_ha("right")

            _, y = main_ax.title.get_position()
            main_ax.title.set_position((1.0, y))

    # ------------------------------------------------------------------
    # Tick font
    # ------------------------------------------------------------------

    for tick in (
        main_ax.get_xticklabels()
        + main_ax.get_yticklabels()
    ):
        tick.set_fontfamily("DejaVu Sans")

    # ------------------------------------------------------------------
    # Return / Show
    # ------------------------------------------------------------------

    if show:
        plt.show()
    else:
        return main_ax