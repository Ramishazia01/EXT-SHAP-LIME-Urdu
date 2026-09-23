"""
shap_rtl.plots.waterfall  –  RTL-aware waterfall plot.

Identical to shap.plots.waterfall except that feature names containing
Urdu/Arabic characters are rendered as properly-shaped HarfBuzz images.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap import Explanation
from shap.utils import format_value
from shap.plots._labels import labels
from shap.plots._style import get_style

from ..rtl_utils import (
    _find_rtl_font,
    is_rtl_text,
    render_rtl_label,
    render_mixed_label,
    apply_rtl_ytick_images,
    hide_rtl_text_ticks,
)


def waterfall(shap_values, max_display=10, show=True):
    """RTL-aware waterfall plot (drop-in for shap.plots.waterfall)."""
    style = get_style()
    if show is False:
        plt.ioff()

    if not isinstance(shap_values, Explanation):
        raise TypeError(
            "The waterfall plot requires an `Explanation` object as the `shap_values` argument."
        )
    sv_shape = shap_values.shape
    if len(sv_shape) != 1:
        raise ValueError(
            f"The waterfall plot needs a single explanation (shape {sv_shape} passed). "
            "Try shap_rtl.plots.waterfall(shap_values[0])."
        )

    base_values  = float(shap_values.base_values)
    features     = (shap_values.display_data if shap_values.display_data is not None
                    else shap_values.data)
    feature_names = shap_values.feature_names
    lower_bounds  = getattr(shap_values, "lower_bounds", None)
    upper_bounds  = getattr(shap_values, "upper_bounds", None)
    values        = shap_values.values

    if isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(len(values))])

    # ── detect RTL ────────────────────────────────────────────────────────────
    rtl_mode  = any(is_rtl_text(str(fn)) for fn in feature_names)
    font_path, _ = _find_rtl_font()

    # ── original SHAP mechanics (unchanged) ───────────────────────────────────
    num_features  = min(max_display, len(values))
    row_height    = 0.5
    rng           = range(num_features - 1, -1, -1)
    order         = np.argsort(-np.abs(values))
    pos_lefts, pos_inds, pos_widths, pos_low, pos_high = [], [], [], [], []
    neg_lefts, neg_inds, neg_widths, neg_low, neg_high = [], [], [], [], []
    loc           = base_values + values.sum()
    yticklabels   = ["" for _ in range(num_features + 1)]

    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    num_individual = num_features if num_features == len(values) else num_features - 1

    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i]);  pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]]); pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i]);  neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]]); neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)

        if num_individual != num_features or i + 4 < num_individual:
            plt.plot([loc, loc], [rng[i] - 1 - 0.4, rng[i] + 0.4],
                     color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)

        raw_fn = str(feature_names[order[i]])
        # CRITICAL FIX: Only store the feature name, NOT the value
        yticklabels[rng[i]] = raw_fn

    if num_features < len(values):
        yticklabels[0] = f"{len(shap_values) - num_features + 1} other features"
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0); pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0); neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points  = (pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths))
               + neg_lefts + list(np.array(neg_lefts) + np.array(neg_widths)))
    dataw   = np.max(points) - np.min(points)

    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02 * dataw,
             left=np.array(pos_lefts) - 0.01 * dataw,
             color=style.primary_color_positive, alpha=0)
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02 * dataw,
             left=np.array(neg_lefts) + 0.01 * dataw,
             color=style.primary_color_negative, alpha=0)

    head_length   = 0.08
    bar_width     = 0.8
    xlen          = plt.xlim()[1] - plt.xlim()[0]
    fig           = plt.gcf()
    ax            = plt.gca()
    bbox          = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox_to_xscale = xlen / bbox.width
    hl_scaled     = bbox_to_xscale * head_length
    renderer      = fig.canvas.get_renderer()

    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(pos_lefts[i], pos_inds[i], dist - hl_scaled, 0,
                              head_length=min(dist, hl_scaled),
                              color=style.primary_color_positive,
                              width=bar_width, head_width=bar_width)
        if pos_low and i < len(pos_low):
            plt.errorbar(pos_lefts[i] + pos_widths[i], pos_inds[i],
                         xerr=np.array([[pos_widths[i] - pos_low[i]],
                                        [pos_high[i] - pos_widths[i]]]),
                         ecolor=style.secondary_color_positive)
        # Add SHAP value INSIDE the bar
        txt_obj = plt.text(pos_lefts[i] + 0.5 * dist, pos_inds[i],
                           format_value(pos_widths[i], "%+0.02f"),
                           horizontalalignment="center", verticalalignment="center",
                           color=style.text_color, fontsize=12)
        if txt_obj.get_window_extent(renderer=renderer).width > arrow_obj.get_window_extent(renderer=renderer).width:
            txt_obj.remove()
            plt.text(pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist, pos_inds[i],
                     format_value(pos_widths[i], "%+0.02f"),
                     horizontalalignment="left", verticalalignment="center",
                     color=style.primary_color_positive, fontsize=12)

    for i in range(len(neg_inds)):
        dist = neg_widths[i]
        arrow_obj = plt.arrow(neg_lefts[i], neg_inds[i], -(-dist - hl_scaled), 0,
                              head_length=min(-dist, hl_scaled),
                              color=style.primary_color_negative,
                              width=bar_width, head_width=bar_width)
        if neg_low and i < len(neg_low):
            plt.errorbar(neg_lefts[i] + neg_widths[i], neg_inds[i],
                         xerr=np.array([[neg_widths[i] - neg_low[i]],
                                        [neg_high[i] - neg_widths[i]]]),
                         ecolor=style.secondary_color_negative)
        # Add SHAP value INSIDE the bar
        txt_obj = plt.text(neg_lefts[i] + 0.5 * dist, neg_inds[i],
                           format_value(neg_widths[i], "%+0.02f"),
                           horizontalalignment="center", verticalalignment="center",
                           color=style.text_color, fontsize=12)
        if txt_obj.get_window_extent(renderer=renderer).width > arrow_obj.get_window_extent(renderer=renderer).width:
            txt_obj.remove()
            plt.text(neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist, neg_inds[i],
                     format_value(neg_widths[i], "%+0.02f"),
                     horizontalalignment="right", verticalalignment="center",
                     color=style.primary_color_negative, fontsize=12)

    # ── y-ticks ───────────────────────────────────────────────────────────────
    # CRITICAL FIX: Only use feature names (no values on the left)
    ytick_pos = list(range(num_features))
    yticklabels_only = yticklabels[:-1]  # Remove the extra empty label
    
    # Set yticks with ONLY feature names (no values)
    plt.yticks(ytick_pos, yticklabels_only, fontsize=13)

    # horizontal feature lines
    for i in range(num_features):
        plt.axhline(i, color=style.hlines_color, lw=0.5, dashes=(1, 5), zorder=-1)

    plt.axvline(base_values, 0, 1 / num_features,
                color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)

    xmin, xmax = ax.get_xlim()

    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks([base_values, base_values + min(1e-8, xmax * 1e-10)])
    ax2.set_xticklabels(
        ["\n$E[f(X)]$", "\n$ = " + format_value(base_values, "%0.03f") + "$"],
        fontsize=12, ha="left")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    ax3.set_xticks([fx, fx + min(1e-8, xmax * 1e-10)])
    ax3.set_xticklabels(["$f(x)$", "$ = " + format_value(fx, "%0.03f") + "$"],
                         fontsize=12, ha="left")
    tl = ax3.xaxis.get_majorticklabels()
    tl[0].set_transform(tl[0].get_transform()
                        + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tl[1].set_transform(tl[1].get_transform()
                        + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tl[1].set_color(style.tick_labels_color)
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    tl2 = ax2.xaxis.get_majorticklabels()
    tl2[0].set_transform(tl2[0].get_transform()
                         + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tl2[1].set_transform(tl2[1].get_transform()
                         + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))
    tl2[1].set_color(style.tick_labels_color)

    tick_labels_mpl = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels_mpl[i].set_color(style.tick_labels_color)

    # ── RTL image injection ───────────────────────────────────────────────────
    if rtl_mode and font_path is not None:
        fig.canvas.draw()
        
        # Get all tick labels
        tick_objs = ax.get_yticklabels()
        
        # Hide ALL existing tick labels (we'll replace with RTL images if needed)
        for tick_obj in tick_objs:
            tick_obj.set_visible(False)
        
        # Only render RTL feature names (no values)
        for j, label_text in enumerate(yticklabels_only):
            if is_rtl_text(str(label_text)):
                _inject_image_label(
                    ax, fig, str(label_text), j,
                    font_path,
                    color="black",
                    font_size_pt=13,
                    mixed=False  # Pure RTL text
                )

    if show:
        plt.show()
    else:
        return plt.gca()


def _inject_image_label(ax, fig, text: str, ydata: float, font_path: str,
                        color: str = "black", font_size_pt: float = 12,
                        mixed: bool = False):
    """Render *text* as an image and place it left of the y-axis at *ydata*."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from ..rtl_utils import render_rtl_label, render_mixed_label
    import matplotlib.colors as mcolors
    
    if not text or not isinstance(text, str):
        return

    try:
        if mixed:
            img_pil = render_mixed_label(text, font_path, font_size_pt=font_size_pt, dpi=150)
        else:
            img_pil = render_rtl_label(text, font_path, font_size_pt=font_size_pt, dpi=150)
    except Exception as e:
        try:
            img_pil = render_rtl_label(text, font_path, font_size_pt=font_size_pt, dpi=150)
        except:
            return
    
    if img_pil is None or img_pil.width < 2:
        return

    arr = np.array(img_pil, dtype=np.uint8).copy()
    
    # Force black color
    try:
        rgb = (0, 0, 0)  # Always black
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j, 3] > 0:
                    arr[i, j, 0] = 0
                    arr[i, j, 1] = 0
                    arr[i, j, 2] = 0
    except Exception:
        pass

    zoom = 72.0 / 150.0
    oi = OffsetImage(arr, zoom=zoom)
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