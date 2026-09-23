"""
shap_rtl.plots.bar - RTL-aware bar plot.

Identical to shap.plots.bar except Urdu/Arabic/Persian feature names are rendered
via HarfBuzz+FreeType images instead of matplotlib text glyphs.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from shap import Cohorts, Explanation
from shap.utils import format_value, ordinal_str
from shap.utils._exceptions import DimensionError
from shap.plots._labels import labels
from shap.plots._style import get_style
from shap.plots._utils import (
    convert_ordering, dendrogram_coords, get_sort_order, merge_nodes, sort_inds,
)

from ..rtl_utils import (
    _find_rtl_font, is_rtl_text,
    apply_rtl_ytick_images, hide_rtl_text_ticks,
)


def _inject_image_label(ax, fig, text, ydata, font_path, color="black", font_size_pt=12):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from ..rtl_utils import render_rtl_label
    import matplotlib.colors as mcolors
    
    img_pil = render_rtl_label(text, font_path, font_size_pt=font_size_pt, dpi=150)
    if img_pil.width < 2:
        return
    arr = np.array(img_pil, dtype=np.uint8).copy()
    try:
        rgb = mcolors.to_rgb(color)
        arr[:,:,0] = int(rgb[0]*255); arr[:,:,1] = int(rgb[1]*255); arr[:,:,2] = int(rgb[2]*255)
    except Exception:
        pass
    oi = OffsetImage(arr, zoom=72.0/150.0)
    oi.image.axes = ax
    ab = AnnotationBbox(oi, xy=(0, ydata), xycoords=("axes fraction","data"),
                        xybox=(-4, 0), boxcoords="offset points",
                        frameon=False, box_alignment=(1.0, 0.5), pad=0)
    ax.add_artist(ab)


def bar(shap_values, max_display=10, order=Explanation.abs,
        clustering=None, clustering_cutoff=0.5, show_data="auto", ax=None, show=True):
    """RTL-aware bar plot (drop-in replacement for shap.plots.bar)."""
    
    # ============================================================
    # FORCE BLACK TEXT COLOR FOR ALL TEXT ELEMENTS
    # ============================================================
    import matplotlib as mpl
    mpl.rcParams['text.color'] = 'black'
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'
    mpl.rcParams['axes.edgecolor'] = 'black'
    # ============================================================
    
    style = get_style()

    if isinstance(shap_values, Explanation):
        cohorts = {"": shap_values}
    elif isinstance(shap_values, Cohorts):
        cohorts = shap_values.cohorts
    elif isinstance(shap_values, dict):
        cohorts = shap_values
    else:
        raise TypeError("shap_values must be Explanation, Cohorts, or dict")

    cohort_labels = list(cohorts.keys())
    cohort_exps   = list(cohorts.values())
    for i, exp in enumerate(cohort_exps):
        if not isinstance(exp, Explanation):
            raise TypeError("Each cohort value must be an Explanation object")
        if len(exp.shape) == 2:
            cohort_exps[i] = exp.abs.mean(0)
        if cohort_exps[i].shape != cohort_exps[0].shape:
            raise DimensionError("All Explanation objects must have the same number of features")

    features      = (cohort_exps[0].display_data if cohort_exps[0].display_data is not None
                     else cohort_exps[0].data)
    feature_names = cohort_exps[0].feature_names
    partition_tree = (None if clustering is False
                      else clustering if clustering is not None
                      else getattr(cohort_exps[0], "clustering", None))
    if partition_tree is not None and (len(partition_tree.shape) != 2 or partition_tree.shape[1] != 4):
        raise TypeError("clustering must be a partition tree (shape Nx4)")

    op_history = cohort_exps[0].op_history
    values     = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])

    if len(values[0]) == 0:
        raise ValueError("Empty Explanation – nothing to plot")

    if show_data == "auto":
        transforms = [op for op in op_history if op.name != "__getitem__"]
        show_data  = len(transforms) == 0

    if issubclass(type(feature_names), str):
        feature_names = [ordinal_str(i) + " " + feature_names for i in range(len(values[0]))]

    xlabel = "SHAP value"
    for op in op_history:
        if op.name == "abs":   xlabel = f"|{xlabel}|"
        elif op.name != "__getitem__": xlabel = f"{op.name}({xlabel})"

    cohort_sizes = []
    for exp in cohort_exps:
        for op in exp.op_history:
            if op.collapsed_instances:
                cohort_sizes.append(op.prev_shape[0]); break

    if isinstance(features, pd.Series):
        if feature_names is None: feature_names = list(features.index)
        features = features.values
    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(len(values[0]))])

    if max_display is None: max_display = len(feature_names)
    num_features = min(max_display, len(values[0]))
    max_display  = num_features

    orig_inds   = [[i] for i in range(len(values[0]))]
    orig_values = values.copy()
    while True:
        feature_order = np.argsort(np.mean(
            [np.argsort(convert_ordering(order, Explanation(values[i]))) for i in range(values.shape[0])], 0))
        if partition_tree is not None:
            clust_order  = sort_inds(partition_tree, np.abs(values).mean(0))
            dist         = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, clustering_cutoff, feature_order)
            if (max_display < len(feature_order)
                    and dist[feature_order[max_display-1], feature_order[max_display-2]] <= clustering_cutoff):
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values).mean(0), partition_tree)
                for _ in range(len(values)):
                    values[:,ind1] += values[:,ind2]; values = np.delete(values,ind2,1)
                    orig_inds[ind1] += orig_inds[ind2]; del orig_inds[ind2]
            else: break
        else: break

    feature_inds = feature_order[:max_display]
    y_pos        = np.arange(len(feature_inds), 0, -1)

    feature_names_new = []
    for inds in orig_inds:
        if len(inds) == 1:
            feature_names_new.append(str(feature_names[inds[0]]))
        else:
            full = " + ".join([str(feature_names[i]) for i in inds])
            if len(full) <= 40: feature_names_new.append(full)
            else:
                mi = np.argmax(np.abs(orig_values).mean(0)[inds])
                feature_names_new.append(f"{feature_names[inds[mi]]} + {len(inds)-1} other features")
    feature_names = feature_names_new

    rtl_mode   = any(is_rtl_text(str(fn)) for fn in feature_names)
    font_path, _ = _find_rtl_font()

    if num_features < len(values[0]):
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features-1, len(values[0]))])
        values[:,feature_order[num_features-1]] = np.sum(
            [values[:,feature_order[i]] for i in range(num_features-1, len(values[0]))], 0)

    yticklabels = []
    for i in feature_inds:
        yticklabels.append(str(feature_names[i]))
        
    if num_features < len(values[0]):
        yticklabels[-1] = f"Sum of {num_cut} other features"

    if ax is None:
        ax  = plt.gca()
        fig = plt.gcf()
        fig.set_size_inches(8, num_features * 0.5 * np.sqrt(len(values)) + 1.5)

    negative_values_present = np.sum(values[:, feature_order[:num_features]] < 0) > 0
    if negative_values_present:
        ax.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    patterns   = (None, "\\\\", "++", "xx", "////", "*", "o", "O", ".", "-")
    total_width = 0.7
    bar_width  = total_width / len(values)
    for i in range(len(values)):
        ypos_offset = -((i - len(values)/2) * bar_width + bar_width/2)
        ax.barh(y_pos + ypos_offset, values[i, feature_inds], bar_width, align="center",
                color=[style.primary_color_negative if values[i, feature_inds[j]] <= 0
                       else style.primary_color_positive for j in range(len(y_pos))],
                hatch=patterns[i], edgecolor=(1,1,1,0.8),
                label=f"{cohort_labels[i]} [{cohort_sizes[i] if i < len(cohort_sizes) else None}]")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(yticklabels, fontsize=13)

    xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    bbox_to_xscale = xlen / bbox.width

    for i in range(len(values)):
        ypos_offset = -((i - len(values)/2) * bar_width + bar_width/2)
        for j in range(len(y_pos)):
            ind = feature_order[j]
            if values[i, ind] < 0:
                ax.text(values[i,ind] - (5/72)*bbox_to_xscale, y_pos[j]+ypos_offset,
                        format_value(values[i,ind], "%+0.02f"),
                        horizontalalignment="right", verticalalignment="center",
                        color="black", fontsize=12)  # ← FORCED BLACK
            else:
                ax.text(values[i,ind] + (5/72)*bbox_to_xscale, y_pos[j]+ypos_offset,
                        format_value(values[i,ind], "%+0.02f"),
                        horizontalalignment="left", verticalalignment="center",
                        color="black", fontsize=12)  # ← FORCED BLACK

    for i in range(num_features):
        ax.axhline(i+1, color="#101010", lw=0.5, dashes=(1,5), zorder=-1)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if negative_values_present: ax.spines["left"].set_visible(False)
    ax.tick_params("x", labelsize=11, colors="black")  # ← FORCED BLACK

    xmin, xmax = ax.get_xlim()
    x_buffer   = (xmax - xmin) * 0.05
    ax.set_xlim(xmin - x_buffer if negative_values_present else xmin, xmax + x_buffer)
    ax.set_xlabel(xlabel, fontsize=13, color="black")  # ← FORCED BLACK
    if len(values) > 1: 
        legend = ax.legend(fontsize=12)
        for text in legend.get_texts():
            text.set_color("black")  # ← FORCED BLACK

    fig = ax.figure
    fig.canvas.draw()
    
    tick_labels_mpl = ax.yaxis.get_majorticklabels()
    for i in range(min(num_features, len(tick_labels_mpl))):
        tick_labels_mpl[i].set_color("black")  # ← FORCED BLACK

    # RTL rendering
    if rtl_mode and font_path is not None:
        fig.canvas.draw()
        ticks_all = ax.get_yticklabels()
        
        for j, (tick_obj, ltext) in enumerate(zip(ticks_all, yticklabels)):
            if is_rtl_text(ltext):
                tick_obj.set_text("")
                tick_obj.set_visible(False)
                _inject_image_label(ax, fig, ltext, y_pos[j], font_path, color="black")
            else:
                tick_obj.set_visible(True)
                tick_obj.set_color("black")  # ← FORCED BLACK
                tick_obj.set_alpha(1.0)

    # Dendrogram visualization tracking
    if partition_tree is not None:
        feature_pos = np.argsort(feature_order)
        ylines, xlines = dendrogram_coords(feature_pos, partition_tree)
        xmin2, xmax2 = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        xl_min, xl_max = np.min(xlines), np.max(xlines)
        ct_pos = (clustering_cutoff/(xl_max-xl_min))*0.1*(xmax2-xmin2)+xmax2
        ax.text(ct_pos+0.005*(xmax2-xmin2), (ymax-ymin)/2,
                "Clustering cutoff = "+format_value(clustering_cutoff,"%0.02f"),
                ha="left", va="center", color="black", fontsize=12, rotation=-90)  # ← FORCED BLACK
        ln = ax.axvline(ct_pos, color="#0e0d0d", dashes=(1,1)); ln.set_clip_on(False)
        for xl, yl in zip(xlines, ylines):
            xv = np.array(xl)/(xl_max-xl_min)
            if np.array(xl).max() <= clustering_cutoff and np.array(yl).max() < max_display:
                ls = ax.plot(xv*0.1*(xmax2-xmin2)+xmax2, max_display-np.array(yl), color="#000000")
                for l in ls: l.set_clip_on(False)

    if show: plt.show()
    else:    return ax