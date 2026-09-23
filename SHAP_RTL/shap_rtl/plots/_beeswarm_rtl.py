"""
shap_rtl.plots.beeswarm-  RTL-aware beeswarm plot.
"""

from __future__ import annotations
from typing import Literal

import matplotlib, matplotlib.pyplot as plt
import numpy as np, pandas as pd
import scipy.cluster, scipy.sparse, scipy.spatial
from matplotlib.figure import Figure
from packaging import version
from scipy.stats import gaussian_kde

from shap import Explanation
from shap.utils import safe_isinstance
from shap.utils._exceptions import DimensionError
from shap.plots import colors
from shap.plots._labels import labels
from shap.plots._utils import (convert_color, convert_ordering,
                                get_sort_order, merge_nodes, sort_inds)

from ..rtl_utils import _find_rtl_font, is_rtl_text

if version.parse(matplotlib.__version__) >= version.parse("3.10"):
    ORIENTATION_KWARG = dict(orientation="horizontal")
else:
    ORIENTATION_KWARG = dict(vert=False)


def _inject_image_label(ax, fig, text, ydata, font_path, color="black", font_size_pt=12):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from ..rtl_utils import render_rtl_label
    import matplotlib.colors as mcolors
    img_pil = render_rtl_label(text, font_path, font_size_pt=font_size_pt, dpi=150)
    if img_pil.width < 2: return
    arr = np.array(img_pil, dtype=np.uint8).copy()
    try:
        rgb = mcolors.to_rgb(color)
        arr[:,:,0]=int(rgb[0]*255); arr[:,:,1]=int(rgb[1]*255); arr[:,:,2]=int(rgb[2]*255)
    except Exception: pass
    oi = OffsetImage(arr, zoom=72.0/150.0)
    oi.image.axes = ax
    ab = AnnotationBbox(oi, xy=(0,ydata), xycoords=("axes fraction","data"),
                        xybox=(-4,0), boxcoords="offset points",
                        frameon=False, box_alignment=(1.0,0.5), pad=0)
    ax.add_artist(ab)


def beeswarm(
    shap_values: Explanation,
    max_display: int | None = 10,
    order=Explanation.abs.mean(0),
    clustering=None, cluster_threshold=0.5, color=None,
    axis_color="#000000", alpha: float = 1.0, ax=None,
    show: bool = True, log_scale: bool = False,
    color_bar: bool = True, s: float = 16,
    plot_size: Literal["auto"] | float | tuple | None = "auto",
    color_bar_label: str = labels["FEATURE_VALUE"],
    group_remaining_features: bool = True,
):
    """RTL-aware beeswarm plot (drop-in for shap.plots.beeswarm)."""
    if not isinstance(shap_values, Explanation):
        raise TypeError("shap_values must be an Explanation object.")
    sv_shape = shap_values.shape
    if len(sv_shape) == 1:
        raise ValueError("Pass a matrix explanation (many instances), not a single row.")
    if len(sv_shape) > 2:
        raise ValueError("Beeswarm does not support >2-D explanations.")
    if ax and plot_size:
        raise ValueError("Cannot pass both ax and plot_size.")

    values   = np.copy(shap_values.values)
    features = shap_values.data
    if scipy.sparse.issparse(features): features = features.toarray()
    feature_names = shap_values.feature_names
    order = convert_ordering(order, values)

    if color is None:
        color = colors.red_blue if features is not None else colors.blue_rgb
    color = convert_color(color)

    idx2cat = None
    if isinstance(features, pd.DataFrame):
        if feature_names is None: feature_names = features.columns
        idx2cat  = features.dtypes.astype(str).isin(["object","category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None: feature_names = features
        features = None
    elif features is not None and len(features.shape)==1 and feature_names is None:
        feature_names = features; features = None

    num_features = values.shape[1]
    if features is not None:
        if num_features - 1 == features.shape[1]:
            raise DimensionError("Shape mismatch – maybe shap_values[:,:-1]?")
        if num_features != features.shape[1]:
            raise DimensionError("Shape mismatch between shap_values and features.")

    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])

    if ax is None: ax = plt.gca()
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    if log_scale: ax.set_xscale("symlog")

    if clustering is None:
        partition_tree = getattr(shap_values, "clustering", None)
        if partition_tree is not None and partition_tree.var(0).sum() == 0:
            partition_tree = partition_tree[0]
        else: partition_tree = None
    elif clustering is False: partition_tree = None
    else: partition_tree = clustering

    if max_display is None: max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    orig_inds   = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()
    while True:
        feature_order = convert_ordering(order, Explanation(np.abs(values)))
        if partition_tree is not None:
            clust_order  = sort_inds(partition_tree, np.abs(values))
            dist         = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, cluster_threshold, feature_order)
            if (max_display < len(feature_order)
                    and dist[feature_order[max_display-1], feature_order[max_display-2]] <= cluster_threshold):
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values), partition_tree)
                for _ in range(len(values)):
                    values[:,ind1]+=values[:,ind2]; values=np.delete(values,ind2,1)
                    orig_inds[ind1]+=orig_inds[ind2]; del orig_inds[ind2]
            else: break
        else: break

    feature_inds = feature_order[:max_display]
    feature_names_new = []
    for inds in orig_inds:
        if len(inds)==1: feature_names_new.append(str(feature_names[inds[0]]))
        elif len(inds)<=2: feature_names_new.append(" + ".join([str(feature_names[i]) for i in inds]))
        else:
            mi = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(f"{feature_names[inds[mi]]} + {len(inds)-1} other features")
    feature_names = feature_names_new

    rtl_mode     = any(is_rtl_text(str(fn)) for fn in feature_names)
    font_path, _ = _find_rtl_font()

    include_grouped = num_features < len(values[0]) and group_remaining_features
    if include_grouped:
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features-1, len(values[0]))])
        values[:,feature_order[num_features-1]] = np.sum(
            [values[:,feature_order[i]] for i in range(num_features-1, len(values[0]))], 0)

    yticklabels = [str(feature_names[i]) for i in feature_inds]
    if include_grouped: yticklabels[-1] = f"Sum of {num_cut} other features"

    row_height = 0.4
    if plot_size == "auto":
        fig.set_size_inches(8, min(len(feature_order), max_display) * row_height + 1.5)
    elif isinstance(plot_size, (list,tuple)): fig.set_size_inches(*plot_size)
    elif plot_size is not None:
        fig.set_size_inches(8, min(len(feature_order), max_display) * plot_size + 1.5)
    ax.axvline(x=0, color="#000000", zorder=-1)

    for pos, i in enumerate(reversed(feature_inds)):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1,5), zorder=-1)
        shaps  = values[:,i]
        fvalues = None if features is None else features[:,i]
        f_inds = np.arange(len(shaps)); np.random.shuffle(f_inds)
        if fvalues is not None: fvalues = fvalues[f_inds]
        shaps = shaps[f_inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]: colored_feature = False
            else: fvalues = np.array(fvalues, dtype=np.float64)
        except Exception: colored_feature = False
        N     = len(shaps)
        nbins = 100
        quant = np.round(nbins*(shaps-np.min(shaps))/(np.max(shaps)-np.min(shaps)+1e-8))
        inds_ = np.argsort(quant+np.random.randn(N)*1e-6)
        layer = 0; last_bin = -1; ys = np.zeros(N)
        for ind in inds_:
            if quant[ind] != last_bin: layer = 0
            ys[ind] = np.ceil(layer/2)*((layer%2)*2-1); layer+=1; last_bin=quant[ind]
        ys *= 0.9*(row_height/np.max(ys+1))

        if safe_isinstance(color,"matplotlib.colors.Colormap") and fvalues is not None and colored_feature:
            vmin=np.nanpercentile(fvalues,5); vmax=np.nanpercentile(fvalues,95)
            if vmin==vmax: vmin=np.nanpercentile(fvalues,1); vmax=np.nanpercentile(fvalues,99)
            if vmin==vmax: vmin=np.min(fvalues); vmax=np.max(fvalues)
            if vmin>vmax: vmin=vmax
            if features is not None and features.shape[0]!=len(shaps):
                raise DimensionError("Feature/SHAP row count mismatch")
            nm = np.isnan(fvalues)
            ax.scatter(shaps[nm], pos+ys[nm], color="#000000", s=s, alpha=alpha, linewidth=0, zorder=3, rasterized=len(shaps)>500)
            cv = fvalues[~nm].astype(np.float64); cvi = cv.copy()
            cvi[np.isnan(cv)]=(vmin+vmax)/2.; cv[cvi>vmax]=vmax; cv[cvi<vmin]=vmin
            ax.scatter(shaps[~nm], pos+ys[~nm], cmap=color, vmin=vmin, vmax=vmax,
                       s=s, c=cv, alpha=alpha, linewidth=0, zorder=3, rasterized=len(shaps)>500)
        else:
            if safe_isinstance(color,"matplotlib.colors.Colormap") and hasattr(color,"colors"): color=color.colors
            ax.scatter(shaps, pos+ys, s=s, alpha=alpha, linewidth=0, zorder=3,
                       color=color if colored_feature else "#000000", rasterized=len(shaps)>500)

    if safe_isinstance(color,"matplotlib.colors.Colormap") and color_bar and features is not None:
        import matplotlib.cm as cm
        m  = cm.ScalarMappable(cmap=color); m.set_array([0,1])
        cb = fig.colorbar(m, ax=ax, ticks=[0,1], aspect=80)
        cb.set_ticklabels([labels["FEATURE_VALUE_LOW"],labels["FEATURE_VALUE_HIGH"]])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0); cb.set_alpha(1); cb.outline.set_visible(False)

    ax.xaxis.set_ticks_position("bottom"); ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False); ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(range(len(feature_inds)), list(reversed(yticklabels)), fontsize=13)
    ax.tick_params("y", length=20, width=0.5, which="major")
    ax.tick_params("x", labelsize=11)
    ax.set_ylim(-1, len(feature_inds))
    ax.set_xlabel(labels["VALUE"], fontsize=13)

    # ── RTL image injection ───────────────────────────────────────────────────
    if rtl_mode and font_path is not None:
        fig.canvas.draw()
        ticks_mpl   = ax.get_yticklabels()
        reversed_labels = list(reversed(yticklabels))
        for j, (tick_obj, ltext) in enumerate(zip(ticks_mpl, reversed_labels)):
            if is_rtl_text(ltext):
                tick_obj.set_visible(False)
                _inject_image_label(ax, fig, ltext, j, font_path, color=axis_color)

    if show: plt.show()
    else:    return ax
