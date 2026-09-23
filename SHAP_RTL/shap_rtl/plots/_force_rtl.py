"""
shap_rtl.plots.force – RTL-aware force (additive) plot.

Leverages the original shap.plots.force core module for robust data tracking,
then post-intercepts the text elements to inject HarfBuzz-rendered labels
displaying true live model metrics.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap

from ..rtl_utils import _find_rtl_font, is_rtl_text, render_rtl_label


def _text_as_image(ax, x, y, text, font_path, fontsize=12, color="#FF0D57", ha="right"):
    """Places a HarfBuzz-rendered image of the text at axes data coordinates."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import matplotlib.colors as mcolors

    img_pil = render_rtl_label(text, font_path, font_size_pt=fontsize, dpi=150)
    if img_pil.width < 2:
        return None

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

    box_align = (1.0, 0.5) if ha == "right" else (0.0, 0.5)

    ab = AnnotationBbox(
        oi,
        xy=(x, y),
        xycoords="data",
        frameon=False,
        box_alignment=box_align,
        pad=0,
    )
    ax.add_artist(ab)
    return ab


def _highlight_text_below_plot(sample_text, feature_names, shap_values, class_name="Prediction"):
    """
    Create a highlighted text visualization below the force plot.
    Red = Positive contribution, Blue = Negative contribution.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create color mapping for features
    color_map = {}
    for name, val in zip(feature_names, shap_values):
        if val > 0:
            color_map[name] = ('red', val)
        else:
            color_map[name] = ('blue', val)
    
    # Create figure for highlighted text
    fig, ax = plt.subplots(figsize=(16, 2.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title
    ax.text(0.5, 0.92, f'📝 Text Highlighting - {class_name} Class', 
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='top')
    ax.text(0.5, 0.82, '🔴 Red = Pushes TOWARD | 🔵 Blue = Pushes AWAY | ⬛ No significant impact', 
            transform=ax.transAxes, fontsize=11, ha='center', va='top')
    
    # Position for text
    x_pos = 0.05
    y_pos = 0.5
    font_size = 20
    
    # Process each word
    for word in sample_text.split():
        found = False
        highlight_color = None
        
        # Check if word or its variants are in color_map
        for feature, (color, val) in color_map.items():
            if feature in word or word in feature:
                highlight_color = color
                found = True
                break
        
        # Create text with highlight (like a highlighter pen)
        if highlight_color == 'red':
            ax.text(x_pos, y_pos, word, 
                   transform=ax.transAxes,
                   fontsize=font_size, 
                   color='black',
                   fontweight='bold',
                   bbox=dict(boxstyle='square,pad=0.05', 
                            facecolor='red', 
                            alpha=0.3,
                            edgecolor='none',
                            linewidth=0))
        elif highlight_color == 'blue':
            ax.text(x_pos, y_pos, word, 
                   transform=ax.transAxes,
                   fontsize=font_size, 
                   color='black',
                   fontweight='bold',
                   bbox=dict(boxstyle='square,pad=0.05', 
                            facecolor='blue', 
                            alpha=0.3,
                            edgecolor='none',
                            linewidth=0))
        else:
            ax.text(x_pos, y_pos, word, 
                   transform=ax.transAxes,
                   fontsize=font_size, 
                   color='black',
                   fontweight='normal')
        
        # Update position
        x_pos += len(word) * 0.013 + 0.005
        
        # Wrap text if needed
        if x_pos > 0.95:
            x_pos = 0.05
            y_pos -= 0.12
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='red', alpha=0.3, edgecolor='none', label='🔴 Pushes TOWARD'),
        patches.Patch(facecolor='blue', alpha=0.3, edgecolor='none', label='🔵 Pushes AWAY'),
        patches.Patch(facecolor='white', alpha=0.5, edgecolor='none', label='⬛ No significant impact')
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
             bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11,
             frameon=False)
    
    plt.tight_layout()
    plt.show()
    
    return color_map


def force(shap_values, figsize=(20, 3), show=True, text_rotation=0, min_perc=0.05, 
          sample_text=None, class_name=None, max_features=10):
    """
    RTL-aware force plot using the original SHAP module directly.
    Extracts live model feature inputs and actual SHAP metrics directly from the graph.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values to plot
    figsize : tuple
        Figure size
    show : bool
        Whether to show the plot
    text_rotation : int
        Rotation angle for text
    min_perc : float
        Minimum percentage threshold
    sample_text : str, optional
        The original sample text to display below the force plot with highlighted words
    class_name : str, optional
        The class name for the title
    max_features : int
        Maximum number of features to show
    """
    # 1. Turn off immediate rendering so we can process the figure elements safely
    plt.ioff()

    # 2. Extract configuration from the Explanation object if present to detect RTL text
    feature_names = getattr(shap_values, "feature_names", None)
    
    rtl_mode = False
    if feature_names is not None:
        rtl_mode = any(is_rtl_text(str(n)) for n in feature_names)
    
    font_path, _ = _find_rtl_font()

    # 3. Call the ORIGINAL native SHAP force plot module directly.
    # This guarantees that the real feature names, model metrics, and internal tracking are preserved perfectly.
    fig = shap.plots.force(
        shap_values,
        matplotlib=True,
        figsize=figsize,
        show=False,
        text_rotation=text_rotation,
        contribution_threshold=min_perc
    )
    
    if fig is None:
        fig = plt.gcf()
    ax = fig.gca()

    # 4. Post-process the generated text elements
    if font_path is not None:
        fig.canvas.draw()
        
        # Capture all raw text labels generated below the bars by original SHAP
        all_texts = list(ax.texts)
        
        for txt in all_texts:
            label_string = txt.get_text()
            
            # Skip empty strings, or layout controls like 'higher', 'lower', or 'f(x)'
            if not label_string or label_string in ["higher", "lower", "f(x)"] or "=" not in label_string:
                continue
                
            # Original SHAP text is structured as: "model_feature_value = feature_name"
            parts = label_string.split("=", 1)
            feat_val_part = parts[0].strip()   # This is the actual feature value from the model pipeline
            feat_name_part = parts[1].strip()  # This is the string label or token word
            
            # Check if this feature name belongs to an RTL script language sequence
            if is_rtl_text(feat_name_part):
                x_pos, y_pos = txt.get_position()
                ha_align = txt.get_transform_radial() if hasattr(txt, 'get_transform_radial') else txt.get_ha()
                txt_color = txt.get_color()
                
                # Format dynamically to show the feature word and the real feature value side by side
                clean_rtl_label = f"{feat_name_part} ({feat_val_part})"
                
                # Turn off the native text element entirely to hide broken LTR square bracket artifacts
                txt.set_text("")
                txt.set_visible(False)
                
                # Inject the clean HarfBuzz script image at the precise native text anchor position
                _text_as_image(
                    ax=ax,
                    x=x_pos,
                    y=y_pos,
                    text=clean_rtl_label,
                    font_path=font_path,
                    fontsize=12,
                    color=txt_color,
                    ha=ha_align
                )

    if show:
        plt.show()
    
    # 5. Display the highlighted sample text below the force plot
    if sample_text is not None:
        # Get SHAP values for highlighting
        if hasattr(shap_values, 'values'):
            if shap_values.values.ndim == 2:
                shap_vals = shap_values.values[0] if len(shap_values.values) > 0 else shap_values.values
            else:
                shap_vals = shap_values.values
        else:
            shap_vals = shap_values
        
        # Get feature names
        if hasattr(shap_values, 'feature_names'):
            f_names = shap_values.feature_names
        else:
            f_names = feature_names
        
        # Create feature to SHAP value mapping
        feature_shap_map = {}
        if isinstance(shap_vals, np.ndarray):
            if len(shap_vals.shape) == 1:
                for i, name in enumerate(f_names):
                    if i < len(shap_vals):
                        feature_shap_map[name] = shap_vals[i]
            elif len(shap_vals.shape) == 2 and shap_vals.shape[0] == 1:
                for i, name in enumerate(f_names):
                    if i < len(shap_vals[0]):
                        feature_shap_map[name] = shap_vals[0][i]
        
        # Get feature names and values
        feature_names_list = list(f_names)
        shap_values_list = []
        
        if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2 and shap_vals.shape[0] == 1:
            shap_values_list = shap_vals[0].tolist()
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 1:
            shap_values_list = shap_vals.tolist()
        else:
            shap_values_list = list(shap_vals) if hasattr(shap_vals, '__iter__') else [shap_vals]
        
        # Ensure we have the right number of features
        if len(feature_names_list) > len(shap_values_list):
            feature_names_list = feature_names_list[:len(shap_values_list)]
        
        # Call the highlighting function
        _highlight_text_below_plot(
            sample_text,
            feature_names_list,
            shap_values_list,
            class_name or "Prediction"
        )
    
    if not show:
        return fig
    else:
        return None