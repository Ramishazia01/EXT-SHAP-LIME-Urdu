# SHAP RTL

**Right-to-left aware SHAP plots and explanations for Urdu, Arabic, and other RTL languages.**

SHAP (SHapley Additive exPlanations) is the standard framework for explaining the output of machine learning models, but its plotting functions were built for left-to-right scripts. Feature names in Urdu or Arabic render as disconnected, reversed, or mis-shaped glyphs when passed through `matplotlib`, which makes SHAP output unusable as-is for models trained on RTL-language text.

SHAP RTL solves this. It provides drop-in replacements for every major `shap.plots` function, each of which detects Arabic and Urdu text in feature names, reshapes and reorders it correctly using the Arabic/Urdu bidirectional text algorithm, and renders it with a bundled Nastaliq-compatible font. Mixed-script labels (Urdu or Arabic tokens alongside English feature names or numeric values) are handled automatically, so nothing needs to be pre-processed by hand before plotting.

The package also includes an LLM-based explainer that converts raw SHAP values into a natural-language explanation of a model's prediction, written directly in Urdu or Arabic rather than translated after the fact.

## Installation

SHAP RTL can be installed from source:

```bash
git clone https://github.com/<your-username>/SHAP_RTL_package.git
cd SHAP_RTL_package
pip install .
```

Or, once published, from PyPI:

```bash
pip install SHAP_RTL_package
```

To use the LLM-based explainer, install the optional `explainer` extra:

```bash
pip install "SHAP_RTL_package[explainer]"
```

## RTL-aware SHAP plots

Every plot in `shap_rtl.plots` mirrors the name and signature of its `shap.plots` counterpart, so existing SHAP code can generally be ported by changing only the import:

```python
from shap_rtl.plots import waterfall, bar, beeswarm, force
from shap_rtl.plots import summary_plot, decision_plot, dependence_plot
from shap_rtl.plots import scatter, heatmap
```

For example, a bar plot over feature names written in Urdu:

```python
import shap
from shap_rtl.plots import bar

explainer = shap.Explainer(model, X_background)
shap_values = explainer(X)

bar(shap_values)
```

Feature names such as "بہت اچھا" or "غير مقبول" are reshaped and right-aligned automatically, with no manual font configuration required. Each of the following is supported:

- `waterfall` — single-prediction waterfall plots
- `bar` — mean absolute SHAP value bar charts
- `beeswarm` — distribution of SHAP values across a dataset
- `force` (also available as `force_plot`) — additive force plots
- `summary_plot` — global feature importance summaries
- `decision_plot` — cumulative decision paths
- `dependence_plot` — feature dependence plots
- `scatter` — SHAP value scatter plots
- `heatmap` — SHAP value heatmaps across instances

## RTL SHAP explainer

`RTLSHAPExplainer` builds on top of standard SHAP values to produce natural-language explanations of individual predictions, written in Urdu or Arabic:

```python
from shap_rtl import RTLSHAPExplainer

explainer = RTLSHAPExplainer(model, language="ur")
explanation = explainer.explain(X.iloc[0])

print(explanation)
```

This component requires the `explainer` extra and an OpenAI-compatible API key, since it uses an LLM to turn SHAP values into readable text rather than translating a fixed template.

## Fonts

SHAP RTL bundles Nastaliq-compatible fonts for Urdu and Arabic rendering, registered automatically with `matplotlib` on import. No system-level font installation is required.

## Why this exists

SHAP is widely used to explain model behavior in NLP research, including work on hate speech detection, sentiment analysis, and text classification in Urdu and Arabic. Without RTL-aware rendering, researchers working in these languages have had to either fall back on transliteration or manually patch plot internals for every figure. SHAP RTL removes that step so that explainability work in RTL languages can be reported the same way it would be in English.

## Contributing

Issues and pull requests are welcome. If you run into a script or font that isn't handled correctly, please open an issue with a minimal reproducible example.

## Citation

If you use SHAP RTL in academic work, please cite it as:

```bibtex
@software{shap_rtl,
  title  = {SHAP RTL: Right-to-left aware SHAP plots for Urdu and Arabic},
  author = {{SHAP RTL contributors}},
  year   = {2026},
  url    = {https://github.com/<your-username>/SHAP_RTL_package}
}
```

## License

MIT
