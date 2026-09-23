def load_your_data():
    import numpy as np
    from pathlib import Path
    LANG = "ara"
    d = Path("data")
    return (np.load(d / f"{LANG}_feature_names.npy", allow_pickle=True),
            np.load(d / f"{LANG}_shap_values.npy"),
            LANG)