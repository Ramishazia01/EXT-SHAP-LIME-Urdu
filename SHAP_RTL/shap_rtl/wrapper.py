# ============================================================
# shap_rtl/wrapper.py - FIXED VERSION
# ============================================================

import numpy as np
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import RTLSHAPExplainer
from .plots import bar, force, waterfall, beeswarm, heatmap, scatter, decision_plot
from .rtl_utils import set_current_language


class SHAPRTL:
    """
    Universal RTL SHAP Analyzer - Works with ANY model!
    """
    
    def __init__(self, language="Arabic", model=None, vectorizer=None, pipeline=None):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        language : str
            'Arabic', 'Urdu', 'Hebrew', or 'Persian'
        model : sklearn model (optional)
            Any sklearn model with .fit() and .predict() methods
        vectorizer : sklearn vectorizer (optional)
            Any sklearn vectorizer with .fit_transform() and .transform() methods
        pipeline : sklearn Pipeline (optional)
            Complete pipeline with vectorizer and model
        """
        self.language = language
        self.model = model
        self.vectorizer = vectorizer
        self.pipeline = pipeline
        self.feature_names = None
        self.class_names = None
        self.explainer_rtl = None
        self.shap_explainer = None
        self.is_fitted = False
        self.X_train_vec = None  # Store training vectors
        self.X_test_vec = None   # Store test vectors
        self.y_train = None
        self.y_test = None
        
        # Set language for RTL rendering
        set_current_language(language)
        
        print(f"✅ SHAP RTL initialized for {language}")
        print(f"   Supported languages: Arabic, Urdu, Hebrew, Persian")
        
        if pipeline is not None:
            print("   Using custom pipeline")
        elif model is not None and vectorizer is not None:
            print("   Using custom model + vectorizer")
        else:
            print("   Using default: TF-IDF + Logistic Regression (use .train())")
    
    def train(self, texts, labels, max_features=2000, test_size=0.2, random_state=42):
        """
        Train using default TF-IDF + Logistic Regression
        """
        print("\n📊 Training with default TF-IDF + Logistic Regression...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Store for later use
        self.y_train = y_train
        self.y_test = y_test
        
        # Vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            token_pattern=r'[^\s]+'
        )
        
        # Transform data
        self.X_train_vec = self.vectorizer.fit_transform(X_train)
        self.X_test_vec = self.vectorizer.transform(X_test)
        self.feature_names = list(self.vectorizer.get_feature_names_out())
        
        # Model
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        ).fit(self.X_train_vec, y_train)
        
        # Store class names
        unique_labels = sorted(set(labels))
        self.class_names = {i: str(label) for i, label in enumerate(unique_labels)}
        
        print(f"✅ Model trained with {len(self.feature_names)} features")
        print(f"   Classes: {self.class_names}")
        print(f"   Train accuracy: {self.model.score(self.X_train_vec, y_train):.3f}")
        print(f"   Test accuracy: {self.model.score(self.X_test_vec, y_test):.3f}")
        
        self.is_fitted = True
        
        # Setup SHAP
        self._setup_shap(self.X_train_vec)
        
        return self
    
    def fit(self, texts, labels):
        """
        Fit using custom model/vectorizer provided in __init__
        """
        if self.pipeline is not None:
            # Use pipeline
            print("\n📊 Fitting custom pipeline...")
            self.pipeline.fit(texts, labels)
            
            # Extract vectorizer and model from pipeline
            self.vectorizer = self.pipeline.named_steps.get('vectorizer')
            self.model = self.pipeline.named_steps.get('classifier') or self.pipeline.named_steps.get('model')
            
            # Transform data
            self.X_train_vec = self.vectorizer.transform(texts)
            self.feature_names = list(self.vectorizer.get_feature_names_out())
            
        elif self.model is not None and self.vectorizer is not None:
            # Use custom model + vectorizer
            print("\n📊 Fitting custom model + vectorizer...")
            self.X_train_vec = self.vectorizer.fit_transform(texts)
            self.model.fit(self.X_train_vec, labels)
            
            try:
                self.feature_names = list(self.vectorizer.get_feature_names_out())
            except:
                self.feature_names = [f"feature_{i}" for i in range(self.X_train_vec.shape[1])]
        
        else:
            raise ValueError("No model/vectorizer provided. Use .train() for default, or provide model+vectorizer in __init__")
        
        # Store class names
        unique_labels = sorted(set(labels))
        self.class_names = {i: str(label) for i, label in enumerate(unique_labels)}
        
        print(f"✅ Model fitted with {len(self.feature_names)} features")
        print(f"   Classes: {self.class_names}")
        
        self.is_fitted = True
        
        # Setup SHAP
        self._setup_shap(self.X_train_vec)
        
        return self
    
    def _setup_shap(self, X_train_vec):
        """Setup SHAP explainers"""
        if X_train_vec.shape[0] > 100:
            X_background = X_train_vec[:100]
        else:
            X_background = X_train_vec
        
        masker = shap.maskers.Independent(data=X_background, max_samples=100)
        
        # Create SHAP explainer based on model type
        try:
            self.shap_explainer = shap.LinearExplainer(self.model, masker=masker)
        except:
            # Fallback to KernelExplainer for non-linear models
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                X_background,
                link="logit"
            )
        
        # RTL explainer
        self.explainer_rtl = RTLSHAPExplainer(
            language=self.language,
            model=self.model,
            vectorizer=self.vectorizer,
            api_type="groq"
        )
        
        print(f"✅ SHAP explainers ready for {self.language}")
    
    def analyze(self, text, plot_type='bar', max_features=10):
        """Analyze a single text"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .train() or .fit() first.")
        
        print(f"\n📝 Text: {text}")
        print(f"   Language: {self.language}")
        
        # Vectorize
        if self.pipeline is not None:
            vec = self.vectorizer.transform([text]).toarray()[0]
        else:
            vec = self.vectorizer.transform([text]).toarray()[0]
        
        present_idx = np.where(vec > 0)[0]
        
        if len(present_idx) == 0:
            print("⚠️ No features found in text")
            return None
        
        # Get prediction
        if self.pipeline is not None:
            pred = self.pipeline.predict([text])[0]
            probs = self.pipeline.predict_proba([text])[0]
        else:
            pred = self.model.predict(vec.reshape(1, -1))[0]
            probs = self.model.predict_proba(vec.reshape(1, -1))[0]
        
        pred_class = self.class_names.get(pred, f'Class {pred}')
        
        print(f"🎯 Prediction: {pred_class}")
        for i, prob in enumerate(probs):
            class_name = self.class_names.get(i, f'Class {i}')
            print(f"   {class_name}: {prob:.3f}")
        
        # Get SHAP values
        shap_values = self.shap_explainer(vec.reshape(1, -1))
        
        # Get present features
        present_feature_names = [self.feature_names[i] for i in present_idx]
        present_shap_values = shap_values.values[0][present_idx]
        present_data = vec[present_idx]
        
        # Sort by absolute SHAP
        sorted_idx = np.argsort(np.abs(present_shap_values))[::-1][:max_features]
        sorted_features = [present_feature_names[i] for i in sorted_idx]
        sorted_shap = present_shap_values[sorted_idx]
        sorted_data = present_data[sorted_idx]
        
        # Create Explanation
        shap_exp = shap.Explanation(
            values=sorted_shap,
            base_values=shap_values.base_values[0],
            data=sorted_data,
            feature_names=sorted_features
        )
        
        # Print features
        print(f"\n📊 Top {max_features} features:")
        print("-" * 60)
        for name, val in zip(sorted_features, sorted_shap):
            direction = "→ TOWARD" if val > 0 else "← AWAY"
            print(f"  {name:<30} {val:>+8.3f} {direction}")
        
        # Plot functions
        plot_functions = {
            'bar': bar,
            'waterfall': waterfall,
            'force': force,
            'beeswarm': beeswarm,
            'heatmap': heatmap,
            'scatter': scatter,
            'decision_plot': decision_plot
        }
        
        plot_fn = plot_functions.get(plot_type)
        if plot_fn is None:
            print(f"⚠️ Plot type '{plot_type}' not supported")
            return None
        
        # Prepare for bar plot
        if plot_type == 'bar':
            values = shap_exp.values.flatten() if len(shap_exp.values.shape) > 1 else shap_exp.values
            shap_exp_bar = shap.Explanation(
                values=values,
                base_values=shap_exp.base_values,
                data=shap_exp.data if hasattr(shap_exp, 'data') else None,
                feature_names=shap_exp.feature_names
            )
            shap_to_use = shap_exp_bar
        else:
            shap_to_use = shap_exp
        
        # Generate RTL plot
        self.explainer_rtl.run_full_analysis(
            shap_values=shap_to_use,
            prediction_label=pred_class,
            plot_fn=plot_fn,
            sample_text=text,
            max_features=max_features,
            allow_positive_negative_split=True
        )
        
        return shap_exp
    
    def analyze_multiple(self, texts, plot_type='beeswarm', max_features=10):
        """Analyze multiple texts"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .train() or .fit() first.")
        
        print(f"\n📊 Analyzing {len(texts)} samples...")
        print(f"   Language: {self.language}")
        
        all_shap_values = []
        all_feature_names = None
        
        for text in texts:
            vec = self.vectorizer.transform([text]).toarray()[0]
            present_idx = np.where(vec > 0)[0]
            
            if len(present_idx) == 0:
                continue
            
            shap_values = self.shap_explainer(vec.reshape(1, -1))
            
            present_feature_names = [self.feature_names[i] for i in present_idx]
            present_shap_values = shap_values.values[0][present_idx]
            
            sorted_idx = np.argsort(np.abs(present_shap_values))[::-1][:max_features]
            sorted_features = [present_feature_names[i] for i in sorted_idx]
            sorted_shap = present_shap_values[sorted_idx]
            
            if all_feature_names is None:
                all_feature_names = sorted_features
                all_shap_values.append(sorted_shap)
            else:
                current_features = sorted_features
                common_features = [f for f in current_features if f in all_feature_names]
                if len(common_features) > 0:
                    common_shap = []
                    for feat in common_features:
                        idx = current_features.index(feat)
                        common_shap.append(sorted_shap[idx])
                    all_shap_values.append(common_shap)
        
        if not all_shap_values:
            print("⚠️ No valid samples found")
            return None
        
        # Pad to same length
        max_len = max(len(v) for v in all_shap_values)
        padded_values = []
        for v in all_shap_values:
            if len(v) < max_len:
                padded = np.pad(v, (0, max_len - len(v)), constant_values=0)
            else:
                padded = v[:max_len]
            padded_values.append(padded)
        
        shap_exp_combined = shap.Explanation(
            values=np.array(padded_values),
            base_values=0,
            data=None,
            feature_names=all_feature_names[:max_len]
        )
        
        plot_functions = {
            'beeswarm': beeswarm,
            'heatmap': heatmap,
            'scatter': scatter
        }
        
        plot_fn = plot_functions.get(plot_type)
        if plot_fn is None:
            print(f"⚠️ Plot type '{plot_type}' not supported for multiple samples")
            return None
        
        self.explainer_rtl.run_full_analysis(
            shap_values=shap_exp_combined,
            prediction_label="Analysis",
            plot_fn=plot_fn,
            sample_text=f"{len(texts)} samples",
            max_features=max_features,
            allow_positive_negative_split=True
        )
        
        return shap_exp_combined


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_analyzer(language="Arabic", model=None, vectorizer=None, pipeline=None):
    """
    Quick way to create an analyzer for any language
    """
    return SHAPRTL(language=language, model=model, vectorizer=vectorizer, pipeline=pipeline)