"""
shap_rtl.explainer
──────────────────
RTL-aware SHAP explanation generation with LLM-powered feature analysis.
"""

import os
import sys
import copy
import time
import re
import numpy as np
from IPython.display import display, HTML

# RTL text processing
import arabic_reshaper
from bidi.algorithm import get_display

# SHAP and visualization
import shap
import matplotlib.pyplot as plt

# LLM integration
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import RTL font utilities
from shap_rtl.rtl_utils import (
    _find_rtl_font, 
    set_rtl_font_path, 
    set_current_language,
    get_current_language
)

load_dotenv()


class RTLSHAPExplainer:
    """
    RTL-aware SHAP explainer with LLM-generated natural language explanations.
    
    Supported Languages:
    - Arabic (العربية)
    - Urdu (اردو)
    - Persian/Farsi (فارسی)
    - Hebrew (עברית)
    """
    
    SUPPORTED_LANGUAGES = ["Arabic", "Urdu", "Persian", "Hebrew"]
    RTL_LANGUAGES = SUPPORTED_LANGUAGES
    
    def __init__(
        self,
        language: str,
        model=None,
        vectorizer=None,
        api_type: str = "groq",
        skip_translation: bool = False,
        max_features_to_translate: int = 50
    ):
        """
        Initialize RTL SHAP explainer.
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported.\n"
                f"Supported languages: {', '.join(self.SUPPORTED_LANGUAGES)}"
            )
        
        self.language = language.strip().capitalize()
        self.model = model
        self.vectorizer = vectorizer
        self.api_type = api_type
        self.skip_translation = skip_translation
        self.max_features_to_translate = max_features_to_translate
        self._llm = None
        
        self.RW = 120
        self.rtl_languages = self.SUPPORTED_LANGUAGES
        self._eng_sample_text = None
        self._cached_features = None
        self._feature_translation_cache = {}
        self._cached_feature_names = None
        self.font_path = None
        self.font_name = None
        
        # Font initialization
        set_current_language(self.language)
        print(f"[DEBUG] Current language set to: {self.language}")
        
        font_path, font_name = _find_rtl_font(language=self.language)
        
        if font_path:
            set_rtl_font_path(font_path)
            self.font_path = font_path
            self.font_name = font_name
            
            try:
                import matplotlib as mpl
                from matplotlib import font_manager
                
                font_manager.fontManager.addfont(font_path)
                prop = font_manager.FontProperties(fname=font_path)
                font_display_name = prop.get_name()
                
                mpl.rcParams['font.family'] = font_display_name
                mpl.rcParams['axes.unicode_minus'] = False
                
                print(f"[INFO] Loaded {self.language} font: {font_name}")
                print(f"[INFO] Font display name: {font_display_name}")
            except Exception as e:
                print(f"[WARNING] Could not register font: {e}")
                print(f"[INFO] Loaded {self.language} font: {font_name}")
        else:
            print(f"[WARNING] No font found for {self.language}")
        
        self._language_fonts = {
            "Arabic": "NotoSansArabic-Regular.ttf",
            "Urdu": "JameelNooriNastaliq.ttf",
            "Persian": "NotoSansArabic-Regular.ttf",
            "Hebrew": "NotoSansHebrew-Regular.ttf",
        }
        
        print(f"\n{'='*60}")
        print(f"🔤 RTL SHAP Explainer Initialized")
        print(f"{'='*60}")
        print(f"   Language: {self.language}")
        print(f"   API Type: {self.api_type}")
        print(f"   Font: {self.font_name or self._language_fonts.get(self.language, 'default')}")
        print(f"   Translations: {'Disabled (fast)' if self.skip_translation else 'Enabled → English'}")
        print(f"   Max features to translate: {self.max_features_to_translate}")
        print(f"{'='*60}\n")
    
    @property
    def llm(self):
        """Lazy-initialize the LLM based on API type."""
        if self._llm is None:
            if self.api_type == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                self._llm = ChatOpenAI(
                    model="llama-3.3-70b-versatile",
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.1,
                )
            else:
                api_key = os.getenv("GITHUB_TOKEN")
                self._llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=api_key,
                    base_url="https://models.inference.ai.azure.com",
                    temperature=0.1,
                )
        return self._llm
    
    @llm.setter
    def llm(self, value):
        self._llm = value
    
    def _invoke_llm(self, prompt: str, retries: int = 2, delay: float = 3.0) -> str:
        """Call the LLM with automatic retry on rate-limit errors."""
        if self.skip_translation:
            return ""
        
        last_exc = None
        for attempt in range(retries + 1):
            try:
                return self.llm.invoke(prompt).content.strip()
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(delay)
        raise last_exc
    
    # ── RTL Text Helpers ──────────────────────────────────────────────────────
    
    def _fix_for_terminal(self, text: str) -> str:
        try:
            return arabic_reshaper.reshape(str(text))
        except Exception:
            return str(text)
    
    def _fix_for_plot(self, text: str) -> str:
        try:
            reshaped = arabic_reshaper.reshape(str(text))
            return get_display(reshaped)
        except Exception:
            return str(text)
    
    # ── Chunked Batch Translation ────────────────────────────────────────────
    
    def _translate_batch(self, texts: list, chunk_size: int = 20) -> list:
        """
        Translate multiple texts in CHUNKS to avoid token limit errors.
        """
        if self.skip_translation:
            return texts
        
        if not texts:
            return []
        
        # Limit how many features to translate
        if len(texts) > self.max_features_to_translate:
            print(f"[INFO] Limiting translation to {self.max_features_to_translate} features (out of {len(texts)})")
            texts = texts[:self.max_features_to_translate]
        
        results = [None] * len(texts)
        
        # Check cache first
        to_translate = []
        to_translate_indices = []
        
        for i, text in enumerate(texts):
            if text in self._feature_translation_cache:
                results[i] = self._feature_translation_cache[text]
            else:
                to_translate.append(text)
                to_translate_indices.append(i)
        
        if not to_translate:
            return results
        
        # Translate in chunks to avoid token limit
        print(f"[INFO] Translating {len(to_translate)} features in chunks of {chunk_size}...")
        
        for chunk_start in range(0, len(to_translate), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(to_translate))
            chunk = to_translate[chunk_start:chunk_end]
            
            try:
                text_list = "\n".join([f"{i+1}. {text}" for i, text in enumerate(chunk)])
                prompt = f"""
                Translate the following {self.language} texts to English.
                Return ONLY a numbered list of translations, one per line, in the same order.
                Do not add any extra text.
                
                Texts to translate:
                {text_list}
                """
                
                response = self._invoke_llm(prompt)
                
                lines = response.strip().split('\n')
                translations = []
                for line in lines:
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    if cleaned:
                        translations.append(cleaned)
                
                while len(translations) < len(chunk):
                    translations.append(chunk[len(translations)])
                
                for j, (idx, text, trans) in enumerate(zip(
                    to_translate_indices[chunk_start:chunk_end],
                    chunk,
                    translations
                )):
                    self._feature_translation_cache[text] = trans
                    results[idx] = trans
                    
            except Exception as e:
                print(f"[WARNING] Chunk translation failed: {e}")
                for j, idx in enumerate(to_translate_indices[chunk_start:chunk_end]):
                    if results[idx] is None:
                        results[idx] = to_translate[chunk_start + j]
        
        return results
    
    def _translate_text(self, text: str) -> str:
        """Translate a single text."""
        if self.skip_translation or not text:
            return text
        
        if text in self._feature_translation_cache:
            return self._feature_translation_cache[text]
        
        try:
            prompt = f"""
            Translate this {self.language} text to English.
            Return ONLY the translation, no extra text.
            
            Text: {text}
            """
            response = self._invoke_llm(prompt)
            translation = response.replace('"', '').strip()
            self._feature_translation_cache[text] = translation
            return translation
        except Exception as e:
            print(f"[WARNING] Translation failed for '{text}': {e}")
            return text
    
    def _translate_sample(self, sample_text: str) -> str:
        """Translate sample text."""
        if self.skip_translation:
            return sample_text
        
        if self._eng_sample_text is not None:
            return self._eng_sample_text
        
        if not sample_text:
            return None
        
        print("\n[TRANSLATING SAMPLE TEXT TO ENGLISH...]")
        self._eng_sample_text = self._translate_text(sample_text)
        
        if self._eng_sample_text and self._eng_sample_text != sample_text:
            print(f"\nOriginal {self.language} Text:")
            print(f"  {self._fix_for_terminal(sample_text)}")
            print(f"\nEnglish Translation:")
            print(f"  {self._eng_sample_text}")
            print("-" * 80)
        else:
            print("[WARNING] Could not translate sample text.")
            self._eng_sample_text = sample_text
        
        return self._eng_sample_text
    
    def _translate_features(self, feature_names: list) -> list:
        """Translate feature names using chunked batch translation."""
        if self.skip_translation or not feature_names:
            return feature_names
        
        if len(feature_names) > self.max_features_to_translate:
            print(f"[INFO] Only translating top {self.max_features_to_translate} features")
            feature_names_to_translate = feature_names[:self.max_features_to_translate]
        else:
            feature_names_to_translate = feature_names
        
        return self._translate_batch(feature_names_to_translate)
    
    def _translate_label(self, label: str) -> str:
        """Translate prediction label."""
        if self.skip_translation:
            return label
        
        try:
            prompt = f"""
            Translate '{label}' from {self.language} to English.
            Return ONLY the translation, no extra text.
            """
            return self._invoke_llm(prompt).replace('"', '')
        except Exception:
            return label
    
    # ── Feature Extraction ──────────────────────────────────────────────────
    
    def get_top_features(self, shap_values, num_features: int = 5) -> list:
        values = shap_values.values
        feature_names = shap_values.feature_names
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(values.shape[-1])]
        
        if values.ndim == 2:
            mean_abs = np.mean(np.abs(values), axis=0)
            mean_vals = np.mean(values, axis=0)
        else:
            mean_abs = np.abs(values)
            mean_vals = values
        
        sorted_idx = np.argsort(mean_abs)[::-1][:num_features]
        return [(str(feature_names[i]), float(mean_vals[i])) for i in sorted_idx]
    
    # ── Translate SHAP Explanation ───────────────────────────────────────────
    
    def _translate_shap_explanation(self, shap_values):
        """Translate feature names in a SHAP Explanation."""
        if self.skip_translation:
            return shap_values
        
        if not hasattr(shap_values, 'feature_names') or shap_values.feature_names is None:
            return shap_values
        
        # Get feature names
        feature_names = list(shap_values.feature_names)
        if len(feature_names) > self.max_features_to_translate:
            feature_names = feature_names[:self.max_features_to_translate]
        
        eng_names = self._translate_features(feature_names)
        
        try:
            values = shap_values.values
            base_values = shap_values.base_values
            data = shap_values.data if hasattr(shap_values, 'data') else None
            display_data = shap_values.display_data if hasattr(shap_values, 'display_data') else None
            
            # Handle 1D values (single feature)
            if values.ndim == 1:
                if len(eng_names) == 0:
                    eng_names = ["Feature"]
                elif len(eng_names) > 1:
                    eng_names = eng_names[:1]
                
                translated = shap.Explanation(
                    values=values,
                    base_values=base_values,
                    data=data,
                    feature_names=eng_names,
                    display_data=display_data
                )
            else:
                # 2D values (multiple features)
                n_features = values.shape[1] if values.ndim == 2 else 1
                if len(eng_names) > n_features:
                    eng_names = eng_names[:n_features]
                elif len(eng_names) < n_features:
                    for i in range(len(eng_names), n_features):
                        eng_names.append(f"Feature_{i}")
                
                translated = shap.Explanation(
                    values=values,
                    base_values=base_values,
                    data=data,
                    feature_names=eng_names,
                    display_data=display_data
                )
            return translated
        except Exception as e:
            print(f"[WARNING] Could not translate SHAP explanation: {e}")
            return shap_values
    
    # ── Dependence Plot ───────────────────────────────────────────────────────
    
    def _show_dependence_plot(self, shap_values, features, feature_names, title_prefix="Hebrew"):
        """Show a dependence plot."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the data
        if hasattr(shap_values, 'values'):
            values = shap_values.values
        else:
            values = shap_values
        
        if features is None:
            features = self._cached_features
        
        if feature_names is None:
            feature_names = shap_values.feature_names if hasattr(shap_values, 'feature_names') else []
        
        # For dependence plot, use first two features
        if values.ndim == 2 and values.shape[1] >= 2:
            x_data = features[:, 0] if features is not None else np.arange(len(values))
            y_data = values[:, 0]
            color_data = features[:, 1] if features is not None else values[:, 1]
            x_label = feature_names[0] if len(feature_names) > 0 else "Feature 1"
            color_label = feature_names[1] if len(feature_names) > 1 else "Feature 2"
        else:
            x_data = features if features is not None else np.arange(len(values))
            y_data = values if values.ndim == 1 else values[:, 0]
            color_data = y_data
            x_label = feature_names[0] if len(feature_names) > 0 else "Feature"
            color_label = x_label
        
        scatter = ax.scatter(
            x_data,
            y_data,
            c=color_data,
            cmap='RdYlBu',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel(f'Feature value: {x_label}', fontsize=12)
        ax.set_ylabel(f'SHAP value for {x_label}', fontsize=12)
        ax.set_title(f'Dependence Plot ({title_prefix}): {x_label}\nColored by {color_label}', 
                     fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'Feature value: {color_label}', fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        return ax
    
    # ── Plot Function Handler ──────────────────────────────────────────────────
    
    def _call_plot_function(self, plot_fn, shap_values):
        """Call a plot function with translated feature names."""
        import inspect
        
        sig = inspect.signature(plot_fn)
        plot_name = plot_fn.__name__ if hasattr(plot_fn, '__name__') else str(plot_fn)
        
        translated_shap = self._translate_shap_explanation(shap_values)
        
        # Handle scatter/dependence_plot
        if plot_name in ['scatter', 'dependence_plot']:
            if hasattr(translated_shap, 'data') and translated_shap.data is not None:
                features = translated_shap.data
            else:
                features = self._cached_features
            
            if 'features' in sig.parameters:
                return plot_fn(shap_values=translated_shap, features=features)
            else:
                return plot_fn(shap_values=translated_shap)
        
        # Handle decision_plot
        elif plot_name == 'decision_plot':
            base_value = getattr(translated_shap, 'base_values', None) or 0.0
            
            if hasattr(translated_shap, 'data') and translated_shap.data is not None:
                features = translated_shap.data
            else:
                features = self._cached_features
            
            if isinstance(translated_shap, list):
                shap_vals = [sv.values for sv in translated_shap]
                features = [sv.data for sv in translated_shap if hasattr(sv, 'data')]
                feature_names = translated_shap[0].feature_names if hasattr(translated_shap[0], 'feature_names') else None
                return plot_fn(base_value=base_value, shap_values=shap_vals, features=features, feature_names=feature_names)
            else:
                shap_vals = translated_shap.values if hasattr(translated_shap, 'values') else translated_shap
                feature_names = translated_shap.feature_names if hasattr(translated_shap, 'feature_names') else None
                
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2 and shap_vals.shape[0] == 1:
                    shap_vals = shap_vals[0]
                    features = features[0] if features is not None and len(features) > 0 else None
                
                return plot_fn(base_value=base_value, shap_values=shap_vals, features=features, feature_names=feature_names)
        
        # Handle heatmap
        elif plot_name == 'heatmap':
            if hasattr(translated_shap, 'feature_names') and translated_shap.feature_names is not None:
                return plot_fn(translated_shap)
            else:
                try:
                    if hasattr(self, '_cached_feature_names') and self._cached_feature_names is not None:
                        new_shap = shap.Explanation(
                            values=translated_shap.values,
                            base_values=translated_shap.base_values,
                            data=translated_shap.data if hasattr(translated_shap, 'data') else None,
                            feature_names=self._cached_feature_names
                        )
                        return plot_fn(new_shap)
                except Exception as e:
                    print(f"[WARNING] Heatmap feature rebuild issue: {e}")
                return plot_fn(translated_shap)
        
        # Handle other plots (bar, waterfall, beeswarm, etc.)
        else:
            return plot_fn(translated_shap)
    
    # ── Prompt Templates ─────────────────────────────────────────────────────
    
    def _rtl_prompt(self, feat_name: str, shap_val: float, prediction_label: str) -> str:
        if self.skip_translation:
            return f"[Translations disabled] Feature: {feat_name} (SHAP: {shap_val:+.4f})"
        
        impact = "positively" if shap_val > 0 else "negatively"
        return f"""
        SYSTEM: You are an AI Explainability Assistant.
        LANGUAGE: {self.language}
        PREDICTION: {prediction_label}
        FEATURE: "{feat_name}"
        SHAP VALUE: {shap_val:+.4f}
        TASK: Explain why the feature "{feat_name}" {impact} impacts the prediction "{prediction_label}" in {self.language} only.
        INSTRUCTIONS: Respond in ONE concise paragraph in {self.language} only.
        Domain and linguistic analysis only. No English, no raw numeric scores.
        """
    
    def _eng_prompt(self, feat_name: str, shap_val: float, prediction_label: str, eng_translation: str) -> str:
        impact = "positively" if shap_val > 0 else "negatively"
        return f"""
        SHAP Analysis: In the context of the prediction '{prediction_label}' 
        (English: '{eng_translation}'), explain in one concise English paragraph 
        why the {self.language} feature '{feat_name}' 
        (SHAP value: {shap_val:+.4f}) {impact} impacts the model output.
        """
    
    # ── Main Workflow ──────────────────────────────────────────────────────
    
    def run_full_analysis(
        self,
        shap_values,
        prediction_label: str = "prediction",
        plot_fn=None,
        max_features: int = 10,
        allow_positive_negative_split: bool = True,
        sample_text: str = None,
        features=None
    ):
        """Run the complete RTL SHAP explanation workflow."""
        if features is not None:
            self._cached_features = features
        elif hasattr(shap_values, 'data') and shap_values.data is not None:
            self._cached_features = shap_values.data
        
        if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
            self._cached_feature_names = shap_values.feature_names
        
        print("\n" + "=" * 80)
        print(f"{self.language.upper()} SHAP EXPLANATIONS".center(80))
        print(f"Prediction: {prediction_label}".center(80))
        print("=" * 80)
        
        # ── Step 1: Display Plot ──────────────────────────────────────────────
        
        # Check if we should show a dependence plot (plot_fn is None but features are provided)
        if plot_fn is None and features is not None:
            print(f"\n[GENERATING {self.language.upper()} DEPENDENCE PLOT...]")
            
            # Get feature names
            feature_names = shap_values.feature_names if hasattr(shap_values, 'feature_names') else []
            
            # Show Hebrew dependence plot
            self._show_dependence_plot(shap_values, features, feature_names, title_prefix=self.language)
            
        elif plot_fn is not None:
            print(f"\n[GENERATING {self.language.upper()} SHAP PLOT...]")
            self._call_plot_function(plot_fn, shap_values)
        else:
            print("\n[INFO] No plot_fn provided and no features for dependence plot. Skipping visualization.")
        
        # ── Step 2: Ask for RTL Explanation ───────────────────────────────────
        print(f"\n[{self.language} Explanation Mode]")
        
        n = None
        while True:
            try:
                user_input = input(f"How many features would you like explained in {self.language}? (Enter 0 or press ESC to skip): ")
                
                if user_input.strip() == "" or user_input.lower() == "escape" or user_input == "0":
                    print("\n[INFO] Skipping RTL explanation.")
                    self._ask_english_plot(shap_values, prediction_label, plot_fn, sample_text, features)
                    return
                
                n = int(user_input)
                if n <= 0:
                    print("[INFO] Skipping RTL explanation.")
                    self._ask_english_plot(shap_values, prediction_label, plot_fn, sample_text, features)
                    return
                
                n = min(n, max_features)
                break
                
            except ValueError:
                print("Please enter a valid number (or press ESC/Enter to skip).")
                continue
            except KeyboardInterrupt:
                print("\n[INFO] Operation cancelled by user.")
                return
        
        top_features = self.get_top_features(shap_values, n)
        
        print("-" * self.RW)
        print(f"PREDICTION: {prediction_label}".rjust(self.RW))
        print("-" * self.RW)
        
        print(f"\nAnalysis of Top {n} SHAP Features ({self.language})".rjust(self.RW))
        print("-" * 50)
        
        for feat_name, shap_val in top_features:
            try:
                display_name = self._fix_for_terminal(feat_name)
                print(f"\nFeature: {display_name} | Value: {shap_val:+.4f}")
                print("-" * 40)
                
                prompt = self._rtl_prompt(feat_name, shap_val, prediction_label)
                response = self._invoke_llm(prompt)
                
                fixed_response = self._fix_for_terminal(response)
                print(fixed_response.rjust(self.RW))
                print("-" * 40)
                
            except Exception as e:
                print(f"Feature: {feat_name} (API unavailable: {str(e)})".rjust(self.RW))
        
        print("\n" + "=" * 80)
        
        # ── Step 3: Ask for English ──────────────────────────────────────────
        if self.language in self.rtl_languages:
            self._ask_english_plot(shap_values, prediction_label, plot_fn, sample_text, features)
    
    # ── English Plot and Explanation Methods ────────────────────────────────
    
    def _ask_english_plot(self, shap_values, prediction_label, plot_fn, sample_text, features=None):
        """Ask user if they want to see the plot in English."""
        while True:
            try:
                choice = input("\nWould you like to see the plot in English? (Yes/No): ").strip().lower()
                
                if choice in ["yes", "y"]:
                    print("\n[INFO] Generating English plot...")
                    
                    # Check if we should show English dependence plot
                    if plot_fn is None and features is not None:
                        print("\n[INFO] Generating English Dependence Plot...")
                        
                        # Get feature names
                        feature_names = shap_values.feature_names if hasattr(shap_values, 'feature_names') else []
                        
                        # Translate feature names
                        eng_feature_names = self._translate_features(feature_names)
                        
                        # Show English dependence plot
                        self._show_dependence_plot(shap_values, features, eng_feature_names, title_prefix="English")
                        
                    else:
                        self._display_english_plot(
                            shap_values=shap_values,
                            prediction_label=prediction_label,
                            plot_fn=plot_fn,
                            sample_text=sample_text,
                            features=features
                        )
                    
                    # Ask for English explanation
                    self._ask_english_explanation(shap_values, prediction_label, sample_text, features)
                    return
                elif choice in ["no", "n", ""]:
                    print("\n[INFO] Exiting. Thank you!")
                    return
                else:
                    print("Please enter Yes or No.")
                    
            except KeyboardInterrupt:
                print("\n[INFO] Exiting. Thank you!")
                return
            except Exception as e:
                print(f"[ERROR] {e}")
                return
    
    def _display_english_plot(self, shap_values, prediction_label, plot_fn, sample_text, features=None):
        """Display the English plot for bar/waterfall/etc."""
        print("\n[INFO] Translating features to English...")
        
        eng_sample_text = None
        if sample_text:
            eng_sample_text = self._translate_sample(sample_text)
        
        if isinstance(shap_values, list):
            feature_names = list(shap_values[0].feature_names or []) if shap_values else []
        else:
            feature_names = list(shap_values.feature_names or [])
        
        if len(feature_names) > self.max_features_to_translate:
            feature_names = feature_names[:self.max_features_to_translate]
        
        eng_names = self._translate_features(feature_names)
        eng_label = self._translate_label(prediction_label)
        print(f"Translated Prediction Label: {eng_label}")
        
        if isinstance(shap_values, list):
            eng_shap = []
            for sv in shap_values:
                eng_shap.append(self._rebuild_explanation_with_names(sv, eng_names))
        else:
            eng_shap = self._rebuild_explanation_with_names(shap_values, eng_names)
        
        if plot_fn is not None:
            print(f"\n[GENERATING ENGLISH SHAP PLOT...]")
            self._call_plot_function(plot_fn, eng_shap)
        else:
            print("\n[INFO] No plot_fn provided. Skipping English visualization.")
        
        self._ask_english_explanation(eng_shap, prediction_label, eng_sample_text, features)
    
    def _ask_english_explanation(self, shap_values, prediction_label, eng_sample_text, features=None):
        """Ask user if they want explanation in English."""
        while True:
            try:
                choice = input("\nWould you like explanation in English? (Yes/No): ").strip().lower()
                
                if choice in ["yes", "y"]:
                    self._run_english_explanations(
                        shap_values=shap_values,
                        prediction_label=prediction_label,
                        eng_sample_text=eng_sample_text,
                        features=features
                    )
                    return
                elif choice in ["no", "n", ""]:
                    print("\n[INFO] Exiting. Thank you!")
                    return
                else:
                    print("Please enter Yes or No.")
                    
            except KeyboardInterrupt:
                print("\n[INFO] Exiting. Thank you!")
                return
            except Exception as e:
                print(f"[ERROR] {e}")
                return
    
    def _run_english_explanations(self, shap_values, prediction_label, eng_sample_text, features=None, max_features=10):
        """Generate English explanations."""
        print("\n" + "=" * 80)
        print("ENGLISH EXPLANATIONS".center(80))
        print("=" * 80)
        
        if eng_sample_text:
            print(f"\nTranslated Sample: {eng_sample_text}")
            print("-" * 80)
        
        print(f"\n[English Explanation Mode]")
        
        while True:
            try:
                user_input = input(
                    "How many features would you like explained in English? "
                    "(Press Enter or type 0 to exit): "
                ).strip()

                if user_input == "" or user_input == "0":
                    print("Exiting explanation mode.")
                    return

                n = int(user_input)

                if n <= 0:
                    print("Exiting explanation mode.")
                    return

                n = min(n, max_features)
                break

            except ValueError:
                print("Please enter a valid number.")
        
        eng_top_features = self.get_top_features(shap_values, n)
        
        eng_label = self._translate_label(prediction_label)
        
        print("-" * 80)
        print(f"PREDICTION: {eng_label}".center(80))
        print("-" * 80)
        
        print(f"\nAnalysis of Top {n} SHAP Features (English)")
        print("-" * 50)
        
        for feat_name, shap_val in eng_top_features:
            try:
                print(f"\nFeature: {feat_name} | Value: {shap_val:+.4f}")
                print("-" * 40)
                
                prompt = self._eng_prompt(feat_name, shap_val, prediction_label, eng_label)
                response = self._invoke_llm(prompt)
                print(response)
                print("-" * 40)
                
            except Exception as e:
                print(f"Feature: {feat_name} (API unavailable: {str(e)})")
        
        print("\n" + "=" * 80)
        print("Analysis complete.")
    
    def _rebuild_explanation_with_names(self, shap_values, new_names):
        try:
            return shap.Explanation(
                values=shap_values.values,
                base_values=shap_values.base_values,
                data=shap_values.data,
                feature_names=new_names,
                display_data=shap_values.display_data,
            )
        except Exception:
            sv_eng = copy.copy(shap_values)
            try:
                object.__setattr__(sv_eng, "feature_names", new_names)
            except Exception:
                sv_eng.__dict__["feature_names"] = new_names
            return sv_eng


# ── Convenience Functions ──────────────────────────────────────────────

def create_explainer(language: str, model=None, vectorizer=None, api_type: str = "groq", skip_translation: bool = False, max_features_to_translate: int = 50):
    """Factory function to create an RTLSHAPExplainer instance."""
    return RTLSHAPExplainer(
        language=language,
        model=model,
        vectorizer=vectorizer,
        api_type=api_type,
        skip_translation=skip_translation,
        max_features_to_translate=max_features_to_translate
    )


def get_supported_languages():
    """Return list of supported RTL languages."""
    return RTLSHAPExplainer.SUPPORTED_LANGUAGES


def print_supported_languages():
    """Print supported languages with their script names."""
    print("\nSupported RTL Languages:")
    print("-" * 40)
    print("  Arabic    (العربية)")
    print("  Urdu      (اردو)")
    print("  Persian   (فارسی)")
    print("  Hebrew    (עברית)")
    print("-" * 40)


if __name__ == "__main__":
    print("=" * 80)
    print("SHAP RTL EXPLAINER - Ready to use!".center(80))
    print("=" * 80)
    print("\n[INFO] Using LLM for translations")
    print_supported_languages()