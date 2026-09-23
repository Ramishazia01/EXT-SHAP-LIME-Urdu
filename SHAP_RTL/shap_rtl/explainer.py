"""
shap_rtl.explainer
──────────────────
RTL-aware SHAP explanation generation with LLM-powered feature analysis.
Supports: Arabic, Urdu, Persian, Hebrew
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
import matplotlib as mpl
from matplotlib import font_manager

# LLM integration
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

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

    SUPPORTED_LANGUAGES = [
        "Arabic",
        "Urdu",
        "Persian",
        "Hebrew"
    ]

    def __init__(
        self,
        language: str = "Urdu",
        model=None,
        vectorizer=None,
        api_type: str = "groq",
        skip_translation: bool = False,
        max_features_to_translate: int = 50,
        custom_font_path: str = None
    ):
        """
        Initialize RTL SHAP explainer.

        Parameters
        ----------
        language : str
            RTL language (Arabic, Urdu, Persian, Hebrew)

        model : object, optional
            ML model for SHAP

        vectorizer : object, optional
            Text vectorizer

        api_type : str
            LLM API type ("groq" or "github")

        skip_translation : bool
            Skip all LLM translations for faster execution

        max_features_to_translate : int
            Maximum features to translate

        custom_font_path : str, optional
            Path to custom RTL font file
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

        # RTL display width
        self.RW = 120

        # Supported RTL languages
        self.rtl_languages = self.SUPPORTED_LANGUAGES

        # -------------------------------------------------------------
        # Sample translation cache
        # -------------------------------------------------------------

        self._original_sample_text = None
        self._eng_sample_text = None

        # -------------------------------------------------------------
        # Feature / plot caches
        # -------------------------------------------------------------

        self._cached_features = None
        self._cached_feature_names = None

        # Feature translation cache
        self._feature_translation_cache = {}

        # Font properties
        self.font_path = None
        self.font_name = None

        # Initialize font
        self._initialize_font(custom_font_path)

        print(f"\n{'=' * 60}")
        print("🔤 RTL SHAP Explainer Initialized")
        print(f"{'=' * 60}")
        print(f"   Language: {self.language}")
        print(f"   API Type: {self.api_type}")
        print(f"   Font: {self.font_name or 'default'}")
        print(
            f"   Translations: "
            f"{'Disabled (fast)' if self.skip_translation else 'Enabled → English'}"
        )
        print(
            f"   Max features to translate: "
            f"{self.max_features_to_translate}"
        )
        print(f"{'=' * 60}\n")

    # =================================================================
    # FONT INITIALIZATION
    # =================================================================

    def _initialize_font(self, custom_font_path=None):
        """Initialize RTL and English-safe fonts for Matplotlib plots."""

        import os
        import matplotlib.font_manager as fm

        # ============================================================
        # FONT PATHS
        # ============================================================

        package_dir = os.path.dirname(os.path.abspath(__file__))

        fonts_dir = os.path.join(
            package_dir,
            "fonts"
        )

        # Your Noto Nastaliq Urdu font
        noto_urdu_path = os.path.join(
            fonts_dir,
            "NotoNastaliqUrdu-Regular.ttf"
        )

        # ============================================================
        # REGISTER NOTO NASTALIQ URDU
        # ============================================================

        if os.path.exists(noto_urdu_path):

            fm.fontManager.addfont(noto_urdu_path)

            urdu_font = fm.FontProperties(
                fname=noto_urdu_path
            )

            urdu_font_name = urdu_font.get_name()

            print(
                f"Noto Nastaliq Urdu loaded: "
                f"{urdu_font_name}"
            )

        else:

            print(
                "WARNING: Noto Nastaliq Urdu font not found:"
            )

            print(noto_urdu_path)

            urdu_font_name = "DejaVu Sans"

        # ============================================================
        # LANGUAGE FONTS
        # ============================================================

        self._language_fonts = {

            "Arabic": "NotoSansArabic-Regular.ttf",

            "Urdu": urdu_font_name,

            "Persian": "NotoSansArabic-Regular.ttf",

            "Hebrew": "NotoSansHebrew-Regular.ttf",

        }

        font_path = custom_font_path
        font_name = self._language_fonts.get(
            self.language,
            "NotoSansArabic-Regular.ttf"
        )

        # English plots must use a Latin-safe font.
        self._english_font_name = "DejaVu Sans"
        self._english_font_path = None

        try:
            english_path = font_manager.findfont(
                font_manager.FontProperties(family="DejaVu Sans"),
                fallback_to_default=True
            )
            if english_path and os.path.exists(english_path):
                self._english_font_path = english_path
                font_manager.fontManager.addfont(english_path)
        except Exception:
            pass

        if not font_path:
            font_dirs = [
                os.path.join(os.path.dirname(__file__), "fonts"),
                os.path.join(os.getcwd(), "fonts"),
                os.path.join(sys.prefix, "share", "fonts", "truetype"),
                os.path.join(
                    os.environ.get("WINDIR", "C:\Windows"),
                    "Fonts"
                ),
                "/usr/share/fonts/truetype",
                "/usr/share/fonts/truetype/noto",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                os.path.expanduser("~/.local/share/fonts"),
                os.path.expanduser("~/Library/Fonts"),
            ]

            # Search recursively because font files may be in subfolders.
            for font_dir in font_dirs:
                if not os.path.isdir(font_dir):
                    continue
                try:
                    for root, _, files in os.walk(font_dir):
                        for filename in files:
                            if filename.lower() == font_name.lower():
                                font_path = os.path.join(root, filename)
                                break
                        if font_path:
                            break
                except Exception:
                    continue
                if font_path:
                    break

        # Final fallback: Matplotlib system font registry.
        if not font_path:
            try:
                for path in font_manager.findSystemFonts(fontext="ttf"):
                    if os.path.basename(path).lower() == font_name.lower():
                        font_path = path
                        break
            except Exception:
                pass

        if font_path and os.path.exists(font_path):
            self.font_path = font_path
            self.font_name = os.path.basename(font_path)

            try:
                font_manager.fontManager.addfont(font_path)
                prop = font_manager.FontProperties(fname=font_path)
                font_display_name = prop.get_name()
                self._rtl_font_name = font_display_name

                mpl.rcParams["font.family"] = [font_display_name]
                mpl.rcParams["font.sans-serif"] = [
                    font_display_name,
                    "Noto Sans Arabic",
                    "DejaVu Sans",
                ]
                mpl.rcParams["axes.unicode_minus"] = False

                print(
                    f"[INFO] Loaded {self.language} font: "
                    f"{self.font_name}"
                )
                print(f"[INFO] Font path: {self.font_path}")
                print(
                    f"[INFO] Font display name: "
                    f"{font_display_name}"
                )

            except Exception as e:
                self._rtl_font_name = None
                print(f"[WARNING] Could not register font: {e}")
        else:
            self._rtl_font_name = None
            print(
                f"[WARNING] No {self.language} font found. "
                f"Expected: {font_name}"
            )

            try:
                available = sorted({
                    f.name for f in font_manager.fontManager.ttflist
                    if "noto" in f.name.lower()
                })
                if available:
                    print("[INFO] Available Noto fonts:")
                    for name in available:
                        print(f"   {name}")
            except Exception:
                pass

    def _set_rtl_plot_font(self):
        """Set the selected RTL font before an RTL plot is rendered."""
        if getattr(self, "_rtl_font_name", None):
            mpl.rcParams["font.family"] = [self._rtl_font_name]
            mpl.rcParams["font.sans-serif"] = [
                self._rtl_font_name,
                "Noto Sans Arabic",
                "DejaVu Sans",
            ]
        mpl.rcParams["axes.unicode_minus"] = False

    def _set_english_plot_font(self):
        """Use a Latin-safe font + BLACK text for English plots."""
        mpl.rcParams["font.family"] = [self._english_font_name]
        mpl.rcParams["font.sans-serif"] = [self._english_font_name, "DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
        # FIX GRAY -> BLACK
        mpl.rcParams["text.color"] = "black"
        mpl.rcParams["axes.labelcolor"] = "black"
        mpl.rcParams["xtick.color"] = "black"
        mpl.rcParams["ytick.color"] = "black"
        mpl.rcParams["axes.edgecolor"] = "black"
    # =================================================================
    # LLM
    # =================================================================

    @property
    def llm(self):
        """Lazy-initialize the LLM based on API type."""

        if self._llm is None:

            if self.api_type == "groq":

                api_key = os.getenv(
                    "GROQ_API_KEY"
                )

                if not api_key:

                    raise ValueError(
                        "GROQ_API_KEY was not found. "
                        "Please add GROQ_API_KEY to your .env file."
                    )

                self._llm = ChatOpenAI(
                    model="openai/gpt-oss-120b",
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.1,
                )

            else:

                api_key = os.getenv(
                    "GITHUB_TOKEN"
                )

                if not api_key:

                    raise ValueError(
                        "GITHUB_TOKEN was not found. "
                        "Please add GITHUB_TOKEN to your .env file."
                    )

                self._llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=api_key,
                    base_url="https://models.inference.ai.azure.com", 
                    temperature=0.1, 
                ) 
 
        return self._llm 
 
    @llm.setter 
    def llm(self, value): 
        """Allow tests to inject a custom LLM or mock.""" 
 
        self._llm = value 
 
    def _invoke_llm( 
        self, 
        prompt: str, 
        retries: int = 2, 
        delay: float = 3.0 
    ) -> str: 
        """Call the LLM with automatic retry.""" 
 
        if self.skip_translation: 
            return "" 
 
        last_exc = None 
 
        for attempt in range(retries + 1): 
 
            try: 
 
                response = self.llm.invoke( 
                    prompt 
                ) 
 
                content = getattr( 
                    response, 
                    "content", 
                    "" 
                ) 
 
                if content is None: 
                    content = "" 
 
                return str(content).strip() 
 
            except Exception as exc: 
 
                last_exc = exc 
 
                if attempt < retries: 
 
                    time.sleep( 
                        delay 
                    ) 
 
        raise last_exc 
 
    # ================================================================= 
    # RTL TEXT HELPERS 
    # ================================================================= 
 
    def _fix_for_terminal(self, text: str) -> str: 
        """Reshape RTL text for terminal display.""" 
 
        try: 
 
            return arabic_reshaper.reshape( 
                str(text) 
            ) 
 
        except Exception: 
 
            return str(text) 
 
    def _fix_for_plot(self, text: str) -> str: 
        """Reshape + BiDi RTL text for Matplotlib.""" 
 
        try: 
 
            reshaped = arabic_reshaper.reshape( 
                str(text) 
            ) 
 
            return get_display( 
                reshaped 
            ) 
 
        except Exception: 
 
            return str(text) 
 
    # ================================================================= 
    # TRANSLATION 
    # ================================================================= 
 
    def _translate_batch( 
        self, 
        texts: list, 
        chunk_size: int = 20 
    ) -> list: 
        """ 
        Translate multiple texts in chunks. 
 
        Used mainly for SHAP feature names. 
        """ 
 
        if self.skip_translation: 
            return texts 
 
        if not texts: 
            return [] 
 
        original_texts = list(texts) 
 
        if len(texts) > self.max_features_to_translate: 
 
            print( 
                f"[INFO] Limiting translation to " 
                f"{self.max_features_to_translate} features " 
                f"(out of {len(texts)})" 
            ) 
 
            texts = texts[ 
                :self.max_features_to_translate 
            ] 
 
        results = [None] * len(texts) 
 
        to_translate = [] 
        to_translate_indices = [] 
 
        for i, text in enumerate(texts): 
 
            if text in self._feature_translation_cache: 
 
                results[i] = ( 
                    self._feature_translation_cache[text] 
                ) 
 
            else: 
 
                to_translate.append(text) 
                to_translate_indices.append(i) 
 
        if not to_translate: 
 
            return results 
 
        print( 
            f"[INFO] Translating {len(to_translate)} features " 
            f"in chunks of {chunk_size}..." 
        ) 
 
        for chunk_start in range( 
            0, 
            len(to_translate), 
            chunk_size 
        ): 
 
            chunk_end = min( 
                chunk_start + chunk_size, 
                len(to_translate) 
            ) 
 
            chunk = to_translate[ 
                chunk_start:chunk_end 
            ] 
 
            try: 
 
                text_list = "\n".join( 
                    [ 
                        f"{i + 1}. {text}" 
                        for i, text in enumerate(chunk) 
                    ] 
                ) 
 
                prompt = f""" 
You are a professional translation engine. 
 
Translate the following {self.language} texts into natural English. 
 
IMPORTANT: 
- This is a research dataset for text classification and explainability. 
- The texts may contain offensive, hateful, threatening, abusive, 
  or violent language. 
- Translate the text faithfully as written. 
- Do NOT refuse the translation. 
- Do NOT censor the text. 
- Do NOT soften the meaning. 
- Do NOT summarize the text. 
- Do NOT explain the text. 
- Do NOT add warnings or opinions. 
- Do NOT moralize about the content. 
- Preserve the original meaning as closely as possible. 
- Return ONLY the numbered English translations. 
- Keep the same order and numbering. 
 
Texts to translate: 
 
{text_list} 
""" 
 
                response = self._invoke_llm( 
                    prompt 
                ) 
 
                lines = response.strip().splitlines() 
 
                translations = [] 
 
                for line in lines: 
 
                    cleaned = re.sub( 
                        r"^\s*\d+\s*[\.\):\-]\s*", 
                        "", 
                        line.strip() 
                    ) 
 
                    if cleaned: 
                        translations.append( 
                            cleaned 
                        ) 
 
                # If the LLM returns exactly the expected number, 
                # use those translations. 
                if len(translations) == len(chunk): 
 
                    for idx, text, trans in zip( 
                        to_translate_indices[ 
                            chunk_start:chunk_end 
                        ], 
                        chunk, 
                        translations 
                    ): 
 
                        self._feature_translation_cache[ 
                            text 
                        ] = trans 
 
                        results[idx] = trans 
 
                else: 
 
                    print( 
                        "[WARNING] Translation response did not " 
                        "contain the expected number of translations." 
                    ) 
 
                    for j, idx in enumerate( 
                        to_translate_indices[ 
                            chunk_start:chunk_end 
                        ] 
                    ): 
 
                        if results[idx] is None: 
 
                            fallback_text = ( 
                                chunk[j] 
                                if j < len(chunk) 
                                else texts[idx] 
                            ) 
 
                            results[idx] = fallback_text 
 
                            self._feature_translation_cache[ 
                                fallback_text 
                            ] = fallback_text 
 
            except Exception as e: 
 
                print( 
                    f"[WARNING] Chunk translation failed: {e}" 
                ) 
 
                for j, idx in enumerate( 
                    to_translate_indices[ 
                        chunk_start:chunk_end 
                    ] 
                ): 
 
                    if results[idx] is None: 
 
                        fallback_text = chunk[j] 
 
                        results[idx] = ( 
                            fallback_text 
                        ) 
 
                        self._feature_translation_cache[ 
                            fallback_text 
                        ] = fallback_text 
 
        return results 
 
    def _translate_text( 
        self, 
        text: str 
    ) -> str: 
        """ 
        Translate a single RTL text into English. 
 
        This method is intentionally designed for research 
        datasets that may contain offensive, hateful, 
        threatening, or violent language. 
        """ 
 
        if self.skip_translation or not text: 
 
            return text 
 
        if text in self._feature_translation_cache: 
 
            return self._feature_translation_cache[ 
                text 
            ] 
 
        try: 
 
            prompt = f""" 
You are a professional translation engine. 
 
Translate the following {self.language} text into natural English. 
 
This is a research text-classification and SHAP explainability 
example from a dataset. 
 
IMPORTANT TRANSLATION RULES: 
1. Translate the text faithfully and directly. 
2. The text may contain offensive, hateful, abusive, 
   threatening, or violent language. 
3. Translate such language instead of refusing the task. 
4. Do NOT censor the text. 
5. Do NOT soften the meaning. 
6. Do NOT replace words with "[censored]" or similar wording. 
7. Do NOT summarize. 
8. Do NOT explain the text. 
9. Do NOT provide a safety warning. 
10. Do NOT add opinions or commentary. 
11. Preserve the original meaning and tone as closely as possible. 
12. Return ONLY the English translation. 
 
Source text: 
{text} 
""" 
 
            response = self._invoke_llm( 
                prompt 
            ) 
 
            translation = ( 
                response 
                .replace('"', "") 
                .replace("“", "") 
                .replace("”", "") 
                .strip() 
            ) 
 
            if not translation: 
 
                translation = text 
 
            self._feature_translation_cache[ 
                text 
            ] = translation 
 
            return translation 
 
        except Exception as e: 
 
            print( 
                f"[WARNING] Translation failed for " 
                f"'{text}': {e}" 
            ) 
 
            return text 
 
    def _translate_sample( 
        self, 
        sample_text: str 
    ) -> str: 
        """ 
        Translate and cache the original sample text. 
 
        The original text is always preserved so that the 
        English stage can display both the original RTL text 
        and its English translation. 
        """ 
 
        if not sample_text: 
 
            return None 
 
        # Always remember the exact original sample. 
        self._original_sample_text = sample_text 
 
        # Prevent repeated translation. 
        if self._eng_sample_text is not None: 
 
            return self._eng_sample_text 
 
        # If translation is disabled, retain original text. 
        if self.skip_translation: 
 
            self._eng_sample_text = sample_text 
 
            return self._eng_sample_text 
 
        print( 
            "\n[TRANSLATING SAMPLE TEXT TO ENGLISH...]" 
        ) 
 
        try: 
 
            # IMPORTANT: 
            # Do NOT use the feature translation cache here. 
            # The sample gets its own translation request. 
            prompt = f""" 
You are a professional translation engine. 
 
Translate the following {self.language} sentence into English. 
 
This sentence comes from a research dataset used for 
hate-speech / offensive-language classification and SHAP 
explainability. 
 
IMPORTANT: 
- Translate it faithfully. 
- The sentence may contain hateful, offensive, threatening, 
  abusive, or violent language. 
- You MUST translate the actual content. 
- Do NOT refuse. 
- Do NOT censor. 
- Do NOT soften the wording. 
- Do NOT summarize. 
- Do NOT explain the sentence. 
- Do NOT add a safety warning. 
- Do NOT moralize. 
- Do NOT express an opinion. 
- Preserve the meaning as closely as possible. 
- Return ONLY the English translation. 
 
Original {self.language} sentence: 
{sample_text} 
""" 
 
            response = self._invoke_llm( 
                prompt 
            ) 
 
            translation = ( 
                response 
                .replace('"', "") 
                .replace("“", "") 
                .replace("”", "") 
                .strip() 
            ) 
 
            if not translation: 
 
                translation = sample_text 
 
            self._eng_sample_text = translation 
 
        except Exception as e: 
 
            print( 
                f"[WARNING] Sample translation failed: {e}" 
            ) 
 
            # Keep the original instead of losing the sample. 
            self._eng_sample_text = sample_text 
 
        # ------------------------------------------------------------- 
        # Always show both versions 
        # ------------------------------------------------------------- 
 
        print( 
            f"\nOriginal {self.language} Sample:" 
        ) 
 
        print( 
            f"  {self._fix_for_terminal(sample_text)}" 
        ) 
 
        print( 
            "\nTranslated English Sample:" 
        ) 
 
        print( 
            f"  {self._eng_sample_text}" 
        ) 
 
        print( 
            "-" * 80 
        ) 
 
        return self._eng_sample_text 
 
    def _translate_features( 
        self, 
        feature_names: list 
    ) -> list: 
        """Translate feature names to English.""" 
 
        if self.skip_translation or not feature_names: 
 
            return feature_names 
 
        if len(feature_names) > self.max_features_to_translate: 
 
            print( 
                f"[INFO] Only translating top " 
                f"{self.max_features_to_translate} features" 
            ) 
 
            feature_names_to_translate = ( 
                feature_names[ 
                    :self.max_features_to_translate 
                ] 
            ) 
 
        else: 
 
            feature_names_to_translate = feature_names 
 
        translated = self._translate_batch( 
            feature_names_to_translate 
        ) 
 
        # Preserve all feature names. 
        # If translation was limited, retain the remaining 
        # original names so SHAP dimensions stay correct. 
 
        if len(translated) < len(feature_names): 
 
            translated = ( 
                translated 
                + feature_names[ 
                    len(translated): 
                ] 
            ) 
 
        return translated 
 
    def _translate_label( 
        self, 
        label: str 
    ) -> str: 
        """Translate prediction label to English.""" 
 
        if self.skip_translation: 
 
            return label 
 
        if not label: 
 
            return label 
 
        try: 
 
            prompt = f""" 
Translate the following {self.language} prediction label 
into English. 
 
Return ONLY the English label. 
Do not add explanations or commentary. 
 
Label: 
{label} 
""" 
 
            response = self._invoke_llm( 
                prompt 
            ) 
 
            result = ( 
                response 
                .replace('"', "") 
                .replace("“", "") 
                .replace("”", "") 
                .strip() 
            ) 
 
            return result if result else label 
 
        except Exception: 
 
            return label 
 
    # ================================================================= 
    # FEATURE EXTRACTION 
    # ================================================================= 
 
    def get_top_features( 
        self, 
        shap_values, 
        num_features: int = 5 
    ) -> list: 
        """Extract top N features ranked by absolute SHAP value.""" 
 
        values = shap_values.values 
        feature_names = shap_values.feature_names 
 
        if feature_names is None: 
 
            feature_names = [ 
                f"Feature {i}" 
                for i in range( 
                    values.shape[-1] 
                ) 
            ] 
 
        if values.ndim == 2: 
 
            mean_abs = np.mean( 
                np.abs(values), 
                axis=0 
            ) 
 
            mean_vals = np.mean( 
                values, 
                axis=0 
            ) 
 
        else: 
 
            mean_abs = np.abs(values) 
            mean_vals = values 
 
        sorted_idx = np.argsort( 
            mean_abs 
        )[::-1][:num_features] 
 
        return [ 
            ( 
                str(feature_names[i]), 
                float(mean_vals[i]) 
            ) 
            for i in sorted_idx 
        ] 
 
    # ================================================================= 
    # SHAP TRANSLATION 
    # ================================================================= 
 
    def _translate_shap_explanation( 
        self, 
        shap_values 
    ): 
        """Translate feature names in a SHAP Explanation.""" 
 
        if self.skip_translation: 
 
            return shap_values 
 
        if ( 
            not hasattr( 
                shap_values, 
                "feature_names" 
            ) 
            or shap_values.feature_names is None 
        ): 
 
            return shap_values 
 
        feature_names = list( 
            shap_values.feature_names 
        ) 
 
        eng_names = self._translate_features( 
            feature_names 
        ) 
 
        try: 
 
            values = shap_values.values 
            base_values = shap_values.base_values 
 
            data = ( 
                shap_values.data 
                if hasattr( 
                    shap_values, 
                    "data" 
                ) 
                else None 
            ) 
 
            display_data = ( 
                shap_values.display_data 
                if hasattr( 
                    shap_values, 
                    "display_data" 
                ) 
                else None 
            ) 
 
            # --------------------------------------------------------- 
            # Keep exactly the same number of feature names as SHAP 
            # features. Never reduce a 1-D explanation to one feature. 
            # --------------------------------------------------------- 
 
            if values.ndim == 1: 
 
                n_features = len(values) 
 
            elif values.ndim == 2: 
 
                n_features = values.shape[1] 
 
            else: 
 
                n_features = values.shape[-1] 
 
            if len(eng_names) > n_features: 
 
                eng_names = eng_names[ 
                    :n_features 
                ] 
 
            elif len(eng_names) < n_features: 
 
                for i in range( 
                    len(eng_names), 
                    n_features 
                ): 
 
                    eng_names.append( 
                        str(feature_names[i]) 
                        if i < len(feature_names) 
                        else f"Feature_{i}" 
                    ) 
 
            translated = shap.Explanation( 
                values=values, 
                base_values=base_values, 
                data=data, 
                feature_names=eng_names, 
                display_data=display_data 
            ) 
 
            return translated 
 
        except Exception as e: 
 
            print( 
                f"[WARNING] Could not translate SHAP " 
                f"explanation: {e}" 
            ) 
 
            return shap_values 
 
    # ================================================================= 
    # DEPENDENCE PLOT 
    # ================================================================= 
 
    def _show_dependence_plot( 
        self, 
        shap_values, 
        features, 
        feature_names, 
        title_prefix="" 
    ): 
        """Show a dependence plot with RTL support.""" 
 
        # Ensure RTL labels use the registered language font. 
        self._set_rtl_plot_font() 
 
        fig, ax = plt.subplots( 
            figsize=(10, 6) 
        ) 
 
        if hasattr( 
            shap_values, 
            "values" 
        ): 
 
            values = shap_values.values 
 
        else: 
 
            values = shap_values 
 
        if features is None: 
 
            features = self._cached_features 
 
        if feature_names is None: 
 
            feature_names = ( 
                shap_values.feature_names 
                if hasattr( 
                    shap_values, 
                    "feature_names" 
                ) 
                else [] 
            ) 
 
        if ( 
            values.ndim == 2 
            and values.shape[1] >= 2 
        ): 
 
            x_data = ( 
                features[:, 0] 
                if features is not None 
                else np.arange( 
                    len(values) 
                ) 
            ) 
 
            y_data = values[:, 0] 
 
            color_data = ( 
                features[:, 1] 
                if features is not None 
                else values[:, 1] 
            ) 
 
            x_label = ( 
                self._fix_for_plot( 
                    str(feature_names[0]) 
                ) 
                if len(feature_names) > 0 
                else "Feature 1" 
            ) 
 
            color_label = ( 
                self._fix_for_plot( 
                    str(feature_names[1]) 
                ) 
                if len(feature_names) > 1 
                else "Feature 2" 
            ) 
 
        else: 
 
            x_data = ( 
                features 
                if features is not None 
                else np.arange( 
                    len(values) 
                ) 
            ) 
 
            y_data = ( 
                values 
                if values.ndim == 1 
                else values[:, 0] 
            ) 
 
            color_data = y_data 
 
            x_label = ( 
                self._fix_for_plot( 
                    str(feature_names[0]) 
                ) 
                if len(feature_names) > 0 
                else "Feature" 
            ) 
 
            color_label = x_label 
 
        scatter = ax.scatter( 
            x_data, 
            y_data, 
            c=color_data, 
            cmap="RdYlBu", 
            alpha=0.6, 
            s=50, 
            edgecolors="black", 
            linewidth=0.5 
        ) 
 
        ax.set_xlabel( 
            f"Feature value: {x_label}", 
            fontsize=12 
        ) 
 
        ax.set_ylabel( 
            f"SHAP value for {x_label}", 
            fontsize=12 
        ) 
 
        ax.set_title( 
            f"Dependence Plot ({title_prefix}): " 
            f"{x_label}\n" 
            f"Colored by {color_label}", 
            fontsize=14, 
            fontweight="bold" 
        ) 
 
        ax.axhline( 
            y=0, 
            color="gray", 
            linestyle="--", 
            alpha=0.5 
        ) 
 
        ax.grid( 
            True, 
            alpha=0.3 
        ) 
 
        cbar = plt.colorbar( 
            scatter, 
            ax=ax 
        ) 
 
        cbar.set_label( 
            f"Feature value: {color_label}", 
            fontsize=11 
        ) 
 
        plt.tight_layout() 
        plt.show() 
 
        return ax 
 
    # ================================================================= 
    # PLOT FUNCTION HANDLER 
    # ================================================================= 
 
    def _call_plot_function( 
        self, 
        plot_fn, 
        shap_values 
    ): 
        """ 
        Call a SHAP/shap_rtl plot function with 
        appropriate arguments. 
        """ 
 
        import inspect 
 
        sig = inspect.signature( 
            plot_fn 
        ) 
 
        plot_name = ( 
            plot_fn.__name__ 
            if hasattr( 
                plot_fn, 
                "__name__" 
            ) 
            else str(plot_fn) 
        ) 
 
        # ------------------------------------------------------------- 
        # Scatter / dependence plot 
        # ------------------------------------------------------------- 
 
        if plot_name in [ 
            "scatter", 
            "dependence_plot" 
        ]: 
 
            if ( 
                hasattr( 
                    shap_values, 
                    "data" 
                ) 
                and shap_values.data is not None 
            ): 
 
                features = shap_values.data 
 
            else: 
 
                features = self._cached_features 
 
            if "features" in sig.parameters: 
 
                return plot_fn( 
                    shap_values=shap_values, 
                    features=features 
                ) 
 
            else: 
 
                return plot_fn( 
                    shap_values=shap_values 
                ) 
 
        # ------------------------------------------------------------- 
        # Decision plot 
        # ------------------------------------------------------------- 
 
        elif plot_name == "decision_plot": 
 
            base_value = getattr( 
                shap_values, 
                "base_values", 
                None 
            ) 
 
            if base_value is None: 
 
                base_value = 0.0 
 
            if ( 
                isinstance( 
                    base_value, 
                    np.ndarray 
                ) 
                and base_value.size == 1 
            ): 
 
                base_value = float( 
                    base_value.reshape(-1)[0] 
                ) 
 
            if ( 
                hasattr( 
                    shap_values, 
                    "data" 
                ) 
                and shap_values.data is not None 
            ): 
 
                features = shap_values.data 
 
            else: 
 
                features = self._cached_features 
 
            if isinstance( 
                shap_values, 
                list 
            ): 
 
                shap_vals = [ 
                    sv.values 
                    for sv in shap_values 
                ] 
 
                features_list = [ 
                    sv.data 
                    for sv in shap_values 
                    if hasattr( 
                        sv, 
                        "data" 
                    ) 
                ] 
 
                feature_names = ( 
                    shap_values[0].feature_names 
                    if hasattr( 
                        shap_values[0], 
                        "feature_names" 
                    ) 
                    else None 
                ) 
 
                return plot_fn( 
                    base_value=base_value, 
                    shap_values=shap_vals, 
                    features=features_list, 
                    feature_names=feature_names 
                ) 
 
            else: 
 
                shap_vals = ( 
                    shap_values.values 
                    if hasattr( 
                        shap_values, 
                        "values" 
                    ) 
                    else shap_values 
                ) 
 
                feature_names = ( 
                    shap_values.feature_names 
                    if hasattr( 
                        shap_values, 
                        "feature_names" 
                    ) 
                    else None 
                ) 
 
                if ( 
                    isinstance( 
                        shap_vals, 
                        np.ndarray 
                    ) 
                    and shap_vals.ndim == 2 
                    and shap_vals.shape[0] == 1 
                ): 
 
                    shap_vals = shap_vals[0] 
 
                    features = ( 
                        features[0] 
                        if features is not None 
                        and len(features) > 0 
                        else None 
                    ) 
 
                return plot_fn( 
                    base_value=base_value, 
                    shap_values=shap_vals, 
                    features=features, 
                    feature_names=feature_names 
                ) 
 
        # ------------------------------------------------------------- 
        # Heatmap 
        # ------------------------------------------------------------- 
 
        elif plot_name == "heatmap": 
 
            if ( 
                hasattr( 
                    shap_values, 
                    "feature_names" 
                ) 
                and shap_values.feature_names is not None 
            ): 
 
                return plot_fn( 
                    shap_values 
                ) 
 
            else: 
 
                try: 
 
                    if ( 
                        self._cached_feature_names 
                        is not None 
                    ): 
 
                        new_shap = shap.Explanation( 
                            values=shap_values.values, 
                            base_values=shap_values.base_values, 
                            data=( 
                                shap_values.data 
                                if hasattr( 
                                    shap_values, 
                                    "data" 
                                ) 
                                else None 
                            ), 
                            feature_names=( 
                                self._cached_feature_names 
                            ) 
                        ) 
 
                        return plot_fn( 
                            new_shap 
                        ) 
 
                except Exception as e: 
 
                    print( 
                        f"[WARNING] Heatmap feature " 
                        f"rebuild issue: {e}" 
                    ) 
 
                return plot_fn( 
                    shap_values 
                ) 
 
        # ------------------------------------------------------------- 
        # Other SHAP plots 
        # ------------------------------------------------------------- 
 
        else: 
 
            return plot_fn( 
                shap_values 
            ) 
 
    # ================================================================= 
    # PROMPTS 
    # ================================================================= 
 
    def _rtl_prompt( 
        self, 
        feat_name: str, 
        shap_val: float, 
        prediction_label: str 
    ) -> str: 
        """Generate RTL-language explanation prompt.""" 
 
        if self.skip_translation: 
 
            return ( 
                f"[Translations disabled] " 
                f"Feature: {feat_name} " 
                f"(SHAP: {shap_val:+.4f})" 
            ) 
 
        impact = ( 
            "positively" 
            if shap_val > 0 
            else "negatively" 
        ) 
 
        return f""" 
SYSTEM: You are an AI Explainability Assistant. 
 
LANGUAGE: {self.language} 
 
PREDICTION: {prediction_label} 
 
FEATURE: "{feat_name}" 
 
SHAP VALUE: {shap_val:+.4f} 
 
TASK:

Explain the feature "{feat_name}" and why it {impact} impacts
the prediction "{prediction_label}" in {self.language}.

The explanation must be based on the ORIGINAL SAMPLE provided above.

Your explanation must naturally contain three connected parts, in this order:

First, explain the general or overall meaning of the feature.
Explain what the word, phrase, or n-gram normally means and how it is
generally used in the language.

Second, explain what the feature means or represents specifically in
the given sample. Consider the exact context in which the feature
appears. If the feature has a different contextual meaning, such as
an insult, metaphor, sarcasm, threat, praise, or another contextual
use, explain that difference.

Third, explain how this contextual use of the feature contributes to
the model's prediction. Connect the meaning of the feature in this
specific sample with the predicted class.

IMPORTANT:

- Respond in ONE natural paragraph only.
- Do NOT use headings.
- Do NOT use labels such as "General Meaning", "Sample Meaning",
  "Contextual Meaning", "Model Contribution", or similar labels.
- Do NOT use bullet points.
- Do NOT number the explanation.
- Do NOT separate the explanation into sections.
- Do NOT repeat the SHAP numeric value.
- Use {self.language} only.
- Do not use English in the explanation.
- Explain the feature generally FIRST and then explain its use in
  the given sample.
- The explanation must refer to the exact sample provided above.
- Do not explain the feature in isolation from the sample.
- If the feature is an n-gram or phrase, explain the meaning of the
  complete phrase.
- Do not invent context that is not present in the sample.
- Clearly connect the contextual meaning of the feature to the
  prediction.
- Keep the explanation concise but informative.
- Output ONLY the explanation paragraph.
"""
 
    def _eng_prompt(
        self,
        feat_name: str,
        shap_val: float,
        prediction_label: str,
        eng_translation: str,
        original_sample_text: str = None,
        eng_sample_text: str = None
    ) -> str:
        """Generate English explanation prompt."""

        impact = (
            "positively"
            if shap_val > 0
            else "negatively"
        )

        original_sample_text = (
            original_sample_text
            if original_sample_text is not None
            else self._original_sample_text
        )

        eng_sample_text = (
            eng_sample_text
            if eng_sample_text is not None
            else self._eng_sample_text
        )

        return f"""
SHAP Analysis:

Prediction label:
'{prediction_label}'

English prediction label:
'{eng_translation}'

Original {self.language} sample:
'{original_sample_text}'

English translation of the sample:
'{eng_sample_text}'

Feature:
'{feat_name}'

SHAP value:
{shap_val:+.4f}

TASK:

Explain the feature '{feat_name}' and why it {impact} impacts
the model prediction.

The explanation must naturally contain three connected parts,
in this order:

First, explain the general or overall meaning of the feature.
Explain what the word, phrase, or n-gram normally means and how it
is generally used.

Second, explain what the feature means or represents specifically
in the given sample. Use the original {self.language} sample and
its English translation to understand the exact context. Explain
whether the feature is being used literally, figuratively,
insultingly, positively, negatively, sarcastically, threateningly,
or in another contextual way when applicable.

Third, explain how this contextual use contributes to the model's
prediction. Connect the feature's contextual meaning with the
predicted class.

IMPORTANT:

- Write ONE natural English paragraph only.
- Do NOT use headings.
- Do NOT use labels such as "General Meaning", "Sample Meaning",
  "Contextual Meaning", "Model Contribution", or similar labels.
- Do NOT use bullet points.
- Do NOT number the explanation.
- Do NOT divide the explanation into sections.
- Explain the general meaning FIRST.
- Then explain the meaning and use in the given sample.
- Then connect that contextual use to the model prediction.
- Base the explanation on the exact sample provided above.
- If the feature is an n-gram or phrase, explain the complete phrase.
- Do not invent context that is not present in the sample.
- Do not repeat the SHAP numeric value unnecessarily.
- Focus on linguistic and model-related interpretation.
- Output ONLY the explanation paragraph.
"""
 
    # ================================================================= 
    # MAIN WORKFLOW 
    # ================================================================= 
 
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
        """ 
        Run complete RTL SHAP explanation workflow. 
 
        Workflow: 
        1. Display RTL plot 
        2. Ask for RTL explanation 
        3. Generate RTL explanation if requested 
        4. Ask for English plot 
        5. Generate English plot if requested 
        6. Ask for English explanation 
        7. Generate English explanation if requested 
        """ 
 
        if features is not None: 
 
            self._cached_features = features 
 
        elif ( 
            hasattr( 
                shap_values, 
                "data" 
            ) 
            and shap_values.data is not None 
        ): 
 
            self._cached_features = ( 
                shap_values.data 
            ) 
 
        if ( 
            hasattr( 
                shap_values, 
                "feature_names" 
            ) 
            and shap_values.feature_names is not None 
        ): 
 
            self._cached_feature_names = ( 
                shap_values.feature_names 
            ) 
 
        print( 
            "\n" + "=" * 80 
        ) 
 
        print( 
            f"{self.language.upper()} SHAP EXPLANATIONS".center(80) 
        ) 
 
        print( 
            f"Prediction: {prediction_label}".center(80) 
        ) 
 
        print( 
            "=" * 80 
        ) 
 
        # ------------------------------------------------------------- 
        # Step 1: RTL Plot 
        # ------------------------------------------------------------- 
 
        if ( 
            plot_fn is None 
            and features is not None 
        ): 
 
            print( 
                f"\n[GENERATING " 
                f"{self.language.upper()} DEPENDENCE PLOT...]" 
            ) 
 
            feature_names = ( 
                shap_values.feature_names 
                if hasattr( 
                    shap_values, 
                    "feature_names" 
                ) 
                else [] 
            ) 
 
            self._show_dependence_plot( 
                shap_values, 
                features, 
                feature_names, 
                title_prefix=self.language 
            ) 
 
        elif plot_fn is not None: 
 
            print( 
                f"\n[GENERATING " 
                f"{self.language.upper()} SHAP PLOT...]" 
            ) 
 
            self._call_plot_function( 
                plot_fn, 
                shap_values 
            ) 
 
        else: 
 
            print( 
                "\n[INFO] No plot_fn provided and no " 
                "features for dependence plot. " 
                "Skipping visualization." 
            ) 
 
        # ------------------------------------------------------------- 
        # Step 2: RTL explanation 
        # ------------------------------------------------------------- 
 
        self._ask_rtl_explanation( 
            shap_values=shap_values, 
            prediction_label=prediction_label, 
            plot_fn=plot_fn, 
            sample_text=sample_text, 
            features=features, 
            max_features=max_features 
        ) 
 
    # ================================================================= 
    # RTL EXPLANATION 
    # ================================================================= 
 
    def _ask_rtl_explanation( 
        self, 
        shap_values, 
        prediction_label, 
        plot_fn, 
        sample_text, 
        features, 
        max_features=10 
    ): 
        """Ask whether the user wants RTL explanation.""" 
 
        while True: 
 
            try: 
 
                choice = input( 
                    f"\nWould you like explanation in " 
                    f"{self.language}? (Yes/No): " 
                ).strip().lower() 
 
                if choice in [ 
                    "yes", 
                    "y" 
                ]: 
 
                    self._run_rtl_explanations( 
                        shap_values=shap_values, 
                        prediction_label=prediction_label, 
                        max_features=max_features 
                    ) 
 
                    self._ask_english_plot( 
                        shap_values=shap_values, 
                        prediction_label=prediction_label, 
                        plot_fn=plot_fn, 
                        sample_text=sample_text, 
                        features=features 
                    ) 
 
                    return 
 
                elif choice in [ 
                    "no", 
                    "n", 
                    "" 
                ]: 
 
                    print( 
                        f"\n[INFO] Skipping " 
                        f"{self.language} explanation." 
                    ) 
 
                    self._ask_english_plot( 
                        shap_values=shap_values, 
                        prediction_label=prediction_label, 
                        plot_fn=plot_fn, 
                        sample_text=sample_text, 
                        features=features 
                    ) 
 
                    return 
 
                else: 
 
                    print( 
                        "Please enter Yes or No." 
                    ) 
 
            except KeyboardInterrupt: 
 
                print( 
                    "\n[INFO] Exiting. Thank you!" 
                ) 
 
                return 
 
            except Exception as e: 
 
                print( 
                    f"[ERROR] {e}" 
                ) 
 
                return 
 
    def _run_rtl_explanations( 
        self, 
        shap_values, 
        prediction_label, 
        max_features=10 
    ): 
        """Generate RTL explanations.""" 
 
        print( 
            f"\n[{self.language} Explanation Mode]" 
        ) 
 
        while True: 
 
            try: 
 
                user_input = input( 
                    f"How many features would you like explained " 
                    f"in {self.language}? " 
                    f"(Enter 0 or press ESC to skip): " 
                ) 
 
                if ( 
                    user_input.strip() == "" 
                    or user_input.lower() == "escape" 
                    or user_input == "0" 
                ): 
 
                    print( 
                        f"\n[INFO] Skipping " 
                        f"{self.language} explanation." 
                    ) 
 
                    return 
 
                n = int( 
                    user_input 
                ) 
 
                if n <= 0: 
 
                    print( 
                        f"[INFO] Skipping " 
                        f"{self.language} explanation." 
                    ) 
 
                    return 
 
                n = min( 
                    n, 
                    max_features 
                ) 
 
                break 
 
            except ValueError: 
 
                print( 
                    "Please enter a valid number " 
                    "(or press ESC/Enter to skip)." 
                ) 
 
            except KeyboardInterrupt: 
 
                print( 
                    "\n[INFO] Operation cancelled by user." 
                ) 
 
                return 
 
        top_features = self.get_top_features( 
            shap_values, 
            n 
        ) 
 
        print( 
            "-" * self.RW 
        ) 
 
        print( 
            f"PREDICTION: {prediction_label}".rjust( 
                self.RW 
            ) 
        ) 
 
        print( 
            "-" * self.RW 
        ) 
 
        print( 
            f"\nAnalysis of Top {n} SHAP Features " 
            f"({self.language})".rjust( 
                self.RW 
            ) 
        ) 
 
        print( 
            "-" * 50 
        ) 
 
        for feat_name, shap_val in top_features: 
 
            try: 
 
                display_name = ( 
                    self._fix_for_terminal( 
                        feat_name 
                    ) 
                ) 
 
                print( 
                    f"\nFeature: {display_name} " 
                    f"| Value: {shap_val:+.4f}" 
                ) 
 
                print( 
                    "-" * 40 
                ) 
 
                prompt = self._rtl_prompt( 
                    feat_name, 
                    shap_val, 
                    prediction_label 
                ) 
 
                response = self._invoke_llm( 
                    prompt 
                ) 
 
                fixed_response = ( 
                    self._fix_for_terminal( 
                        response 
                    ) 
                ) 
 
                print( 
                    fixed_response.rjust( 
                        self.RW 
                    ) 
                ) 
 
                print( 
                    "-" * 40 
                ) 
 
            except Exception as e: 
 
                print( 
                    f"Feature: {feat_name} " 
                    f"(API unavailable: {str(e)})" 
                    .rjust(self.RW) 
                ) 
 
        print( 
            "\n" + "=" * 80 
        ) 
 
    # ================================================================= 
    # ENGLISH PLOT 
    # ================================================================= 
 
    def _ask_english_plot( 
        self, 
        shap_values, 
        prediction_label, 
        plot_fn, 
        sample_text, 
        features=None 
    ): 
        """Ask whether the user wants English plot.""" 
 
        while True: 
 
            try: 
 
                choice = input( 
                    "\nWould you like to see the plot in English? " 
                    "(Yes/No): " 
                ).strip().lower() 
 
                if choice in [ 
                    "yes", 
                    "y" 
                ]: 
 
                    print( 
                        "\n[INFO] Generating English plot..." 
                    ) 
 
                    self._set_english_plot_font() 
 
                    # ------------------------------------------------- 
                    # English Dependence Plot 
                    # ------------------------------------------------- 
 
                    if ( 
                        plot_fn is None 
                        and features is not None 
                    ): 
 
                        print( 
                            "\n[INFO] Generating " 
                            "English Dependence Plot..." 
                        ) 
 
                        feature_names = ( 
                            shap_values.feature_names 
                            if hasattr( 
                                shap_values, 
                                "feature_names" 
                            ) 
                            else [] 
                        ) 
 
                        eng_feature_names = ( 
                            self._translate_features( 
                                list(feature_names) 
                            ) 
                        ) 
 
                        self._show_dependence_plot( 
                            shap_values, 
                            features, 
                            eng_feature_names, 
                            title_prefix="English" 
                        ) 
 
                        eng_shap = ( 
                            self._rebuild_explanation_with_names( 
                                shap_values, 
                                eng_feature_names 
                            ) 
                        ) 
 
                        eng_sample_text = None 
 
                        if sample_text: 
 
                            eng_sample_text = ( 
                                self._translate_sample( 
                                    sample_text 
                                ) 
                            ) 
 
                    # ------------------------------------------------- 
                    # English SHAP Plot 
                    # ------------------------------------------------- 
 
                    else: 
 
                        ( 
                            eng_shap, 
                            eng_sample_text 
                        ) = self._display_english_plot( 
                            shap_values=shap_values, 
                            prediction_label=prediction_label, 
                            plot_fn=plot_fn, 
                            sample_text=sample_text 
                        ) 
 
                    # ------------------------------------------------- 
                    # English explanation 
                    # ------------------------------------------------- 
 
                    self._ask_english_explanation( 
                        shap_values=eng_shap, 
                        prediction_label=prediction_label, 
                        eng_sample_text=eng_sample_text, 
                        features=features 
                    ) 
 
                    return 
 
                elif choice in [ 
                    "no", 
                    "n", 
                    "" 
                ]: 
 
                    print( 
                        "\n[INFO] Exiting. Thank you!" 
                    ) 
 
                    return 
 
                else: 
 
                    print( 
                        "Please enter Yes or No." 
                    ) 
 
            except KeyboardInterrupt: 
 
                print( 
                    "\n[INFO] Exiting. Thank you!" 
                ) 
 
                return 
 
            except Exception as e: 
 
                print( 
                    f"[ERROR] {e}" 
                ) 
 
                return 
 
    def _display_english_plot( 
        self, 
        shap_values, 
        prediction_label, 
        plot_fn, 
        sample_text 
    ): 
        """ 
        Display the English plot. 
 
        This method only generates the English plot. 
        """ 
 
        print( 
            "\n[INFO] Translating features to English..." 
        ) 
 
        # English feature names must be rendered with a Latin-safe font. 
        self._set_english_plot_font() 
 
        eng_sample_text = None 
 
        if sample_text: 
 
            eng_sample_text = ( 
                self._translate_sample( 
                    sample_text 
                ) 
            ) 
 
        if isinstance( 
            shap_values, 
            list 
        ): 
 
            feature_names = list( 
                shap_values[0].feature_names or [] 
            ) if shap_values else [] 
 
        else: 
 
            feature_names = list( 
                shap_values.feature_names or [] 
            ) 
 
        eng_names = self._translate_features( 
            feature_names 
        ) 
 
        eng_label = self._translate_label( 
            prediction_label 
        ) 
 
        print( 
            f"Translated Prediction Label: " 
            f"{eng_label}" 
        ) 
 
        if isinstance( 
            shap_values, 
            list 
        ): 
 
            eng_shap = [] 
 
            for sv in shap_values: 
 
                eng_shap.append( 
                    self._rebuild_explanation_with_names( 
                        sv, 
                        eng_names 
                    ) 
                ) 
 
        else: 
 
            eng_shap = ( 
                self._rebuild_explanation_with_names( 
                    shap_values, 
                    eng_names 
                ) 
            ) 
 
        if plot_fn is not None: 
 
            print( 
                "\n[GENERATING ENGLISH SHAP PLOT...]" 
            ) 
 
            self._call_plot_function( 
                plot_fn, 
                eng_shap 
            ) 
 
        else: 
 
            print( 
                "\n[INFO] No plot_fn provided. " 
                "Skipping English visualization." 
            ) 
 
        return ( 
            eng_shap, 
            eng_sample_text 
        ) 
 
    # ================================================================= 
    # ENGLISH EXPLANATION 
    # ================================================================= 
 
    def _ask_english_explanation( 
        self, 
        shap_values, 
        prediction_label, 
        eng_sample_text, 
        features=None 
    ): 
        """Ask whether the user wants English explanation.""" 
 
        while True: 
 
            try: 
 
                choice = input( 
                    "\nWould you like explanation in English? " 
                    "(Yes/No): " 
                ).strip().lower() 
 
                if choice in [ 
                    "yes", 
                    "y" 
                ]: 
 
                    self._run_english_explanations( 
                        shap_values=shap_values, 
                        prediction_label=prediction_label, 
                        eng_sample_text=eng_sample_text, 
                        features=features 
                    ) 
 
                    return 
 
                elif choice in [ 
                    "no", 
                    "n", 
                    "" 
                ]: 
 
                    print( 
                        "\n[INFO] Exiting. Thank you!" 
                    ) 
 
                    return 
 
                else: 
 
                    print( 
                        "Please enter Yes or No." 
                    ) 
 
            except KeyboardInterrupt: 
 
                print( 
                    "\n[INFO] Exiting. Thank you!" 
                ) 
 
                return 
 
            except Exception as e: 
 
                print( 
                    f"[ERROR] {e}" 
                ) 
 
                return 
 
    def _run_english_explanations( 
        self, 
        shap_values, 
        prediction_label, 
        eng_sample_text, 
        features=None, 
        max_features=10 
    ): 
        """Generate English explanations.""" 
 
        print( 
            "\n" + "=" * 80 
        ) 
 
        print( 
            "ENGLISH EXPLANATIONS".center(80) 
        ) 
 
        print( 
            "=" * 80 
        ) 
 
        # ------------------------------------------------------------- 
        # Always display original + translated sample 
        # ------------------------------------------------------------- 
 
        if self._original_sample_text: 
 
            print( 
                f"\nOriginal {self.language} Sample:" 
            ) 
 
            print( 
                f"  {self._fix_for_terminal(self._original_sample_text)}" 
            ) 
 
            print( 
                "\nTranslated English Sample:" 
            ) 
 
            print( 
                f"  {eng_sample_text or self._eng_sample_text}" 
            ) 
 
            print( 
                "-" * 80 
            ) 
 
        elif eng_sample_text: 
 
            print( 
                "\nTranslated English Sample:" 
            ) 
 
            print( 
                f"  {eng_sample_text}" 
            ) 
 
            print( 
                "-" * 80 
            ) 
 
        print( 
            "\n[English Explanation Mode]" 
        ) 
 
        while True: 
 
            try: 
 
                user_input = input( 
                    "How many features would you like explained " 
                    "in English? " 
                    "(Press Enter or type 0 to exit): " 
                ).strip() 
 
                if ( 
                    user_input == "" 
                    or user_input == "0" 
                ): 
 
                    print( 
                        "Exiting explanation mode." 
                    ) 
 
                    return 
 
                n = int( 
                    user_input 
                ) 
 
                if n <= 0: 
 
                    print( 
                        "Exiting explanation mode." 
                    ) 
 
                    return 
 
                n = min( 
                    n, 
                    max_features 
                ) 
 
                break 
 
            except ValueError: 
 
                print( 
                    "Please enter a valid number." 
                ) 
 
            except KeyboardInterrupt: 
 
                print( 
                    "\n[INFO] Operation cancelled." 
                ) 
 
                return 
 
        eng_top_features = ( 
            self.get_top_features( 
                shap_values, 
                n 
            ) 
        ) 
 
        eng_label = ( 
            self._translate_label( 
                prediction_label 
            ) 
        ) 
 
        print( 
            "-" * 80 
        ) 
 
        print( 
            f"PREDICTION: {eng_label}".center( 
                80 
            ) 
        ) 
 
        print( 
            "-" * 80 
        ) 
 
        print( 
            f"\nAnalysis of Top {n} SHAP Features " 
            f"(English)" 
        ) 
 
        print( 
            "-" * 50 
        ) 
 
        for feat_name, shap_val in eng_top_features: 
 
            try: 
 
                print( 
                    f"\nFeature: {feat_name} " 
                    f"| Value: {shap_val:+.4f}" 
                ) 
 
                print( 
                    "-" * 40 
                ) 
 
                prompt = self._eng_prompt( 
                    feat_name, 
                    shap_val, 
                    prediction_label, 
                    eng_label 
                ) 
 
                response = self._invoke_llm( 
                    prompt 
                ) 
 
                print( 
                    response 
                ) 
 
                print( 
                    "-" * 40 
                ) 
 
            except Exception as e: 
 
                print( 
                    f"Feature: {feat_name} " 
                    f"(API unavailable: {str(e)})" 
                ) 
 
        print( 
            "\n" + "=" * 80 
        ) 
 
        print( 
            "Analysis complete." 
        ) 
 
    # ================================================================= 
    # SHAP EXPLANATION REBUILD 
    # ================================================================= 
 
    def _rebuild_explanation_with_names( 
        self, 
        shap_values, 
        new_names 
    ): 
        """Rebuild SHAP Explanation with new feature names.""" 
 
        try: 
 
            return shap.Explanation( 
                values=shap_values.values, 
                base_values=shap_values.base_values, 
                data=shap_values.data, 
                feature_names=new_names, 
                display_data=shap_values.display_data, 
            ) 
 
        except Exception: 
 
            sv_eng = copy.copy( 
                shap_values 
            ) 
 
            try: 
 
                object.__setattr__( 
                    sv_eng, 
                    "feature_names", 
                    new_names 
                ) 
 
            except Exception: 
 
                sv_eng.__dict__[ 
                    "feature_names" 
                ] = new_names 
 
            return sv_eng 
 
 
# ===================================================================== 
# CONVENIENCE FUNCTIONS 
# ===================================================================== 
 
def create_explainer( 
    language: str = "Urdu", 
    model=None, 
    vectorizer=None, 
    api_type: str = "groq", 
    skip_translation: bool = False, 
    max_features_to_translate: int = 50, 
    custom_font_path: str = None 
): 
    """Factory function to create an RTLSHAPExplainer instance.""" 
 
    return RTLSHAPExplainer( 
        language=language, 
        model=model, 
        vectorizer=vectorizer, 
        api_type=api_type, 
        skip_translation=skip_translation, 
        max_features_to_translate=max_features_to_translate, 
        custom_font_path=custom_font_path 
    ) 
 
 
def get_supported_languages(): 
    """Return list of supported RTL languages.""" 
 
    return RTLSHAPExplainer.SUPPORTED_LANGUAGES 
 
 
def print_supported_languages(): 
    """Print supported languages with their script names.""" 
 
    print( 
        "\nSupported RTL Languages:" 
    ) 
 
    print( 
        "-" * 40 
    ) 
 
    print( 
        "  Arabic    (العربية)" 
    ) 
 
    print( 
        "  Urdu      (اردو)" 
    ) 
 
    print( 
        "  Persian   (فارسی)" 
    ) 
 
    print( 
        "  Hebrew    (עברית)" 
    ) 
 
    print( 
        "-" * 40 
    ) 
 
 
# ===================================================================== 
# DIRECT EXECUTION 
# ===================================================================== 
 
if __name__ == "__main__": 
 
    print( 
        "=" * 80 
    ) 
 
    print( 
        "SHAP RTL EXPLAINER - Ready to use!".center( 
            80 
        ) 
    ) 
 
    print( 
        "=" * 80 
    ) 
 
    print( 
        "\n[INFO] Using LLM for translations" 
    ) 
 
    print_supported_languages() 
