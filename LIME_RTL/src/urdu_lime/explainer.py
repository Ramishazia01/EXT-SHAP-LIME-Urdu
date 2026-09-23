from email.mime import text
import os                                                                        # Import operating system interfaces
import copy                                                                      # Import shallow and deep copy operations
import numpy as np                                                               # Import numerical computing library
import matplotlib.pyplot as plt                                                  # Import plotting library
from IPython.display import HTML, display                                        # Import tools for HTML rendering in notebooks
from lime.lime_text import LimeTextExplainer                                     # Import LIME explainer for text models
from langchain_openai import ChatOpenAI                                          # Import LangChain's OpenAI model wrapper
from dotenv import load_dotenv                                                   # Import tool to load environment variables from .env

# For RTL support
import arabic_reshaper                                                           # Import library to reshape Arabic characters
from bidi.algorithm import get_display                                           # Import BiDi algorithm for right-to-left layout

load_dotenv()                                                                    # Load environment variables (like API keys)

class RTLLimeExplainer:                                                          # Define a class for RTL-aware LIME explanations
    def __init__(self, model, vectorizer, language="Urdu"):                      # Initialize with model, vectorizer, and target language
        self.model = model                                                       # Store the trained machine learning model
        self.vectorizer = vectorizer                                             # Store the text vectorizer (e.g., TF-IDF)
        self.language = language                                                 # Set the target RTL language
        self.class_names = [str(c) for c in model.classes_]                      # Get class names from the model
        # Force LIME to split by whitespace so it perturbs actual words
        self.explainer = LimeTextExplainer(class_names=self.class_names, split_expression=r'\s+')         # Initialize the standard LIME text explainer
        
        # --- UPDATED FOR GITHUB MODELS ---
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=os.getenv("GITHUB_TOKEN"),                                   # Using your GITHUB_TOKEN from .env
            base_url="https://models.inference.ai.azure.com",                    # Pointing to GitHub's inference endpoint
            temperature=0.1
        )

    def _predict_proba(self, texts_list):                                        # Helper to get probability predictions from the model
        return self.model.predict_proba(self.vectorizer.transform(texts_list))   # Vectorize text and return probability scores

    def _fix_for_terminal(self, text):
        try:
            return arabic_reshaper.reshape(text) + "\u200F" # Add RLM to terminal too
        except:
            return text                                                          # Return the original text as a fallback

    def _fix_for_terminal(self, text):                                           # Helper to prepare RTL text for terminal output
        """Fixes broken Arabic in terminals."""
        try:                                                                     # Attempt to process the text
            return arabic_reshaper.reshape(text)                                 # Return reshaped text for joining letters
        except:                                                                  # If processing fails
            return text                                                          # Return original text

    def explain_instance(self, text, num_features=10):                           # Generate LIME explanation for a single text input
        probs = self._predict_proba([text])[0]                                   # Get prediction probabilities for the input
        predicted_idx = np.argmax(probs)                                         # Identify the index of the highest probability class
        exp = self.explainer.explain_instance(                                   # Call LIME to explain the prediction
            text, self._predict_proba, num_features=num_features, labels=[predicted_idx]
        )                                                                        # Pass text, prediction function, and target label
        return exp, predicted_idx                                                # Return the explanation object and predicted index

    def plot_matplotlib(self, exp, predicted_idx, title_prefix=None):            # Create a Matplotlib bar chart of feature importance
        if title_prefix is None: title_prefix = self.language                    # Default title prefix to the language name
        
        # FIX: Set global font family to support Arabic/Urdu glyphs
        plt.rcParams['font.family'] = 'sans-serif'                               # Set font family to sans-serif
        plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans']      # Prioritize fonts that typically include RTL glyphs
        
        fig = exp.as_pyplot_figure(label=predicted_idx)                          # Generate the base LIME pyplot figure
        ax = plt.gca()                                                           # Get the current axes object
        
        labels = [item.get_text() for item in ax.get_yticklabels()]              # Extract existing y-axis labels (features)
        fixed_labels = [self._fix_for_plot(lbl) for lbl in labels]               # Apply RTL fixes to each label
        ax.set_yticklabels(fixed_labels)                                         # Re-apply fixed labels to the y-axis
        
        plt.title(f"{title_prefix} LIME - Predicted: {self.class_names[predicted_idx]}") # Set the chart title
        plt.tight_layout()                                                       # Adjust layout to prevent label clipping
        plt.show()                                                               # Display the plot

    def run_full_analysis(self, exp, user_tweet, predicted_idx):                 # Execute complete analysis including LLM insights
        target_class = self.class_names[predicted_idx]                           # Get the name of the predicted class
        other_class = self.class_names[1 - predicted_idx] if len(self.class_names) == 2 else f"Not {target_class}" # Get opposite class
        
        print("\n" + "="*80)                                                     # Print separator line
        print(f"{self.language.upper()} LIME VISUALIZATIONS (Predicted Class: {target_class})") # Print header
        
        print(f"\n[DISPLAYING {self.language.upper()} HTML VISUALIZATION...]")   # Print status message
        display(HTML(exp.as_html(labels=[predicted_idx])))                       # Render LIME's interactive HTML view in notebook

        print(f"\n[GENERATING {self.language.upper()} MATPLOTLIB PLOT...]")      # Print status message
        self.plot_matplotlib(exp, predicted_idx)                                 # Call the RTL-fixed Matplotlib plotter

        num_v = int(input(f"\nHow many top features of class '{target_class}' to explain? ")) # Ask user for positive feature count
        num_nv = int(input(f"How many top features of class '{other_class}' to explain? ")) # Ask user for negative feature count

        exp_list = exp.as_list(label=predicted_idx)                              # Get list of feature weights from explanation
        v_rtl = sorted([f for f in exp_list if f[1] > 0], key=lambda x: x[1], reverse=True)[:num_v] # Filter top positive features
        nv_rtl = sorted([f for f in exp_list if f[1] < 0], key=lambda x: x[1])[:num_nv] # Filter top negative features

        RW = 120                                                                 # Set right-alignment width
        print("-" * RW)                                                          # Print separator line
        print(f"Text Input: {self._fix_for_terminal(user_tweet)}".rjust(RW))     # Display reshaped input tweet right-aligned
        print(f"Model Prediction: {target_class}".rjust(RW))                     # Display model prediction right-aligned
        print("-" * RW)                                                          # Print separator line

        def rtl_prompt(feat_name, is_pos):                                       # Define prompt generator for RTL linguistic analysis
            target = target_class if is_pos else other_class                     # Determine which class to justify
            return f"""
            SYSTEM: You are a Safety Research Assistant.
            LANGUAGE: {self.language}
            TEXT: "{user_tweet}"
            WORD: "{feat_name}"
            TASK: Explain why the model associated the word "{feat_name}" with the class "{target}" in {self.language} only.
            INSTRUCTIONS: Respond in ONE concise paragraph in {self.language} only. 
            Linguistic analysis only. No English, no scores or values.
            FORMAT: "Word: {feat_name} in {self.language} only:"
            """                                                                  # Return the formatted prompt string

        if v_rtl:                                                                # If there are positive features to analyze
            print(f"Analysis of Features for {target_class} ({self.language})".rjust(RW)) # Print section header
            for f in v_rtl:                                                      # Iterate through positive features
                try:
                    response = self.llm.invoke(rtl_prompt(f[0], True)).content.strip() 
                    print(self._fix_for_terminal(response).rjust(RW))
                except Exception:
                    print(f"Word: {f[0]} (GitHub API Rate Limit Hit)".rjust(RW))

        if nv_rtl:                                                               # If there are negative features to analyze
            print(f"\nAnalysis of Features for {other_class} ({self.language})".rjust(RW)) # Print section header
            for f in nv_rtl:                                                     # Iterate through negative features
                try:
                    response = self.llm.invoke(rtl_prompt(f[0], False)).content.strip() 
                    print(self._fix_for_terminal(response).rjust(RW))
                except Exception:
                    print(f"Word: {f[0]} (GitHub API Rate Limit Hit)".rjust(RW))

        print("\n" + "="*80)                                                     # Print separator line
        choice = input("Do you want English Plot and Analysis? (yes/no): ").lower() # Prompt user for English translation option
        if choice == 'yes':                                                      # If user chooses yes
            self._run_english_logic(exp, exp_list, user_tweet, v_rtl, nv_rtl, target_class, other_class, predicted_idx) # Run English logic

    def _run_english_logic(self, exp, exp_list, user_tweet, v_rtl, nv_rtl, target_class, other_class, predicted_idx): # Logic for English translations
        print("\n[INFO] Generating Safety-Aware English Translation...")         # Print status message
        try:
            trans_res = self.llm.invoke(f"Translate this {self.language} text to literal English only for safety analysis, no other sentence except the literal translation: {user_tweet}").content.strip() # Translate text
            trans = trans_res.replace('"', '')                                       # Clean up quotes from translation
        except Exception:
            trans = "[Translation Unavailable due to Rate Limit]"

        tokens = trans.split()                                                   # Split translated text into words
        weights = [f[1] for f in exp_list]                                       # Extract original weights from LIME
        eng_map = [(tokens[i] if i < len(tokens) else "...", w) for i, w in enumerate(weights)] # Map weights to English tokens
        
        exp_en = copy.deepcopy(exp)                                              # Create a deep copy of the explanation object
        exp_en.as_list = lambda *args, **kwargs: eng_map                         # Monkey-patch weights list with English mapping
        exp_en.domain_mapper.indexed_string.raw = trans                          # Replace raw text with translation
        exp_en.domain_mapper.indexed_string.as_list = tokens                     # Replace token list with English tokens

        print(f"ENGLISH LIME VISUALIZATION (Class: {target_class})")             # Print section header
        display(HTML(exp_en.as_html(labels=[predicted_idx])))                   # Display modified HTML in English
        self.plot_matplotlib(exp_en, predicted_idx, title_prefix="English")      # Plot the English version
        
        print(f"English Translation: {trans}\n" + "-"*80)                        # Print the full English translation

        def eng_prompt(f_rtl, is_pos):                                           # Define prompt generator for English-language analysis
            target = target_class if is_pos else other_class                     # Determine target class
            return f"Linguistic Analysis: In the context of '{trans}', explain in one concise English paragraph why the {self.language} word '{f_rtl}' impacts the '{target}' classification." # Return prompt

        if v_rtl:                                                                # If positive features exist
            print(f"Top Features for {target_class} (English Analysis)")         # Print section header
            for f in v_rtl:                                                      # Iterate through features
                try:
                    print(self.llm.invoke(eng_prompt(f[0], True)).content.strip() + "\n") # Get and print LLM English analysis
                except Exception:
                    print(f"Analysis for {f[0]} unavailable (Rate Limit).\n")