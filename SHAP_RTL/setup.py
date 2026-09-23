from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "RTL-aware SHAP plots for Urdu and Arabic"

setup(
    name="SHAP_RTL_package",
    version="1.0.0",
    description="Right-to-left (RTL) aware SHAP plots for Urdu, Arabic, and other RTL languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SHAP_RTL contributors",
    python_requires=">=3.8",
    packages=find_packages(),
    package_data={
        "shap_rtl": ["fonts/*.ttf", "fonts/*.otf"],
    },
    install_requires=[
        "shap>=0.41.0",
        "matplotlib>=3.5.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "arabic-reshaper>=3.0.0",
        "python-bidi>=0.4.2",
        # Part B: LLM explainer dependencies 
        "langchain-openai>=0.1.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "explainer": [
            "langchain-openai>=0.1.0",
            "python-dotenv>=1.0.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
