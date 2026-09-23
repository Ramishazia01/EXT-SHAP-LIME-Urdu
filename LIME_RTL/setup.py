from setuptools import setup, find_packages

setup(
    name="urdu_lime",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "lime",
        "langchain-openai",
        "python-dotenv",
        "arabic-reshaper",
        "python-bidi",
        "openpyxl"
    ],
    author="Rameesha Zia, Dr. Shahid Iqbal Malik",
    description="A framework for interpreting Urdu NLP models using LIME and GenAI",
    # We removed 'requires-python' and 'classifiers' to avoid conflict warnings
)