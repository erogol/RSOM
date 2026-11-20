from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RSOM",
    version="0.1.0",
    author="Eren Gölge",
    author_email="",
    description="A Rectifying Self-Organizing Map (RSOM) implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erogol/RSOM",
    py_modules=["rsom"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "torch",
        "scikit-learn",
    ],
)
