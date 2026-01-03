import setuptools

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pauc",
    version="0.2.1",
    author="Manas Mahale, Srijit Seal",
    description="A Python library for ROC curve analysis, comparison, and visualization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/srijitseal/pauc",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
)

