"""Setup script for PhenoCell."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="phenocell",
    version="1.0.0",
    author="PhenoCell Authors",
    description="PhenoCell: Learning Perturbation Effects in Cell Painting via Text-Guided Contrastive Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/phenocell",
    packages=find_packages(
        include=["src", "src.*", "vendor", "vendor.*"]
    ),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
