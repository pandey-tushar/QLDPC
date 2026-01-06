"""
Setup script for QLDPC package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="qldpc",
    version="0.1.0",
    description="Quantum Low-Density Parity-Check Code Simulation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tushar Pandey",
    author_email="tusharp@tamu.edu",
    url="https://github.com/tusharpandey13/QLDPC",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "ldpc>=0.1.0",
        "bposd>=0.1.0",
        "pymatching>=2.0.0",
        "matplotlib>=3.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="quantum error correction, LDPC, quantum computing, simulation",
)

