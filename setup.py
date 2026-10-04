"""
PEP 517 build frontend shim.

The actual build is driven by scikit-build-core via pyproject.toml.
This file exists only for legacy tooling (pip install .) that does not
yet read pyproject.toml build-system tables directly.
"""
from skbuild import setup  # scikit-build-core exposes a setup() shim

setup(
    name="vecengine",
    version="0.1.0",
    description="High-performance C++ vector distance library with Python bindings",
    author="Maximilian Miller",
    python_requires=">=3.9",
    packages=["vecengine"],
    package_dir={"vecengine": "bindings"},
    cmake_install_dir="vecengine",
    cmake_args=[
        "-DCMAKE_BUILD_TYPE=Release",
    ],
    extras_require={
        "dev": ["numpy", "pytest"],
    },
)
