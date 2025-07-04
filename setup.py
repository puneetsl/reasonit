#!/usr/bin/env python3
"""
Setup script for ReasonIt - Advanced LLM Reasoning Architecture.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements from pyproject.toml or fallback to basic requirements
requirements = [
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-ai>=0.0.13",
    "click>=8.0.0",
    "rich>=13.0.0",
    "aiohttp>=3.8.0",
    "asyncio-throttle>=1.0.0",
    "tiktoken>=0.5.0",
    "numpy>=1.24.0",
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "python-dotenv>=1.0.0"
]

setup(
    name="reasonit",
    version="0.1.0",
    author="ReasonIt Team",
    author_email="team@reasonit.ai",
    description="Advanced LLM reasoning architecture that pushes the limits of small models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reasonit/reasonit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest-cov>=4.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "benchmarks": [
            "datasets>=2.10.0",
            "evaluate>=0.4.0",
            "transformers>=4.30.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "reasonit=reasonit:main",
        ],
    },
    include_package_data=True,
    package_data={
        "reasonit": [
            "*.md",
            "*.toml",
            "*.yaml",
            "*.yml",
            "examples/*.py",
            "examples/*.md",
        ],
    },
    zip_safe=False,
    keywords="llm reasoning ai machine-learning nlp artificial-intelligence",
    project_urls={
        "Bug Reports": "https://github.com/reasonit/reasonit/issues",
        "Source": "https://github.com/reasonit/reasonit",
        "Documentation": "https://reasonit.readthedocs.io/",
    },
)