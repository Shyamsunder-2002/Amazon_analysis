"""
Setup script for Amazon India Analytics Platform
"""

from setuptools import setup, find_packages

setup(
    name="amazon-india-analytics",
    version="1.0.0",
    description="Comprehensive E-Commerce Analytics Platform for Amazon India Data",
    author="Data Science Team",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.1",
        "pandas>=2.1.1",
        "numpy>=1.24.3",
        "plotly>=5.17.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "scikit-learn>=1.3.0",
        "sqlalchemy>=2.0.21",
        "scipy>=1.11.3",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Data Scientists",
        "Programming Language :: Python :: 3.8+",
    ],
)
