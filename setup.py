from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if not line.startswith('#') and line.strip()]

# Read long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="options_visualizer",
    version="0.1.0",
    author="Reeve Francis",
    author_email="your.email@example.com",
    description="A web application for visualizing stock options data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/options-visualizer",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "options-visualizer=main:main",
        ],
    },
) 