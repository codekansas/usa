#!/usr/bin/env python

import re

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()


with open("requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: list[str] = f.read().splitlines()


with open("requirements-ipynb.txt", "r", encoding="utf-8") as f:
    requirements_ipynb: list[str] = f.read().splitlines()


with open("usa/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in ml/__init__.py"
version: str = version_re.group(1)


setup(
    name="usa-net",
    version=version,
    description="USA net project",
    author="Benjamin Bolte",
    url="https://github.com/codekansas/usa-net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=requirements,
    tests_require=requirements_dev,
    extras_require={
        "dev": requirements_dev,  # Testing, linting, etc.
        "ipynb": requirements_ipynb,  # Jupyter notebook
    },
)
