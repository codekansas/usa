#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()


with open("requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: list[str] = f.read().splitlines()


with open("usa/__version__.txt", "r", encoding="utf-8") as f:
    version: str = f.read().strip()


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
    extras_require={"dev": requirements_dev},
)
