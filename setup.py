# Copyright 2023 Felix. All Rights Reserved.
# Check the LICENSE file for the license information
# ==================================================

import setuptools
from setuptools import setup
from pathlib import Path
from pip._vendor.packaging import tags
import re


__version__ = "0.0.1"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


long_description = re.sub(
    r"<img [^>]*class=\"only-dark\"[^>]*>",
    "",
    long_description,
    flags=re.MULTILINE,
)

long_description = re.sub(
    r"<a [^>]*class=\"only-dark\"[^>]*>((?:(?!<\/a>).)|\s)*<\/a>\n",
    "",
    long_description,
    flags=re.MULTILINE,
)


setup(
    name="fpl",
    version=__version__,
    author="Felix Hirwa Nshuti",
    author_email="felixhirwa9@gmail.com",
    description=(
        "Fantasy premier league python API. Get data from the official FPL API."
        "Includes some analytical tools for machine learning predictions."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://fnhirwa.com",
    project_urls={
        "Docs": "https://fnhirwa.com",
        "Source": "https://github.com/hirwa-nshuti/fpl",
    },
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        line.split(" ")[0].strip()
        for line in open("requirements.txt", "r", encoding="utf-8")
    ],
    classifiers=["License :: OSI Approved :: MIT License"],
    license="MIT License",
)



