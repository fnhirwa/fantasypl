from setuptools import setup
import os

version = None
parent_dir = os.path.dirname(__file__)
init_path = os.path.join(parent_dir, "fantasypl", "__init__.py")
with open(init_path) as f:
    exec(f.read(), version)
DESCRIPTION = "Fantasy Premier League API Wrapper"

setup(
    name="fantasypl",
    version=version,
    author="Felix (Felix Hirwa Nshuti)",
    author_email="<hirwanshutiflx@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/fnhirwa/fantasypl",
    long_description=open("README.md").read(),
    keywords=["fantasy premier league", "fpl", "crawler"],
    install_requires=[line for line in open("requirements.txt", "r", encoding="utf-8")],
)
