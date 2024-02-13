from setuptools import setup

__version__ = "0.0.0"
DESCRIPTION = "Fantasy Premier League API Wrapper"

setup(
    name="fantasypl",
    version=__version__,
    author="Felix (Felix Hirwa Nshuti)",
    author_email="<hirwanshutiflx@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/fnhirwa/fantasypl",
    long_description=open("README.md").read(),
    keywords=["fantasy premier league", "fpl", "crawler"],
    install_requires=[line for line in open("requirements.txt", "r", encoding="utf-8")],
)
