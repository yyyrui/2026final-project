"""Python setup.py for project_name package"""
import io
import os
from setuptools import find_packages, setup

# 可以先放著，不需要手動執行，也不用逐行研究。
# 主要是套件的打包與安裝設定檔。在這個專案中，安裝工具可以透過它取得套件名稱、版本，以及需要安裝哪些依賴。


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]


setup(
    name="tactis",
    version="0.1.3",
    description="Transformer-Attentional Copulas for Multivariate Time Series",
    url="https://github.com/ServiceNow/tactis/tree/tactis-2",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Arjun Ashok, Étienne Marcotte, Valentina Zantedeschi, Nicolas Chapados, Alexandre Drouin",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"research": read_requirements("requirements_research.txt")},
)
