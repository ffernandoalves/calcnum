"""
Gera o pacote do biblioteca calcnum, veja a seção
Desenvolvimento no README.md.
"""

import re
import pathlib

from setuptools import setup, find_packages

# root path
here = pathlib.Path(__file__).parent.resolve()

# get calcnum version
VERSION_FILE = here/"src"/"calcnum"/"__init__.py"
with open(VERSION_FILE, "rt", encoding="utf-8") as vf:
    getversion = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vf.read(), re.M)
if getversion is not None:
    new_version = getversion.group(1)
else:
    raise RuntimeError(f"Unable to find version string in {VERSION_FILE}.")

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# get requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="calcnum",
    version=new_version,
    # install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["numpy", "matplolib", "calcnum"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # description="",
    # author_email="",
    # python_requires=">=3.9",
    # author="",
    # license="",
    # url="",
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
