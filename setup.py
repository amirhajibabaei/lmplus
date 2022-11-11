# +
from __future__ import annotations

from setuptools import find_packages, setup

with open("lmplus/version.py") as f:
    version: dict[str, str] = {}
    exec(f.read(), version)
__version__ = version["__version__"]


setup(
    name="lmplus",
    version=__version__,
    author="Amir Hajibabaei",
    author_email="a.hajibabaei.86@gmail.com",
    description="LAMMPS utility Python scripts",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["numpy", "scipy", "ase"],
    url="https://github.com/amirhajibabaei/lmplus",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
