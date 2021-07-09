# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:20:03 2019

===============================================================================
@author:    Manuel Martinez
===============================================================================
"""
import codecs
from pathlib import Path

from setuptools import find_packages, setup


def read(rel_path: str) -> str:
    here = Path(__file__).parent.absolute()
    with codecs.open(str(here.joinpath(rel_path)), 'r') as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path=rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')


with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='stochatreat',
    version=get_version('stochatreat/__init__.py'),
    author='Manuel Martinez',
    author_email='manmartgarc@gmail.com',
    description='Stratified random assignment using pandas',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/manmartgarc/stochatreat',
    keywords=[
        'randomization',
        'block randomization',
        'stratified randomization'
        'stratified',
        'strata',
    ],
    packages=find_packages(),
    install_requires=[
        'pandas',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
