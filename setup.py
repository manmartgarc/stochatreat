# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:20:03 2019

===============================================================================
@author:    Manuel Martinez
===============================================================================
"""
import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
        name='stochatreat',
        version='0.0.1',
        author='Manuel Martinez',
        author_email='manmartgarc@gmail.com',
        description='Randomized block assignment using pandas',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/manmartgarc/stochatreat',
        packages=setuptools.find_packages(),
        classifiers=[
                'Programming Language :: Python :: 3',
                'License :: OSI Approved :: MIT License',
                'Operating System :: OS Independent',
                ]
        )
