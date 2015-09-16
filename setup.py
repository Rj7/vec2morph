#!/usr/bin/env python

from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(name='vec2morph',
      version='0.2',
      description='',
      author='Joachim Daiber',
      author_email='daiber.joachim@gmail.com',
      url='',
	  #ext_modules = cythonize("extract.pyx"),
      install_requires=['joblib', 'gensim', 'DAWG', 'argparse', 'networkx', 'annoy'],
     )
