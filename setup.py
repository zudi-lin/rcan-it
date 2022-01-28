import os
import sys
import numpy as np
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

requirements = [
    'jupyter>=1.0',
    'scipy>=1.5',
    'scikit-learn>=0.23.1',
    'scikit-image>=0.17.2',
    'matplotlib>=3.3.0',
    'yacs>=0.1.8',
    'imageio>=2.9.0',
    'GPUtil>=1.4.0',
    'tqdm>=4.62.0'
]


def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]


def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/zudi-lin/rcan-it'

    setup(name='ptsr',
          description='A PyTorch framework for image super-resolution',
          version=__version__,
          url=url,
          license='MIT',
          author='Zudi Lin',
          install_requires=requirements,
          include_dirs=getInclude(),
          packages=find_packages(),
          )


if __name__ == '__main__':
    # pip install --editable .
    setup_package()
