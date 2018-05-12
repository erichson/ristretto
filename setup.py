#! /usr/bin/env python
#
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0
from __future__ import print_function

import os
import shutil
from distutils.command.clean import clean as Clean
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

DISTNAME = 'ristretto'
DESCRIPTION = 'ristretto: Randomized Dimension Reduction Library'
with open('README.md') as f:
    LONG_DESRIPTION = f.read()
AUTHOR = 'N. Benjamin Erichson'
AUTHOR_EMAIL = 'erichson@uw.edu'
URL = 'https://github.com/erichson/ristretto'
LICENSE = 'GNU'
KEYWORDS = ['randomized algorithms',
            'dimension reduction',
            'singular value decomposition',
            'matrix approximations']

# import restricted version of ristretto to get version
import ristretto
VERSION = ristretto.__version__

# Custom clean command to remove build artifacts from scikit-learn setup.py
# https://github.com/scikit-learn/scikit-learn/blob/master/setup.py
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('ristretto'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))

extensions = [
    Extension("ristretto.externals.cdnmf_fast", ["ristretto/externals/cdnmf_fast.pyx"]),
]
cmdclass = {'build_ext' : build_ext, 'clean' : CleanCommand}


def setup_package():
    metadata = dict(name=DISTNAME,
                    author=AUTHOR,
                    author_email=AUTHOR_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    version=VERSION,
                    keywords=KEYWORDS,
                    long_description=LONG_DESRIPTION,
                    classifiers=[
                        'Development Status :: 4 - Beta',
                        'Intended Audience :: Science/Research',
                        'Topic :: Scientific/Engineering :: Mathematics',
                        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                        #'Programming Language :: Python :: 2.7',
                        #'Programming Language :: Python :: 3.5',
                        'Programming Language :: Python :: 3.6',
                    ],
                    test_suite='nose.collector',
                    install_requires=['numpy', 'scipy', 'Cython'],
                    tests_require=['numpy', 'scipy'],
                    cmdclass=cmdclass,
                    ext_modules=cythonize(extensions))

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
