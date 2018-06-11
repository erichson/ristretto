#! /usr/bin/env python
#
# Authors: N. Benjamin Erichson
#          Joseph Knox
# License: GNU General Public License v3.0
from __future__ import print_function
import ast
import io
import os
import re
import sys
import shutil
from setuptools import setup, Extension, find_packages
from distutils.command.clean import clean as Clean
from distutils.version import LooseVersion


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


if sys.version_info[0] < 3:
    # Python 2.*
    import __builtin__ as builtins
else:
    import builtins


def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(here, 'ristretto/__init__.py')
    _version_re = re.compile(r'__version__\s+=\s+(?P<version>.*)')
    with io.open(init_file, 'r', encoding='utf8') as f:
        match = _version_re.search(f.read())
        version = match.group('version') if match is not None else '"unknown"'
    return str(ast.literal_eval(version))


VERSION = get_version()

SCIPY_MIN_VERSION = '0.0.13'
NUMPY_MIN_VERSION = '1.8.2'
CYTHON_MIN_VERSION = '0.23'

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

# Custom cythonize command to check if Cython installed and up to date from scikit-learn
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/_build_utils/__init__.py#L63
def cythonize_extensions(extensions):
    """Tweaks for building extensions between release and development mode."""
    message = ('\nPlease install cython with a version >= {0} in order '
               'to build a ristretto development version.').format(
                   CYTHON_MIN_VERSION)
    try:
        import Cython
        if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
            message += ' Your version of Cython was {0}.'.format(Cython.__version__)
            raise ValueError(message)
        from Cython.Build import cythonize
    except (ModuleNotFoundError, ImportError) as exc:
        args = exc.args[0] + message
        raise type(exc)(args)

    return cythonize(extensions)


extensions = [
    Extension("ristretto.externals.cdnmf_fast", ["ristretto/externals/cdnmf_fast.pyx"]),
]
ext_modules = cythonize_extensions(extensions)

cmdclass = {'clean': CleanCommand}

extra_setuptools_args = dict(
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        'numpy >= {0}'.format(NUMPY_MIN_VERSION),
        'scipy >= {0}'.format(SCIPY_MIN_VERSION),
        ]
)


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
                    long_description_content_type='text/markdown',
                    classifiers=[
                        'Development Status :: 4 - Beta',
                        'Intended Audience :: Science/Research',
                        'Topic :: Scientific/Engineering :: Mathematics',
                        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                        'Programming Language :: Python :: 2.7',
                        'Programming Language :: Python :: 3.5',
                        'Programming Language :: Python :: 3.6',
                    ],
                    test_suite='nose.collector',
                    cmdclass=cmdclass,
                    ext_modules=ext_modules,
                    packages=find_packages(exclude=['tests']),
                    **extra_setuptools_args)

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
