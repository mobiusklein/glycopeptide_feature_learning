import os
import sys
import traceback

from setuptools import find_packages, Extension, setup
from distutils.command.build_ext import build_ext
from distutils.errors import (CCompilerError, DistutilsExecError,
                              DistutilsPlatformError)


def make_extensions():
    try:
        import numpy
    except ImportError:
        print("Installation requires `numpy`")
        raise
    try:
        import brainpy
    except ImportError:
        print("Installation requires `brainpy`, install with `python -m pip install brain-isotopic-distribution`")
        raise
    try:
        import ms_peak_picker
    except ImportError:
        print("Installation requires `ms_peak_picker`")
        raise
    from Cython.Build import cythonize
    cython_directives = {
        'embedsignature': True,
        "profile": True
    }
    extensions = cythonize([
        Extension(name='feature_learning._c.data_source', sources=["feature_learning/_c/data_source.pyx"],
                  include_dirs=[brainpy.get_include(), ms_peak_picker.get_include(), numpy.get_include()]),
        Extension(name='feature_learning._c.peak_relations', sources=["feature_learning/_c/peak_relations.pyx"],
                  include_dirs=[brainpy.get_include(), ms_peak_picker.get_include(), numpy.get_include()]),
    ], compiler_directives=cython_directives)
    return extensions


ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)
if sys.platform == 'win32':
    # 2.6's distutils.msvc9compiler can raise an IOError when failing to
    # find the compiler
    ext_errors += (IOError,)


class BuildFailed(Exception):

    def __init__(self):
        self.cause = sys.exc_info()[1]  # work around py 2/3 different syntax

    def __str__(self):
        return str(self.cause)


class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            traceback.print_exc()
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors:
            traceback.print_exc()
            raise BuildFailed()
        except ValueError:
            # this can happen on Windows 64 bit, see Python issue 7511
            traceback.print_exc()
            if "'path'" in str(sys.exc_info()[1]):  # works with both py 2/3
                raise BuildFailed()
            raise


cmdclass = {}

cmdclass['build_ext'] = ve_build_ext


def status_msgs(*msgs):
    print('*' * 75)
    for msg in msgs:
        print(msg)
    print('*' * 75)


setup(name='feature_learning',
      packages=find_packages(),
      ext_modules=make_extensions(),
      cmdclass=cmdclass)
