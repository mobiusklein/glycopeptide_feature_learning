import os
import sys
import traceback
import platform

from setuptools import find_packages, Extension as _Extension, setup
from distutils.command.build_ext import build_ext
from distutils.errors import (CCompilerError, DistutilsExecError,
                              DistutilsPlatformError)


def has_option(name):
    try:
        sys.argv.remove('--%s' % name)
        return True
    except ValueError:
        pass
    # allow passing all cmd line options also as environment variables
    env_val = os.getenv(name.upper().replace('-', '_'), 'false').lower()
    if env_val == "true":
        return True
    return False


include_diagnostics = has_option("include-diagnostics")
force_cythonize = has_option("force-cythonize")
no_openmp = has_option('no-openmp')

with_openmp = not no_openmp


def configure_openmp(ext):
    # http://www.microsoft.com/en-us/download/confirmation.aspx?id=2092 was required.
    if os.name == 'nt' and with_openmp:
        ext.extra_compile_args.append("/openmp")
    elif platform.system() == 'Darwin':
        pass
    elif with_openmp:
        ext.extra_compile_args.append("-fopenmp")
        ext.extra_link_args.append("-fopenmp")


def Extension(*args, **kwargs):
    ext = _Extension(*args, **kwargs)
    return ext


def OpenMPExtension(*args, **kwargs):
    ext = Extension(*args, **kwargs)
    configure_openmp(ext)
    return ext



def make_extensions():
    try:
        import numpy
    except ImportError:
        print("Installation requires `numpy`")
        raise
    try:
        import ms_deisotope
    except ImportError:
        print("Installation requires `ms_deisotope`, install with `python -m pip install ms_deisotope`")
        raise
    try:
        import glycopeptidepy
    except ImportError:
        print("Installation requires `glycopeptidepy`")
        raise
    from Cython.Build import cythonize
    cython_directives = {
        'embedsignature': True,
        "profile": include_diagnostics
    }
    extensions = cythonize([
        Extension(name='glycopeptide_feature_learning._c.data_source',
                  sources=["src/glycopeptide_feature_learning/_c/data_source.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension(name='glycopeptide_feature_learning._c.peak_relations',
                  sources=["src/glycopeptide_feature_learning/_c/peak_relations.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension(name='glycopeptide_feature_learning._c.amino_acid_classification',
                  sources=["src/glycopeptide_feature_learning/_c/amino_acid_classification.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension(name='glycopeptide_feature_learning._c.approximation',
                  sources=["src/glycopeptide_feature_learning/_c/approximation.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension(name='glycopeptide_feature_learning._c.model_types',
                  sources=["src/glycopeptide_feature_learning/_c/model_types.pyx"],
                  include_dirs=[numpy.get_include()]),
        OpenMPExtension(name='glycopeptide_feature_learning.scoring._c.scorer',
                  sources=["src/glycopeptide_feature_learning/scoring/_c/scorer.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension(name='glycopeptide_feature_learning.scoring._c.score_set',
                  sources=["src/glycopeptide_feature_learning/scoring/_c/score_set.pyx"],
                  include_dirs=[numpy.get_include()]),
    ], compiler_directives=cython_directives, force=force_cythonize)
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


setup(name='glycopeptide_feature_learning',
      zip_safe=False,
      packages=find_packages("src"),
      package_dir={"": "src"},
      ext_modules=make_extensions(),
      include_package_data=True,
      package_data={
          "glycopeptide_feature_learning": ["src/glycopeptide_feature_learning/data/*"],
      },
      entry_points={
          'console_scripts': [
              "glycopeptide-feature-learning = glycopeptide_feature_learning.tool:cli"
          ],
      },
      cmdclass=cmdclass)
