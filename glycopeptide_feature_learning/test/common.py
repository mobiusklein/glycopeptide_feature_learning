import os
import gzip
import pickle
import sys

try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass

data_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_data"))


def datafile(name):
    path = os.path.join(data_path, name)
    if not os.path.exists(path):
        raise Exception(
            ("The path %r could not be located in the test suite's data package." % (path, )) +
            "If you are NOT running the glycopeptide_feature_learning test suite, you should not"
            " be using this function.")
    return path


def gzload(path):
    with gzip.open(path, 'rb') as fh:
        if sys.version_info.major > 2:
            return pickle.load(fh, encoding='latin1')
        else:
            return pickle.load(fh)
