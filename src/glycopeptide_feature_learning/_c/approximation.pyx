# cython: embedsignature=True
cimport cython
import numpy as np
cimport numpy as np

np.import_array()

from matplotlib import pyplot as plt


@cython.boundscheck(False)
cdef size_t position_before(double[::1] x, double val) nogil:
    cdef:
        size_t i, n
    n = x.shape[0]
    for i in range(n):
        if x[i] > val:
            return i
    return n


@cython.boundscheck(False)
cdef size_t binary_search(double[::1] x, double val) nogil:
    cdef:
        size_t lo, hi, mid, n
        double y

    n = x.shape[0]
    lo = 0
    hi = n
    while hi != lo:
        mid = (hi + lo) // 2
        y = x[mid]
        if y == val:
            return min(mid + 1, n)
        elif hi == (lo + 1):
            return min(mid + 1, n)
        elif y < val:
            lo = mid
        else:
            hi = mid
    return n


cdef class StepFunction(object):
    """
    A basic step function adapted from statsmodels

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array-like
    y : array-like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].
    """
    def __init__(self, x, y, ival=0., sorted=False, side='left'):

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = self._npx = np.r_[-np.inf, _x]
        self.y = self._npy = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self._npx = np.take(self.x, asort, 0)
            self.x = self._npx
            self._npy = np.take(self.y, asort, 0)
            self.y = self._npy
        self.n = self.x.shape[0]

    @cython.nonecheck(False)
    cpdef scalar_or_array interpolate(self, scalar_or_array xval):
        if scalar_or_array is not double:
            return self._npy[np.searchsorted(self.x, xval, self.side) - 1]
        else:
            return self.interpolate_scalar(xval)

    cpdef position_before(self, double xval):
        return binary_search(self.x, xval)

    @cython.nonecheck(False)
    cdef double interpolate_scalar(self, double xval) except -1:
        cdef:
            size_t index
        index = binary_search(self.x, xval) - 1
        return self.y[index]

    def __call__(self, xval):
        return self.interpolate(xval)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        x = np.linspace(np.array(self.x)[1:].min(), np.array(self.x).max())
        ax.plot(x, self(x))
        return ax

    def __eq__(self, other):
        return np.allclose(self.x, other.x
                           ) and np.allclose(self.y, other.y)

    def __ne__(self, other):
        return not (self == other)


cdef class ECDF(StepFunction):
    """
    Return the Empirical CDF of an array as a step function.

    Parameters
    ----------
    x : array-like
        Observations
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Empirical CDF as a step function.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import ECDF
    >>>
    >>> ecdf = ECDF([3, 3, 1, 4])
    >>>
    >>> ecdf([3, 55, 0.5, 1.5])
    array([ 0.75,  1.  ,  0.  ,  0.25])
    """
    def __init__(self, x, side='right'):
        x = np.array(x, copy=True)
        x.sort()
        nobs = len(x)
        y = np.linspace(1. / nobs, 1, nobs)
        super(ECDF, self).__init__(x, y, side=side, sorted=True)
