# cython: embedsignature=True
cimport cython
import numpy as np
cimport numpy as np

np.import_array()

from matplotlib import pyplot as plt


@cython.boundscheck(False)
cdef size_t position_before(double[:] x, double val) nogil:
    cdef:
        size_t i, n
    n = x.shape[0]
    for i in range(n):
        if x[i] > val:
            return i
    return n


cdef class StepFunction(object):
    """
    A basic step function.

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

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import StepFunction
    >>>
    >>> x = np.arange(20)
    >>> y = np.arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print(f(3.2))
    3.0
    >>> print(f([[3.2,4.5],[24,-3.1]]))
    [[  3.   4.]
     [ 19.   0.]]
    >>> f2 = StepFunction(x, y, side='right')
    >>>
    >>> print(f(3.0))
    2.0
    >>> print(f2(3.0))
    3.0
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
        return position_before(self.x, xval)

    cdef double interpolate_scalar(self, double xval) except -1:
        cdef:
            size_t index
        index = position_before(self.x, xval) - 1
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
