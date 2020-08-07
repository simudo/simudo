import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

__all__ = ["IV_params"]

# From Matt Wilkins


def ordertest(A):
    """Test if list ``A`` is strictly ascending."""
    return all(A[k] <= A[k + 1] for k in range(len(A) - 1))


class IV_params:
    """
Takes arrays of current and voltage, and interpolates Voc, Isc, and Pmax.

``voc`` and ``isc`` are linearly interpolated, ``pmax`` is found
with a spline fit.

If v and i are in V and A respectively, then pmax is in W.

If the the power quadrant has negative I (as for Simudo), use optional
argument ``powersign=-1.0``.

Example
-------

::

    p = iv_params(current_array, voltage_array)
    print(p.voc, p.isc, p.pmax, p.ff, p.mppv, p.mppi)
    """

    def __init__(self, i, v, powersign=1.0):
        self.i = np.array(i)
        self.v = np.array(v)
        assert len(i) == len(v), "Unequal length arrays"
        if ordertest(v) == False:
            self.i = self.i[::-1]
            self.v = self.v[::-1]
        assert ordertest(
            self.v
        ), "Voltage array out of order, must be monotonic"
        assert (
            max(i) >= 0.0 and min(i) <= 0.0 and max(i) > min(i)
        ), "Current array does not cross zero"
        assert (
            max(v) >= 0.0 and min(v) <= 0.0 and max(v) > min(v)
        ), "Voltage array does not cross zero"

        interp_i = interp1d(self.v, self.i, kind="linear")
        self.isc = float(interp_i(0.0))

        interp_v = interp1d(self.i, self.v, kind="linear", bounds_error=False)
        self.voc = float(interp_v(0.0))

        spline_P = UnivariateSpline(
            self.v, powersign * self.i * self.v, k=4, s=0
        )
        dPdV = spline_P.derivative()
        roots = dPdV.roots()
        Pmax = -1e10
        for k in range(len(roots)):
            P = spline_P(roots[k])
            if P > Pmax:
                Pmax = P
                self.mppv = roots[k]
        self.mppi = interp_i(self.mppv)

        self.pmax = spline_P(self.mppv)  # self.mppi*self.mppv
        self.ff = self.pmax / (self.voc * self.isc)
