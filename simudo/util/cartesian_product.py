import numpy

__all__ = ['cartesian_product']

# https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/45378609#45378609

# Generalized N-dimensional products
def cartesian_product(arrays):
    la = len(arrays)
    dtype = numpy.find_common_type([a.dtype for a in arrays], [])
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
