"""Collection of functions for converting between magnitude systems."""

import numpy

_usno_to_sdss_matrix = numpy.array([
    [1, 0,      0,      0,              0], #u
    [0, 1.06,   -0.06,  0,              0], #g
    [0, 0,      1.035,  -0.035,         0], #r
    [0, 0,      0.041,  1.0 - 0.041,    0], #i
    [0, 0,      0,      -0.03,          1.03]  #z
])

_sdss_to_usno_matrix = numpy.asarray(numpy.asmatrix(_usno_to_sdss_matrix).I)

_sdss_to_usno_offset = numpy.array([
    [0],
    [0.06 * 0.53],
    [0.035 * 0.21],
    [0.041 * 0.21],
    [-0.03 * 0.09]
])

def sdss_to_usno(sdss_ugriz):
    """
    Return the estimated USNO 1m estimated magnitudes from SDSS 2.5m ones.

    Args:
        sdss_ugriz(5xN numpy.array):    The values of the u, g, r, and z
            magnitudes in the SDSS 2.5m system. Each magnitude is a column.

    Returns:
        5 x N numpy array:
            The values of the u', g', r', i', and z' magnitudes in the USNO 1m
            system in the same format as the input.
    """

    assert sdss_ugriz.shape[0] == 5
    return numpy.tensordot(
        _sdss_to_usno_matrix,
        (sdss_ugriz.T + _sdss_to_usno_offset.T).T,
        1
    )

def usno_to_sdss(usno_ugriz):
    """
    Return the estimated SDSS 2.5m estimated magnitudes from UNSO 1m ones.

    Args:
        sdss_ugriz(5xN numpy.array):    The values of the u', g', r', i', and z'
            magnitudes in the UNSO 1m system. Each magnitude is a column.

    Returns:
        5 x N numpy array:
            The values of the u, g, r, i, and z magnitudes in the SDSS 2.5m
            system in the same format as the input.
    """

    return numpy.asarray(_usno_to_sdss_matrix * usno_ugriz
                         -
                         _sdss_to_usno_offset)

def deredden_usno(usno_ugriz, reddening_bv):
    """
    Apply reddening correction to USNO u', g', r', i', and z' magnitudes.

    Args:
        usno_ugriz:     Iterable of the u', g', r', i', and z' magnitudes in the
            UNSO 1m system in that order.

        reddening_bv(float):    The value of E(B-V) reddening to base the
            correction on.

    Returns:
        5xN numpy array:
            The values of the reddening corrected magnitudes, with each
            magnitude as its own column.
    """

    print('USNO ugriz shape: ' + repr(usno_ugriz.shape))
    result = numpy.empty(shape=(5, len(usno_ugriz[0])), dtype=float)
    print('Result shape: ' + repr(result.shape))
    reddening_coef = [4.879, 3.708, 2.722, 2.089, 1.519]
    for mag_index in range(5):
        result[mag_index] = usno_ugriz[mag_index] - (reddening_coef[mag_index]
                                                     *
                                                     reddening_bv)

    return result
