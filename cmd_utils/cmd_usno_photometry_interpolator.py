#!/usr/bin/env python3

"""
Define a class that works with interpolated SDSS photometry from CMD isochrones.
"""

import os.path

from matplotlib import pyplot
import scipy

from general_purpose_python_modules.planetary_system_io import \
    read_cds_pipe_table
from general_purpose_python_modules.magnitude_transformations import \
    sdss_to_usno

#False positive
#pylint: disable=import-error
from cmd_photometry_interpolator import CMDPhotometryInterpolator
#pylint: enable=import-error

class CMDUSNOPhotometryInterpolator(CMDPhotometryInterpolator):
    """Interpolate SDSS photometry from CMD isochrones for a single cluster."""

    def __init__(self, isochrone_fname, distance_modulus):
        """Interpolate within the given isochrone grid."""

        super().__init__(isochrone_fname, distance_modulus)

        #False positive: parent's __init__ defines it
        #pylint: disable=access-member-before-definition
        for filchar in 'ugriz':
            assert filchar in self.available_filters

        self.grid_mag = sdss_to_usno(
            self.grid_mag[
                [self.available_filters.index(filchar)
                 for filchar in 'ugriz']
            ]
        )
        #pylint: enable=access-member-before-definition

        self.available_filters = list('ugriz')

    def __call__(self, interp_mass):
        """Estimate UNSO u', g', r', i', z' photometry for given mass(es)."""

        return sdss_to_usno(
            super().__call__(
                scipy.array(interp_mass, copy=False, ndmin=1)
            )
        )
