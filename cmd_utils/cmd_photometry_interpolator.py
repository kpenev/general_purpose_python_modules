#!/usr/bin/env python3

"""
Define a class that works with interpolated photometry from CMD isochrones.
"""

import re
import os.path

from matplotlib import pyplot
import numpy

#pylint: disable=wrong-import-position
from general_purpose_python_modules.planetary_system_io import \
    read_cds_pipe_table
#pylint: disable=import-error
from cmd_isochrone_interpolator import CMDInterpolator
#pylint: enable=import-error
#pylint: enable=wrong-import-position

class CMDPhotometryInterpolator(CMDInterpolator):
    """
    Interpolate photometry from CMD isochrones for a single cluster.

    Attributes:
        age(float):    The age of the isochrones.

        feh(float):    The metallicity ([Fe/H]) of the isochrones.

        min_mass(astropy Quantity):    The minimum stellar mass in the
            isochrones.

        max_mass(astropy Quantity):    The maximum stellar mass in the
            isochrones.

        extinction(float):    The assumed extinction (Av) applied to the
            isochrone by the CMD interface.
    """

    def _parse_header(self):
        """Extract the useful information from the isochrone file header."""

        extinction_parse_rex = re.compile(
            'photometry includes extinction of Av=(?P<Av>[^ ,]*)[, ]'
        )
        for line in self.header:
            if ':' in line:
                keyword, value = line[1:].strip().split(':', 1)
                if keyword.strip() == 'Attention':
                    parsed_extinction = extinction_parse_rex.match(
                        value.strip()
                    )
                    assert parsed_extinction
                    self.extinction = float(parsed_extinction['Av'])

        self.available_filters = [col_name[:-3]
                                  for col_name in self.data[0].dtype.names
                                  if col_name.endswith('mag')]


    def __init__(self, isochrone_fname, distance_modulus):
        """Interpolate within the given isochrone grid."""

        super().__init__(isochrone_fname)
        self._parse_header()
        assert len(self.data) == 1
        #False positive
        #pylint: disable=no-member
        self.min_mass = self.data[0]['Mini'][0]
        self.max_mass = self.data[0]['Mini'][-1]
        self.feh = self.data[0]['MH'][0]
        self.distance_modulus = distance_modulus
        #pylint: enable=no-member
        assert numpy.unique(self.data[0]['logAge']).size == 1
        #False positive
        #pylint: disable=no-member
        self.age = 10.0**(self.data[0]['logAge'][0] - 9.0)
        #pylint: enable=no-member

        self.grid_mag = numpy.stack(
            [
                self.data[0][filchar + 'mag']
                for filchar in self.available_filters
            ]
        )

    def __call__(self, interp_mass):
        """Return the available photometry for the given mass(es)."""

        return numpy.stack(
            [
                (
                    self.get_interpolated(mag_letter + 'mag', interp_mass, None)
                    +
                    self.distance_modulus
                )
                for mag_letter in self.available_filters
            ]
        )

    def get_binary_magnitudes(self, primary_mass, secondary_mass):
        """Estimate SDSS u, g, r, i, z for a binary, given mass(es)."""

        primary_mags = self(primary_mass)
        secondary_mags = self(secondary_mass)
        #False positive
        #pylint: disable=invalid-unary-operand-type
        return -2.5 * numpy.log10(10.0**(-primary_mags/2.5)
                                  +
                                  10.0**(-secondary_mags/2.5))
        #pylint: enable=invalid-unary-operand-type
