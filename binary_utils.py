"""A collection of functions useful when modeling binaries."""

import logging

import numpy

from astropy import constants

_logger = logging.getLogger(__name__)

def calculate_secondary_mass(primary_mass,
                             orbital_period,
                             rv_semi_amplitude,
                             eccentricity=0,
                             inclination=numpy.pi/2):
    """
    Calculate the mass of a secondary object for a given RV observations.

    All arguments must have the appropriate units set using the astropy module.

    Args:
        primary_mass:    The mass of the primary star.

        orbital_period:    The orbital period for the binary.

        rv_semi_amplitude:    The semi-amplitude of the radial velocity best-fit
            curve, complete with astropy units.

        eccentricity:    The eccentricity of the orbit.

        inclination:    The inclination to assume for the orbit in radians.

    Returns:
        The mass of the secondary star, complete with astropy units.
    """

    mass_ratio_function = (
        (
            orbital_period * rv_semi_amplitude**3
            /
            #False positive
            #pylint: disable=no-member
            (2.0 * numpy.pi * constants.G)
            #pylint: enable=no-member
            *
            (1.0 - eccentricity**2)**1.5
        )
        /
        primary_mass
    ).to_value('')

    solutions = numpy.roots([numpy.sin(inclination)**3,
                             -mass_ratio_function,
                             -2.0 * mass_ratio_function,
                             -mass_ratio_function])
    mass_ratio = None
    for candidate_mass_ratio in solutions:
        if (
                candidate_mass_ratio.imag == 0
                and
                candidate_mass_ratio.real > 0
                and
                candidate_mass_ratio.real <= 1
        ):
            assert mass_ratio is None
            mass_ratio = candidate_mass_ratio.real
    if mass_ratio is None:
        _logger.critical(
            'No valid binary mass ratio exists found for binary with M1 = %s'
            'Porb = %s, Krv = %s, e = %s, i = %s. Solutions: %s',
            repr(primary_mass),
            repr(orbital_period),
            repr(rv_semi_amplitude),
            repr(eccentricity),
            repr(inclination),
            repr(solutions)
        )
    assert mass_ratio is not None
    return mass_ratio * primary_mass

def rv_semi_amplitude_scale(primary_mass,
                            secondary_mass,
                            orbital_period):
    """
    Calculate the radial velocity semi-amplitude of a circular edge-on orbit.

    Args:
        primary_mass:    The mass of the primary star in the binary. Must
            include units.

        secondary_mass:    The mass of the secondary star in the binary. Must
            include units.

        orbital_period:    The orbital period of the binary. Must include units.

    Returns:
        The radial velocity semi-amplitude of a circular edge-on orbit with the
        given parameters.
    """

    return (
        secondary_mass
        *
        (
            2.0 * numpy.pi * constants.G
            /
            (
                orbital_period
                *
                (primary_mass + secondary_mass)**2
            )
        )**(1.0/3.0)
    )


