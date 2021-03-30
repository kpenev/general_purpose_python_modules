#!/usr/bin/env python3

"""Various numerical solutions for Rice distribution parameters."""

import logging

from matplotlib import pyplot
from scipy.optimize import root_scalar
from scipy.stats import norm, rice
import numpy

def rice_from_error_bars(value, abs_plus_error, abs_minus_error):
    """Return a Rice distribution reproducing the given mode & standard dev."""

    def get_distro(rice_b, scale):
        """Get the distribution at given parameters with normal backup."""

        return (
            rice(b=rice_b, scale=scale)
            if rice_b < 50.0 else
            norm(loc=scale * numpy.sqrt(rice_b**2 + 1.0), scale=scale)
        )

    def lower_diff(rice_b, scale):
        """Return the error of quantile of value - abs_minus_error."""

        return (
            get_distro(rice_b, scale).cdf(value - abs_minus_error)
            -
            norm.cdf(-1.0)
        )

    def upper_diff(scale, rice_b):
        """Return the error of quantile of value + abs_plus_error."""

        return (
            get_distro(rice_b, scale).cdf(value + abs_plus_error)
            -
            norm.cdf(1.0)
        )

    def find_b(scale):
        """Return the b parameter that matches the lower bound given scale."""

        b_min = 0.0
        b_min_lower_diff = lower_diff(b_min, scale)
        if b_min_lower_diff < 1e-8:
            return b_min
        assert b_min_lower_diff >= 0

        b_max = (value + abs_plus_error) / scale
        while lower_diff(b_max, scale) > 0:
            b_max *= 10.0
        solution = root_scalar(f=lower_diff,
                               args=(scale,),
                               bracket=(b_min, b_max))
        assert solution.converged

        return solution.root

    def scale_equation(scale):
        """The equation to solve in order to find the scale."""

        return upper_diff(scale, find_b(scale))

    def find_scale():
        """Return the scale that matches the upper bound at b matching lower."""

        scale_max = (value - abs_minus_error) / rice.ppf(norm.cdf(-1.0), b=0)

        if scale_equation(scale_max) > 0:
            return None

        scale_min = 0.1 * min(abs_minus_error, abs_plus_error)

        while scale_equation(scale_min) < 0:
            scale_max = scale_min
            scale_min /= 10.0

        solution = root_scalar(f=scale_equation,
                               bracket=(scale_min, scale_max))
        assert solution.converged
        return solution.root

    scale = find_scale()
    if scale is None:
        logging.getLogger(__name__).warning(
            'Lower and upper limits of %s and %s cannot be matched to 16-th '
            'and 84-th percentiles of a Rice distribution. Matching upper '
            'limit and assuming b=0',
            repr(value - abs_minus_error),
            repr(value + abs_plus_error)
        )
        return rice(
            b=0.0,
            scale=((value + abs_plus_error) / rice.ppf(norm.cdf(1.0), b=0))
        )
    rice_b = find_b(scale)
    return rice(b=rice_b, scale=scale)

def test():
    """Avoid polluting global namespace."""

    distro = rice_from_error_bars(0.11, 0.11, 0.07)
    plot_x = numpy.linspace(0, 0.5, 1000)
    pyplot.plot(plot_x, distro.pdf(plot_x))
    for mark in [0.04, 0.11, 0.22]:
        pyplot.axvline(x=mark)
    for mark_cdf in [norm.cdf(-1.0), 0.5, norm.cdf(1.0)]:
        pyplot.axvline(x=distro.ppf(mark_cdf), linestyle=':', color='red')
    pyplot.show()

if __name__ == '__main__':
    test()
