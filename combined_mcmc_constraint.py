"""Define class for combining unrelated MCMC constraints of a parameter."""

import numpy
from scipy.stats import rdist
from scipy.integrate import cumtrapz as cumulative_trapezoid
from scipy.interpolate import BPoly
from scipy.optimize import root_scalar

class CombinedMCMCConstraint:
    """
    Calculate the product of PDFs of independent MCMC constraints on a grid.

    """


    def _get_cdf_slice(self, pdf_slice):
        """Return the CDF at single slice of PDF along grid."""

        interp_y = numpy.empty((self._grid.size, 2))
        interp_y[:, 0] = cumulative_trapezoid(pdf_slice,
                                              self._grid,
                                              initial=0)
        interp_y[:, 1] = pdf_slice
        interp_y /= interp_y[-1, 0]
        return BPoly.from_derivatives(self._grid, interp_y)


    def _find_ppf(self, cdf_func, cdf_value):
        """Equation to solve when finding PPF."""

        solution = root_scalar(lambda x: cdf_func(x) - cdf_value,
                               bracket=(self._grid[0], self._grid[-1]))
        assert solution.converged
        return solution.root


    def __init__(self, grid, kernel_width):
        """Prepare the grid on which combined PDF will be calculated."""

        self._grid = grid
        self._pdf = None
        self._cdf = None
        self.eval_kernel = rdist(c=4, scale=kernel_width).pdf


    def add_samples(self,
                    samples,
                    prior_range=(-numpy.inf, numpy.inf),
                    include_samples=slice(None)):
        """
        Update the PDF with the given samples.

        If samples is more than 1-D the last axis should iterate over sample
        index (other axes are treated as separate variables).
        """

        eval_grid = numpy.copy(self._grid)
        eval_grid[self._grid < prior_range[0]] = prior_range[0]
        eval_grid[self._grid > prior_range[1]] = prior_range[1]
        new_pdf = numpy.mean(
            self.eval_kernel(
                (
                    samples.flatten() - eval_grid[:, None]
                ).reshape(
                    eval_grid.shape + samples.shape
                )
            ),
            -1
        )

        if self._pdf is None:
            self._pdf = numpy.ones(new_pdf.shape)
        self._pdf[:, include_samples] *= new_pdf[:, include_samples]
        self._cdf = None

        assert (self._pdf >= 0).all()


    def ppf(self, cdf_value):
        """Return the percent point function (inverse of CDF)."""

        if self._cdf is None:
            self._cdf = numpy.apply_along_axis(self._get_cdf_slice,
                                               0,
                                               self._pdf)
        assert 0 <= cdf_value <= 1
        return numpy.vectorize(self._find_ppf)(self._cdf, cdf_value)


    def __call__(self):
        """Return the PDF on the given grid."""

        return self._pdf / self._pdf.sum(0)
