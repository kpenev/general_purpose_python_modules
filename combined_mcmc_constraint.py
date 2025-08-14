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
        assert solution.converged, repr(solution)
        return solution.root


    def _apply_ignore_mask(self, pdf, ignore_past_quantile):
        """Replace values of the PDF that must be ignored with 1.0."""

        def _get_masked_pdf_slice(pdf_slice):
            """Return the input slice of the PDF masked on the selected side."""

            cdf = self._get_cdf_slice(pdf_slice)
            critical_value = root_scalar(
                lambda x: cdf(x) - ignore_past_quantile[0],
                bracket=(self._grid[0], self._grid[-1]),
                fprime=cdf.derivative(),
            ).root
            index_below = numpy.nonzero(self._grid <= critical_value)[0].max()
            grid_below, grid_above = self._grid[index_below: index_below + 2]
            pdf_below, pdf_above = pdf_slice[index_below: index_below + 2]
            assert grid_below <= critical_value
            assert grid_above > critical_value
            fraction = (critical_value - grid_below) / (grid_above - grid_below)
            replace_value = pdf_below + fraction * (pdf_above - pdf_below)
            if ignore_past_quantile[1] > 0:
                pdf_slice[index_below + 1:] = replace_value
            else:
                assert ignore_past_quantile[1] < 0
                pdf_slice[:index_below + 1] = replace_value
            return pdf_slice

        return numpy.apply_along_axis(_get_masked_pdf_slice, 0, pdf)


    def __init__(self, grid, kernel_width):
        """Prepare the grid on which combined PDF will be calculated."""

        self._grid = grid
        self._pdf = None
        self._cdf = None
        self.eval_kernel = rdist(c=4, scale=kernel_width).pdf


    def add_samples(self,
                    samples,
                    prior_range=(-numpy.inf, numpy.inf),
                    include_samples=slice(None),
                    ignore_past_quantile=None):
        """
        Update the PDF with the given samples.

        Args:
            samples(array):    The samples to add. if samples is more than 1-D
                the last axis should iterate over sample index (other axes are
                treated as separate variables).

            prior_range(2-tuple):    Limit the range over which the distribution
                is evaluated to the given upper and lower limits. For grid
                values outside of this range, the value of the PDF at the
                specified limit is used.

            include_samples(slice):    Slice to apply to the first index of
                samples limiting the range of the additional variable over which
                the constraint is used.

            ignore_past_quantile(2-tuple):    Allows ignoring part of the
                distribution when the constraint is being updated. The first
                entry specifies a quartile beyond which the distribution should
                be ignored and the second entry specifies a direction. For
                example, ``ignore_past_quantile = (0.5, 1)`` results in using
                a distribution with PDF equal to the new distribution up to the
                median and a constant value equal to the new PDF at the median
                past that.
        """

        eval_grid = numpy.copy(self._grid)
        eval_grid[self._grid < prior_range[0]] = prior_range[0]
        eval_grid[self._grid > prior_range[1]] = prior_range[1]

        new_pdf = numpy.zeros(shape=(eval_grid.shape + samples.shape)[:-1],
                              dtype=float)
        split = max(1, samples.shape[-1] // (int(samples.size / 1e5) + 1))
        for start in range(0, samples.shape[-1], split):
            sub_samples = samples[
                ...,
                start : min(start + split, samples.shape[-1])
            ]
            new_pdf += numpy.sum(
                self.eval_kernel(
                    (
                        sub_samples.flatten() - eval_grid[:, None]
                    ).reshape(
                        eval_grid.shape + sub_samples.shape
                    )
                ),
                -1
            )
        new_pdf /= samples.shape[-1]
        if ignore_past_quantile:
            self._apply_ignore_mask(new_pdf, ignore_past_quantile)

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
