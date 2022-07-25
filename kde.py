"""Class for working with KDE approximated distributions."""

import numpy
from scipy.stats import rv_continuous, rdist

class KDEDistribution(rv_continuous):
    """Distribution-like object based on KDE estimate given a set of samples."""

    def _kernel_arg(self, x):
        """Get difference between samples (inner index) and x (outer index)."""

        rhs, lhs = numpy.meshgrid(self._samples, x)
        return lhs - rhs

    def __init__(self, samples, kernel, weights=None):
        """
        Set the "distribution" based on the given samples and kernel.

        Args:
            samples(1D array):    The samples to base the distribution on.

            kernel(rv_continuous):    A scipy stats like distribution to use as
                the kernel. This is assumed to apply to all samples and should
                already have the correct bandwidth set.

        Returns:
            None
        """

        self._kernel = kernel
        self._samples = numpy.ravel(samples)
        self._weights = weights
        kernel_min, kernel_max = kernel.support()
        self._support = ()
        super().__init__(a=(self._samples.min() + kernel_min),
                         b=(self._samples.max() + kernel_max))


    def set_weights(self, weights):
        """Define weights for the samples."""

        assert(self._samples.shape == weights.shape)
        self._weights = weights


    def _pdf(self, x):
        """Evaluate the KDE estimated probability density."""

        return numpy.average(self._kernel.pdf(self._kernel_arg(x)),
                             axis=-1,
                             weights=self._weights)


    def _cdf(self, x):
        """
        Evaluate the CDF.

        Args:
            x(1D array):    The locations at which to evaluate the CDF.

        Returns:
            array:
                The values of the CDF.
        """

        return numpy.average(self._kernel.cdf(self._kernel_arg(x)),
                             axis=-1,
                             weights=self._weights)


    def eval_sample_pdf(self, x):
        """Evaluate the kernel around each sample at x."""

        return self._kernel.pdf(self._kernel_arg(x))

    def eval_sample_cdf(self, x):
        """Evaluate the kernel around each sample at x."""

        return self._kernel.cdf(self._kernel_arg(x))


if __name__ == '__main__':
    from matplotlib import pyplot
    epi_kde = KDEDistribution([-1.0, 1.0], rdist(c=4))
    plot_x = numpy.linspace(-5.0, 5.0, 1000)
    pyplot.plot(plot_x, epi_kde.ppf(plot_x))
    pyplot.show()
