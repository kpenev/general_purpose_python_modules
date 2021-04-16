"""Define scipy type distribution that has a powerlaw PDF in a finite range."""

from matplotlib import pyplot
from scipy import stats
import numpy

#Public methods inherited from rv_continuous
#pylint: disable=too-few-public-methods
class BoundedPowerlawDistribution(stats.rv_continuous):
    """Distribution where pdf is a powerlaw in given range, 0 outside."""

    #Shape parameter specified through __init__ (too ugly othrewise).
    #pylint: disable=arguments-differ
    def _pdf(self, x):
        """The fully normalized PDF."""

        x = numpy.atleast_1d(x)
        min_x, max_x = self.support()
        result = numpy.zeros(x.shape, dtype=float)
        result[numpy.logical_and(min_x <= x, x <= max_x)] = (
            self._norm
            *
            x**(self._powerlaw - 1)
        )
        return result

    def _lodpdf(self, x):
        """More accurate expression for log of :meth:`pdf`."""

        x = numpy.atleast_1d(x)
        min_x, max_x = self.support()
        result = numpy.zeros(x.shape, dtype=float)
        result[numpy.logical_and(min_x <= x, x <= max_x)] = (
            numpy.log(self._norm)
            +
            (self._powerlaw - 1) * numpy.log(x)
        )
        return result

    def _cdf(self, x):
        """The cumulative distribution function."""

        x = numpy.atleast_1d(x)
        min_x, max_x = self.support()
        result = numpy.empty(x.shape, dtype=float)

        result[x < min_x] = 0.0

        result[x > max_x] = 1.0

        result[numpy.logical_and(min_x <= x, x <= max_x)] = (
            self._norm
            *
            (x**self._powerlaw - min_x**self._powerlaw)
            /
            self._powerlaw
        )
        return result

    def _ppf(self, quantile):
        """The inverse of :meth:`_cdf`."""

        quantile = numpy.atleast_1d(quantile)
        assert (quantile >= 0.0).all()
        assert (quantile <= 1.0).all()

        min_x, max_x = self.support()
        result = numpy.empty(quantile.shape, dtype=float)

        result[quantile == 0] = min_x

        result[quantile == 1] = max_x

        result[numpy.logical_and(quantile > 0, quantile < 1)] = (
            quantile * self._powerlaw / self._norm + min_x**self._powerlaw
        )**(1.0 / self._powerlaw)

        return result

    def __init__(self, powerlaw, *args, **kwargs):
        """Normalize distribution per specified support."""

        super().__init__(*args, **kwargs)

        self._powerlaw = powerlaw
        min_x, max_x = self.support()
        self._norm = (powerlaw / (max_x**powerlaw - min_x**powerlaw))

    #pylint: enable=arguments-differ
#pylint: enable=too-few-public-methods

def create_check_plots():
    """Create plots comparing the distribution methods."""

    distro = BoundedPowerlawDistribution(a=0.5, b=20.0, powerlaw=1.46)

    plot_x = 10.0**numpy.linspace(-0.5, 1.5, 1000)
    pyplot.semilogx(
        plot_x,
        distro.pdf(plot_x),
        label='PDF'
    )
    pyplot.semilogx(
        plot_x,
        numpy.exp(distro.logpdf(plot_x)),
        '--',
        label='exp(ln(PDF))'
    )

    pyplot.title('PDF')
    pyplot.legend()
    pyplot.show()

    pyplot.semilogx(
        plot_x,
        distro.cdf(plot_x),
        label='CDF'
    )
    plot_quantiles = numpy.linspace(0, 1, 1000)
    pyplot.semilogx(
        distro.ppf(plot_quantiles),
        plot_quantiles,
        '--',
        label='PPF$^{-1}$'
    )
    pyplot.title('CDF')
    pyplot.legend()
    pyplot.show()

    pyplot.semilogx(
        plot_x,
        distro.logpdf(plot_x),
        label='log-pdf'
    )
    pyplot.semilogx(
        plot_x,
        numpy.log(distro.pdf(plot_x)),
        '--',
        label='ln(PDF)'
    )
    pyplot.title('ln(PDF)')
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    create_check_plots()
