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

        min_x, max_x = self.support()
        if min_x < x < max_x:
            return self._norm * x**(self._powerlaw - 1)
        return 0.0

    def _lodpdf(self, x):
        """More accurate expression for log of :meth:`pdf`."""

        return numpy.log(self._norm) + (self._powerlaw - 1) * numpy.log(x)

    def _cdf(self, x):
        """The cumulative distribution function."""

        min_x, max_x = self.support()

        if x < min_x:
            return 0.0

        if x > max_x:
            return 1.0

        return (
            self._norm
            *
            (x**self._powerlaw - min_x**self._powerlaw)
            /
            self._powerlaw
        )


    def _ppf(self, quantile):
        """The inverse of :meth:`_cdf`."""

        assert 0.0 <= quantile <= 1.0

        min_x, max_x = self.support()

        if quantile == 0:
            return min_x

        if quantile == 1:
            return max_x

        return (
            quantile * self._powerlaw / self._norm + min_x**self._powerlaw
        )**(1.0 / self._powerlaw)

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
        numpy.vectorize(distro.pdf)(plot_x),
        label='PDF'
    )
    pyplot.semilogx(
        plot_x,
        numpy.exp(
            numpy.vectorize(distro.logpdf)(plot_x)
        ),
        '--',
        label='exp(ln(PDF))'
    )

    pyplot.title('PDF')
    pyplot.legend()
    pyplot.show()

    pyplot.semilogx(
        plot_x,
        numpy.vectorize(distro.cdf)(plot_x),
        label='CDF'
    )
    plot_quantiles = numpy.linspace(0, 1, 1000)
    pyplot.semilogx(
        numpy.vectorize(distro.ppf)(plot_quantiles),
        plot_quantiles,
        '--',
        label='PPF$^{-1}$'
    )
    pyplot.title('CDF')
    pyplot.legend()
    pyplot.show()

    pyplot.semilogx(
        plot_x,
        numpy.vectorize(distro.logpdf)(plot_x),
        label='log-pdf'
    )
    pyplot.semilogx(
        plot_x,
        numpy.log(numpy.vectorize(distro.pdf)(plot_x)),
        '--',
        label='ln(PDF)'
    )
    pyplot.title('ln(PDF)')
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    create_check_plots()
