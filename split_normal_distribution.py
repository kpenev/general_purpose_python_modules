#!/usr/bin/env python3
"""Defines a Gaussian with different positive/negative variance."""

from matplotlib import pyplot
from scipy.stats import rv_continuous, norm
import numpy
from scipy.integrate import solve_ivp

class SplitNormal(rv_continuous):
    r"""
    Gaussian distribution with different positive/negative variance.

    The location (``loc``) keyword specifies the mean.
    The scale (``scale``) keyword specifies the average of the two standard
    devations.

    `split_normal` takes :math:`stddev_ratio` as shape parameter, specifying the
    ratio of the positive to negative standard devations.
    """

    @staticmethod
    def _norm_pdf_arg(x, stddev_ratio):
        """Return the argument to pass to standard normal matching PDF value."""

        norm_arg = x * (stddev_ratio + 1) / 2.0
        return numpy.where(x < 0, norm_arg, norm_arg / stddev_ratio)

    #Different signature follows from shape parameter
    #pylint: disable=arguments-differ
    def _pdf(self, x, stddev_ratio):

        return norm.pdf(self._norm_pdf_arg(x, stddev_ratio))

    def _logpdf(self, x, stddev_ratio):

        return (
            norm.logpdf(self._norm_pdf_arg(x, stddev_ratio))
        )

    def _cdf(self, x, stddev_ratio):

        norm_arg = self._norm_pdf_arg(x, stddev_ratio)

        return numpy.where(
            x < 0,
            2.0 / (stddev_ratio + 1) * norm.cdf(norm_arg),
            (
                1.0
                +
                stddev_ratio * (2.0 * norm.cdf(norm_arg) - 1.0)
            ) / (
                stddev_ratio + 1
            )
        )

    def _logcdf(self, x, stddev_ratio):

        norm_arg = self._norm_pdf_arg(x, stddev_ratio)
        return numpy.where(
            x < 0,
            (
                numpy.log(2.0 / (stddev_ratio + 1))
                +
                norm.logcdf(norm_arg)
            ),
            numpy.log(
                (
                    1.0
                    +
                    stddev_ratio * (2.0 * norm.cdf(norm_arg) - 1.0)
                )
                /
                (
                    stddev_ratio + 1
                )
            )
        )

    def _sf(self, x, stddev_ratio):

        return self._cdf(-x, 1.0 / stddev_ratio)

    def _logsf(self, x, stddev_ratio):

        return self._logcdf(-x, 1.0 / stddev_ratio)

    def _ppf(self, q, stddev_ratio):

        q_split = 1.0 / (1.0 + stddev_ratio)


        return 2.0 * numpy.where(
            q < q_split,
            norm.ppf(q / q_split / 2.0),
            (
                stddev_ratio
                *
                norm.ppf((q + 1 - 2.0 * q_split) / (2.0 * (1.0 - q_split)))
            )
        ) / (1.0 + stddev_ratio)

    def _isf(self, q, stddev_ratio):

        return -self._ppf(q, 1.0 / stddev_ratio)
    #pylint: disable=arguments-differ

split_normal = SplitNormal(name='split-normal')

if __name__ == '__main__':
    plot_distro = split_normal(stddev_ratio=2.0, scale=3.0)
    plot_x = numpy.linspace(-10.0, 10.0, 1000)

    pyplot.plot(plot_x,
                plot_distro.pdf(plot_x),
                '-r')
    pyplot.plot(plot_x,
                numpy.exp(plot_distro.logpdf(plot_x)),
                ':g',
                linewidth=3)
    pyplot.title('PDF')
    pyplot.show()

    pyplot.plot(plot_x,
                plot_distro.cdf(plot_x),
                '-r')
    pdf_integral = solve_ivp(lambda t, y: plot_distro.pdf(t),
                             (-20.0, 20.0),
                             y0=numpy.array([0.0]),
                             t_eval=plot_x).y.flatten()
    pyplot.plot(plot_x,
                pdf_integral,
                ':g',
                linewidth=3)
    pyplot.plot(plot_x[::10],
                numpy.exp(plot_distro.logcdf(plot_x[::10])),
                'xb',
                linewidth=3)
    pyplot.plot(plot_x[::10],
                1.0 - plot_distro.sf(plot_x[::10]),
                '+c',
                linewidth=3)

    pyplot.title('CDF')
    pyplot.show()

    plot_q = numpy.linspace(0, 1, 1000)
    pyplot.plot(plot_distro.cdf(plot_x),
                plot_x,
                '-r')
    pyplot.plot(plot_q,
                plot_distro.ppf(plot_q),
                ':g',
                linewidth=3)
    pyplot.plot(plot_q[::10],
                plot_distro.isf(1.0 - plot_q[::10]),
                'xb',
                linewidth=3)

    pyplot.plot('PPF')
    pyplot.show()
