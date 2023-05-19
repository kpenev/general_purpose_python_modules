#!/usr/bin/env python3

"""Define eccentricity distro with e being U(0,1) instead of esinw, ecosw."""

from matplotlib import pyplot
from scipy import stats, special
from scipy.integrate import quad
import numpy

#Naming convention comes from scipy.stats
#Method arguments also come from scipy.stats
#pylint: disable=invalid-name
#pylint: disable=arguments-differ
class eccentricity_circular_kde_distro_gen(stats.rv_continuous):
    r"""RV with PDF given by KDE of (esinw,ecosw) samples for circular kernel.

    %(before_notes)s

    Notes
    -----
    The probability density function:

    .. math::

        f(x, {e_i}, w) = \sum_i
        \exp(- \frac{x^2 + e_i^2}{2w^2})
        I_0\left(\frac{x e_i}{w}\right)

    for :math:`x >= 0`, :math:`e_i > 0 \forall i` and :math:`w>0`. :math:`I_0`
    is the modified Bessel function of order zero (`scipy.special.i0`). In other
    words, the PDF of the distribution is the sum of Rice distribution PDF
    divided by x centered on each sample. The division by x is to change the
    prior from uniform in :math:`e\sin\omega` and :math:`e\cos\omega` to
    uniform in :math:`e`.

    The distribution takes an arrray of the samples and the kernel width (``w``)
    on `__init__`. This is required instead of specifying those as shape
    parameters so that the normalization constants for each sample cane be
    computed using numerical integration before evaluating the distribution.

    %(after_notes)s

    %(example)s

    """

    @staticmethod
    def _norm_integrand(x, s_to_n):
        """The function to integrate to find the norm for ecah sample."""

        return (
            numpy.exp(-(x-s_to_n)*(x-s_to_n)/2.0)
            *
            special.i0e(x*s_to_n)
        )

    def _calc_norm(self, center, uncertainty):
        """Calculate the normalization for single center/uncertainty combo."""

        s_to_n = center/uncertainty
        points = numpy.linspace(s_to_n - 10.0,
                                s_to_n + 10.0,
                                10)
        points = numpy.concatenate(([0], points[points>0]))
        result = (
            quad(
                self._norm_integrand,
                a=0,
                b=points[-1],
                points=points,
                args=(s_to_n,)
            )[0]
            +
            quad(
                self._norm_integrand,
                a=points[-1],
                b=numpy.inf,
                args=(s_to_n,)
            )[0]
        )
        if self._uniform_e_samples:
            return result * center
        return result


    def _pdf(self, x):
        # Like in scipy, this function uses (x**2 + b**2)/2 = ((x-b)**2)/2 + xb.
        # The factor of np.exp(-xb) is then included in the i0e function
        # in place of the modified Bessel function, i0, improving
        # numerical stability for large values of xb.

        eval_x = numpy.atleast_1d(x).flatten() / self._width

        result = (
            (
                numpy.exp(-(eval_x-self._s_to_n)*(eval_x-self._s_to_n)/2.0)
                *
                special.i0e(eval_x*self._s_to_n)
            )
        )

        if self._uniform_e_samples:
            result *= self._s_to_n * self._width

        result = result.sum(axis=0) * self._norm

        try:
            return result.reshape(x.shape)
        except AttributeError:
            return result[0]


    def _cdf_single(self, x):
        """Print what is being evaluated."""

        points = numpy.linspace(self._s_to_n.min() - 2,
                                self._s_to_n.max() + 2,
                                10) * self._width
        points = numpy.concatenate((
            [0],
            points[numpy.logical_and(points > 0, points < x)]
        ))
        return (
            quad(
                self._pdf,
                a=0,
                b=points[-1],
                points=points,
                limit=200
            )[0]
            +
            quad(
                self._pdf,
                a=points[-1],
                b=x,
                limit=200
            )[0]
        )


    def __init__(self,
                 e_samples,
                 kernel_width,
                 uniform_e_samples=False,
                 **kwargs):
        r"""
        Specify the samples, kernel width, and distribution config.

        Args:
            e_samples(array):    The samples of eccentricity values to build the
                distribution from.

            kernel_width(float):    The with of the kernel in
                :math:`e\sin\omega` and `e\cos\omega` space (assumed the same).

            uniform_e_samples(bool):    Were the samples generated assuming a
                uniform prior on e?

            **kwargs:    Any arguments other than ``a``, ``b``, and ``name`` to
                pass directly to `scipy.stats.rv_continuous.__init__`.

        Returns:
            None
        """

        super().__init__(a=0.0, name="EccentricityDistro", **kwargs)
        e_samples = numpy.atleast_1d(e_samples)[:, None]
        self._width = kernel_width
        self._s_to_n = e_samples / kernel_width
        self._uniform_e_samples = uniform_e_samples
        self._norm = 1.0 / numpy.vectorize(self._calc_norm)(
            e_samples,
            kernel_width
        ).sum() / kernel_width

class eccentricity_noncircular_kde_distro_gen(stats.rv_continuous):
    r"""RV with PDF given by KDE of (esinw,ecosw) samples for non-circular kern.

    %(before_notes)s

    Notes
    -----

    Each eccentricity component is allowed to have a different kernel. The
    probability density function is built by numerical integration over the
    periapsis angle.

    .. math::

        f(e) = \int_0^{2\pi} d\omega
            \sum_i k_c[e'\cos\omega'-(e\cos\omega)_i]
                   k_s[e'\sin\omega'-(e\sin\omega)_i] e_i^n

    Where :math:`k_s` and :math:`k_c` are the kernels to use for
    :math:`e\sin\omega` and :math:`e\cos\omega` respectively,
    :math:`(e\sin\omega)_i` and :math:`(e\cos\omega)_i` are the samples we are
    building the distribution from, and :math:`n` is zero if samples were built
    with independent uniform priors on :math:`e\sin\omega` and
    :math:`e\cos\omega` were used for sampling and 1 if uniform prior on
    :math:`e` was used. Note that the area element in the integration is taken
    as :math:`de\,d\omega` instead of :math:`e de d\omega` to impose uniform
    distribution on :math:`e` around a given sample.

    %(after_notes)s

    %(example)s
    """

    def _get_support(self):

        if numpy.logical_and(
                numpy.isfinite(self._kernel['sin'].support()).any(),
                numpy.isfinite(self._kernel['cos'].support()).any()
        ):
            ranges = dict()
            for component in ['sin', 'cos']:
                ranges[component] = dict(
                    min=(self._samples[component]
                         +
                         self._kernel[component].support()[0]),
                    max=(self._samples[component]
                         +
                         self._kernel[component].support()[1]),

                )
            corner_e2 = numpy.concatenate(
                (
                    numpy.square(ranges['sin'][sin_bound])
                    +
                    numpy.square(ranges['cos'][cos_bound])
                )
                for sin_bound in ['min', 'max']
                for cos_bound in ['min', 'max']
            )

            if numpy.logical_and(
                numpy.logical_and(ranges['sin']['min'] <= 0,
                                  ranges['sin']['max'] >= 0),
                numpy.logical_and(ranges['cos']['min'] <= 0,
                                  ranges['cos']['max'] >= 0)
            ).any():
                support_lower = 0.0
            else:
                support_lower = numpy.sqrt(corner_e2.min())
            return support_lower, min(1.0, numpy.sqrt(corner_e2.max()))

        return 0.0, 1.0


    def _get_w_integration_points(self):
        """Return the ``points`` argument to use for periapsis integrals."""

        means = {
            component: numpy.mean(self._samples[component])
            for component in ['sin', 'cos']
        }
        stddevs = {
            component: numpy.std(self._samples[component])
            for component in ['sin', 'cos']
        }
        return (
            numpy.arctan2(esinw, ecosw)
            for esinw in [means['sin'] - stddevs['sin'],
                          means['sin'],
                          means['sin'] + stddevs['sin']]
            for ecosw in [means['cos'] - stddevs['cos'],
                          means['cos'],
                          means['cos'] + stddevs['cos']]
        )


    def _kernel_arg(self, e_component, which):
        """Get difference between samples (inner index) and x (outer index)."""

        rhs, lhs = numpy.meshgrid(self._samples[which], e_component)
        return numpy.squeeze(lhs - rhs)


    def _pdf_integrand(self, eccentricity, periapsis):
        """The function to integrate over periapsis to get PDF(e)."""

        numpy.average(
            self._kernel['sin'].pdf(
                self._kernel_arg(eccentricity * numpy.sin(periapsis), 'sin')
            )
            *
            self._kernel['cos'].pdf(
                self._kernel_arg(eccentricity * numpy.cos(periapsis), 'cos')
            ),
            axis=-1,
            weights=self._weights
        )


    def _calculate_pdf(self, eccentricity):
        """Numerically integrate over periapsis to get PDF(e)."""

        if eccentricity < 0 or eccentricity > 1:
            return 0.0

        return quad(
            self._pdf_integrand,
            0,
            2.0 * numpy.pi,
            (eccentricity,),
            points=self._w_integration_points
        )


    #False positive, kernel accessed through locals()
    #pylint: disable=unused-argument
    def __init__(self,
                 *,
                 esinw_samples,
                 ecosw_samples,
                 sin_kernel,
                 cos_kernel,
                 uniform_e_samples=False,
                 **kwargs):

        r"""
        Specify the samples, the kernels, and distribution config.

        The kernels should be scipy stats like distributions, assumed to apply
        to all samples of a given eccentricity component and should already have
        the correct bandwidth set. If picklable object is required, specify the
        kernel as 3-tuple:
                    * str:    The name of the distribution in `scipy.stats`
                    * tuple:    Positional arguments for kernel ``__init__``
                    * dict:    Keyword arguments for kernel ``__init__``

        Args:
            esinw_samples(array):    The samples of :math:`e\sin\omega`  to
                build the distribution from.

            ecosw_samples(array):    The samples of :math:`e\cos\omega`  to
                build the distribution from.

            sin_kernel(rv_continuous):    The kernel to convolve
                :math:`e\sin\omega` samples with.

            cos_kernel(rv_continuous):    The kernel to convolve
                :math:`e\cos\omega` samples with.

            uniform_e_samples(bool):    Were the samples generated assuming a
                uniform prior on e?

            **kwargs:    Any arguments other than ``a``, ``b``, and ``name`` to
                pass directly to `scipy.stats.rv_continuous.__init__`.

        Returns:
            None
        """

        self._kernel_config = dict()
        self._kernel = dict()
        self._samples = dict()
        self._w_integration_points = []

        for component in ['sin', 'cos']:
            kernel = locals()[component + '_kernel']
            if isinstance(kernel, tuple):
                self._kernel_config[component] = kernel
                self._kernel[component] = getattr(stats, kernel[0])(*kernel[1],
                                                                    **kernel[2])
            else:
                self._kernel_config[component] = None
                self._kernel[component] = kernel

            self._samples[component] = numpy.ravel(
                locals()['e' + component + 'wsamples']
            )

        if uniform_e_samples:
            self._weights = numpy.sqrt(
                numpy.square(esinw_samples)
                +
                numpy.square(ecosw_samples)
            )
        else:
            self._weights = None

        self._w_integration_points = self._get_w_integration_points()

        support_lower, support_upper = self._get_support()
        super().__init__(a=support_lower,
                         b=support_upper,
                         name="EccentricityDistro",
                         **kwargs)
    #pylint: enable=unused-argument

#pylint: enable=invalid-name
#pylint: enable=arguments-differ


if __name__ == '__main__':
    val=0.05
    unc=0.001
    e_distro = eccentricity_circular_kde_distro_gen([0.0, val], unc)

    plot_x = numpy.linspace(0.0, val + 5.0 * unc, 1000)
    pyplot.plot(
        plot_x,
        e_distro.pdf(plot_x),
        label='e-distro 1 PDF'
    )

    pyplot.plot(
        plot_x,
        e_distro.cdf(plot_x),
        label='e-distro CDF'
    )
    pyplot.axhline(y=1)

#    pyplot.plot(
#        plot_x,
#        gauss.pdf(plot_x) / loc,
#        label='Gauss'
#    )
    pyplot.legend()
    pyplot.show()
