"""
Utilities for analyzing the autocorrelation of emcee results.

Many of the functions come directly from emcee documentation.
"""

import numpy
import numpy.fft
import celerite
from celerite import terms
from scipy.optimize import minimize

def autocorr_func_1d(chain, norm=True):
    """Calculate the autocorrelation of a 1D array using FFT."""

    #Perfectly good name for the argument
    #pylint: disable=invalid-name
    def next_pow_two(n):
        i = 1
        while i < n:
            i = i << 1
        return i
    #pylint: enable=invalid-name

    chain = numpy.atleast_1d(chain)
    if len(chain.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    max_freq = next_pow_two(len(chain))

    # Compute the FFT and then (from that) the auto-correlation function
    chain_fft = numpy.fft.fft(chain - numpy.mean(chain), n=2 * max_freq)
    acf = numpy.fft.ifft(
        chain_fft * numpy.conjugate(chain_fft)
    )[: len(chain)].real
    acf /= 4 * max_freq

    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

def auto_window(cumulative_autocorr, scale):
    """Automated windowing procedure following Sokal (1989)."""

    check = numpy.arange(len(cumulative_autocorr)) < scale * cumulative_autocorr
    if numpy.any(check):
        return numpy.argmin(check)
    return len(cumulative_autocorr) - 1


def autocorr_gw2010(all_chains, scale=5.0, time=True):
    """Combine walkers to estimate autocorr. per Goodman & Weare (2010)."""
    autocorr_mean = autocorr_func_1d(numpy.mean(all_chains, axis=0))
    if not time:
        return autocorr_mean

    taus = 2.0 * numpy.cumsum(autocorr_mean) - 1.0
    window = auto_window(taus, scale)
    return taus[window]


def average_autocorr(all_chains, scale=5.0, time=True):
    """Average the autocerrelation of individual walkers."""

    mean_autocorr = numpy.zeros(all_chains.shape[1])
    nchains = 0
    for chain in all_chains:
        if numpy.unique(chain).size > 1:
            mean_autocorr += autocorr_func_1d(chain)
            nchains += 1
        else:
            print('Chain of %d constant values = %s found'
                  %
                  (chain.size, repr(chain[0])))
    mean_autocorr /= nchains
    if not time:
        return mean_autocorr

    taus = 2.0 * numpy.cumsum(mean_autocorr) - 1.0
    window = auto_window(taus, scale)
    return taus[window]

def max_likelihood_autocorr(all_chains, thin, scale=5.0):
    """Maximum likelihood estimate of the auto-correlation."""

    # Compute the initial estimate of tau using the standard method
    init = average_autocorr(all_chains, scale=scale)
    thinned_chains = all_chains[:, ::thin]
    nsamples = thinned_chains.shape[1]

    # Build the GP model
    tau = max(1.0, init / thin)
    kernel = terms.RealTerm(
        numpy.log(0.9 * numpy.var(thinned_chains)),
        -numpy.log(tau),
        bounds=[(-5.0, 5.0), (-numpy.log(nsamples), 0.0)],
    )
    kernel += terms.RealTerm(
        numpy.log(0.1 * numpy.var(thinned_chains)),
        -numpy.log(0.5 * tau),
        bounds=[(-5.0, 5.0), (-numpy.log(nsamples), 0.0)],
    )
    gaus_proc = celerite.GP(kernel, mean=numpy.mean(thinned_chains))
    gaus_proc.compute(numpy.arange(thinned_chains.shape[1]))

    def nll(params):
        """Define the objective."""

        # Update the GP model
        gaus_proc.set_parameter_vector(params)

        # Loop over the chains and compute likelihoods
        #pylint: disable=invalid-name
        v, g = zip(*(gaus_proc.grad_log_likelihood(chain, quiet=True)
                     for chain in thinned_chains))

        # Combine the datasets
        return -numpy.sum(v), -numpy.sum(g, axis=0)
        #pylint: enable=invalid-name

    # Optimize the model
    initial_params = gaus_proc.get_parameter_vector()
    bounds = gaus_proc.get_parameter_bounds()
    soln = minimize(nll, initial_params, jac=True, bounds=bounds)
    gaus_proc.set_parameter_vector(soln.x)

    # Compute the maximum likelihood tau
    #pylint: disable=invalid-name
    a, c = kernel.coefficients[:2]
    tau = thin * 2 * numpy.sum(a / c) / numpy.sum(a)
    #pylint: enable=invalid-name

    return tau
