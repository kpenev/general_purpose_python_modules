#!/usr/bin/env python3
"""Test the convergence of quantile estimates from MCMC chains."""

from matplotlib import pyplot
import numpy

from discrete_markov import DiscreteMarkov

def get_approximate_markov(chain, num_states=None):
    """
    Return order 1 Markov process fit to thinned chain and the thinning used.

    Args:
        chain(array):    A chain of discrete values that will be thinned until
            it is well approximated by an order 1 Markov process.

        num_states(int):    The number of states the process samples over  (in
            case not all are represented in `chain`). If None, `chain.max()` is
            used.

    Returns:
        float, float:
            The maximum likelihood estimet of the transition probabilities 0->1
            and 1->0 respectively.

        int:
            The thinning applied to the chain to ensure first order Markov
            process is a good approximation.
    """


    fitted_markov = DiscreteMarkov(samples_size=0)
    if num_states is None:
        num_states = chain.max() + 1

    thinning_statistic = 1
    thin = 0
    while thinning_statistic > 0:
        thin += 1
        if thin >  0.1 * chain.size:
            print('Thin %d too big. Giving up. Chain:' % thin)
            print(repr(chain))
            return None, 0
        thinning_statistic = (
            DiscreteMarkov(samples_size=0).fit(chain[::thin],
                                               num_states,
                                               2,
                                               True)
            -
            fitted_markov.fit(chain[::thin], num_states, 1, True)
            -
            num_states * (num_states - 1)**2 / 2.0
            *
            numpy.log(chain[::thin].size)
        )

    return fitted_markov, thin


def get_approximate_binary_markov(binary_chain):
    """
    Convenience wrapper around get_approximate_markov() for 2-state chains.

    Args:
        binary_chain(array):    A chain of 0/1 values that will be approximated
            by an order 1 Markov process.

    Returns
        float, float:
            The maximum likelihood estimet of the transition probabilities 0->1
            and 1->0 respectively.

        int:
            The thinning applied to the chain to ensure first order Markov
            process is a good approximation.
    """

    fitted_markov, thin = get_approximate_markov(binary_chain, 2)

    if fitted_markov is None:
        assert thin == 0
        return numpy.nan, numpy.nan, 0

    return (fitted_markov.transition_probabilities[0, 1],
            fitted_markov.transition_probabilities[1, 0],
            thin)


def get_raftery_lewis_burnin(binary_chain, burn_in_tolerance):
    """Find the burn-in a chain calculated per Raftery & Lewis (1995)."""

    if numpy.unique(binary_chain).size == 1:
        return 0
    alpha, beta, thin = get_approximate_binary_markov(binary_chain)
    if not alpha + beta < 1:
        print('Burn in determination from %d steps: '
              'alpha = %s, beta = %s, thin = %s!'
              %
              (binary_chain.size, alpha, beta, thin))

    if numpy.isnan(alpha) or numpy.isnan(beta):
        return binary_chain.size

    return thin * int(
        numpy.ceil(
            numpy.log((alpha + beta) * burn_in_tolerance / max(alpha, beta))
            /
            numpy.log(numpy.abs(1.0 - alpha - beta))
        )
    )


def get_iterative_raftery_lewis_burnin(binary_chain, burn_in_tolerance):
    """
    Find burn-in per Raftery & Lewis (1995), iterated to avoid under-estimate.

    Because it may be that after burn-in the auto-correlation is smaller, using
    the entire chain to calculate burn-in may result in under-estimate. To avoid
    this, this function re-calculates a new burn-in value based on the last
    estimate until the first iteration when the predicted values drops.
    """

    burn_in = get_raftery_lewis_burnin(binary_chain, burn_in_tolerance)
    while burn_in > 0:
        new_burn_in = get_raftery_lewis_burnin(binary_chain[:burn_in],
                                               burn_in_tolerance)
        if new_burn_in <= burn_in:
            return burn_in
        burn_in = new_burn_in

    return 0

def get_raftery_lewis_quantile_stddev(binary_chain):
    """
    Find the standard dev. of a quantile estimate per Raftery & Lewis (1995).

    Args:
        binary_chain(array):    1-D array of zeros and ones indicating whether a
            sample is below the threshold for which convergence is being
            estimated (0: below, 1: above).
    Returns:
        float:
            The estimated standard deviation from the entire chain (no burn-in
            is applied).

        int:
            The thinning needed to ensure first order Markov
            process is a good approximation.
    """

    if numpy.unique(binary_chain).size <= 1:
        return numpy.nan, 1

    alpha, beta, thin = get_approximate_binary_markov(binary_chain)

    if thin == 0:
        return numpy.inf, 0

    nsamples = numpy.floor(binary_chain.size / thin)

    return (
        (
            (2.0 - alpha - beta) * alpha * beta / (alpha + beta)**3
            /
            nsamples
        )**0.5,
        thin
    )


def get_raftery_lewis_diagnostics(binary_chain, burn_in_tolerance):
    """
    Return the quantile convergence diagnostics per Raftery & Lewis (1995).

    Args:
        binary_chain(array):    1-D array of zeros and ones indicating whether a
            sample is below the threshold for which convergence is being
            estimated (0: below, 1: above).

        burn_in_tolerance(float):    Burn-in samples are determined to ensure
            that the probability that the first non-discarded state in the
            binary chain is within this tolerance of the theoretical value based
            on maximum likelihood estimates of the transition probabilities of
            the thinned chain approximated as order-1 Markov process.

    Returns:
        float:
            The expected standard deviation of the quantile estimate (average of
            chain after burn-in)

        int:
            The thinning that was applied to the chain to ensure it is well
            approximated by an order-1 Markov process.

        int:
            The number of burn-in samples from the start of the (not thinned)
            chain to discard to ensure steady state has been reached.
    """


    burn_in = get_raftery_lewis_burnin(binary_chain, burn_in_tolerance)

    return (
        get_raftery_lewis_quantile_stddev(binary_chain[burn_in:])
        +
        (burn_in,)
    )


def show_raftery_lewis_plots():
    """Display plots showing the Raftery & Lewis thinning algorithm at work."""

    transition_probabilities = [numpy.empty((2,1)),
                                numpy.empty((2,2,1)),
                                numpy.empty((2,2,1))]
    transition_probabilities[0][0, 0] = 0.7
    transition_probabilities[0][1, 0] = 0.2

    transition_probabilities[1][0, 0, 0] = 0.69
    transition_probabilities[1][1, 0, 0] = 0.71
    transition_probabilities[1][1, 1, 0] = 0.19
    transition_probabilities[1][0, 1, 0] = 0.21

    transition_probabilities[2][0, 0, 0] = 0.3
    transition_probabilities[2][1, 1, 0] = 0.7
    transition_probabilities[2][0, 1, 0] = 0.5
    transition_probabilities[2][1, 0, 0] = 0.5

    nsamples = 2**numpy.arange(25)

    generators = [
        DiscreteMarkov(
            transition_probabilities=prob,
            initial_state=numpy.array([0, 1])[:len(prob.shape) - 1],
            samples_size=nsamples[-1]
        )
        for prob in transition_probabilities
    ]

    fitter = DiscreteMarkov(samples_size=0)

    chains = [
        gen.extend_chain(nsamples[-1])
        for gen in generators
    ]

    penalty = numpy.log(nsamples)
    plot_y = [
        [
            (
                fitter.fit(chn[:thin * i:thin], 2, 2, True)
                -
                fitter.fit(chn[:thin * i:thin], 2, 1, True)
            )
            for i in nsamples[:nsamples.size - thin + 1]
        ]
        for chn in chains
        for thin in [1, 2]
    ]
    labels = ['Chain %d, thin %d' % (chn, thin)
              for chn in range(len(chains))
              for thin in [1, 2]]
    curves = [
        pyplot.semilogx(nsamples[:len(curve)], curve, '-o')[0]
        for curve in plot_y
    ]
    pyplot.semilogx(nsamples, penalty, '-k')
    pyplot.legend(curves, labels)
    pyplot.show()

if __name__ == '__main__':
    show_raftery_lewis_plots()
