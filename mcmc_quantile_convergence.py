#!/usr/bin/env python3
"""Test the convergence of quantile estimates from MCMC chains."""

from matplotlib import pyplot
import numpy

from discrete_markov import DiscreteMarkov

def get_raferty_lewis_thinning(chain, num_states):
    """Select thinning per the Raferty & Lewis (1995) algorithm."""

    fitter = DiscreteMarkov(samples_size=0)
    statistic = 1
    thin = 0
    while statistic > 0:
        thin += 1
        statistic = (
            fitter.fit(chain[::thin], num_states, 2, True)
            -
            fitter.fit(chain[::thin], num_states, 1, True)
            -
            numpy.log(chain[::thin].size)
        )

    return thin


def show_raferty_lewis_plots():
    """Display plots showing the Raferty & Lewis thinning algorithm at work."""

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
    show_raferty_lewis_plots()
