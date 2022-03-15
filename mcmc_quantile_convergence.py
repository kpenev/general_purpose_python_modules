"""Test the convergence of quantile estimates from MCMC chains."""

import numpy

def fit_markov_probabilities(chain, num_states, markov_order):
    """
    Get max likelihood probabilities of a Markov process given a sampled chain.

    Args:
        chain(array):    The chain of discrete states that is assumed to be the
            result of a Markov process. States are labeled by integers stating
            at zero.

        num_states(int):    The number of states the chain can access (in case
            not all states are represented in the input chain.

        markov_order(int):    The order of the markov process to fit.

    Return:
        arary:
            Maximum likelihood estimates of the conditional probabilities to
            transition to each of the possible states given the last N states of
            the chain. The last index is the new state, the indices before are
            ordered in the same order as the states in the chain (e.g. for 2-nd
            order markov index 2 is the state the chain will transition to,
            index 0 is the state before last in the chain, and index 1 is the
            last state in the chain).
    """

    assert len(chain.shape) == 1
    result = numpy.zeros((markov_order + 1)* (num_states,), dtype=float)
    for i in range(markov_order, chain.size):
        result[tuple(chain[i - markov_order : i + 1])] += 1

    with numpy.errstate(divide='ignore', invalid='ignore'):
        result /= numpy.expand_dims(result.sum(axis=-1), axis=markov_order)

    return result
