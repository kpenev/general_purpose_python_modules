#!/usr/bin/env python3
"""Test the convergence of quantile estimates from emcee walker samples."""

#from sys import float_info
from functools import partial

import numpy

from mcmc_quantile_convergence import get_approximate_markov


def regularize_discrete_chain(input_chain):
    """
    Return a chain over only states represented in the input chain.

    Args:
        input_chain(array):    A chain of integers, possibly non-sequential.

    Returns:
        array(int):
            An equivalent chain with all states in the input chain relabeled to
            form a sequence of consecutive integers starting at 0.
    """

    represented_states = numpy.unique(input_chain)
    if represented_states.size == input_chain.max() + 1:
        print('Chain contains all states')
        return input_chain

    output_chain = numpy.empty(input_chain.shape, dtype=input_chain.dtype)
    for new, old in enumerate(represented_states):
        update = (input_chain == old)
        output_chain[update] = new
    return output_chain


def get_emcee_burnin(below_quantile_chain, burn_in_tolerance):
    """
    Find burn-in for chain of number of walkers below quantile threshold.

    Args:
        below_quantile_chain(array):    Array of integers indicating the number
            of walkers at each stap that are below the quantile being estimated.

        burn_in_tolerance(float):    Distribution after burn-in of number
            walkers below threshold is within this value of the limiting
            distribution of the approximate Markov process.

    Returns:
        int:
            The number of burn-in steps.
    """

#    def eval_burnin_equation_svd(leftover_tensor,
#                                 leftover_singular_vals,
#                                 burn_in_tolerance,
#                                 nsteps):
#        """Calculate excess deviation from equilibrium PDF after nsteps."""
#
#        return leftover_tensor.dot(
#            numpy.power(leftover_singular_vals, nsteps)
#        ).max() - burn_in_tolerance


    def eval_burnin_equation_powers(transition_probabilities,
                                    expected_distro,
                                    nsteps):
        """Calculate excess deviation from equilibrium PDF after nsteps."""

        return (
            numpy.linalg.matrix_power(transition_probabilities, nsteps)
            -
            expected_distro
        ).max() - burn_in_tolerance


    def get_burnin_equation(fitted_markov):
        """Return the equation to solve to find the burnin."""

        return partial(eval_burnin_equation_powers,
                       fitted_markov.transition_probabilities,
                       fitted_markov.get_equilibrium_distro())

#        left, singular_vals, right_trans = numpy.linalg.svd(
#            fitted_markov.transition_probabilities.T
#        )
#        print('Left: ' + repr(left))
#        print('SV: ' + repr(singular_vals))
#        print('Right: ' + repr(right_trans))
#        tolerance = 10.0 * float_info.epsilon * singular_vals.size
#        assert numpy.abs(singular_vals[0] - 1) < tolerance
#        assert (
#            numpy.abs(
#                left[:, 0] * right_trans[0, :]
#                -
#                fitted_markov.get_equilibrium_distro()
#            )
#            <
#            tolerance
#        )
#
#        leftover_tensor = left[:, None, 1:] * right_trans[1:, :].T[None, :, :]
#
#        return partial(eval_burnin_equation,
#                       leftover_tensor,
#                       singular_vals[1:],
#                       burn_in_tolerance)

    def find_burnin_bracket(burnin_equation):
        """Return burnin range (min, max) that brackets the true value."""

        burnin_min = 1
        assert burnin_equation(burnin_min) > 0
        burnin_max = 2
        while burnin_equation(burnin_max) > 0:
            burnin_min = burnin_max
            burnin_max *= 2

        return burnin_min, burnin_max

    if numpy.unique(below_quantile_chain).size == 1:
        return 0

    regular_chain = regularize_discrete_chain(below_quantile_chain)
    fitted_markov, thin = get_approximate_markov(regular_chain)

    burnin_equation = get_burnin_equation(fitted_markov)
    burnin_min, burnin_max = find_burnin_bracket(burnin_equation)

    while burnin_max - burnin_min > 1:
        midpoint = (burnin_max + burnin_min) // 2
        if burnin_equation(midpoint) > 0:
            burnin_min = midpoint
        else:
            burnin_max = midpoint

    return burnin_max * thin
