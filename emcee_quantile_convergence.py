#!/usr/bin/env python3
"""Test the convergence of quantile estimates from emcee walker samples."""

#from sys import float_info
from functools import partial

import numpy
from scipy.stats import rdist

from mcmc_quantile_convergence import get_approximate_markov
from kde import KDEDistribution


def regularize_discrete_chain(input_chain):
    """
    Return a chain over only states represented in the input chain.

    Args:
        input_chain(array):    A chain of integers, possibly non-sequential.

    Returns:
        array(int):
            An equivalent chain with all states in the input chain relabeled to
            form a sequence of consecutive integers starting at 0.

        array(int):
            The state in the input chain that corresponds to each of the states
            of the new chain.
    """

    while input_chain.size > 1 and (input_chain == input_chain[-1]).sum() == 1:
        input_chain = input_chain[:-1]

    if input_chain.size <= 1:
        return None, None

    represented_states = numpy.unique(input_chain)
    if represented_states.size == input_chain.max() + 1:
        return input_chain, represented_states

    output_chain = numpy.empty(input_chain.shape, dtype=input_chain.dtype)
    for new, old in enumerate(represented_states):
        update = (input_chain == old)
        output_chain[update] = new

    return output_chain, represented_states


def get_emcee_burnin(regular_indicator_chain, burnin_tolerance):
    """
    Find burn-in for chain of number of walkers below quantile threshold.

    Args:
        regular_indicator_chain(array):    Array of consecutive integers
            aliasing the number of walkers at each stap that are below the
            quantile being estimated. Usually created by
            regularize_discrete_chain().

        burnin_tolerance(float):    See find_emcee_quantiles().

    Returns:
        int:
            The number of burn-in steps.
    """

#    def eval_burnin_equation_svd(leftover_tensor,
#                                 leftover_singular_vals,
#                                 burnin_tolerance,
#                                 nsteps):
#        """Calculate excess deviation from equilibrium PDF after nsteps."""
#
#        return leftover_tensor.dot(
#            numpy.power(leftover_singular_vals, nsteps)
#        ).max() - burnin_tolerance


    def eval_burnin_equation_powers(transition_probabilities,
                                    expected_distro,
                                    nsteps):
        """Calculate excess deviation from equilibrium PDF after nsteps."""

        return (
            numpy.linalg.matrix_power(transition_probabilities, nsteps)
            -
            expected_distro
        ).max() - burnin_tolerance


#    def get_burnin_equation(fitted_markov):
#        """Return the equation to solve to find the burnin."""
#
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
#                       burnin_tolerance)

    def find_burnin_bracket(burnin_equation):
        """Return burnin range (min, max) that brackets the true value."""

        burnin_min = 1
        assert burnin_equation(burnin_min) > 0
        burnin_max = 2
        while burnin_equation(burnin_max) > 0:
            burnin_min = burnin_max
            burnin_max *= 2
            if burnin_max > 1e6:
                return burnin_min, None

        return burnin_min, burnin_max

    if numpy.unique(regular_indicator_chain).size == 1:
        return 0

    fitted_markov, thin = get_approximate_markov(regular_indicator_chain)

    if fitted_markov is None:
        return 0

    burnin_equation = partial(eval_burnin_equation_powers,
                              fitted_markov.transition_probabilities,
                              fitted_markov.get_equilibrium_distro())

    burnin_min, burnin_max = find_burnin_bracket(burnin_equation)

    if burnin_max is None:
        print('The following transition matrix fails to converge:\n'
              +
              repr(fitted_markov.transition_probabilities))
        return 0


    while burnin_max - burnin_min > 1:
        midpoint = (burnin_max + burnin_min) // 2
        if burnin_equation(midpoint) > 0:
            burnin_min = midpoint
        else:
            burnin_max = midpoint

    return burnin_max * thin


#The point is the initialization.
#pylint: disable=too-few-public-methods
class DrawRandomCDF:
    """Generator of random realizations of CDF at quantile."""

    def __init__(self,
                 fitted_markov,
                 chain_length,
                 fraction_below_states):
        """Prepare to draw random realizations of the CDF per given process."""

        self._fitted_markov = fitted_markov
        self._chain_length = chain_length
        self.equilibrium_distro = fitted_markov.get_equilibrium_distro()
        self._fraction_below_states = fraction_below_states

    def __call__(self):
        """Generate another independent random sample."""

        initial_state_rv = numpy.random.rand()
        for initial_state in range(self._fraction_below_states.size):
            initial_state_rv -= self.equilibrium_distro[initial_state]
            if initial_state_rv < 0:
                break
        simulated_chain = self._fitted_markov.extend_chain(self._chain_length,
                                                           initial_state,
                                                           True)
        return self._fraction_below_states[simulated_chain].mean()
#pylint: enable=too-few-public-methods


def diagnose_emcee_quantile(regular_indicator_chain=None,
                            num_below_states=None,
                            num_walkers=0,
                            variance_realizations=0):
    """
    Compute diagnostics of a quantile estimate based on emcee samples.

    Args:
        regular_indicator_chain(array):    The first return value of
            regularize_discrete_chain() given the input chain of number walkers
            below a threshold value.

        num_below_states(array):    The second return value of
            regularize_discrete_chain().

        num_walkers(int):    How many walkers were used when running emcee.

    Returns:
        Same as get_emcee_quantile_diagnostics() return value except burnin.
    """

    fail_result = (numpy.full(2, numpy.nan), numpy.nan, None)

    if regular_indicator_chain is None:
        return fail_result

    fitted_markov, thin = get_approximate_markov(regular_indicator_chain)
    if fitted_markov is None:
        assert thin == 0
        return fail_result

    sample_cdf = DrawRandomCDF(fitted_markov,
                               regular_indicator_chain.size // thin,
                               num_below_states / num_walkers)

    cdf_realizations = numpy.empty(variance_realizations)
    for realiz in range(variance_realizations):
        cdf_realizations[realiz] = sample_cdf()

    print('Mean(CDF realizations): ' + repr(cdf_realizations.mean()))

    return (
        numpy.array([
            num_below_states[regular_indicator_chain].mean() / num_walkers,
            sample_cdf.equilibrium_distro.dot(num_below_states) / num_walkers
        ]),
        numpy.var(cdf_realizations, ddof=1)**0.5,
        thin
    )



def get_emcee_quantile_diagnostics(binary_chain,
                                   burnin_tolerance,
                                   variance_realizations):
    """
    Return quantile convergence diagnostics like R&E95 but tuned to EMCEE chain.

    Args
        binary_chain(2D array):    The chain of 0 if a particular sample
            is above the target quantile, 1 if it is below. Should have shape
            (number of steps, number of walkers).

        burnin_tolerance(float):    See find_emcee_quantiles().

        variance_realizations(int):    See find_emcee_quantiles().

    Returns:
        array(float):
            Estimated CDF(`quantile`) using several estimates in order of
            reliability:

                - average number of walkers below `quantile' over emcee steps
                  after burn-in period.

                - Steady state of the maximum likelihood order-1 Markov process
                  approximating the thinned number walkers below quantile chain.

        float:
            The variance of the quantile estimate above.

        int:
            The thinning applied to the chain of number walkers below `quantile`
            in order to make it well approximated by a Markov process.

        int:
            The number of burn-in steps discarded.
    """

    num_below_chain = binary_chain.sum(axis=1)
    regular_indicator_chain, num_below_states = regularize_discrete_chain(
        num_below_chain
    )

    if regular_indicator_chain is None:
        return diagnose_emcee_quantile() + (None,)

    burnin = get_emcee_burnin(regular_indicator_chain, burnin_tolerance)
    if burnin >= regular_indicator_chain.size + 1:
        return diagnose_emcee_quantile() + (burnin,)

    regular_indicator_chain, num_below_states = regularize_discrete_chain(
        num_below_chain[burnin:]
    )
    return diagnose_emcee_quantile(regular_indicator_chain,
                                   num_below_states,
                                   binary_chain.shape[1],
                                   variance_realizations)


def find_emcee_quantiles(samples,
                         cdf_value,
                         burnin_tolerance,
                         variance_realizations):
    """
    Iterate between burn-in and PPF(cdf_value) to find quantiles & diagnostics.

    Args:
        samples(array):    The emcee samples of the quantity we are trying to
            find the quantile of.

        cdf_value(float):    The value of the cumulative distribution of the
            quantile we wish to estimate.

        burnin_tolerance(float):    Distribution after burn-in of number
            walkers below threshold is within this value of the limiting
            distribution of the approximate Markov process.

        variance_realizations(int):    Variance of the estimated CDF is
            calculated by generating this many independent random realizations
            of the best fit Markov process to the thinned chain of number of
            walkers below `quantile`.

    Returns:
        float:
            The quantile, i.e. estimate of the value at which the CDF of the
            distribution from which samples are drawn is equal to `cdf_value`.

        array(float):
            Estimated CDF(`quantile`) using several estimates in order of
            reliability:

                - average number of walkers below `quantile' over emcee steps
                  after burn-in period.

                - Steady state of the maximum likelihood order-1 Markov process
                  approximating the thinned number walkers below quantile chain.

        float:
            The standard deviation of the CDF estimates above.

        int:
            The thinning applied to the chain of number walkers below `quantile`
            in order to make it well approximated by a Markov process.

        int:
            The number of burn-in steps discarded.

    """

    for burnin in range(samples.shape[0]):
        quantile = KDEDistribution(
            samples[burnin:].flatten(),
            rdist(c=4, scale=0.05)
        ).ppf(
            cdf_value
        )

        regular_indicator_chain, num_below_states = regularize_discrete_chain(
            (samples < quantile).astype(int).sum(axis=1)
        )
        min_burnin = get_emcee_burnin(regular_indicator_chain, burnin_tolerance)
        if burnin == 0:
            full_chain_burnin = min_burnin
            full_chain_quantile = quantile
        if min_burnin > 0 and burnin >= min_burnin:
            break

    print(
        '\tBurn-in (CDF=%.2f) is %d/%d'
        %
        (
            cdf_value,
            (burnin if burnin >= min_burnin else full_chain_burnin),
            samples.shape[0]
        )
    )

    if burnin < min_burnin:
        return (
            (full_chain_quantile,)
            +
            diagnose_emcee_quantile()
            +
            (full_chain_burnin,)
        )

    return (
        (quantile,)
        +
        diagnose_emcee_quantile(regular_indicator_chain,
                                num_below_states,
                                samples.shape[1],
                                variance_realizations)
        +
        (burnin,)
    )
