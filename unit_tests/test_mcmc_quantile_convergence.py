#!/usr/bin/env python3

"""Test suite for the functions diagnosing MCMC quantile convergence."""

import unittest

import numpy

from discrete_markov import DiscreteMarkov
#We wish to test all functions there
#pylint: disable=wildcard-import
from mcmc_quantile_convergence import *
#pylint: enable=wildcard-import

class TestMCMCQuantileConvergence(unittest.TestCase):
    """Unit tests for MCMC quantile convergence diagnostics."""

    def _check_fit_probabilities_deterministic(self, order, nstates, ntests):
        """
        Test fitting random transition matrices contaning only 0 and 1.

        Args:
            order(int):    The order of the markov process to test.

            nstates(int):    How many states should the chain iterate over.

            ntestst(int):    How many random tests to run.

        Returns:
            None
        """


        nsamples = nstates * order * 10
        for _ in range(ntests):
            true_probabilities = numpy.zeros((order + 1) * (nstates,))

            all_from_states = numpy.dstack(
                numpy.mgrid[
                    order * (slice(0, nstates),)
                ].reshape(
                    order,
                    nstates**order
                )
            )[0]
            for from_states in all_from_states:
                true_probabilities[
                    tuple(from_states)
                    +
                    (numpy.random.choice(numpy.arange(nstates)),)
                ] = 1.0

            process = DiscreteMarkov(true_probabilities[..., :-1],
                                     samples_size=nsamples)
            fit_probabilities = numpy.full(true_probabilities.shape, numpy.nan)
            for initial_state in range(nstates):
                chain = process.extend_chain(
                    nsamples,
                    initial_state=numpy.full(order, initial_state),
                    reset=True
                )
                this_fit = fit_markov_probabilities(chain, nstates, order)
                compare = numpy.logical_and(
                    numpy.isfinite(fit_probabilities),
                    numpy.isfinite(this_fit)
                )
                replace = numpy.logical_and(
                    numpy.logical_not(numpy.isfinite(fit_probabilities)),
                    numpy.isfinite(this_fit)
                )
                self.assertTrue(
                    (fit_probabilities[compare] == this_fit[compare]).all()
                )
                fit_probabilities[replace] = this_fit[replace]
            self.assertTrue((true_probabilities == fit_probabilities).all())

    def test_fit_probabilities_deterministic_order1(self):
        """Test fitting random transition matrices contaning only 0 and 1."""

        self._check_fit_probabilities_deterministic(1, 1, 3)
        self._check_fit_probabilities_deterministic(1, 2, 100)
        self._check_fit_probabilities_deterministic(1, 7, 1000)

    def test_fit_probabilities_deterministic_order2(self):
        """Test fitting random transition matrices contaning only 0 and 1."""

        self._check_fit_probabilities_deterministic(2, 1, 3)
        self._check_fit_probabilities_deterministic(2, 2, 100)
        self._check_fit_probabilities_deterministic(2, 7, 1000)



if __name__ == '__main__':
    unittest.main()
