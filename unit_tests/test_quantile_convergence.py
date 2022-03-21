#!/usr/bin/env python3

"""Unit tests of functions for detecting quantiles have converged."""

import unittest

import numpy

from mcmc_quantile_convergence import get_raftery_lewis_burnin
from emcee_quantile_convergence import get_emcee_burnin
from discrete_markov import DiscreteMarkov


class TestQuantileConvergenceDiagnostics(unittest.TestCase):
    """Test suite for detecting quantiles have converged."""

    def test_binary_burnin_match(self):
        """Test that emcee & MCMC burnin calculators give consistent results"""

        alpha = 0.01
        beta = 0.1
        process = DiscreteMarkov(
            transition_probabilities=numpy.array([[1.0 - alpha], [beta]]),
            initial_state = 0
        )
        chain = process.extend_chain(999999)

        mcmc_burnin = get_raftery_lewis_burnin(chain, 1e-6)
        emcee_burnin = get_emcee_burnin(chain, 1e-6)
        print('MCMC burnin: ' + repr(mcmc_burnin))
        print('EMCEE burnin: ' + repr(emcee_burnin))
        self.assertEqual(mcmc_burnin, emcee_burnin)

if __name__ == '__main__':
    unittest.main()
