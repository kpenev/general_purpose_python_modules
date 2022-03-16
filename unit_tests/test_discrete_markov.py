#!/usr/bin/env python3

"""Test suite for discrete markov procesess."""

import unittest

import numpy

from discrete_markov import DiscreteMarkov

class TestDiscreteMarkov(unittest.TestCase):
    """Unit test for DiscreteMarkov processes."""

    @staticmethod
    def _get_equivalent_order1_process(probabilities_1d,
                                       process_order,
                                       samples_size):
        """Create higher order process identical to order 1 process."""

        num_states = probabilities_1d.shape[0]
        probabilities = numpy.empty(process_order * (num_states,)
                                    +
                                    (num_states -1,))
        probabilities[..., :, :] = probabilities_1d
        return DiscreteMarkov(transition_probabilities=probabilities,
                              samples_size=samples_size)


    def _check_fixed_order1(self, num_states, num_samples, process_order=1):
        """
        Check that order 1 process defined to hold a fixed value does so.

        Args:
            num_states(int):    The number of possible states the Markov process
                can access.

            num_samples(int):    How long of a chain to test.

            process_order(int):    The process is defined to have this order but
                behave as a first order Markov process.

        Returns:
            None
        """

        process = self._get_equivalent_order1_process(
            probabilities_1d=numpy.diag(
                numpy.ones(num_states)
            )[:, :num_states - 1],
            process_order=process_order,
            samples_size=num_samples
        )
        for fixed_state in range(num_states):
            chain = process.extend_chain(num_samples - 1,
                                         numpy.full(process_order, fixed_state),
                                         reset=True)
            self.assertTrue(numpy.unique(chain) == fixed_state)

    def _check_converging_order1(self,
                                 num_states,
                                 converge_state,
                                 num_samples,
                                 process_order=1):
        """
        Check order 1 process that gets to fixed value after first step.

        Args:
            num_states:   See _check_fixed_order1()

            converge_state(int):    The state to jump to and hold.

            num_states:    See _check_fixed_order1()

            process_order:    See _check_fixed_order1()

        Returns:
            None
        """

        probabilities = numpy.zeros((num_states, num_states - 1))
        if converge_state < num_states - 1:
            probabilities[..., converge_state] = 1.0

        process = self._get_equivalent_order1_process(
            probabilities_1d=probabilities,
            samples_size=num_samples,
            process_order=process_order
        )

        for initial_state in range(num_states):
            chain = process.extend_chain(
                num_samples - process_order,
                numpy.full(process_order, initial_state),
                reset=True
            )
            self.assertEqual(chain[process_order - 1], initial_state)
            self.assertTrue(numpy.unique(chain[process_order:])
                            ==
                            converge_state)


    def _check_alternating_order1(self,
                                  *,
                                  num_states,
                                  state1,
                                  state2,
                                  num_samples,
                                  process_order=1):
        """Check order 1 process s.t. state1 -> state2, all others -> state1."""

        probabilities = numpy.zeros((num_states, num_states))
        probabilities[:, state1] = 1.0
        probabilities[state1, state1] = 0.0
        probabilities[state1, state2] = 1.0

        process = self._get_equivalent_order1_process(
            probabilities_1d=probabilities[..., :num_states-1],
            samples_size=num_samples,
            process_order=process_order
        )

        for initial_state in range(num_states):
            chain = process.extend_chain(
                num_samples - 1,
                numpy.full(process_order, initial_state),
                reset=True
            )

            self.assertEqual(chain[0], initial_state)

            self.assertTrue(numpy.unique(chain[process_order::2])
                            ==
                            (state2 if initial_state == state1 else state1))
            self.assertTrue(numpy.unique(chain[process_order + 1::2])
                            ==
                            (state1 if initial_state == state1 else state2))


    def _check_cycling_ordern(self, order, num_states, num_samples):
        """
        Check order N process staying on a state N times then moving to next.

        Args:
            order(int):    The order of the process, also the number of times it
                should stay on a value.

            num_states(int):    The number of states the chain can visit.

            num_samples(int):    How many samples of the chain to generate for
                testing.

        Returns:
            None
        """

        probabilities = numpy.empty((order + 1) * (num_states,))
        last_match_prob = numpy.ones((order + 1) * (num_states,))
        for i in range(0, order - 1):
            last_match_prob *= numpy.expand_dims(
                numpy.diag(numpy.ones(num_states)),
                axis=tuple(range(i)) + tuple(range(i+2, order + 1))
            )

        shift_prob = numpy.expand_dims(
            numpy.diag(numpy.ones(num_states - 1), k=1),
            axis=tuple(range(0, order - 1))
        )
        shift_prob[..., num_states - 1, 0] = 1.0

        stay_prob = numpy.expand_dims(
            numpy.diag(numpy.ones(num_states), k=0),
            axis=tuple(range(0, order - 1))
        )

        probabilities = (
            shift_prob
            *
            last_match_prob
            +
            stay_prob
            *
            (1.0 - last_match_prob)
        )
        process = DiscreteMarkov(
            transition_probabilities=probabilities[..., :-1],
            samples_size=num_samples
        )
        num_shifts = int(numpy.ceil(num_samples / order))
        num_samples = num_shifts * order

        for initial_state in range(num_states):
            expected = numpy.expand_dims(
                numpy.arange(initial_state, initial_state + num_shifts)
                %
                num_states,
                axis=1
            )

            chain = process.extend_chain(
                num_samples - order,
                numpy.full(order, initial_state),
                reset=True
            )

            self.assertTrue(
                (
                    chain.reshape(num_shifts, order)
                    ==
                    expected
                ).all()
            )


    #No reasonable way to simplify
    #pylint: disable=too-many-locals
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
        for progress in range(ntests):
            print('Progress: %d/%d (%d%%)' % (progress,
                                              ntests,
                                              int(100 * progress/ntests)),
                  end='\r')
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

            true_process = DiscreteMarkov(true_probabilities[..., :-1],
                                          samples_size=nsamples)
            fit_process = DiscreteMarkov(samples_size=0)

            fit_probabilities = numpy.full(true_probabilities.shape, numpy.nan)
            for initial_state in all_from_states:
                chain = true_process.extend_chain(
                    nsamples,
                    initial_state=initial_state,
                    reset=True
                )
                fit_process.fit(chain, nstates, order)
                compare = numpy.logical_and(
                    numpy.isfinite(fit_probabilities),
                    numpy.isfinite(fit_process.transition_probabilities)
                )
                replace = numpy.logical_and(
                    numpy.logical_not(numpy.isfinite(fit_probabilities)),
                    numpy.isfinite(fit_process.transition_probabilities)
                )
                self.assertTrue(
                    (
                        fit_probabilities[compare]
                        ==
                        fit_process.transition_probabilities[compare]
                    ).all()
                )
                fit_probabilities[replace] = (
                    fit_process.transition_probabilities[replace]
                )
            self.assertTrue((true_probabilities == fit_probabilities).all())
    #pylint: enable=too-many-locals


    def test_fixed_order1(self):
        """Test order 1 process defined to hold a fixed value."""

        for process_order in range(1, 5):
            self._check_fixed_order1(1, 1000, process_order)
            self._check_fixed_order1(2, 1000, process_order)
            self._check_fixed_order1(6, 1000, process_order)


    def test_converging_order1(self):
        """Test order 1 process set to immediately jump and hold a value."""

        for process_order in range(1, 5):
            for num_states in [1, 2, 6]:
                for converge_state in range(num_states):
                    self._check_converging_order1(num_states,
                                                  converge_state,
                                                  1000,
                                                  process_order=process_order)


    def test_alternating_order1(self):
        """Test order 1 process s.t. state1 -> state2, all others -> state1."""

        for process_order in range(1, 3):
            for num_states in [1, 2, 6]:
                for state1 in range(num_states):
                    for state2 in range(num_states):
                        self._check_alternating_order1(
                            num_states=num_states,
                            state1=state1,
                            state2=state2,
                            num_samples=1000,
                            process_order=process_order
                        )


    def test_cycling_ordern(self):
        """
        Test order N process staying on a state N times then moving to next.
        """

        self._check_cycling_ordern(3, 7, 1000)


    def test_fit_probabilities_deterministic_order1(self):
        """Test fitting random transition matrices contaning only 0 and 1."""

        self._check_fit_probabilities_deterministic(1, 1, 3)
        self._check_fit_probabilities_deterministic(1, 2, 100)
        self._check_fit_probabilities_deterministic(1, 7, 1000)


    def test_fit_probabilities_deterministic_order2(self):
        """Test fitting random transition matrices contaning only 0 and 1."""

        self._check_fit_probabilities_deterministic(2, 1, 3)
        self._check_fit_probabilities_deterministic(2, 2, 100)
        self._check_fit_probabilities_deterministic(2, 6, 300)

    def test_fit_probabilities_deterministic_order5(self):
        """Test fitting random transition matrices contaning only 0 and 1."""

        self._check_fit_probabilities_deterministic(2, 1, 3)
        self._check_fit_probabilities_deterministic(2, 2, 100)
        self._check_fit_probabilities_deterministic(2, 4, 1000)


if __name__ == '__main__':
    unittest.main()
