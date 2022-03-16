"""Define a discrete Markov process of arbitrary order."""

import numpy

class DiscreteMarkov:
    """Allawo simulating a discrete Markov process of arbitrary order."""


    def __init__(self,
                 transition_probabilities=None,
                 initial_state=None,
                 samples_size=1000000):
        """Create the Markov process."""

        if transition_probabilities is not None:
            self.set_probabilities(transition_probabilities)
        else:
            self.transition_probabilities = None

        self._samples = numpy.empty(samples_size, dtype=int)
        self._num_samples = 0
        if initial_state is not None:
            self.set_initial_state(initial_state)


    def set_probabilities(self, transition_probabilities):
        """
        Set the transition probabilities (order + 1) rank tensor.

        Args:
            transition_probabilities(array):    The conditional probabilities
                to transition to each of the possible states except the last
                one, given the last N states of the chain. The last index is
                the new state, the indices before are ordered in the same order
                as the states in the chain (e.g. for 2-nd order markov index 2
                is the state the chain will transition to, index 0 is the state
                before last in the chain, and index 1 is the last state in the
                chain).

            initial_state(array or None):    If not None, the initial state of
                the chain. Should contain exactly a number of entries equal to
                the order of the process.

            samples_size(int):    Initially the internally stored chain
                allocates space for this many samples. If a longer chain is
                needed, it gets resized, but this requires moving the data so
                should be avoided if possible.

        Returns:
            None
        """

        self._order = len(transition_probabilities.shape) - 1
        self._num_states = transition_probabilities.shape[0]
        assert transition_probabilities.shape == (
            self._order * (self._num_states,)
            +
            (self._num_states - 1,)
        )

        total_prob = numpy.sum(transition_probabilities, axis=self._order)
        assert total_prob.min() >= 0.0
        assert total_prob.max() <= 1.0

        self.transition_probabilities = numpy.empty(
            (self._order + 1) * (self._num_states,),
        )
        self.transition_probabilities[..., :self._num_states -1] = (
            transition_probabilities
        )
        self.transition_probabilities[..., self._num_states - 1] = (1.0
                                                                     -
                                                                     total_prob)


    def set_initial_state(self, initial_state, reset=False):
        """Define the inital state to start the chain from."""

        initial_state = numpy.atleast_1d(initial_state)
        if reset:
            self._num_samples = 0
        else:
            assert self._num_samples == 0

        assert initial_state.shape == (self._order,)
        self._samples[:self._order] = initial_state
        self._num_samples = self._order


    def draw_sample(self, chain_tail):
        """Draw a single sample (not adding to chain) given tail of chain."""

        probabilities = self.transition_probabilities[
            tuple(chain_tail[-self._order:]) + (slice(None),)
        ]

        return numpy.random.choice(numpy.arange(self._num_states),
                                   p=probabilities)

    def extend_chain(self,
                     size,
                     initial_state=None,
                     reset=False):
        """Extend the chain according to the specified process."""

        if initial_state is not None:
            self.set_initial_state(initial_state, reset)

        assert self._num_samples >= self._order

        assert size > self._order

        if self._samples.size < self._num_samples + size:
            self._samples.resize(self._num_samples + size)

        for _ in range(size):
            self._samples[self._num_samples] = self.draw_sample(
                self._samples[self._num_samples - self._order
                              :
                              self._num_samples]
            )
            self._num_samples += 1

        return self._samples[:self._num_samples]

    def fit(self, chain, num_states, order, return_max_loglikelihood=False):
        """
        Use maximum likelihood estimates of the probabilities given a chain.

        If not all transitions are represented in the input chain, some of the
        transition probabilities will be NaN.

        Args:
            chain(array):    The chain of discrete states that is assumed to be
                the result of a Markov process. States are labeled by integers
                stating at zero.

            num_states(int):    The number of states the chain can access (in
                case not all states are represented in the input chain.

            order(int):    The order of the markov process to fit.

            return_max_loglikelihood(bool):    Should the maximum log-likelihood
                be calculated and returned?

        Returns:
            None
        """

        assert len(chain.shape) == 1
        num_transitions = numpy.zeros((order + 1) * (num_states,), dtype=float)
        for i in range(order, chain.size):
            num_transitions[tuple(chain[i - order : i + 1])] += 1

        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.transition_probabilities = num_transitions / numpy.expand_dims(
                num_transitions.sum(axis=-1),
                axis=order
            )

        if return_max_loglikelihood:
            include = num_transitions > 0
            return (
                num_transitions[include]
                *
                numpy.log(self.transition_probabilities[include])
            ).sum()

        return None
