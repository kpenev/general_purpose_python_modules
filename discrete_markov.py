"""Define a discrete Markov process of arbitrary order."""

import numpy

class DiscreteMarkov:
    """Allawo simulating a discrete Markov process of arbitrary order."""


    def __init__(self,
                 transition_probabilities,
                 initial_state=None,
                 samples_size=1000000):
        """
        Set the transition probabilities (order + 1) rank tensor.

        Args:
            transition_probabilities(array):    The conditional probabilities
                to transition to each of the possible states given the last N
                states of the chain. The first index is the new state, the
                indices after are ordered in the same order as the states in the
                chain (e.g. for 2-nd order markov index 0 is the state the chain
                will transition to, index 1 is the state before last in the
                chain, and index 2 is the last state in the chain).

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
        self._num_states = transition_probabilities.shape[-1]
        assert transition_probabilities.shape == (
            (self._num_states - 1,)
            +
            self._order * (self._num_states,)
        )

        total_prob = numpy.sum(transition_probabilities, axis=0)
        assert total_prob.min() >= 0.0
        assert total_prob.max() <= 1.0

        self._transition_probabilities = numpy.empty(
            (self._order + 1) * (self._num_states,),
        )
        self._transition_probabilities[:self._num_states -1] = (
            transition_probabilities
        )
        self._transition_probabilities[self._num_states - 1] = 1.0 - total_prob

        self._samples = numpy.empty(samples_size, dtype=int)
        self._num_samples = 0
        if initial_state is not None:
            self.set_initial_state(initial_state)


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

        probabilities = self._transition_probabilities[
            (slice(None),) + tuple(chain_tail[-self._order:])
        ]

        probabilities=probabilities.flatten()
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

        for i in range(size):
            self._samples[self._num_samples] = self.draw_sample(
                self._samples[self._num_samples - self._order
                              :
                              self._num_samples]
            )
            self._num_samples += 1

        return self._samples[:self._num_samples]
