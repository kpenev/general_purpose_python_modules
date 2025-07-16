"""Define a discrete Markov process of arbitrary order."""

import logging

import numpy


class DiscreteMarkov:
    """Allawo simulating a discrete Markov process of arbitrary order."""

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        transition_probabilities=None,
        **initial_state_kwargs,
    ):
        """Create the Markov process."""

        if transition_probabilities is not None:
            self.set_probabilities(transition_probabilities)
        else:
            self.transition_probabilities = None

        self._samples = None
        self._num_samples = 0
        self._num_states = None
        self._order = None
        self.set_initial_state(reset=True, **initial_state_kwargs)

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
            self._order * (self._num_states,) + (self._num_states - 1,)
        )

        total_prob = numpy.sum(transition_probabilities, axis=self._order)
        assert total_prob.min() >= 0.0
        assert total_prob.max() <= 1.0

        self.transition_probabilities = numpy.empty(
            (self._order + 1) * (self._num_states,),
        )
        self.transition_probabilities[..., : self._num_states - 1] = (
            transition_probabilities
        )
        self.transition_probabilities[..., self._num_states - 1] = (
            1.0 - total_prob
        )

    def set_initial_state(
        self, initial_state=None, reset=False, samples_size=10000
    ):
        """Define the inital state to start the chain from and reset."""

        if initial_state is not None:
            initial_state = numpy.atleast_1d(initial_state)
        if reset:
            self._samples = numpy.empty(
                max(
                    samples_size,
                    0 if initial_state is None else initial_state.size,
                ),
                dtype=int,
            )
            self._num_samples = 0

        if initial_state is not None:
            initial_state = numpy.atleast_1d(initial_state)
            if reset:
                self._num_samples = 0
            else:
                assert self._num_samples == 0

            if initial_state.shape != (self._order,):
                raise ValueError(
                    f"Initial state must have {self._order} entries, got "
                    f"{initial_state.shape}: {initial_state!r}"
                )
            self._samples[: self._order] = initial_state
            self._num_samples = self._order

    def draw_sample(self, chain_tail):
        """Draw a single sample (not adding to chain) given tail of chain."""

        probabilities = self.transition_probabilities[
            tuple(chain_tail[-self._order :]) + (slice(None),)
        ]

        return numpy.random.choice(
            numpy.arange(self._num_states), p=probabilities
        )

    def extend_chain(self, size, initial_state=None, reset=False):
        """Extend the chain according to the specified process."""

        if initial_state is not None:
            self.set_initial_state(initial_state, reset)

        assert self._num_samples >= self._order

        assert size > self._order

        if self._samples.size < self._num_samples + size:
            self._samples.resize(self._num_samples + size)

        for _ in range(size):
            self._samples[self._num_samples] = self.draw_sample(
                self._samples[
                    self._num_samples - self._order : self._num_samples
                ]
            )
            self._num_samples += 1

        return self._samples[: self._num_samples]

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

        self._logger.debug(
            "Fitting %d order Markov process to %d state chain of length %d.",
            order,
            num_states,
            chain.size,
        )
        self._num_states = num_states
        self._order = order

        assert len(chain.shape) == 1
        num_transitions = numpy.zeros((order + 1) * (num_states,), dtype=float)
        self._logger.debug(
            "Filling number of transitions of size %s",
            repr(num_transitions.size),
        )
        for i in range(order, chain.size):
            num_transitions[tuple(chain[i - order : i + 1])] += 1

        with numpy.errstate(divide="ignore", invalid="ignore"):
            self.transition_probabilities = num_transitions / numpy.expand_dims(
                num_transitions.sum(axis=-1), axis=order
            )
        print(
            "Calculated transition probabilities of size %s",
            self.transition_probabilities.size,
        )

        self._logger.debug(
            "Transition normalization coefficients:\n%s",
            numpy.array2string(
                num_transitions.sum(axis=-1), threshold=numpy.inf
            ),
        )

        if (
            not numpy.isfinite(self.transition_probabilities).all()
            and order == 1
        ):
            self._logger.error(
                "Not all transition probabilities finite for %d order, "
                "%d state chain:\n" % (order, num_states)
                + repr(chain)
                + "\nNum transitions:\n"
                + repr(num_transitions)
            )

        if return_max_loglikelihood:
            include = num_transitions > 0
            return (
                num_transitions[include]
                * numpy.log(self.transition_probabilities[include])
            ).sum()

        return None

    def get_equilibrium_distro(self):
        """
        Return the equilibrium distribution of the currently defined process.

        Must haveth transition probabilities defined either by
        set_probabilities() or by fit(), defining a regular process.

        Args:
            None

        Returns:
            array:
                The probability of each state in the equilibrium distribution.
        """

        equilibrium_coef = numpy.empty((self._num_states + 1, self._num_states))
        equilibrium_coef[: self._num_states] = (
            self.transition_probabilities.T
            - numpy.diag(numpy.ones(self._num_states))
        )
        equilibrium_coef[self._num_states] = 1.0

        equilibrium_rhs = numpy.zeros(self._num_states + 1)
        equilibrium_rhs[self._num_states] = 1.0

        self._logger.debug(
            "Solving equilibrium distribution with A of shape %s and b of "
            "shape %s",
            repr(equilibrium_coef.shape),
            repr(equilibrium_rhs.shape),
        )
        try:
            distro, residuals, _, _ = numpy.linalg.lstsq(
                equilibrium_coef, equilibrium_rhs, rcond=None
            )
        except:
            self._logger.critical(
                "Failed to solve LSTSQ with A=\n%s\nb=%s",
                repr(equilibrium_coef),
                repr(equilibrium_rhs),
            )
            raise

        assert residuals < 1e-16

        return distro
