"""Hidden Markov Model implementation."""

import numba
import numpy as np
import numpy.typing as npt

from numpy.random import choice
from hmm.types import FloatArray
from itertools import product


# Unnecessarily fast.
@numba.njit(parallel=True, fastmath=True)
def logsumexp(x: npt.ArrayLike[float]) -> float:
    """Compute log-sum-exp using the trick.

    Args:
        x (npt.ArrayLike[float]): Array of log-probabilities.
    
    Returns:
        float: Log-sum-exp of x.
    """
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


class HMM:
    """Hidden Markov Model."""

    def __init__(
        self,
        prior: FloatArray,
        transition: FloatArray,
        emission: FloatArray,
    ) -> None:
        """Initialise hidden markov model with relevant probabilities.

        Args:
            prior (FloatArray): Prior vector of probabilities for each latent state.
            transition (FloatArray): Transition probability matrix for model states.
            emission (FloatArray): Emission probability matrix for observations.
        """
        self.prior = prior
        self.transition = transition  # Denoted uppercase gamma (Γ).
        self.emission = emission  # P(X = x | Z = z) = Poisson - with mean λ_z > 0.

        # May come in handy.
        self._num_hiddens = transition.shape[0]
        self._num_states = emission.shape[1]



    @numba.jit
    def forward(self, observations: npt.NDArray) -> FloatArray:
        """Compute forward trellis using forward algorithm.
        
        Notes:
            This one is the O(2TN^T) forward-pass.

        Args:
            observations (npt.NDArray): Observation sequence.

        Raises:
            ValueError: If observations are not 1D.

        Returns:
            FloatArray: Two-dimensional forward trellis (α) matrix.
        """
        if len(observations.shape) > 1:
            raise ValueError("Observations should be 1D. What are you doing?")

        alpha: FloatArray = np.zeros([self._num_hiddens, observations.shape[0]])

        # Set initial state probabilities by first observation.
        for state in range(self._num_hiddens):
            alpha[state, 0] = np.log(self.prior[state]) + np.log(
                self.emission[state, observations[0]]
            )

        # Forward!
        for t, observation, state in product(
            enumerate(observations), range(self._num_hiddens)
        ):
            alpha[state, t] = logsumexp(
                [
                    alpha[state_, t - 1]
                        + np.log(self.transition[state_, state])
                        + np.log(self.emission[state, observation])
                    for state_ in range(self._num_hiddens)
                ]
            )

        return alpha