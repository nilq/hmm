"""Hidden Markov Model implementation."""

import numpy as np
import numpy.typing as npt

from hmm.types import FloatArray, IntArray

from typing import Callable


def sample_poisson_stimuli(z_values: IntArray, rates: IntArray):
    sample_rates = [rates[z] for z in z_values]
    return np.random.poisson(sample_rates)


class HMM:
    def __init__(
        self,
        transition: FloatArray,
        alpha: list[float],
        sample_stimuli: Callable[[IntArray], IntArray],
        states: list[int],
    ) -> None:
        """Initialise HMM.

        Args:
            transition (FloatArray): Transition probability matrix.
            alpha (list[float]): Alpha probabilities.
            sample_stimuli (Callable[[IntArray], IntArray]): Stimuli-sampling function. Just the Poisson one.
            states (list[int]): States for HMM.
        """
        self.alpha = alpha
        self.transition = transition  # Uppercase gamma.

        # Sampling from Poisson distribution; P(X_{t,i} = x | Z_{t,i} = z).
        self.sample_stimuli = sample_stimuli
        self.states = states

    def sample_hidden_c(self, current_c: int) -> int:
        """Sample C-value weighted by transition matrix.

        Args:
            current_c (int): Current C-value to transition from.

        Returns:
            int: New C-value.
        """
        return np.random.choice(self.states, p=self.transition[current_c])

    def sample_hidden_z(self, size: int, current_c: int) -> IntArray:
        """Sample hidden Z-value.

        Args:
            size (int): Size of binomial sample.
            current_c (int): Current C-value to sample Z.

        Returns:
            IntArray: Sampled Z-values.
        """
        match current_c:
            case 0:
                p = 1 - self.alpha
            case 1:
                p = self.alpha
            case 2:
                p = 0.5

        return np.random.binomial(n=1, p=p, size=size)

    def forward(
        self, num_nodes: int, time_steps=100, initial_c=2
    ) -> tuple[list[int], FloatArray, FloatArray]:
        """Run forward simulation with specified node amount and time steps.

        Args:
            num_nodes (int): Number of C-nodes.
            time_steps (int, optional): Time steps in forward simulation. Defaults to 100.
            initial_c (int, optional): Initial C-value. Defaults to 2.

        Returns:
            tuple[list[int], FloatArray, FloatArray]: Processing modes (C), focus (Z), activations (X).
        """
        current_c: int = initial_c

        activations: FloatArray = np.array([])

        focus: npt.NDArray[np.int32] = np.array([])
        processing_modes: list[int] = []

        for t in range(time_steps):
            z = self.sample_hidden_z(num_nodes, current_c)
            x = self.sample_stimuli(z)

            processing_modes.append(current_c)

            # After first step, we have values.
            if t > 0:
                focus = np.vstack([focus, z])
                activations = np.vstack([activations, x])
            else:
                focus = z
                activations = x

            current_c = self.sample_hidden_c(current_c)

        return processing_modes, focus, activations
