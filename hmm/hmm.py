"""Hidden Markov Model implementation."""

import numpy as np
import numpy.typing as npt

from hmm.types import FloatArray, IntArray

import math
from itertools import product
from typing import Callable

from scipy.special import factorial


def sample_poisson_stimuli(z_values: IntArray, rates: IntArray):
    sample_rates = [rates[z] for z in z_values]
    return np.random.poisson(sample_rates)


def poisson_pmf(x: int, lambda_z: float) -> float:
    """Compute the Poisson probability mass function (PMF) for given observation x and rate parameter lambda_z."""
    return np.exp(-lambda_z) * (lambda_z**x) / factorial(x)


def compute_emission_probabilities(possible_x, possible_z, rates):
    """Compute emission probabilities P(X | Z).

    Args:
        possible_x (IntArray): All possible observations.
        possible_z (IntArray): All possible hidden states.
        rates (list[int]): Poisson distribution rates for each hidden state Z.

    Returns:
        FloatArray: Emission probability matrix, shape (len(possible_x), len(possible_z)).
    """
    emission_probs = np.zeros((len(possible_x), len(possible_z)))

    for i, x in enumerate(possible_x):
        for j, z in enumerate(possible_z):
            rate = rates[z]
            emission_probs[i, j] = np.exp(-rate) * (rate**x) / factorial(x)

    return emission_probs


class HMM:
    def __init__(
        self,
        transition: FloatArray,
        alpha: list[float] | float,
        sample_stimuli: Callable[[IntArray], IntArray],
        states: list[int],
        rates: list[int],
    ) -> None:
        """Initialise HMM.

        Args:
            transition (FloatArray): Transition probability matrix.
            alpha (list[float] | float): Alpha probabilities or probability.
            sample_stimuli (Callable[[IntArray], IntArray]): Stimuli-sampling function.
                Just the Poisson one.
            states (list[int]): States for HMM.
        """
        self.alpha = alpha
        self.transition = transition  # Uppercase gamma.

        # Sampling from Poisson distribution; P(X_{t,i} = x | Z_{t,i} = z).
        self.sample_stimuli = sample_stimuli
        self.states = states
        self.rates = rates

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
            time_steps (int, optional): Time steps in forward simulation.
                Defaults to 100.
            initial_c (int, optional): Initial C-value. Defaults to 2.

        Returns:
            tuple[list[int], FloatArray, FloatArray]:
                Processing modes (C), focus (Z), activations (X).
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

    def emission_probabilities(self, x: int, current_c: int) -> float:
        match current_c:
            case 0:
                p = 1 - self.alpha
            case 1:
                p = self.alpha
            case 2:
                p = 0.5

        # Compute probabilities for Z = 0 and Z = 1
        emission_probs = np.zeros(2)
        for z in range(2):
            rate = self.rates[z]
            emission_probs[z] = sum((np.exp(-rate) * (rate**x)) / factorial(x))

        # Weighted probability based on the relationship between C and Z
        return p * emission_probs[1] + (1 - p) * emission_probs[0]

    def forward_pass(self, observations: IntArray) -> tuple[FloatArray, FloatArray]:
        time_steps: int = len(observations)
        num_states: int = len(self.states)

        forward_prob = np.zeros([time_steps, num_states])
        scaling_factors = np.zeros(time_steps)

        for t in range(time_steps):
            if t == 0:
                for i in range(num_states):
                    emission_prob = self.emission_probabilities(observations[t], i)
                    forward_prob[t, i] = self.alpha * emission_prob
            else:
                for j in range(num_states):
                    emission_prob = self.emission_probabilities(observations[t], j)
                    forward_prob[t, j] = (
                        np.dot(forward_prob[t - 1], self.transition[:, j])
                        * emission_prob
                    )

            scaling_factors[t] = np.sum(forward_prob[t])
            forward_prob[t] /= scaling_factors[t]

        return forward_prob, scaling_factors

    def backward_pass(
        self, observations: IntArray, scaling_factors: FloatArray
    ) -> FloatArray:
        time_steps: int = len(observations)
        num_states: int = len(self.states)
        backward_prob = np.zeros([time_steps, num_states])

        # Base case.
        backward_prob[-1] = 1

        for t in range(time_steps - 2, -1, -1):
            for i in range(num_states):
                emission_prob = self.emission_probabilities(observations[t + 1], i)
                backward_prob[t, i] = np.dot(
                    self.transition[i], emission_prob * backward_prob[t + 1]
                )
            backward_prob[t] /= scaling_factors[t + 1]

        return backward_prob

    def infer(self, observations: IntArray) -> FloatArray:
        forward_prob, scaling_factors = self.forward_pass(observations)
        backward_prob = self.backward_pass(observations, scaling_factors)

        time_steps, num_states = forward_prob.shape
        joint_prob = np.zeros((time_steps - 1, num_states, num_states))

        for t in range(time_steps - 1):
            for i in range(num_states):
                for j in range(num_states):
                    emission_prob = self.emission_probabilities(observations[t + 1], j)
                    joint_prob[t, i, j] = (
                        forward_prob[t, i]
                        * self.transition[i, j]
                        * emission_prob
                        * backward_prob[t + 1, j]
                    )

        joint_prob /= np.sum(joint_prob, axis=(1, 2), keepdims=True)

        return joint_prob
