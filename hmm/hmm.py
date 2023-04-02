"""Hidden Markov Model implementation."""

import numpy as np
import numpy.typing as npt

from hmm.types import FloatArray, IntArray

from itertools import product
from typing import Callable

from scipy.special import factorial
from scipy.stats import poisson


class HMM:
    def __init__(
        self,
        transition: FloatArray,
        alpha: float,
        processing_modes: list[int],
        rates: list[int],
    ) -> None:
        """Initialise HMM.

        Args:
            transition (FloatArray): Transition probability matrix.
            alpha (list[float] | float): Alpha probabilities or probability.
            processing_modes (list[int]): States for HMM.
            rates (list[int]): Rates for Poisson sampling of stimuli.
        """
        self.alpha = alpha
        self.transition = transition
        self.processing_modes = processing_modes
        self.rates = rates

        self.p_z_given_c_mat = np.array(
            [[1 - self.alpha, self.alpha, 0.5],
             [self.alpha, 1 - self.alpha, 0.5]]
        )

    def sample_poisson_stimuli(self, z_values: IntArray) -> IntArray:
        """Sample Poisson stimuli.

        Args:
            z_values (IntArray): List of Z-values.
            rates (IntArray): Rate-lookup indexed by Z-values.

        Returns:
            IntArray: Drawn samples from the parameterized Poisson distribution.
        """
        # Sampling from Poisson distribution; P(X_{t,i} = x | Z_{t,i} = z).
        sample_rates = [self.rates[z] for z in z_values]
        return np.random.poisson(sample_rates)

    def sample_hidden_c(self, current_c: int) -> int:
        """Sample C-value weighted by transition matrix.

        Args:
            current_c (int): Current C-value to transition from.

        Returns:
            int: New C-value.
        """
        return np.random.choice(self.processing_modes, p=self.transition[current_c])

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

    def p_z_given_c(self, z: int, c: int) -> float:
        """Probability of Z=z given C=c.

        Args:
            z (int): Z value.
            c (int): C value.

        Returns:
            float: Probability $P(Z=z | C=c)$.
        """
        match c:
            case 0:
                p = 1 - self.alpha
            case 1:
                p = self.alpha
            case 2:
                p = 0.5

        return p if z == 1 else 1 - p

    def p_c_given_c(self, c_next, current_c):
        return self.transition[current_c][c_next]

    def forward(
        self, num_nodes: int, time_steps=100, initial_c=2, seed=1
    ) -> tuple[list[int], FloatArray, FloatArray]:
        """Run forward simulation with specified node amount and time steps.

        Args:
            num_nodes (int): Number of C-nodes.
            time_steps (int, optional): Time steps in forward simulation.
                Defaults to 100.
            initial_c (int, optional): Initial C-value. Defaults to 2.
            seed: Seed for random sampling

        Returns:
            tuple[list[int], FloatArray, FloatArray]:
                Processing modes (C), focus (Z), activations (X).
        """
        current_c: int = initial_c

        activations: FloatArray = np.array([])

        focus: npt.NDArray[np.int32] = np.array([])
        processing_modes: list[int] = []
        np.random.seed(seed)
        for t in range(time_steps):
            z = self.sample_hidden_z(num_nodes, current_c)
            x = self.sample_poisson_stimuli(z)

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

    def compute_messages_from_x_and_z(
        self, t: int, num_nodes: int, observations: IntArray
    ):
        """Compute clique message of clique X-Z at timestep t.

        Args:
            t (int): Time step t.

        Returns:
            _type_: Clique message.
        """
        messages_from_x_and_z = []

        for z_t_i in range(num_nodes):
            # Observed stimuli corresponding to Z_t
            observed_stimuli = observations[t][z_t_i]

            messages_from_x_and_z.append(
                [
                    poisson.pmf(observed_stimuli, self.rates[0]),
                    poisson.pmf(observed_stimuli, self.rates[1]),
                ]
            )

        return messages_from_x_and_z

    def compute_messages_from_z_and_c(
        self, messages_from_x_and_z, z_index: int | None = None
    ):
        messages_from_z_and_c = []

        for c in self.processing_modes:
            # Snoop Dogg approved.
            joint_x_given_c = 1

            if z_index is None:
                all_messages = messages_from_x_and_z
            else:
                all_messages = (
                    messages_from_x_and_z[:z_index] + messages_from_x_and_z[z_index:]
                )

            for message in all_messages:
                # P(X|C) = P(X | Z = 0)P(Z = 0 | C) + P(X | Z = 1)P(Z = 1 | C)
                probability_of_x_given_c = sum(
                    message[z] * self.p_z_given_c(z, c) for z in range(2)
                )
                # P(X_1 | C)P(X_2 | C) = P(X_1, X_2 | C)
                joint_x_given_c *= probability_of_x_given_c

            messages_from_z_and_c.append(joint_x_given_c)

        return np.array(messages_from_z_and_c)

    def compute_messages_from_clique_zc_to_cc(
            self, observations: IntArray
    ) -> FloatArray:
        # P(X|Z)
        p_x_given_z = np.array(
            [
                poisson.pmf(observations, self.rates[1]),
                poisson.pmf(observations, self.rates[0]),
            ]
        )

        # return np.array([p_x_given_z[0] * self.p_z_given_c_mat[0, c] + p_x_given_z[1] * self.p_z_given_c_mat[1, c] for c in [0, 1, 2]])
        return np.einsum("ijk, il -> ljk", p_x_given_z, self.p_z_given_c_mat)

    def clique_tree_forward(
        self, observations: IntArray, timestep: int, initial_c: int = 2
    ) -> FloatArray:
        """_summary_

        Returns:
            FloatArray: _description_
        """
        time_steps, num_nodes = observations.shape

        # Forward pass.

        # P(C_1)
        forward_prob = np.eye(len(self.processing_modes))[initial_c]

        for t in range(timestep):
            messages_from_x_and_z = self.compute_messages_from_x_and_z(t, num_nodes, observations)
            messages_from_z_and_c = self.compute_messages_from_z_and_c(
                messages_from_x_and_z, t
            )

            # if t == 0:
            #     forward_prob = (
            #         messages_from_z_and_c[initial_c] * self.transition[initial_c]
            #     )
            # else:
            forward_prob = np.einsum(
                "i, j, ij -> j",
                forward_prob,  # P(C, X_prev)
                messages_from_z_and_c,  # P(X | C)
                self.transition,  # P(C_next | C)
            )

        return forward_prob

    def clique_tree_backward(self, observations: IntArray, timestep: int) -> FloatArray:
        """_summary_

        Args:
            observations (IntArray): _description_
            timestep (int): _description_

        Returns:
            FloatArray: _description_
        """
        # Backward pass
        time_steps, num_nodes = observations.shape

        backward_prob = np.ones(len(self.processing_modes))  # P(X_(1:T)|C_t)
        for t in range(time_steps - 1, timestep, -1):
            messages_from_x_and_z = self.compute_messages_from_x_and_z(t, num_nodes, observations)
            messages_from_z_and_c = self.compute_messages_from_z_and_c(  # P(X_t|C_t)
                messages_from_x_and_z, t
            )

            # Sum over C_t for P(X_(1:T)|C_t)P(C_t|C_prev)P(X_t|C_t)=P(X_t|C_prev)
            backward_prob = np.einsum(
                "i, ij, j -> j",
                backward_prob,
                self.transition,  # self.transition[i][j] -> P(C_t = j| C_prev = i)
                messages_from_z_and_c,
            )

        return backward_prob

    def clique_tree_forward_backward(
        self, observations: IntArray, timestep: int, initial_c: int = 2
    ):
        """Compute forward and backward probabilities at given timestep.

        Args:
            observations (IntArray): _description_
            timestep (int): _description_
            initial_c (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        return (
            self.clique_tree_forward(observations, timestep, initial_c),
            self.clique_tree_backward(observations, timestep),
        )

    def infer_marginal_c(
        self, observations: IntArray, time_of_c: int, initial_c: int = 2
    ) -> FloatArray:
        """_summary_

        Returns:
            FloatArray: _description_
        """
        time_steps, num_nodes = observations.shape

        forward_prob, backward_prob = self.clique_tree_forward_backward(
            observations, time_of_c, initial_c
        )

        # At time t of c_t.
        messages_from_x_and_z = self.compute_messages_from_x_and_z(time_of_c, num_nodes, observations)
        messages_from_z_and_c = self.compute_messages_from_z_and_c(
            messages_from_x_and_z, time_of_c
        )

        # P(C_t, X_1,...,X_{t-1}) P(X_T,...,X_{t+1}| C_t) P(X_t|C_t) = P(C_t, X^(1:T))
        joint_with_evidence = forward_prob * backward_prob * messages_from_z_and_c

        return joint_with_evidence / np.sum(joint_with_evidence)

    def infer_marginal_z(
        self, observations: IntArray, timestep: int, z_index: int, initial_c: int = 2
    ):
        time_steps, num_nodes = observations.shape

        messages_from_x_and_z = self.compute_messages_from_x_and_z(timestep, num_nodes, observations)
        messages_from_z_and_c = self.compute_messages_from_z_and_c(
            messages_from_x_and_z, z_index
        )

        forward_prob, backward_prob = self.clique_tree_forward_backward(
            observations, timestep, initial_c
        )
        joint_with_evidence = forward_prob * backward_prob * messages_from_z_and_c

        z_given_c = np.array(
            [[self.p_z_given_c(z, c) for c in self.processing_modes] for z in (0, 1)]
        )
        poisson_probs = np.array(
            [poisson.pmf(observations[timestep][z_index], self.rates[z]) for z in (0, 1)]
        )

        z_given_x = poisson_probs * np.einsum("ij,j->i", z_given_c, joint_with_evidence)

        return z_given_x / np.sum(z_given_x)