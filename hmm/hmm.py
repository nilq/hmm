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

        self.p_z_given_c_mat = np.array([
            [1 - self.alpha, self.alpha, 0.5],
            [self.alpha, 1 - self.alpha, 0.5]
        ])

        # Used for belief calibration
        self.mu_cz = None
        self.mu_cc = None
        self.beta_c = None

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

    def clique_tree_forward(
            self, observations: IntArray, timestep: int, initial_c: int = 2
    ) -> FloatArray:
        """_summary_

        Returns:
            FloatArray: _description_
        """
        time_steps, num_nodes = observations.shape

        # Forward pass.

        forward_prob = np.ones(len(self.processing_modes))

        for t in range(timestep):
            messages_from_x_and_z = self.compute_messages_from_x_and_z(t, num_nodes, observations)
            messages_from_z_and_c = self.compute_messages_from_z_and_c(
                messages_from_x_and_z, t
            )

            if t == 0:
                forward_prob = (
                        messages_from_z_and_c[initial_c] * self.transition[initial_c]
                )
            else:
                forward_prob = np.einsum(
                    "i, j, ij -> i",
                    forward_prob,
                    messages_from_z_and_c,
                    self.transition,
                )

        return forward_prob

    def messages_from_clique_zc_to_cc(self, observations):
        # P(X|Z)
        p_x_given_z = np.array([
            poisson.pmf(observations, self.rates[0]),
            poisson.pmf(observations, self.rates[1]),
        ])

        self.mu_cz = p_x_given_c = (
            np.einsum('ijk, il -> ljk',
                      p_x_given_z,
                      self.p_z_given_c_mat
                      )
        )
        # Above is equivalent to
        # np.array([p_x_given_z[0] * self.p_z_given_c_mat[0, c] + p_x_given_z[1] * self.p_z_given_c_mat[1, c] for c in [0, 1, 2]])
        return p_x_given_c

    def belief_propagation(self, observations, initial_c: int = 2):
        T = len(observations)
        num_processing_modes = len(self.processing_modes)

        # Pass up from zc to cc
        self.messages_from_clique_zc_to_cc(observations)
        mu_cc = []
        beta_cs = np.zeros((T, num_processing_modes))

        # Forward pass
        forward_prob = np.ones(num_processing_modes)
        for t in range(T-1):
            if t == 0:
                # P(X1|C1=2)P(X2|C1=2)...P(Xn|C1=2) P(C2|C1=2)P(C1=2)
                # P(X1,...,Xn,C2)
                # Yields P(C2, x1,...,xn)
                forward_prob = (
                    np.prod(self.mu_cz[initial_c, t, :]) * self.transition[initial_c]  # * 1, which is P(C1=2)
                )
            else:
                # We take the product with np.prod
                # forward_prob is
                # The following computes Sum_C P(C_next | C)*P(X, C) = P(X, C_next)
                forward_prob = np.dot(
                    self.transition.T,
                    # P(X_prev, C) *  P(X1|C)P(X2|C)...P(Xn|C) = P(X, C)
                    forward_prob * np.prod(self.mu_cz[:, t, :], axis=1)
                )
            # P(C_next| X) = P(C_next,X)/Sum_C P(C_next,X)
            forward_prob = forward_prob/sum(forward_prob)
            mu_cc.append(forward_prob)
        self.mu_cc = mu_cc
        # We are now at the T'th node, this node is fully informed
        beta_cs[-1] = forward_prob * np.prod(self.mu_cz[:, -1, :], axis=1)
        beta_cs[-1] = beta_cs[-1]/sum(beta_cs[-1])
        # We proceed to do downward pass
        for t in range(T-1, -1, -1):
            self.mu_cc[t]

        # Should return list inferred of C probabilities and Z probabilities
        return

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

        backward_prob = np.ones(len(self.processing_modes))

        for t in range(time_steps - 1, timestep, -1):
            messages_from_x_and_z = self.compute_messages_from_x_and_z(t, num_nodes, observations)
            messages_from_z_and_c = self.compute_messages_from_z_and_c(
                messages_from_x_and_z, t
            )

            # Sum over C_prev for P(C|C_prev)P(X1,X2,...|C_prev)
            backward_prob = np.einsum(
                "i, ij, j -> j",
                backward_prob,
                self.transition,
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

        # P(X_{1,1},...,X_{t-1,n}, C_t) P(X_{} C_t)
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

    def learned_parameters(
            self, c_values: IntArray, z_values: IntArray, x_values: IntArray
    ) -> tuple[float, float, float, float, float]:
        """Learn parameters from observed C, Z and X values.

        Args:
            c_values (IntArray): Observed processing modes.
            z_values (IntArray): Observed focus.
            x_values (IntArray): Observed stimuli.

        Returns:
            tuple[float, float, float, float, float]:
                Tuple containing learned parameters:
                    - lambda_0_hat
                    - lambda_1_hat
                    - alpha_hat
                    - beta_hat
                    - gamma_hat
        """
        time_steps: int = len(c_values)

        # We are interested in these when computing lambda-hat values.
        z_0_mask = z_values == 0  # Indices where Z_{t,i} = 0
        z_1_mask = z_values == 1  # ...

        # Compute the lambdas as the average stimulis for respective Z-values.
        lambda_0_hat: float = x_values[z_0_mask].sum() / z_0_mask.sum()
        lambda_1_hat: float = x_values[z_1_mask].sum() / z_1_mask.sum()

        # NumPy trick to get sum of (Z_{t,i} = C_t = 0 or 1)
        alpha_mask = (
                (z_values == c_values[:, None]).any(axis=1) & (c_values <= 1)
        ).sum()

        alpha_count: int = alpha_mask.sum()
        alpha_hat: float = alpha_count / c_values.size

        # Used to count cases of beta and gamma transition cases.
        beta_count: int = 0
        gamma_count: int = 0

        # Trivial variable.
        total_transitions: int = time_steps - 1

        # Count cases of transitions.
        for t in range(total_transitions):
            # This is so nice.
            match (c_values[t], c_values[t + 1]):
                case (2, 0) | (2, 1):  # From 2 -> {0,1}
                    beta_count += 1
                case (0, 2) | (1, 2):  # From {0,1} -> 2
                    gamma_count += 1

        beta_hat: float = beta_count / total_transitions
        gamma_hat: float = gamma_count / total_transitions

        return (lambda_0_hat, lambda_1_hat, alpha_hat, beta_hat, gamma_hat)


def expectation_maximisation_hard_assignment(
        joint_prob: FloatArray, num_nodes: int
) -> tuple[IntArray, IntArray]:
    """Compute Z and C hard-assignments.

    Args:
        joint_prob (FloatArray): Infered normalised joint probabilities.
            You can get this from `HMM.infer`.

    Returns:
        tuple[IntArray, IntArray]: C a
    """
    # Recall dimensions of joint probability tensor:
    # ... (T-1, num possible Cs at each t, num possible Cs at t + 1)
    # Thus, time_steps here will be (T-1)
    time_steps: int = joint_prob.shape[0]

    # Preparation.
    z_hat = np.zeros((time_steps, num_nodes), dtype=int)
    c_hat = np.zeros(time_steps, dtype=int)

    for t in range(time_steps):
        # Star struck. This one is literally in the task description.
        c_hat[t] = np.argmax(np.sum(joint_prob[t], axis=1))

        for i in range(num_nodes):
            # Compute marginals, i.e. $P(Z | X, C)$.
            marginal_probs = np.zeros(2)
            for z in range(2):
                marginal_probs[z] = joint_prob[t, c_hat[t], z]

            # No way.
            z_hat[t, i] = np.argmax(marginal_probs)

    return z_hat, c_hat
