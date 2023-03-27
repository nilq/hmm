"""Hidden Markov Model implementation."""

import numpy as np
import numpy.typing as npt

from hmm.types import FloatArray, IntArray

from itertools import product
from typing import Callable

from scipy.special import factorial
from scipy.stats import poisson


def sample_poisson_stimuli(z_values: IntArray, rates: IntArray) -> IntArray:
    """Sample Poisson stimuli.

    Args:
        z_values (IntArray): List of Z-values.
        rates (IntArray): Rate-lookup indexed by Z-values.

    Returns:
        IntArray: Drawn samples from the parameterized Poisson distribution.
    """
    sample_rates = [rates[z] for z in z_values]
    return np.random.poisson(sample_rates)


class HMM:
    def __init__(
            self,
            transition: FloatArray,
            alpha: float,
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

    def p_z_given_c(self, z, c):
        match c:
            case 0:
                p = 1 - self.alpha
            case 1:
                p = self.alpha
            case 2:
                p = 0.5
        return p if z else 1 - p

    def p_c_given_c(self, c_next, current_c):
        return self.transition[current_c][c_next]

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

    def emission_probabilities(self, observation: int, current_c: int) -> float:
        """Compute probability of observing observation given current processing mode.

        Notes:
            This is the emission probability computed as follows:
                P(X=x|C=c) = p⋅P(X=x|Z=1)+(1-p)⋅P(X=x|Z=0)

        Args:
            observation (int): _description_
            current_c (int): _description_

        Returns:
            float: _description_
        """

        # Emission probabilities for C. Super hardcoded.
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
            emission_probs[z] = sum((np.exp(-rate) * (rate ** observation)) / factorial(observation))

        # Weighted probability based on the relationship between C and Z
        return p * emission_probs[1] + (1 - p) * emission_probs[0]

    def forward_pass(self, observations: IntArray) -> tuple[FloatArray, FloatArray]:
        """Forward pass of forward-backward algorithm.

        Notes:
            The forward pass computes the scaled forward probabilites
                for each state at each time step.

        Args:
            observations (IntArray): Array of observations.

        Returns:
            tuple[FloatArray, FloatArray]: Forward probabilities and scaling factors.
                - forward_prob: A 2D array containing the scaled
                    forward probabilities for each state at each time step.
                - scaling_factors: A 1D array containing the scaling factors used
                    to normalize the forward probabilities at each time step.
        """
        time_steps: int = len(observations)
        num_states: int = len(self.states)

        forward_prob = np.zeros([time_steps, num_states])
        scaling_factors = np.zeros(time_steps)

        for t in range(time_steps):
            for i in range(num_states):
                # Emission probability of observation at time step t given mode i.
                emission_prob = self.emission_probabilities(observations[t], i)

                # If it's the first time step, we initialise the forward
                # ... probabilities using the alpha value.
                if t == 0:
                    forward_prob[t, i] = self.alpha * emission_prob
                else:
                    # For all other steps, compute the forward probability.
                    forward_prob[t, i] = (
                            np.dot(forward_prob[t - 1], self.transition[:, i])
                            * emission_prob
                    )

            # Scaling factor at t is the sum of the forward probabilities at t.
            scaling_factors[t] = np.sum(forward_prob[t])
            # Normalise the forward probabilities at time step t.
            forward_prob[t] /= scaling_factors[t]

        return forward_prob, scaling_factors  # Au revoir.

    def backward_pass(
            self, observations: IntArray, scaling_factors: FloatArray
    ) -> FloatArray:
        """Backward pass of forward-backward algorithm.

        Notes:
            The backward pass computes the scaled backward probabilities
                for each state at each time step.

        Args:
            observations (IntArray): Array of observations.
            scaling_factors (FloatArray): Scaling factors from forward-pass.

        Returns:
            FloatArray: A 2D array containing the scaled backward probabilities
                for each state at each time step.
        """
        time_steps: int = len(observations)
        num_states: int = len(self.states)

        backward_prob = np.zeros([time_steps, num_states])

        # Base case.
        backward_prob[-1] = 1

        # Moonwalk across observed time steps, starting form second-to-last.
        for t in range(time_steps - 2, -1, -1):
            # For each processing mode (i.e. state) we calculate the backward
            # ... probability of state i at time step t.
            for i in range(num_states):
                # Emission probability of observation at time (t + 1)
                # ... given processing mode i.
                emission_prob = self.emission_probabilities(observations[t + 1], i)
                backward_prob[t, i] = np.dot(
                    self.transition[i],  # Transition probability from i to all states.
                    emission_prob * backward_prob[t + 1]  # Element-wise product. :) 
                )

            # Normalise backward probabilities.
            backward_prob[t] /= scaling_factors[t + 1]

        return backward_prob  # See you later. 

    def infer(self, observations: IntArray) -> FloatArray:
        """Infer the joint probabilities of processing modes for observed time steps.

        Args:
            observations (IntArray): Array of observations.

        Returns:
            FloatArray: A 3D array containing the joint probabilities for each tuple
                of states at consecutive time steps.
        """
        forward_prob, scaling_factors = self.forward_pass(observations)
        backward_prob = self.backward_pass(observations, scaling_factors)

        time_steps, num_states = forward_prob.shape
        joint_prob = np.zeros((time_steps - 1, num_states, num_states))

        # For each time step, we compute a the probabilities of transitioning
        # from state i at time t, to state j at time $t + 1$.
        for t in range(time_steps - 1):
            # Iterate over all possible pairs of processing modes.
            for i, j in product(range(num_states), range(num_states)):
                # Calculate the emission probability of observing stimulus
                # ... at (t + 1) given mode $j$.
                emission_prob = self.emission_probabilities(observations[t + 1], j)
                joint_prob[t, i, j] = (
                        forward_prob[t, i]  # Forward probability of state i at time t.
                        * self.transition[i, j]  # Transition probability from i to j.
                        * emission_prob  # Guess what.
                        * backward_prob[t + 1, j]  # Backward probability at j at (t + 1)
                )

        # Ensure row stochasticity. 
        joint_prob /= np.sum(joint_prob, axis=(1, 2), keepdims=True)

        return joint_prob  # Bye.

    def infer_C(self, time, observations):
        time_steps: int = len(observations)
        num_nodes: int = len(observations[0])

        forward_pass = np.array([1, 1, 1])
        for i in range(time):
            message_from_clique_xz = []

            for z_i in range(num_nodes):
                message_from_clique_xz.append([poisson.pmf(observations[i][z_i], self.rates[0]),  # P(X|Z=0)
                                               poisson.pmf(observations[i][z_i], self.rates[1])])  # P(X|Z=1)

            # Here we sum P(X_i|Z_i)P(Z_i|C) over Z_i, then take the product the sums over i < num_states

            message_from_clique_zc = []
            for c in (0, 1, 2):
                joint_x_given_c = 1
                for pxz_i in message_from_clique_xz:
                    # P(X|C) =P(X| Z = 0)P(Z = 0| C)+P(X| Z = 1)P(Z = 1| C)
                    p_x_given_c = pxz_i[0] * self.p_z_given_c(0, c) + pxz_i[1] * self.p_z_given_c(1, c)
                    joint_x_given_c *= p_x_given_c  # P(X1|C)P(X2|C)=P(X1,X2|C)
                message_from_clique_zc.append(joint_x_given_c)

            if i == 0:
                # Sum over C1, which can only be 2,
                # P(C1)P(X1, X2, ..| C1)P(C2 | C1) = P(C2, X1, X2, ..)
                forward_pass = message_from_clique_zc[2] * self.transition[2]
            else:
                # Sum over C2
                # P(X11,..., C)P(X21, X22, ..| C2)P(C3 | C2) = P(C3, X1, X2, ..)
                forward_pass = [
                    sum([forward_pass[i] * message_from_clique_zc[i] * self.p_c_given_c(c_, i) for i in (0, 1, 2)])
                    for c_ in (0, 1, 2)]

        #### BACKWARD PASS
        backward_pass = np.array([1, 1, 1])
        for i in range(time_steps - 1, time, -1):
            message_from_clique_xz = []

            for z_i in range(num_nodes):
                message_from_clique_xz.append([poisson.pmf(observations[i][z_i], self.rates[0]),  # P(X|Z=0)
                                               poisson.pmf(observations[i][z_i], self.rates[1])])  # P(X|Z=1)

            message_from_clique_zc = []
            for c in (0, 1, 2):
                joint_x_given_c = 1
                for pxz_i in message_from_clique_xz:
                    # P(X|C) =P(X| Z = 0)P(Z = 0| C)+P(X| Z = 1)P(Z = 1| C)
                    p_x_given_c = pxz_i[0] * self.p_z_given_c(0, c) + pxz_i[1] * self.p_z_given_c(1, c)
                    joint_x_given_c *= p_x_given_c  # P(X1|C)P(X2|C)=P(X1,X2|C)
                message_from_clique_zc.append(joint_x_given_c)  # P(X1,X2|C = 0,1,2)

            backward_pass = [
                # Sum over C_prev for P(C|C_prev)P(X1,X2,...|C_prev)
                sum([backward_pass[i] * self.p_c_given_c(c_, i) for i in (0, 1, 2)]) * message_from_clique_zc[c_]
                for c_ in (0, 1, 2)]

        # P(X_{1,1},...,X_{t-1,n}, C_t) P(X_{} C_t)

        ### At time t
        message_from_clique_xz = []

        for z_i in range(num_nodes):
            message_from_clique_xz.append([poisson.pmf(observations[time][z_i], self.rates[0]),  # P(X|Z=0)
                                           poisson.pmf(observations[time][z_i], self.rates[1])])  # P(X|Z=1)

        # Here we sum P(X_i|Z_i)P(Z_i|C) over Z_i, then take the product the sums over i < num_states
        message_from_clique_zc = []
        for c in (0, 1, 2):
            joint_x_given_c = 1
            for pxz_i in message_from_clique_xz:
                # P(X|C) =P(X| Z = 0)P(Z = 0| C)+P(X| Z = 1)P(Z = 1| C)
                p_x_given_c = pxz_i[0] * self.p_z_given_c(0, c) + pxz_i[1] * self.p_z_given_c(1, c)
                joint_x_given_c *= p_x_given_c  # P(X1|C)P(X2|C)=P(X1,X2|C)
            message_from_clique_zc.append(joint_x_given_c)

        joint_with_evidence = np.array(forward_pass) * np.array(backward_pass) * np.array(message_from_clique_zc)
        marginal = joint_with_evidence / np.sum(joint_with_evidence)
        return marginal

    def infer_Z(self, time, index, observations):
        time_steps: int = len(observations)
        num_nodes: int = len(observations[0])

        forward_pass = np.array([1, 1, 1])
        for i in range(time):
            message_from_clique_xz = []

            for z_i in range(num_nodes):
                message_from_clique_xz.append([poisson.pmf(observations[i][z_i], self.rates[0]),  # P(X|Z=0)
                                               poisson.pmf(observations[i][z_i], self.rates[1])])  # P(X|Z=1)

            # Here we sum P(X_i|Z_i)P(Z_i|C) over Z_i, then take the product the sums over i < num_states

            message_from_clique_zc = []
            for c in (0, 1, 2):
                joint_x_given_c = 1
                for pxz_i in message_from_clique_xz:
                    # P(X|C) =P(X| Z = 0)P(Z = 0| C)+P(X| Z = 1)P(Z = 1| C)
                    p_x_given_c = pxz_i[0] * self.p_z_given_c(0, c) + pxz_i[1] * self.p_z_given_c(1, c)
                    joint_x_given_c *= p_x_given_c  # P(X1|C)P(X2|C)=P(X1,X2|C)
                message_from_clique_zc.append(joint_x_given_c)

            if i == 0:
                # Sum over C1, which can only be 2,
                # P(C1)P(X1, X2, ..| C1)P(C2 | C1) = P(C2, X1, X2, ..)
                forward_pass = message_from_clique_zc[2] * self.transition[2]
            else:
                # Sum over C2
                # P(X11,..., C)P(X21, X22, ..| C2)P(C3 | C2) = P(C3, X1, X2, ..)
                forward_pass = [
                    sum([forward_pass[i] * message_from_clique_zc[i] * self.p_c_given_c(c_, i) for i in (0, 1, 2)])
                    for c_ in (0, 1, 2)]

        #### BACKWARD PASS
        backward_pass = np.array([1, 1, 1])
        for i in range(time_steps - 1, time, -1):
            message_from_clique_xz = []

            for z_i in range(num_nodes):
                message_from_clique_xz.append([poisson.pmf(observations[i][z_i], self.rates[0]),  # P(X|Z=0)
                                               poisson.pmf(observations[i][z_i], self.rates[1])])  # P(X|Z=1)

            message_from_clique_zc = []
            for c in (0, 1, 2):
                joint_x_given_c = 1
                for pxz_i in message_from_clique_xz:
                    # P(X|C) =P(X| Z = 0)P(Z = 0| C)+P(X| Z = 1)P(Z = 1| C)
                    p_x_given_c = pxz_i[0] * self.p_z_given_c(0, c) + pxz_i[1] * self.p_z_given_c(1, c)
                    joint_x_given_c *= p_x_given_c  # P(X1|C)P(X2|C)=P(X1,X2|C)
                message_from_clique_zc.append(joint_x_given_c)  # P(X1,X2|C = 0,1,2)

            backward_pass = [
                # Sum over C_prev for P(C|C_prev)P(X1,X2,...|C_prev)
                sum([backward_pass[i] * self.p_c_given_c(c_, i) for i in (0, 1, 2)]) * message_from_clique_zc[c_]
                for c_ in (0, 1, 2)]

        # P(X_{1,1},...,X_{t-1,n}, C_t) P(X_{} C_t)

        ### At time t
        message_from_clique_xz = []

        for z_i in range(num_nodes):
            message_from_clique_xz.append([poisson.pmf(observations[time][z_i], self.rates[0]),  # P(X|Z=0)
                                           poisson.pmf(observations[time][z_i], self.rates[1])])  # P(X|Z=1)

        # Here we sum P(X_i|Z_i)P(Z_i|C) over Z_i, then take the product the sums over i < num_states
        message_from_clique_zc = []
        for c in (0, 1, 2):
            joint_x_given_c = 1
            for pxz_i in message_from_clique_xz[:index] + message_from_clique_xz[index:]:
                # calculates P(X|C) =P(X| Z = 0)P(Z = 0| C)+P(X| Z = 1)P(Z = 1| C)
                p_x_given_c = pxz_i[0] * self.p_z_given_c(0, c) + pxz_i[1] * self.p_z_given_c(1, c)
                joint_x_given_c *= p_x_given_c  # P(X1|C)P(X2|C)=P(X1,X2|C) all except X_index
            message_from_clique_zc.append(joint_x_given_c)

        # P(Ct, X_all_except_index)= P(Ct, X_before_t)P(X_after_t|Ct)P(X_under_Ct|Ct)
        joint_with_evidence = np.array(forward_pass) * np.array(backward_pass) * np.array(message_from_clique_zc)

        z_given_x = [poisson.pmf(observations[time][index], self.rates[z]) *
                     sum([joint_with_evidence[c] * self.p_z_given_c(z, c) for c in (0, 1, 2)]) for z in
                     (0, 1)]
        marginal = z_given_x / np.sum(z_given_x)
        return marginal

    def learned_parameters(
            self,
            c_values: IntArray,
            z_values: IntArray,
            x_values: IntArray
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
        alpha_mask = ((z_values == c_values[:, None]).any(axis=1) & (c_values <= 1)).sum()

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

        return (
            lambda_0_hat,
            lambda_1_hat,
            alpha_hat,
            beta_hat,
            gamma_hat
        )


def expectation_maximisation_hard_assignment(joint_prob: FloatArray, num_nodes: int) -> tuple[IntArray, IntArray]:
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
