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
