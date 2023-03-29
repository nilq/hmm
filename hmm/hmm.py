"""Hidden Markov Model implementation."""

import numpy as np
import numpy.typing as npt

from hmm.types import FloatArray, IntArray

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



    def nielslief_propagation(
        self, observations: IntArray, initial_c: int = 2
    ) -> tuple[FloatArray, FloatArray]:
        """Use message passing to compute marginal C and Z propabilities given observed X.

        Args:
            observations (IntArray): Observed stimuli X.
            initial_c (int, optional): Initial C value. Defaults to 2.

        Returns:
            tuple[FloatArray, FloatArray]: Marginals for C and Z.
        """
        time_steps, num_nodes = observations.shape
        num_states: int = len(self.processing_modes)

        # Universal message from Z-C cliques to C-C cliques.
        mu_cz = self.compute_messages_from_clique_zc_to_cc(observations)

        # Messages between C-C cliques.
        # TODO: Contains forward probs (i.e. CC message) for each t. (double check this)
        mu_cc = np.zeros([time_steps - 1, num_states])
        beta_cs = np.zeros([time_steps, num_states])  # This is our final beliefs to be updated.

        # Forward pass.
        forward_prob = np.ones(num_states)

        # Informing nodes up until the last node, which will be handled after.
        for t in range(time_steps - 1):
            if t == 0:
                # P(X1|C1=2)P(X2|C1=2)...P(Xn|C1=2) P(C2|C1=2)P(C1=2)
                # Yields P(C2, x1,...,xn)
                forward_prob = (
                    np.prod(mu_cz[initial_c, t, :])
                    * self.transition[initial_c]  # * 1, which is P(C1=2)
                )
            else:
                # P(C | X_prev)P(X1|C)P(X2|C)...P(Xn|C) = P(C|X_prev)P(X| C)=P(X,C|X_prev)
                beta_cs[t] = forward_prob * np.prod(mu_cz[:, t, :], axis=1)
                beta_cs[t] /= beta_cs[t].sum()
                # We take the product with np.prod.
                # The following computes
                # P(C_next, X|X_prev) = Sum_C P(C_next| C)*P(X,C|X_prev)
                # (by cond indep) = Sum_C P(C_next|X,C)*P(X,C|X_prev)
                # = Sum_C P(C_next, X, C|X_prev)
                forward_prob = np.dot(self.transition.T, beta_cs[t])
                forward_prob /= forward_prob.sum()

            mu_cc[t] = forward_prob

        # Mesages between Cs.

        # P(X_T,C_T|X_prev) = P(C_T|X_prev) P(X_T|C_T)
        sigma = beta_cs[-1] = forward_prob * np.prod(mu_cz[:, -1, :], axis=1)
        # P(C_T|X_all)
        beta_cs[-1] = beta_cs[-1] / sum(beta_cs[-1])

        # Backward pass.
        backward_prob = np.ones(num_states)

        for t in range(time_steps - 2, -1, -1):
            # Sum_C_T P(C_T|C_T-1)P(X_T|C_T) = P(X_T|C_T-1)
            backward_prob = np.dot(
                self.transition, backward_prob * sigma
            )

            backward_prob /= backward_prob.sum()
            beta_cs[t] *= backward_prob
            beta_cs[t] /= beta_cs[t].sum()

            sigma = beta_cs[t]

        z_marginals = np.zeros((time_steps, num_nodes, 2))
        
        # Compute marginal joint distribution for Z
        for t in range(time_steps):
            for i in range(num_nodes):
                for c in range(num_states):
                    z_marginals[t, i, 1] += beta_cs[t, c] * mu_cz[c, t, i]
                z_marginals[t, i, 0] = 1 - z_marginals[t, i, 1]

        return beta_cs, z_marginals

    def belief_propagation(self, observations, initial_c: int = 2):
        T = len(observations)
        num_processing_modes = len(self.processing_modes)

        # Pass up from zc to cc
        mu_cz = self.compute_messages_from_clique_zc_to_cc(observations)
        mu_cc = []
        beta_cs = np.zeros((T, num_processing_modes))

        # Forward pass
        forward_prob = np.ones(num_processing_modes)
        for t in range(T - 1):
            if t == 0:
                # P(X1|C1=2)P(X2|C1=2)...P(Xn|C1=2) P(C2|C1=2)P(C1=2)
                # Yields P(C2, x1,...,xn)
                forward_prob = (
                    np.prod(mu_cz[initial_c, t, :])
                    * self.transition[initial_c]  # * 1, which is P(C1=2)
                )
            else:
                # P(C | X_prev)P(X1|C)P(X2|C)...P(Xn|C) = P(C|X_prev)P(X| C)=P(X,C|X_prev)
                beta_cs[t] = forward_prob * np.prod(mu_cz[:, t, :], axis=1)
                # We take the product with np.prod
                # forward_prob is
                # The following computes
                # P(C_next, X|X_prev) = Sum_C P(C_next| C)*P(X,C|X_prev)
                # (by cond indep) = Sum_C P(C_next|X,C)*P(X,C|X_prev)
                # = Sum_C P(C_next, X, C|X_prev)
                forward_prob = np.dot(self.transition.T, beta_cs[t])
                # Get P(C|X)
                beta_cs[t] /= sum(beta_cs[t])
            # P(C_next| X_till_now) = P(C_next, X|X_prev)/Sum_C P(C_next, X|X_prev)
            # TODO: Double check.
            forward_prob = forward_prob / sum(forward_prob)
            mu_cc.append(forward_prob)

        # Messages between Cs.
        self.mu_cc = mu_cc
        # We are now at the T'th node, this node is fully informed
        # P(X_T,C_T|X_prev) = P(C_T|X_prev) P(X_T|C_T)
        beta_cs[-1] = sigma = forward_prob * np.prod(mu_cz[:, -1, :], axis=1)
        # P(C_T|X_all)
        beta_cs[-1] = beta_cs[-1] / sum(beta_cs[-1])
        # We proceed to do downward pass
        downward_prob = np.ones(num_processing_modes)
        for t in range(T - 2, -1, -1):
            # Sum_C_T P(C_T|C_T-1)P(X_T|C_T) = P(X_T|C_T-1)
            a = np.dot(self.transition, sigma / self.mu_cc[t])
            downward_prob = a * beta_cs[t]

        # Another one. Zs.

        # Should return list inferred of C probabilities and Z probabilities
        return beta_cs

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
            messages_from_x_and_z = self.compute_messages_from_x_and_z(
                t, num_nodes, observations
            )
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
        messages_from_x_and_z = self.compute_messages_from_x_and_z(
            time_of_c, num_nodes, observations
        )
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

        messages_from_x_and_z = self.compute_messages_from_x_and_z(
            timestep, num_nodes, observations
        )
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
            [
                poisson.pmf(observations[timestep][z_index], self.rates[z])
                for z in (0, 1)
            ]
        )

        z_given_x = poisson_probs * np.einsum("ij,j->i", z_given_c, joint_with_evidence)

        return z_given_x / np.sum(z_given_x)


