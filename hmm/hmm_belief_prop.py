"""Hidden Markov Model implementation."""
import numpy as np

from .hmm import HMM


class HMM2(HMM):
    def __init__(
        self, *args, **kwargs
    ) -> None:
        """Initialise HMM.

        Args:
            transition (FloatArray): Transition probability matrix.
            alpha (list[float] | float): Alpha probabilities or probability.
            processing_modes (list[int]): States for HMM.
            rates (list[int]): Rates for Poisson sampling of stimuli.
        """
        super().__init__(*args, **kwargs)

    def infer_hidden_belief_propagation(self, observations, initial_c: int = 2):
        num_modes = len(self.processing_modes)
        T, n = observations.shape

        mu_c = np.zeros((T-1, num_modes))
        beta_c = np.zeros((T, num_modes))
        beta_z = np.zeros((T, n, 2))
        p_x_given_z, p_x_given_c = self.compute_messages_from_clique_zc_to_cc(observations)

        # Upward pass
        forward_prob = np.eye(len(self.processing_modes))[initial_c] # sigma^(t)(C_{t+1})

        c_t = np.zeros(T)
        for t in range(T-1):
            beta_c[t] = np.prod(p_x_given_c[:, t, :], axis=1) * forward_prob
            normalization = np.sum(beta_c[t])
            beta_c[t] /= normalization  # now is sigma^(t)(C_t), current belief state

            c_t[t] = normalization

            forward_prob = np.einsum(
                "i, ij -> j",
                beta_c[t],
                self.transition,
            )  # P(C_{t+1} | X^(1:t)) = sigma^(t)(C_{t+1})
            mu_c[t] = forward_prob  # message between CC and CX cliques

        beta_c[-1] = np.prod(p_x_given_c[:, -1, :], axis=1) * forward_prob
        normalization = np.sum(beta_c[-1])
        beta_c[-1] /= normalization  # sigma^(T)(C_T), last clique
        c_t[-1] = normalization

        # Downward pass
        downward_prob = beta_c[-1] # P(C_T | X^(1:T))
        for t in range(T-2, -1, -1):
            delta_t_bar = downward_prob/mu_c[t]  # P(C_{t+1} | X^(1:T))/sigma^(t)(C_{t+1})
            beta_c[t] = np.dot(self.transition, delta_t_bar) * beta_c[t]
            downward_prob = beta_c[t]

        # Downward pass to Z's
        for t in range(T):
            p_c_given_x = beta_c[t]
            for i in range(n):
                beta_z[t, i] = np.sum(
                    p_c_given_x/p_x_given_c[:, t, i] * p_x_given_z[:, t, (i,)] * self.p_z_given_c_mat,
                    axis=1
                )

        return beta_c, beta_z, c_t

