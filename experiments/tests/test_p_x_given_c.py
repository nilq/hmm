import numpy as np
from scipy.stats import poisson

from hmm.hmm import HMM
import numpy as np


def compute_messages_from_x_and_z(
        self, t: int, observations
):
    """Compute clique message of clique X-Z at timestep t.

    Args:
        t (int): Time step t.

    Returns:
        _type_: Clique message.
    """
    messages_from_x_and_z = []

    for z_t_i in range(observations.shape[1]):
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
    self, messages_from_x_and_z
):
    messages_from_z_and_c = []

    for c in self.processing_modes:
        # Snoop Dogg approved.
        joint_x_given_c = 1

        for message in messages_from_x_and_z:
            # P(X|C) = P(X | Z = 0)P(Z = 0 | C) + P(X | Z = 1)P(Z = 1 | C)
            probability_of_x_given_c = sum(
                message[z] * self.p_z_given_c(z, c) for z in range(2)
            )
            # P(X_1 | C)P(X_2 | C) = P(X_1, X_2 | C)
            joint_x_given_c *= probability_of_x_given_c

        messages_from_z_and_c.append(joint_x_given_c)

    return np.array(messages_from_z_and_c)


def compute_messages_from_clique_zc_to_cc(self, observations):
    # P(X|Z)
    p_x_given_z = np.array(
        [
            poisson.pmf(observations, self.rates[1]),
            poisson.pmf(observations, self.rates[0]),
        ]
    )

    # return np.array([p_x_given_z[0] * self.p_z_given_c_mat[0, c] + p_x_given_z[1] * self.p_z_given_c_mat[1, c] for c in [0, 1, 2]])
    return np.einsum("ijk, il -> ljk", p_x_given_z, self.p_z_given_c_mat)


gamma = 0.5
beta = 0.8
alpha = 0.9
rates = [1, 20]

# This is uppercase-gamma.
transition_matrix = np.array(
    [[1 - gamma, 0, gamma], [0, 1 - gamma, gamma], [beta / 2, beta / 2, 1 - beta]]
)
hmm = HMM(
    transition_matrix,
    alpha,
    rates=rates,
    processing_modes=[0, 1, 2],
)

n = 5
t = 10
processing_modes, focus, activations = hmm.forward(n, t)


def test_p_x_given_c():
    probs_old_method = []
    for t_ in range(t):
        messages_from_x_and_z = compute_messages_from_x_and_z(hmm, t_, activations)
        probs_x_given_c = compute_messages_from_z_and_c(hmm, messages_from_x_and_z)
        probs_old_method.append(probs_x_given_c)

    probs_einsum_method = compute_messages_from_clique_zc_to_cc(hmm, activations)
    assert np.all(np.prod(probs_einsum_method, axis=2).T == np.array(probs_old_method))


def infer_all_c_and_z_values(hmm: HMM, observations, initial_c: int = 2):
    time_steps, num_nodes = observations.shape
    num_processing_modes = len(hmm.processing_modes)

    c_marginals = []
    z_marginals = []

    for timestep in range(time_steps):
        forward_prob, backward_prob = hmm.clique_tree_forward_backward(observations, timestep, initial_c)

        # At time t of c_t.
        messages_from_x_and_z = hmm.compute_messages_from_x_and_z(timestep, num_nodes, observations)
        messages_from_z_and_c = hmm.compute_messages_from_z_and_c(messages_from_x_and_z, timestep)

        # P(C_t, X_1,...,X_{t-1}) P(X_T,...,X_{t+1}| C_t) P(X_t|C_t) = P(C_t, X^(1:T))
        joint_with_evidence = forward_prob * backward_prob * messages_from_z_and_c

        c_marginals.append(joint_with_evidence / np.sum(joint_with_evidence))

        z_marginals_t = []
        for z_index in range(num_nodes):
            z_given_x = hmm.infer_marginal_z(observations, timestep, z_index, initial_c)
            z_marginals_t.append(z_given_x)

        z_marginals.append(z_marginals_t)

    return np.array(c_marginals), np.array(z_marginals)

test_p_x_given_c()

print(infer_all_c_and_z_values(hmm, activations))