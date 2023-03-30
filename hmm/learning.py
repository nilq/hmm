import numpy as np
from hmm.hmm import HMM

from hmm.types import IntArray, FloatArray


def learn_parameters_everything_observed(
        c_values: IntArray, z_values: IntArray, x_values: IntArray
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

    alpha_1 = z_values[c_values == 1].flatten()
    if len(alpha_1) == 0:
        alpha_0 = z_values[c_values == 0].flatten()
        alpha_hat = 1 - (sum(alpha_0) / len(alpha_0))
    else:
        alpha_hat: float = sum(alpha_1) / len(alpha_1)

    # Used to count cases of beta and gamma transition cases.
    beta_count: int = 0
    beta_total: int = 0
    gamma_count: int = 0
    gamma_total: int = 0

    # Trivial variable.
    total_transitions: int = time_steps - 1

    # Count cases of transitions.
    for t in range(total_transitions):
        # This is so nice.
        match (c_values[t], c_values[t + 1]):
            case (2, 1) | (2, 0):
                beta_count += 1
                beta_total += 1
            case (2, 2):
                beta_total += 1
            case (0, 2) | (1, 2):
                gamma_count += 1
                gamma_total += 1
            case (0, 0) | (1, 1):
                gamma_total += 1

    beta_hat: float = beta_count / beta_total
    gamma_hat: float = gamma_count / gamma_total

    return (lambda_0_hat, lambda_1_hat, alpha_hat, beta_hat, gamma_hat)


def hard_assignment_em(
    x_values: IntArray,
    hmm: HMM,
    max_iterations: int = 100,
) -> HMM:
    time_steps, num_nodes = x_values.shape

    for i in range(max_iterations):
        # E-step
        # c_marginals, z_marginals = hmm.nielslief_propagation(x_values)
        # c_marginals = [hmm.infer_marginal_c(x_values, t) for t in range(time_steps)]
        # z_marginals = [[hmm.infer_marginal_z(x_values, t, z) for z in range(8)] for t in range(time_steps)]
        # c_argmax = np.argmax(c_marginals, axis=1)        
        # z_argmax = [[z[0] > z[1] for z in z_marg] for z_marg in z_marginals]
        # z_argmax = np.array(z_argmax).astype(int)
        joint_probabilities_normalised = hmm.infer(x_values)
        # Compute the marginal probabilities of C at each time step
        marginal_prob_C = np.sum(joint_probabilities_normalised, axis=2)
        # print(joint_probabilities_normalised)
        # print(marginal_prob_C)
        # Calculate the estimated C at each time step
        c_argmax = np.argmax(marginal_prob_C, axis=1)
        # Compute the most likely Z given the estimated C
        z_argmax = np.zeros((c_argmax.shape[0], num_nodes), dtype=int)
        for t, c in enumerate(c_argmax):
            z_argmax[t] = hmm.sample_hidden_z(num_nodes, c)

        # M-step
        (
            lambda_0_hat,
            lambda_1_hat,
            learned_alpha,
            learned_beta,
            learned_gamma
        ) = learn_parameters_everything_observed(c_argmax, z_argmax, x_values[:-1])

        learned_rates = [lambda_0_hat, lambda_1_hat]

        learned_transition_matrix = np.array(
            [[1 - learned_gamma, 0, learned_gamma],
             [0, 1 - learned_gamma, learned_gamma],
             [learned_beta / 2, learned_beta / 2, 1 - learned_beta]]
        )

        diff_rates = abs(np.array(hmm.rates) - np.array(learned_rates)).sum()
        diff_transition = abs(hmm.transition - learned_transition_matrix).sum()
        diff_alpha = abs(hmm.alpha - learned_alpha)

        if (diff_rates**2 + diff_transition**2 + diff_alpha**2) < 1e-9:
            print(f"Found good after {i} iterations!")
            break

        hmm.alpha = learned_alpha
        hmm.transition = learned_transition_matrix
        hmm.rates = learned_rates

    return hmm
