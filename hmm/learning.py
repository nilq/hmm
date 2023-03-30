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
    alpha_hat: float = sum(alpha_1) / len(alpha_1)

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


def hard_assignment_em(
    x_values: IntArray,
    observed_focus: IntArray,
    hmm: HMM,
    max_iterations: int = 100,
) -> HMM:
    time_steps, num_nodes = x_values.shape

    for i in range(max_iterations):
        # E-step
        c_marginals, _ = hmm.nielslief_propagation(x_values)
        c_argmax = np.argmax(c_marginals, axis=1)
        
        # M-step
        (
            lambda_0_hat,
            lambda_1_hat,
            learned_alpha,
            learned_beta,
            learned_gamma
        ) = learn_parameters_everything_observed(c_argmax, observed_focus, x_values)

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