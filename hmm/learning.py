import numpy as np

from hmm.hmm_belief_prop import HMM2
from hmm.types import IntArray


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

    alpha_0 = z_values[c_values == 0].flatten()
    alpha_1 = z_values[c_values == 1].flatten()
    alpha_hat0 = 1 - (sum(alpha_0) / alpha_0.size) if alpha_0.size else 0
    alpha_hat1 = (sum(alpha_1) / alpha_1.size) if alpha_1.size else 0

    alpha_hat = (alpha_hat1 + alpha_hat0) / sum((alpha_hat1 != 0, alpha_hat0 != 0))
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

    return lambda_0_hat, lambda_1_hat, alpha_hat, beta_hat, gamma_hat


def hard_assignment_em(
    x_values: IntArray,
    max_iterations: int = 100,
    initial_gamma=0.5,
    initial_alpha=0.51,
    initial_beta=0.8,
    initial_rates=(1, 5)
):
    # initial values
    transition_matrix = make_transition_matrix_from_values(initial_gamma, initial_beta)

    learned_hmm = HMM2(transition_matrix, initial_alpha, rates=initial_rates, processing_modes=[0, 1, 2])
    print('Learning with initial parameters:')
    print('Initial rates:', initial_rates)
    print('Initial gamma:', initial_gamma)
    print('Initial beta:', initial_beta)
    print('Initial alpha:', initial_alpha)

    small_diff_count = 0
    learned_parameters_iterations = [[initial_gamma, initial_beta, initial_alpha, initial_rates[0], initial_rates[1]]]
    for i in range(max_iterations):
        print(f"Learning iteration {i}...")
        # E-step
        c_marginals, z_marginals, x_probs = learned_hmm.infer_hidden_belief_propagation(x_values)
        c_argmax = np.argmax(np.log(c_marginals), axis=1)
        z_argmax = np.argmax(np.log(z_marginals), axis=2)

        # M-step
        (
            lambda_0_hat,
            lambda_1_hat,
            learned_alpha,
            learned_beta,
            learned_gamma
        ) = learn_parameters_everything_observed(c_argmax, z_argmax, x_values)
        learned_parameters_iterations.append([learned_gamma, learned_beta, learned_alpha, lambda_0_hat, lambda_1_hat])
        learned_rates = [lambda_0_hat, lambda_1_hat]

        print('Learned rates:', learned_rates)
        print('Learned gamma:', learned_gamma)
        print('Learned beta:', learned_beta)
        print('Learned alpha:', learned_alpha)
        print('X probs:', np.prod(x_probs))
        learned_transition_matrix = make_transition_matrix_from_values(learned_gamma, learned_beta)

        diff_rates = abs(np.array(learned_hmm.rates) - np.array(learned_rates)).sum()
        diff_transition = abs(learned_hmm.transition - learned_transition_matrix).sum()
        diff_alpha = abs(learned_hmm.alpha - learned_alpha)

        if (diff_rates**2 + diff_transition**2 + diff_alpha**2) < 1e-29:
            small_diff_count += 1
            if small_diff_count > 3:
                print(f"Found good after {i} iterations!")
                break
        else:
            small_diff_count = 0

        learned_hmm = HMM2(learned_transition_matrix, learned_alpha, rates=learned_rates, processing_modes=[0, 1, 2])

    return (learned_gamma, learned_beta, learned_alpha, *learned_rates), np.array(learned_parameters_iterations)


def make_transition_matrix_from_values(gamma, beta):
    return np.array(
        [[1 - gamma, 0, gamma],
         [0, 1 - gamma, gamma],
         [beta / 2, beta / 2, 1 - beta]]
    )