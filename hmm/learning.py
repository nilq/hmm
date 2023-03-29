import numpy as np

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
