from hmm.hmm import HMM 
import numpy as np
from scipy.stats import poisson

import numpy as np

def forward_algorithm(init_probs, trans_probs, emission_probs, observations):
    T = len(observations)
    N = len(init_probs)
    alpha = np.zeros((T, N))

    alpha[0, :] = init_probs * emission_probs[:, observations[0]]

    for t in range(1, T):
        alpha[t, :] = np.dot(alpha[t-1, :], trans_probs) * emission_probs[:, observations[t]]

    return alpha

def backward_algorithm(trans_probs, emission_probs, observations):
    T = len(observations)
    N = trans_probs.shape[0]
    beta = np.zeros((T, N))

    beta[-1, :] = 1

    for t in range(T-2, -1, -1):
        beta[t, :] = np.dot(trans_probs, emission_probs[:, observations[t+1]] * beta[t+1, :])

    return beta

def hmm_inference(init_probs, trans_probs, emission_probs, observations):
    alpha = forward_algorithm(init_probs, trans_probs, emission_probs, observations)
    beta = backward_algorithm(trans_probs, emission_probs, observations)

    return alpha, beta


gamma = 0.1
beta = 0.2
alpha = 0.6
rates = [1, 5]

# This is uppercase-gamma.
transition_matrix = np.array(
    [[1 - gamma, 0, gamma], [0, 1 - gamma, gamma], [beta / 2, beta / 2, 1 - beta]]
)

num_nodes = 8
time_steps = 1000
initial_c = 2

hmm = HMM(transition_matrix, alpha, processing_modes=[0, 1, 2], rates=rates)

observed_processing_modes, observed_focus, observed_stimuli = hmm.forward(
    num_nodes,
    time_steps,
    initial_c,
)

# Example HMM parameters
init_probs = np.array([1/3, 1/3, 1/3]) # Initial state probabilities (2x1)
trans_probs = transition_matrix # Transition probabilities (2x2)
emission_probs = np.array(
            [
                poisson.pmf(observed_stimuli, hmm.rates[1]),
                poisson.pmf(observed_stimuli, hmm.rates[0]),
            ]
        )

# Example observation sequence (replace this with your actual observations)
observations = [0, 1, 2, 1]

# Run inference
alpha, beta = hmm_inference(init_probs, trans_probs, emission_probs, observed_stimuli)

# Compute the marginal probabilities of each state at each time step
marginal_probs = alpha * beta
marginal_probs /= marginal_probs.sum(axis=1, keepdims=True)

print("Forward probabilities (alpha):")
print(alpha)

print("\nBackward probabilities (beta):")
print(beta)

print("\nMarginal probabilities:")
print(marginal_probs)
