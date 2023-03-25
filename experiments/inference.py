""" Hidden markov model inference. """
from scipy.stats import poisson
from hmm.hmm import *
import numpy as np


def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    """Forward-backward algorithm."""
    # Forward part of the algorithm
    fwd = []
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k] * trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)

    # Backward part of the algorithm
    bkw = []
    for i, observation_i_plus in enumerate(reversed(observations[1:] + (None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(
                    trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l]
                    for l in states
                )

        bkw.insert(0, b_curr)

    p_bkw = sum(
        start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states
    )

    # Merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    assert p_fwd == p_bkw
    return fwd, bkw, posterior


gamma = 0.1
beta = 0.2
alpha = 0.5
lambda_0 = 1
lambda_1 = 5
rates = [lambda_0, lambda_1]

Gamma = np.array(
    [[1 - gamma, 0, gamma], [0, 1 - gamma, gamma], [beta / 2, beta / 2, 1 - beta]]
)

# simulate
hmm = HMM(Gamma, alpha, lambda Z: sample_poisson_stimuli(Z, rates), states=[0, 1, 2])
n = 10
t = 100
C_true, z, X_generated = hmm.forward(n, t)


start_prob = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
trans_prob = {
    0: {0: Gamma[0, 0], 1: Gamma[0, 1], 2: Gamma[0, 2]},
    1: {0: Gamma[1, 0], 1: Gamma[1, 1], 2: Gamma[1, 2]},
    2: {0: Gamma[2, 0], 1: Gamma[2, 1], 2: Gamma[2, 2]},
}

# Define the range of spike counts you want to consider, e.g., 0 to 10
spike_counts = list(range(n + 1))

# Calculate Poisson probabilities for each spike count and each lambda
poisson_probs_0 = [poisson.pmf(x, lambda_0) for x in spike_counts]
poisson_probs_1 = [poisson.pmf(x, lambda_1) for x in spike_counts]

# Normalize the probabilities to ensure they sum to 1
poisson_probs_0 = np.array(poisson_probs_0) / sum(poisson_probs_0)
poisson_probs_1 = np.array(poisson_probs_1) / sum(poisson_probs_1)

# Create the emm_prob dictionary
emm_prob = {
    0: {x: poisson_probs_0[i] for i, x in enumerate(spike_counts)},
    1: {x: poisson_probs_1[i] for i, x in enumerate(spike_counts)},
}


fwd, bkw, posterior = fwd_bkw(
    X_generated, [0, 1, 2], start_prob, trans_prob, emm_prob, "e"
)
