"""Hidden Markov Model implementation."""

import numpy as np


def poisson_stimuli_sample_method(Z, rates):
    sample_rates = [rates[z] for z in Z]

    return np.random.poisson(sample_rates)


class HMM2:
    PROCESSING_STATES = [0, 1, 2]

    def __init__(self, Gamma, alpha, stimuli_sample_method):
        self.alpha = alpha
        self.Gamma = Gamma # Large Gamma
        self.stimuli_sample_method = stimuli_sample_method # Sampling from P(X_{t,i} = x | Z_{t,i} = z) poisson

    def sample_hidden_C(self, current_C): 
        return np.random.choice(HMM2.PROCESSING_STATES, p=self.Gamma[current_C])

    def sample_hidden_Z(self, n, current_C):
        if current_C == 0:
            p = 1-self.alpha
        elif current_C == 1:
            p = self.alpha
        elif current_C == 2:
            p = 0.5
        
        return np.random.binomial(n=1, p=p, size=n)

    def forward(self, n, t=100, initial_C=2):
        """
            simulate for n neurons and up to time t
        """
        current_C = initial_C
        observations = {}
        activations = np.array([])
        focus = np.array([])
        processing_modes = []
        for current_t in range(t):
            Z = self.sample_hidden_Z(n, current_C)
            X = self.stimuli_sample_method(Z)
            processing_modes.append(current_C)
            focus = np.vstack([focus, Z]) if focus.any() else Z
            activations = np.vstack([activations, X]) if activations.any() else X

            # observations[current_t] = {
            #     'C': current_C,
            #     'Z': Z,
            #     'X': X
            # }

            current_C = self.sample_hidden_C(current_C)
        return processing_modes, focus, activations
