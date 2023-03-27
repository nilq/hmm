import numpy as np

from hmm.hmm import HMM, sample_poisson_stimuli

gamma = 0.1
beta = 0.2
alpha = 0.9
rates = [1, 20]

# This is uppercase-gamma.
transition_matrix = np.array(
    [[1 - gamma, 0, gamma], [0, 1 - gamma, gamma], [beta / 2, beta / 2, 1 - beta]]
)
#%%
hmm = HMM(transition_matrix, alpha, lambda z: sample_poisson_stimuli(z, rates), rates=rates, states=[0, 1, 2])

n = 5
t = 3

c, z, x = hmm.forward(n, t)
print(c, z, x)

hmm.infer_Z(1, 1, x)



