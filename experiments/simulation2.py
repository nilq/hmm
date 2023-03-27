import numpy as np

from hmm.hmm import HMM, sample_poisson_stimuli

gamma = 0.5
beta = 0.8
alpha = 0.9
rates = [1, 20]

# This is uppercase-gamma.
transition_matrix = np.array(
    [[1 - gamma, 0, gamma], [0, 1 - gamma, gamma], [beta / 2, beta / 2, 1 - beta]]
)
#%%
hmm = HMM(transition_matrix, alpha, lambda z: sample_poisson_stimuli(z, rates), rates=rates, states=[0, 1, 2])

n = 5
t = 10

c, z, x = hmm.forward(n, t)
print(c, z, x)


# Infer C for every time step
for i in range(t):
    marginal = hmm.infer_C(i, x)
    print('C Guess vs actual:', np.argmax(marginal), c[i])


# Infer first Z in every timestep
for i in range(t):
    marginal = hmm.infer_Z(i, 1, x)
    print('Z Guess vs actual:', np.argmax(marginal), z[i][1])



