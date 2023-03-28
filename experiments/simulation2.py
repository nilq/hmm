import numpy as np

from hmm.hmm import HMM

gamma = 0.5
beta = 0.8
alpha = 0.9
rates = [1, 20]

# This is uppercase-gamma.
transition_matrix = np.array(
    [[1 - gamma, 0, gamma], [0, 1 - gamma, gamma], [beta / 2, beta / 2, 1 - beta]]
)
# %%
hmm = HMM(
    transition_matrix,
    alpha,
    rates=rates,
    processing_modes=[0, 1, 2],
)

n = 5
t = 10

c, z, x = hmm.forward(n, t, seed=2)
print(c, z, x)

hmm.belief_propagation(x)

# # Infer C for every time step
# for i in range(t):
#     marginal = hmm.infer_marginal_c(x, i)
#     print("C Guess vs actual:", np.argmax(marginal), c[i])
#
# print()
# # Infer first Z in every timestep
# for i in range(t):
#     marginal = hmm.infer_marginal_z(x, i, 1)
#     print("Z Guess vs actual:", np.argmax(marginal), z[i][1])
