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
hmm = HMM(
    transition_matrix,
    alpha,
    rates=rates,
    processing_modes=[0, 1, 2],
)

n = 5
T = 10

c, z, x = hmm.forward(n, T, seed=1)
print(c, z, x)


def test_marginal_c_when_t_is(t):
    marginal = hmm.infer_marginal_c(x, t)
    print(marginal)


# test_marginal_c_when_t_is_T()
#test_marginal_c_when_t_is(0)  ## == [0, 0, 1]

test_marginal_c_when_t_is(3)
