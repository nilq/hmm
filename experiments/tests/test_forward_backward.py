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
# test_marginal_c_when_t_is(0)  ## == [0, 0, 1]

# test_marginal_c_when_t_is(3)
# test_marginal_c_when_t_is(T - 1)  # Inferring last node

def test_c_inference_single(N, t):
    for_current_t = []
    for _ in range(N):
        c_sim, z_sim, x_sim = hmm.forward(n, T)
        true_c = c_sim[t]  # 0, 1, 2
        inferred_probabilities = hmm.infer_marginal_c(x_sim, t)

        diff = true_c - inferred_probabilities[true_c]

        for_current_t.append(diff)
    return for_current_t


def test_c_inference(N=100):
    diff_indicator_c_sub_inferred_c = []
    for t in range(T):
        diffs = test_c_inference_single(N, t)
        diff_indicator_c_sub_inferred_c.append(diffs)
    return diff_indicator_c_sub_inferred_c


res = test_c_inference_single(10000, 3)
print(np.mean(res))
