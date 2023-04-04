import numpy as np

from hmm.hmm import HMM
from hmm.hmm_belief_prop import HMM2

gamma = 0.5
beta = 0.8
alpha = 0.9
rates = [1, 20]
n = 1
T = 2

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

hmmBU = HMM2(
    transition_matrix,
    alpha,
    rates=rates,
    processing_modes=[0, 1, 2],
)


def test_compare_with_fwbw():
    c, z, x = hmm.forward(n, T, seed=449)
    print(c, z, x)
    marginal_fw_bw = np.array([hmm.infer_marginal_c(x, t) for t in range(T)])
    marginal_fw_bw_BU = hmmBU.infer_c_belief_propagation(x)

    for p1, p2 in zip(marginal_fw_bw, marginal_fw_bw_BU):
        print('FW-BW vs BU', p1,p2)


def test_c_inference2(N=100):
    diff_indicator_c_sub_inferred_c = []
    for round in range(N):
        print('Inferring for round', round)
        c, z, x = hmmBU.forward(n, T)

        inferred_probs = hmmBU.infer_c_belief_propagation(x)

        arr = np.zeros((T, 3))
        arr[np.arange(T), c] = 1
        diffs = np.abs(arr - inferred_probs)
        diff_indicator_c_sub_inferred_c.append(diffs)
    return np.array(diff_indicator_c_sub_inferred_c)


# test_compare_with_fwbw()
res = test_c_inference2(10000)
print(np.mean(res[:100, :, :]), np.mean(res[:1000, :, :]), np.mean(res[:5000, :, :]), np.mean(res))