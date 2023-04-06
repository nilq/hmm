import numpy as np

from hmm.hmm import HMM
from hmm.hmm_belief_prop import HMM2

import seaborn as sns
import matplotlib.pyplot as plt

gamma = 0.1
beta = 0.2
alpha = 0.9
rates = [1, 20]
n = 11
T = 5

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


def test_compare_c_with_fwbw():
    c, z, x = hmm.forward(n, T, seed=449)
    print(c, z, x)
    marginal_fw_bw, _ = np.array([hmm.infer_marginal_c(x, t) for t in range(T)])
    marginal_fw_bw_BU, _ = hmmBU.infer_hidden_belief_propagation(x)

    for p1, p2 in zip(marginal_fw_bw, marginal_fw_bw_BU):
        print('FW-BW vs BU', p1, p2)


def test_inference_absolute_error(N=100, start_seed=1):
    absolute_error_c = []
    absolute_error_z = []

    np.random.seed(start_seed)
    for _ in range(N):
        print('Inferring for round', _)
        c, z, x = hmmBU.forward(n, T)

        inferred_probs_c, inferred_probs_z = hmmBU.infer_hidden_belief_propagation(x)

        # for C
        arr_c = np.zeros(inferred_probs_c.shape)
        arr_c[np.arange(T), c] = 1
        # ae_c = np.abs(arr_c - inferred_probs_c)
        ae_c = arr_c - inferred_probs_c
        absolute_error_c.append(ae_c)

        # for Z
        arr_z = np.zeros(inferred_probs_z.shape)
        arr_z[:, :, 0] = z == 0
        arr_z[:, :, 1] = z == 1
        # ae_z = np.abs(arr_z - inferred_probs_z)
        ae_z = arr_z - inferred_probs_z
        absolute_error_z.append(ae_z)
    return np.array(absolute_error_c), np.array(absolute_error_z)


def sim_and_plot(N=10000):
    absolute_error_c, absolute_error_z = test_inference_absolute_error(N)

    c_errors = []
    z_errors = []
    for i in range(1, N):
        c_errors.append((i, np.mean(absolute_error_c[:i])))
        z_errors.append((i, np.mean(absolute_error_z[:i])))

    c_errors = np.array(c_errors)
    z_errors = np.array(z_errors)

    sns.relplot(x=c_errors[:, 0], y=c_errors[:, 1], kind="line")
    sns.relplot(x=z_errors[:, 0], y=z_errors[:, 1], kind="line")
    sns.set(style="darkgrid")
    sns.set_palette("muted")

    plt.show()

    return c_errors, z_errors


c_errors, z_errors = sim_and_plot()
print(c_errors, z_errors, sep='\n')
# absolute_error_c, absolute_error_z = test_inference_absolute_error(1000)
# print(np.mean(absolute_error_c), np.mean(absolute_error_z))
