import itertools

import numpy as np

from hmm.hmm import HMM
from hmm.hmm_belief_prop import HMM2

import seaborn as sns
import matplotlib.pyplot as plt

gamma = 0.1
beta = 0.2
alpha = 0.9
rates = [1, 3]
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
    marginal_fw_bw_BU, _, _ = hmmBU.infer_hidden_belief_propagation(x)

    for p1, p2 in zip(marginal_fw_bw, marginal_fw_bw_BU):
        print('FW-BW vs BU', p1, p2)


def test_inference_absolute_error(N=100, start_seed=1):
    absolute_error_c = []
    absolute_error_z = []

    np.random.seed(start_seed)
    for _ in range(N):
        print('Inferring for round', _)
        c, z, x = hmmBU.forward(n, T)

        inferred_probs_c, inferred_probs_z, _ = hmmBU.infer_hidden_belief_propagation(x)

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


def test_inference_brute_force(N=1000000):
    T = 3  # should be considered fixed
    n = 1  # should be considered fixed

    c_sim = np.zeros((N, T), dtype=int)
    x_sim = np.zeros((N, T, n), dtype=int)

    counts = np.zeros((20, 20, 20))  # 20 is chosen arbitrarily, just needs to be larger than max rate
    for i in range(N):
        print('Iteration:', i)
        c, _, x = hmmBU.forward(n, T)
        c_sim[i] = c
        x_sim[i] = x
        counts[(*x.flatten(),)] += 1  # hacks, 3-dimensional dictionary

    prob = 0
    sim_prob = 0
    for comb in itertools.product([0, 1, 2, 3], repeat=3):
        _, _, c_t = hmmBU.infer_hidden_belief_propagation(np.array(comb).reshape(-1, 1))
        prob += np.prod(c_t)
        sim_prob += counts[comb] / N
    print('X probabilities match', prob, sim_prob)  # if they are the same then that's good

    index_argmax = np.unravel_index(np.argmax(counts), counts.shape)
    print(index_argmax, counts[index_argmax], counts[index_argmax] / N)

    # test x probabilities
    x_argmax = np.array(index_argmax).reshape(-1, 1)
    c_marginal_bu, _, c_t = hmmBU.infer_hidden_belief_propagation(x_argmax)
    c_marginal_fwbw = infer_fw_bw(x_argmax)
    print(np.prod(c_t))

    # test c probabilities
    x_sim_squeeze = np.squeeze(x_sim)
    arg_max_where = np.all(x_sim_squeeze == index_argmax, axis=1)
    c_where_x_is_argmax = c_sim[arg_max_where, :]
    c2_counts = [c_where_x_is_argmax[:, 1] == 0, c_where_x_is_argmax[:, 1] == 1, c_where_x_is_argmax[:, 1] == 2]
    c3_counts = [c_where_x_is_argmax[:, 2] == 0, c_where_x_is_argmax[:, 2] == 1, c_where_x_is_argmax[:, 2] == 2]
    c2_sim_probs = np.mean(c2_counts, axis=1)
    c3_sim_probs = np.mean(c3_counts,
                    axis=1)

    print('Simulated:', np.array([c2_sim_probs, c3_sim_probs]), sep='\n')
    print('C2 counts:', np.sum(c2_counts, axis=1), sep='\n')
    print('C3 counts:', np.sum(c3_counts, axis=1), sep='\n')
    # Might not be enough for asymptotic behavior
    print('Effective observations:', c_where_x_is_argmax.shape[0], sep='\n')
    print('Inferred BU:', c_marginal_bu[1:, :], sep='\n')
    print('Inferred FW-BW:', c_marginal_fwbw[1:, :], sep='\n')


def infer_fw_bw(x):
    T = len(x)
    return np.array([hmm.infer_marginal_c(x, t) for t in range(T)])


test_inference_brute_force()

# c_errors, z_errors = sim_and_plot()
# print(c_errors, z_errors, sep='\n')
# absolute_error_c, absolute_error_z = test_inference_absolute_error(1000)
# print(np.mean(absolute_error_c), np.mean(absolute_error_z))
