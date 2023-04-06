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


hmmBU = HMM2(
    transition_matrix,
    alpha,
    rates=rates,
    processing_modes=[0, 1, 2],
)


c, z, x = hmmBU.forward(n, T, seed=449)
print(c, z, x, sep='\n')
pc, pz = hmmBU.infer_c_belief_propagation(x)

print(pc, pz)
