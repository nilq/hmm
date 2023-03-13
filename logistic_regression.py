# %%
from sklearn.linear_model import LogisticRegression
from hmm import *
import numpy as np
import matplotlib.pyplot as plt

gamma = 0.1
beta = 0.2
alpha = 0.01
rates = [1, 20]

Gamma = np.array([
    [1 - gamma, 0, gamma],
    [0, 1 - gamma, gamma],
    [beta / 2, beta / 2, 1 - beta]
])

# simulate
hmm = HMM2(Gamma, alpha, lambda Z: poisson_stimuli_sample_method(Z, rates))
n = 10
t = 100

def run_and_test(times: int):
    p_modes = []
    sim_Xs = []
    # We do multiclass logistic regression on processing_modes (C) given the activations (X)
    for _ in range(times):
        processing_modes, sim_Z, sim_X = hmm.forward(n, t)
        p_modes += processing_modes
        sim_Xs += sim_X.tolist()

    linear_model = LogisticRegression(random_state=0, max_iter=1000)
    linear_model.fit(sim_Xs, p_modes)


    processing_modes, sim_Z, sim_X = hmm.forward(n, t)
    # We then see how well the model predicts the processing modes
    print(linear_model.score(sim_X, processing_modes))


for i in range(10):
    run_and_test(i+1)


