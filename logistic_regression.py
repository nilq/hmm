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


def run_and_test(t = 1):
    # simulate
    hmm = HMM2(Gamma, alpha, lambda Z: poisson_stimuli_sample_method(Z, rates))
    n = 10

    # We do multiclass logistic regression on processing_modes (C) given the activations (X)
    processing_modes, sim_Z, sim_X = hmm.forward(n, t=t)

    linear_model = LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1)
    linear_model.fit(sim_X, processing_modes)

    processing_modes, sim_Z, sim_X = hmm.forward(n, t=1000)

    # We then see how well the model predicts the processing modes
    print(f"t = {t}, accuracy = {linear_model.score(sim_X, processing_modes)}")

for t in [10, 50, 100, 200, 500, 1000]:
    run_and_test(t)


