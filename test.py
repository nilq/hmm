# %%
from hmm.hmm import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    gamma = 0.1
    beta = 0.2
    alpha = 0.01
    rates = [1, 20]

    Gamma = np.array(
        [[1 - gamma, 0, gamma], [0, 1 - gamma, gamma], [beta / 2, beta / 2, 1 - beta]]
    )

    # simulate
    hmm = HMM(
        Gamma, alpha, lambda Z: sample_poisson_stimuli(Z, rates), states=[0, 1, 2]
    )
    n = 10
    t = 100
    processing_modes, sim_Z, sim_X = hmm.forward(n, t)
    sim_Z_transformed = (max(rates) * 2) * sim_Z - 5
    # Create a figure and axes
    fig, ax = plt.subplots(n, figsize=(15, 15), sharex=True)

    # for i in range(n):  # loop through every neuron up to n
    #     for current_t in range(t):
    #         current_C = processing_modes[current_t]
    #         color = ['green', 'red', 'blue'][current_C]
    #         shape = ['o', 's'][sim_Z]
    #         ax[i].plot(sim_X[:, i], '-o', markersize=4)
    #         ax[i].set_ylabel(f"NN {i}")
    # Loop over each column of the matrix and plot it as a line
    for i in range(n):
        ax[i].plot(sim_X[:, i], "-o", markersize=4)
        ax[i].plot(sim_Z_transformed[:, i], alpha=0.3)
        ax[i].set_ylabel(f"NN {i}")

        for current_t in range(t):
            current_C = processing_modes[current_t]
            color = ["green", "red", "blue"][current_C]

            ax[i].scatter(current_t, sim_Z_transformed[current_t, i], color=color)

    # Set the x-axis label and title
    fig.text(0.5, 0.04, "Time", ha="center")
    fig.suptitle("Neuron activation plot")

    # Display the plot
    plt.show()
