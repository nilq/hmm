# %%
from hmm import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    gamma = 0.1
    beta = 0.2
    alpha = 0.9
    rates = [1, 5]

    Gamma = np.array([
        [1-gamma, 0, gamma],
        [0, 1-gamma, gamma],
        [beta/2, beta/2, 1-beta]
    ])

    # simulate
    hmm = HMM2(Gamma, alpha, lambda Z: poisson_stimuli_sample_method(Z, rates))
    sim_X, observations = hmm.forward(10)

    # Create a figure and axes
    fig, ax = plt.subplots(sim_X.shape[1], figsize=(15, 15), sharex=True)

    # Loop over each column of the matrix and plot it as a line
    for i in range(sim_X.shape[1]):
        ax[i].plot(sim_X[:, i])
        ax[i].set_ylabel(f"NN {i}")

    # Set the x-axis label and title
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.suptitle('Neuron activation plot')

    # Display the plot
    plt.show()
