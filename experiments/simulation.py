from hmm.hmm import HMM, sample_poisson_stimuli
import numpy as np
import matplotlib.pyplot as plt

from hmm.types import FloatArray
from typing import Callable


DEFAULT_RATES = [1, 20]


def default_transition(gamma: float, beta: float):
    return np.array(
        [[1 - gamma, 0, gamma], [0, 1 - gamma, gamma], [beta / 2, beta / 2, 1 - beta]]
    )


def forward_simulation(
    n: int,
    t: int,
    gamma: float = 0.1,
    beta: float = 0.2,
    alpha: float = 0.01,
    rates: list[int] = DEFAULT_RATES,
    transition: Callable[[float, float], FloatArray] = default_transition,
) -> tuple[list[int], FloatArray, FloatArray]:
    """_summary_

    Args:
        gamma (float, optional): _description_. Defaults to 0.1.
        beta (float, optional): _description_. Defaults to 0.2.
        alpha (float, optional): _description_. Defaults to 0.01.
        rates (list[int], optional): _description_.
            Defaults to DEFAULT_RATES.
        transition (Callable[[float, float], FloatArray], optional): _description_.
            Defaults to default_transition.

    Returns:
        tuple[list[int], FloatArray, FloatArray]:
            Processing modes (C), focus (Z), activations (X).
    """
    transition = transition(gamma, beta)
    # simulate
    hmm = HMM(
        transition, alpha, lambda Z: sample_poisson_stimuli(Z, rates), states=[0, 1, 2]
    )

    return hmm.forward(n, t)


def plot_simulation_values(
    n: int,
    t: int,
    processing_modes: list[int],
    z: FloatArray,
    x: FloatArray,
    rates: list[int] = DEFAULT_RATES,
):
    z_transformed = (max(rates) * 2) * z - 5
    # Create a figure and axes
    fig, ax = plt.subplots(n, figsize=(15, 15), sharex=True)

    for i in range(n):
        ax[i].plot(x[:, i], "-o", markersize=4)
        ax[i].plot(z_transformed[:, i], alpha=0.3)
        ax[i].set_ylabel(f"NN {i}")

        for current_t in range(t):
            current_C = processing_modes[current_t]
            color = ["green", "red", "blue"][current_C]

            ax[i].scatter(current_t, z_transformed[current_t, i], color=color)

    # Set the x-axis label and title
    fig.text(0.5, 0.04, "Time", ha="center")
    fig.suptitle("Neuron activation plot")

    # Display the plot
    plt.show()