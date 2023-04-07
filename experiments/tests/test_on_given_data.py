import numpy as np

from hmm.learning import hard_assignment_em


def test_hard_em(filename):
    # Read in the data from a CSV file
    data = np.loadtxt(f'../../data/{filename}', delimiter=',', skiprows=1)

    print(data[:, 1:])
    x = data[:, 1:]
    (learned_gamma,
     learned_beta,
     learned_alpha,
     learned_rates0,
     learned_rates1) = hard_assignment_em(x,
                                          initial_gamma=0.2,
                                          initial_beta=0.2,
                                          initial_alpha=0.7,
                                          initial_rates=(1, 6))


test_hard_em('Ex_8.csv')
