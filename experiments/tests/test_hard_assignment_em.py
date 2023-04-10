import numpy as np

from hmm.hmm_belief_prop import HMM2
from hmm.learning import hard_assignment_em

# Parameters from assignment
gamma = 0.3
beta = 0.5
alpha = 0.9
rates = (1, 9)
n = 10
T = 10000

# This is uppercase-gamma.
transition_matrix = np.array([
    [1 - gamma, 0, gamma],
    [0, 1 - gamma, gamma],
    [beta / 2, beta / 2, 1 - beta]
])

hmmBU = HMM2(
    transition_matrix,
    alpha,
    rates=rates,
    processing_modes=[0, 1, 2],
)

c, z, x = hmmBU.forward(n, T, seed=None)


def test_hard_assignment_em_initial(observations, **kwargs):
    print('True parameters:')
    print('True rates:', rates)
    print('True gamma:', gamma)
    print('True beta:', beta)
    print('True alpha:', alpha)
    return hard_assignment_em(observations, **kwargs)


def test_hard_assignment_em_iterations(iterations=10):
    learned_parameters = []
    for i in range(iterations):
        print('----- LEARNING -----', i)
        x = hmmBU.forward(n, T, seed=None)[2]
        learned_parameters.append(test_hard_assignment_em_initial(x))

    learned_parameters = np.array(learned_parameters)
    print('Learned parameters:', np.mean(learned_parameters, axis=0))
    return learned_parameters


def test_hard_assignment_em_multiple_start():
    learned_parameters = []

    gamma_values_to_try = np.linspace(0.1, 1, 10)
    for gamma_ in gamma_values_to_try:
        print('----- LEARNING -----', gamma_)
        x = hmmBU.forward(n, T, seed=None)[2]
        learned_parameters.append(
            test_hard_assignment_em_initial(
                x, initial_gamma=gamma_,
                initial_beta=beta,
                initial_alpha=alpha,
                initial_rates=rates
            )
        )
    return learned_parameters


# test_hard_assignment_em()
# test_hard_assignment_em_initial(x, initial_gamma=gamma, initial_beta=beta, initial_alpha=alpha, initial_rates=rates)
# learned_parameters = test_hard_assignment_em_iterations(1000)

learned_parameters = test_hard_assignment_em_multiple_start()
print(learned_parameters)
print('Done')
