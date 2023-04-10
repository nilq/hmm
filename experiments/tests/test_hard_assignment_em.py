import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from hmm.hmm_belief_prop import HMM2
from hmm.learning import hard_assignment_em

# Parameters from assignment
gamma = 0.3
beta = 0.5
alpha = 0.9
rates = (1, 9)
n = 10
T = 1000

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


def test_hard_assignment_em_multiple_start(param=0,  values_to_try=[*range(1, 10), 9.99]):
    learned_parameters = {}
    for value in values_to_try:
        print('----- LEARNING -----', value/10)
        final_params, learned_parameters_iterations = test_hard_assignment_em_initial(
                x,
                initial_gamma=value/10 if param == 0 else gamma,
                initial_beta=value/10 if param == 1 else beta,
                initial_alpha=alpha,
                initial_rates=rates
            )
        learned_parameters[value] = learned_parameters_iterations
    return learned_parameters


# test_hard_assignment_em()
# test_hard_assignment_em_initial(x, initial_gamma=gamma, initial_beta=beta, initial_alpha=alpha, initial_rates=rates)
# learned_parameters = test_hard_assignment_em_iterations(1000)

def test_and_plot(param=0, param_name=r'$\gamma$', values_to_try=[*range(1, 10), 9.99]):
    res = test_hard_assignment_em_multiple_start(param)

    data = []
    for key, values in res.items():
        for index, value in enumerate(values[:, param]):
            data.append([index, value, key/10])
    df = pd.DataFrame(data,
                      columns=['x', 'y', 'key'],
                      )

    sns.set(font_scale=1.1, style='whitegrid', font='serif')
    plt.figure(dpi=200)
    plt.title(f'Convergence of HMM prediction of {param_name}')

    y_ticks = np.arange(min(df['y']), max(df['y'])+0.1, 0.1)
    g = sns.lineplot(data=df, x='x', y='y', hue='key', legend='full')
    g.set_yticks(y_ticks)
    g.set(xlabel='Iteration', ylabel=param_name)
    g.legend_.set_title(f'Initial {param_name}')

    plt.show()


test_and_plot()
test_and_plot(param=1, param_name=r'$\beta$')

print('Done')
