import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from hmm.hmm_belief_prop import HMM2
from hmm.learning import hard_assignment_em

# Parameters from assignment
gamma = 0.1
beta = 0.2
alpha = 0.9
rates = (1, 5)
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


def test_hard_assignment_em_initial(observations, max_itter, start_values):
    print('True parameters:')
    print('True rates:', rates)
    print('True gamma:', gamma)
    print('True beta:', beta)
    print('True alpha:', alpha)
    initial_gamma, initial_alpha, initial_beta, *initial_rates = start_values
    return hard_assignment_em(observations, max_itter, initial_gamma=initial_gamma, initial_alpha=initial_alpha, initial_beta=initial_beta, initial_rates=initial_rates)


# def calculate_error(true_params, learned_hmm):
#     learned_gamma, learned_beta, learned_alpha, *learned_rates = learned_hmm
#     gamma_error = np.abs(gamma - learned_gamma)
#     beta_error = np.abs(beta - learned_beta)
#     alpha_error = np.abs(alpha - learned_alpha)
#     rates_0_error = np.abs(rates[0] - learned_rates[0])
#     rates_1_error = np.abs(rates[1] - learned_rates[1])
#     return gamma_error, beta_error, alpha_error, rates_0_error, rates_1_error

def calculate_error(true_valus, learned_hmm_parameters):
    learned_gamma, learned_beta, learned_alpha, lambda_0, lambda_1 = learned_hmm_parameters
    true_c, true_z, true_x = true_valus
    learned_hmm = HMM2(
        np.array([
            [1 - learned_gamma, 0, learned_gamma],
            [0, 1 - learned_gamma, learned_gamma],
            [learned_beta / 2, learned_beta / 2, 1 - learned_beta]
        ]),
        learned_alpha,
        rates=[lambda_0, lambda_1],
        processing_modes=[0, 1, 2],
    )
    c_marginals, z_marginals, x_probs = learned_hmm.infer_hidden_belief_propagation(
        true_x)
    c_argmax = np.argmax(c_marginals, axis=1)
    z_argmax = np.argmax(z_marginals, axis=2)

    return np.abs(true_c - c_argmax).sum() / len(true_c), np.abs(true_z - z_argmax).sum() / true_z.size


def test_hard_assignment_em_iterations(iterations=10, start_values=(0.9, 0.1, 0.8, 4, 17)):
    errors = []
    for i in range(iterations):
        print('----- LEARNING -----', i)
        c, z, x = hmmBU.forward(n, T, seed=None)
        try:
            learned_hmm = test_hard_assignment_em_initial(x, i+1, start_values)
        except ZeroDivisionError:
            return None
        errors.append(calculate_error((c, z, x), learned_hmm))

    return np.array(errors)


def calculate_errors_for_start_value(args):
    idx, start_value, j = args
    errors = test_hard_assignment_em_iterations(j, start_values=start_value)
    error_data = []
    if errors is None:
        return None
    
    for i, error_tuple in enumerate(errors):
        c_error, _ = error_tuple
        error_data.append([i, idx, 'C error', c_error])

    return error_data


def run_experiment(j):
    start_values = [
        (0.1, 0.9, i/10, 1, 5) for i in range(1, 10)
    ]

    all_error_data = []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(calculate_errors_for_start_value, [(idx, sv, j) for idx, sv in enumerate(start_values)]))

    for error_data in results:
        if error_data:
            all_error_data.extend(error_data)

    df = pd.DataFrame(all_error_data, columns=['Iteration', 'Start Value', 'Parameter', 'Error'])
    sns.relplot(data=df, kind="line", x='Iteration', y='Error', hue='Start Value', style='Parameter', ci="sd")
    plt.title('Convergence of HMM prediction')
    plt.show()


# Run the experiment with the desired number of trials, e.g., 10
run_experiment(10)
