import numpy as np

from hmm.learning import hard_assignment_em


def test_hard_em(filename):
    # Read in the data from a CSV file
    data = np.loadtxt(f'../../data/{filename}', delimiter=',', skiprows=1)

    print(data[:, 1:])
    x = data[:, 1:]
    return hard_assignment_em(
        x,
        initial_gamma=0.2,
        initial_beta=0.2,
        initial_alpha=0.7,
        initial_rates=(1, 6)
    )


res = []
for i in range(1, 10):
    try:
        res.append((i, *test_hard_em(f'Ex_{i}.csv')))
    except Exception as e:
        print('Couldn\'t run', i, e)

print(np.array(res))
