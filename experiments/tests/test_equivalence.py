import numpy as np
from scipy.stats import poisson

# Consider a single tree structure where we have C->Zi->Xi for i <= n of some n.
# If we observe Xi=xi, then we can run VE on each of the branches and eliminate to get P(X=x, C)
# From this we have the option of calculating Sum_C P(X=x, C)P(Zi|C)  and obtain P(X=x, Zi) for a specific i
# Is this equivalent to eliminating all other branches than the i'th branch, and then eliminating C?
# I.e. with n = 3, we first eliminate Z3, Z2, then C to obtain the final factor that is an argument of Z1

rates = [5, 20]
n = 3
alpha = 1

p_c = np.array([0.3, 0.2, 0.5])
p_z_given_c = np.array([[alpha, 1-alpha, 0.5],  # P(Z = 0 | C = 0,1,2)
                       [1-alpha, alpha, 0.5]])  # P(Z = 1 | C = 0,1,2)


# Calculate P(X=x, Z1) by eliminating all branches
def both_methods(observations):
    p_x_given_z = np.array([
        poisson.pmf(observations, rates[0]),
        poisson.pmf(observations, rates[1])
    ])

    # p_x_given_c = []
    # for i in range(p_x_given_z.shape[1]):
    #     p_xi_given_c = np.sum(p_z_given_c * p_x_given_z[:, i][:, np.newaxis], axis=0)
    #     p_x_given_c.append(p_xi_given_c)

    p_x_given_c = np.einsum("ij, il -> jl", p_x_given_z, p_z_given_c)  # P(xi|C) for each i
    p_xc = np.prod(p_x_given_c, axis=0) * p_c  # P(x|C)P(C) = P(x, C)

    p_z1_x = np.sum(p_xc * p_z_given_c, axis=1)  # Sum_C P(Z|C)P(x, C) = P(Z, x)
    p_z1_given_x = p_z1_x/np.sum(p_z1_x)


    # Calculate P(X=x, Z1) by eliminating 2nd and 3rd branch, without eliminating 1st branch
    p_x2_up_c = np.prod(p_x_given_c[1:, ], axis=0) * p_c  # P(X2=x2, X3=x3, C)
    p_z1_x2_up = np.sum(p_x2_up_c * p_z_given_c, axis=1)  # Sum_C P(X2=x2, X3=x3, C)P(Z1|C) = P(X2=x2, X3=x3, Z1)
    p_z1_x_all = p_z1_x2_up * p_x_given_z[:, 0]  # P(X2=x2, X3=x3, Z1) * P(X1|Z1)
    p_z1_given_x_all = p_z1_x_all/np.sum(p_z1_x_all)
    print(p_xc/np.sum(p_xc), p_z1_given_x, 'vs', p_z1_given_x_all, p_x2_up_c/np.sum(p_x2_up_c))  # They are not the same, at all


for i in range(30):
    print(i)
    both_methods(np.array([i, 1, 1]))


# Solution to paradox, P(Z|C) =/= P(Z|C, X)
