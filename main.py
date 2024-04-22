import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from pyDOE import lhs

np.random.seed(1221)

# Constants for Branin function
a = 1
b = 5.1 / (4 * np.pi**2)
c = 5 / np.pi
r = 6
s = 10
t = 1 / (8 * np.pi)

# Branin function
def branin(x):
    x1 = x[0]
    x2 = x[1]
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s

# Correlation function
def corr(xi, xj, theta0, theta1, ph0, ph1):
    # sum of distances
    sum_dist = 0
    sum_dist += theta0*((xi[0] - xj[0])**ph0)
    sum_dist += theta1*((xi[1] - xj[1])**ph1)
    return np.exp(-sum_dist)

def corr_matrix(X, theta0, theta1, ph0, ph1):
    n = X.shape[0]
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = corr(X[i], X[j], theta0, theta1, ph0, ph1)
        R[i, i] += 1e-8  # small value to the diagonal to ensure positive definiteness
    return R

def neg_log_likelihood(params, X, y):
    theta0, theta1, ph0, ph1 = params
    # Equation (4)
    try:
        R = corr_matrix(X, theta0, theta1, ph0, ph1)
        L = np.linalg.cholesky(R)
        L_inv = np.linalg.inv(L)
        R_inv = np.dot(L_inv.T, L_inv)
        ones = np.ones(len(y))
        mu = np.dot(ones.T, np.dot(R_inv,y)) / np.dot(ones.T, np.dot(R_inv, ones))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - mu))
        term1 = np.dot(y - mu, alpha)
        sigma2 = term1 / len(y)
        log_likelihood = -0.5 * len(y) * np.log(2 * np.pi*sigma2) - 0.5 * np.log(np.linalg.det(R)) - 0.5 * term1 / sigma2
        return -log_likelihood
    except np.linalg.LinAlgError:
        return np.inf

def kriging_parameters(X,y):
    initial_guess = [1.0, 1.0, 2, 2]  # initial guess
    bounds = [ (0, 5), (0, 5), (1, 2), (1, 2)]
    res = minimize(neg_log_likelihood, initial_guess, args=(X, y), bounds=bounds)
    return res.x


def predict(X, y, X_star, sigma2, mu, theta0, theta1, ph0, ph1):
    R = corr_matrix(X, theta0, theta1, ph0, ph1)
    L = np.linalg.cholesky(R)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - mu))

    # Compute correlations for new points
    R_star = np.zeros((len(X_star), len(X)))
    for i in range(len(X_star)):
        for j in range(len(X)):
            R_star[i, j] = corr(X_star[i], X[j], theta0, theta1, ph0, ph1)
    
    # Equation (7)
    mu_star = mu + np.dot(R_star, alpha) # Mean for each prediction point
    
    # Compute variance using Equation (9)
    L = np.linalg.cholesky(R)
    L_inv = np.linalg.inv(L)
    R_inv = np.dot(L_inv.T, L_inv)
    s2_star = np.zeros(len(X_star))
    one = np.ones((len(X), 1)) 

    for i in range(len(X_star)):
        r = R_star[i, :].reshape(-1, 1)  # column vector r for the ith new point
        term1 = np.dot(np.dot(r.T, R_inv), r)
        term2_part1 = np.dot(np.dot(one.T, R_inv), r)
        term2 = (term2_part1 - 1) ** 2 / np.dot(one.T, np.dot(R_inv, one))
        s2_star[i] = sigma2 * (1 - term1 + term2)

   
    return mu_star, np.sqrt(s2_star)

def expected_improvement(X, y, X_star, sigma2, mu, theta0, theta1, ph0, ph1):
    # Equation (15)
    mu_star, sigma_star = predict(X, y, X_star, sigma2, mu, theta0, theta1, ph0, ph1)
    f_min = np.min(y)
    err = f_min - mu_star
    gamma = err / sigma_star
    ei = err * norm.cdf(gamma) + sigma_star * norm.pdf(gamma)
    return ei


def compute_sigma2_mu(params, X, y):
    theta0, theta1, ph0, ph1 = params
    R = corr_matrix(X, theta0, theta1, ph0, ph1)
    L = np.linalg.cholesky(R)
    L_inv = np.linalg.inv(L)
    R_inv = np.dot(L_inv.T, L_inv)
    ones = np.ones(len(y))
    mu = np.dot(ones.T, np.dot(R_inv,y)) / np.dot(ones.T, np.dot(R_inv, ones))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - mu))
    term1 = np.dot(y - mu, alpha)
    sigma2 = term1 / len(y)
    return sigma2, mu

# Optimization

# initial sampling
X_sample = lhs(2, samples=200) * 15 - 5
y_sample = np.array([branin(x) for x in X_sample])

# predict the Kriging function output at the grid points
x = np.linspace(-5, 10, 50)
y = np.linspace(0, 15, 50)
X1, X2 = np.meshgrid(x, y)
X_grid = np.column_stack([X1.ravel(), X2.ravel()])


n_iter = 28
for i in range(n_iter):
    # Step 1: fit the Kriging Model
    theta0, theta1, ph0, ph1 = kriging_parameters(X_sample, y_sample)
    sigma2, mu = compute_sigma2_mu([theta0, theta1, ph0, ph1], X_sample, y_sample)
    print('Current best Parameters: ', sigma2, mu, theta0, theta1, ph0, ph1)
    print('Current best f_min:', np.min(y_sample))

    # Step 2: compute the maximum improvement and find the next point to sample
    ei = expected_improvement(X_sample, y_sample, X_grid, sigma2, mu, theta0, theta1, ph0, ph1)
    X_next = X_grid[np.argmax(ei)]
    y_next = branin(X_next)
    print('Maximized Expected Improvement: ', np.max(ei))
    # append X_next and y_next to X_samples and y_sample
    X_sample = np.vstack([X_sample, X_next])
    y_sample = np.append(y_sample, y_next)

    print('\n')

# Final results
print('X_min: ', X_sample[np.argmin(y_sample)], '\t', 'f_min:', np.min(y_sample))
