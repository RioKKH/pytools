#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class GaussianProcess:
    """ Gaussian Process regression. """
    def __init__(self, kernel, beta=1.):
        self.kernel = kernel
        self.beta =beta


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.K = self.kernel(X, X)
        self.L = np.linalg.cholesky(self.K + self.beta * np.eye(len(self.X_train)))

    def predict(self, X):
        self.K_x = self.kernel(self.X_train, X)
        self.K_xx = self.kernel(X, X)

        self.mu = self.K_x.T.dot(np.linalg.inv(self.K)).dot(self.y_train)
        self.sigma = np.sqrt(np.diag(self.K_xx - self.K_x.T.dot(np.linalg.inv(self.K)).dot(self.K_x)))

        return self.mu.squeeze(), self.sigma.squeeze()

    def lml(self, params):
        """ Log Marginal Likelihood
            対数周辺尤度はガウス過程回帰のパラメータを推定するための指標で、
            ハイパーパラメータの最適化に用いられる
        """
        # Update kernel parameters
        self.kernel.params = params

        # Refit model
        self.fit(self.X_train, self.y_train)

        # Compute log marginal likelihood
        return np.sum(np.log(np.diagonal(self.L))) + \
                      0.5 * self.y_train.T.dot(
                          np.linalg.inv(self.K + self.beta
                                        * np.eye(len(self.X_train)))).dot(self.y_train) + \
                      0.5 * len(self.X_train) * np.log(2 * np.pi)

# Objective function
def f(x):
    """ The function to predict """
    return x * np.sin(x)

# Kernel function
def RBF(X1, X2, l=1.0, sigma_f=1.0):
    """
    Radial basis function (RBF) or Gaussian Kernel.
        Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        l: Kernel vertical variation parameter.
    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def golden_section_search(gp, a, b, tol=1e-5):
    """ 1D optimization using golden section search.
    Args:
        gp: A GaussianProcess instance.
        a: Lower bound.
        b: Upper bound.
        tol: Tolerance for stopping criterion.
    Returns:
        x_opt: Optimal input.
    """
    gr = (np.sqrt(5) - 1) / 2 # golden ratio
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    while abs(c - d) > tol:
        # np.atleast_2d: 入力された配列を少なとも2次元のnp.ndarrayに変換する
        fc = -gp.predict(np.atleast_2d(c))[0]
        fd = -gp.predict(np.atleast_2d(d))[0]
        if fc < fd:
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may
        # lead to incorrect results or infinite loop
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    return (b + a) / 2

# Observations and noise
X_train = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
# ravel()は多次元配列を1次元配列にフラット化するメソッド
y_train = f(X_train).ravel()
dy = 0.5 + 1.0 * np.random.random(y_train.shape)
noise = np.random.normal(0, dy)
y_train += noise

# Instantiate a Gaussian Process model
gp = GaussianProcess(RBF, beta=1/dy**2)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X_train, y_train)
                                                     
# Make the prediction on the meshed x-axis (ask for MSE as well)
x = np.atleast_2d(np.linspace(0, 10, 1000)).T
y_pred, sigma = gp.predict(x)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X_train, y_train, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

# optimization
#result = minimize(lambda x: -gp.predict(np.atleast_2d(x))[0], x0=[2])
#plt.plot(result.x, gp.predict(result.x.reshape(-1, 1))[0], 'go', markersize=10)
x_opt = golden_section_search(gp, 0, 10)
plt.plot(x_opt, gp.predict(x_opt.reshape(-1, 1))[0], 'go', markersize=10)
plt.show()

