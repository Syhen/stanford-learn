# -*- coding: utf-8 -*-
"""
create on 2021-01-29 20:46
author @66492
"""
import numpy as np

from cs229.assignment.problem_set_1 import solver
from cs229.assignment.utils import activations


class LogisticRegression(object):
    def __init__(self, C=0.0, solver="newtons-method", objective="average-empirical-loss", fit_interception=True,
                 theta_initial="zero"):
        self.C_ = C
        self.solver_ = solver
        self.objective_ = objective
        self.fit_interception_ = fit_interception
        self.theta_initial_ = theta_initial
        self.theta_ = None
        self.interception_ = None
        self.optimal_theta_ = None
        self.fitted = False

    def _initial_theta(self, X):
        n_features = X.shape[1]
        if self.fit_interception_:
            n_features += 1
        theta_shape = (n_features, 1)
        if self.theta_initial_ == "zero":
            return np.zeros(theta_shape)
        if self.theta_initial_ == "random":
            return np.random.random(theta_shape)
        raise ValueError("`theta_initial` should be 'zero' or 'random'")

    def _get_objective_derivative_function(self, X, y):
        m, n = X.shape
        theta_shape = (n, 1)
        if self.objective_ in {"average-empirical-loss", "aml"}:
            def objective_derivative(theta):
                j = np.zeros(theta_shape)
                for i in range(m):
                    j += (y[i, :] * X[i, :] / (1 + np.e ** (y[i, :] * theta.T @ X[i, :]))).reshape(*theta_shape)
                return -j / m

            return objective_derivative

    def _get_hessian(self, X, y):
        m, n = X.shape
        hessian_shape = (n, n)
        if self.objective_ in {"average-empirical-loss", "aml"}:
            def hessian(theta: np.ndarray):
                hessian_matrix = np.zeros(hessian_shape)
                for k in range(m):
                    y_k, x_k = y[k, :], X[k, :]
                    for i in range(n):
                        for j in range(n):
                            # try:
                            exp_term = np.exp(y_k * theta.T @ x_k)
                            # except ValueError:
                            #     print(theta.shape, x_k.shape, X.shape)
                            under = np.square(1 + exp_term)
                            hessian_matrix[i][j] += (y_k * y_k * exp_term * x_k[i] * x_k[j]) / under
                return hessian_matrix / m

            return hessian

    def _get_solver(self):
        if self.solver_ in ("newtons-method", "newton"):
            return solver.newton_method
        raise ValueError("`solver` should be 'newtons-method(alias: newton)'")

    def _check_X(self, X):
        if not self.fit_interception_:
            return X
        n_samples = X.shape[0]
        return np.concatenate((np.array([[1]] * n_samples), X), axis=1)

    def _check_y(self, y):
        if len(y.shape) == 1:
            return y.reshape(-1, 1)
        return y

    def fit(self, X, y=None):
        theta = self._initial_theta(X)
        X = self._check_X(X)
        y = self._check_y(y)
        derivative_function = self._get_objective_derivative_function(X, y)
        # TODO: generate hessian.
        hessian = self._get_hessian(X, y)
        solver = self._get_solver()
        self.fitted = True
        optimal_theta = solver(derivative_function, hessian, theta, max_iter=100, tol=1e-16)
        if self.fit_interception_:
            self.theta_ = optimal_theta.reshape(-1, )[1:]
            self.interception_ = optimal_theta.reshape(-1, )[0]
        else:
            self.theta_ = optimal_theta.reshape(-1, )
        self.optimal_theta_ = optimal_theta
        return self

    def predict_proba(self, X):
        if not self.fitted:
            raise RuntimeWarning("must call `fit` function first.")
        X = self._check_X(X)
        proba = activations.sigmoid(X@self.optimal_theta_)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba > 0.5, 1, -1)


if __name__ == '__main__':
    from cs229.assignment.utils import metrics
    from cs229.assignment.utils.dataset import load_logistic_data

    x, y = load_logistic_data()
    print(x.shape, y.shape)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x, y)
    y_pred = logistic_regression.predict(x)
    print("optimal_theta:", logistic_regression.optimal_theta_)
    print("accuracy: %.5f" % metrics.accuracy(y, y_pred))
