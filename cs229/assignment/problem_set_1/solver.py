# -*- coding: utf-8 -*-
"""
create on 2021-01-29 20:47
author @66492
"""
import warnings

import autograd

import numpy as np


def newton_method_autograd(objective, parameter, max_iter=100, tol=1e-16):
    """Apply Newton's Method in `\mathbb{R}`

    :param objective: callable. objective function. It's argument should be `parameter`.
    :param parameter: parameter of objective to optimize. it should be init.
    :param max_iter: int. max iteration of the algorithm.
    :param tol: float. algorithm will stop when `abs(function(parameter_{i+1}) - function(parameter_i)) < tol`.
    :return: np.array. optimized parameter.
    """
    parameter = np.array(parameter)
    final_parameter = parameter.astype("float64")
    first_grad_function = autograd.elementwise_grad(objective)

    for _ in range(max_iter):
        objective_val = objective(final_parameter)
        first_derivative = first_grad_function(final_parameter)
        final_parameter = final_parameter - objective_val / first_derivative
        after_objective = objective(final_parameter)
        if np.linalg.norm(after_objective, 2) < tol:
            break
    else:
        warnings.warn("algorithm don't converge after iteration %s" % max_iter)
    return final_parameter


def newton_method(objective_derivative, hessian, parameter, max_iter=100, tol=1e-16):
    final_parameter = parameter.astype("float64")
    for _ in range(max_iter):
        hessian_inverse = np.linalg.inv(hessian(final_parameter))
        final_parameter = final_parameter - hessian_inverse @ objective_derivative(final_parameter)
        after_objective = objective_derivative(final_parameter)
        # print("hessian:", hessian_inverse)
        # print("derivative:", objective_derivative(final_parameter))
        if np.linalg.norm(after_objective, 2) < tol:
            break
    else:
        warnings.warn("algorithm don't converge after iteration %s" % max_iter)
    return final_parameter


if __name__ == '__main__':
    # print(newton_method(lambda x: (x + 1) ** 4, 0))
    print(newton_method_autograd(lambda x: (x + 1) ** 4, np.array([10000, 10000]), max_iter=100, tol=1e-16))
