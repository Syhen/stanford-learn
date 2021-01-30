# -*- coding: utf-8 -*-
"""
create on 2021-01-30 15:41
author @66492
"""
import numpy as np


def accuracy(y_true: np.ndarray, y_hat: np.ndarray):
    y_true = y_true.reshape(-1, )
    y_hat = y_hat.reshape(-1, )
    return (y_true == y_hat).sum() / y_true.shape[0]
