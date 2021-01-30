# -*- coding: utf-8 -*-
"""
create on 2021-01-30 16:04
author @66492
"""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
