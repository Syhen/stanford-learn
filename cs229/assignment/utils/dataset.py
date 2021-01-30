# -*- coding: utf-8 -*-
"""
create on 2021-01-30 15:42
author @66492
"""
import os

import numpy as np

from cs229.assignment.config import PROJECT_PATH


def load_logistic_data():
    x = []
    filename = os.path.join(PROJECT_PATH, "problem_set_1/dataset/logistic_x.txt")
    print(filename)
    with open(filename, encoding="utf8", mode="r") as f:
        for line in f.read().strip().split("\n"):
            line = line.strip()
            x.append([float(i.strip()) for i in line.split(" ") if i.strip()])
    x = np.array(x)
    filename = os.path.join(PROJECT_PATH, "problem_set_1/dataset/logistic_y.txt")
    with open(filename, encoding="utf8", mode="r") as f:
        y = np.array([int(float(i.strip())) for i in f.read().strip().split("\n") if i.strip()])
    return x, y


if __name__ == '__main__':
    x, y = load_logistic_data()
    print(x)
    print(y)
