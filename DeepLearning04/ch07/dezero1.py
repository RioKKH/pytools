#!/usr/bin/env python

import numpy as np
from dezero import Variable
import dezero.functions as F


def main():
    # Inner products
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    # 明示的にVariableに変換しなくても
    # F.matmul()にnp.arrayを渡せば自動的に変換される
    a, b = Variable(a), Variable(b)
    c = F.matmul(a, b)
    print(c)

    # Matrix products
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    c = F.matmul(a, b)
    print(c)


if __name__ == "__main__":
    main()
