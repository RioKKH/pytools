#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from sympy import symbols, Matrix, solve


def solve_matrix_using_np():
    C = np.array([[2, 2, -1],
                  [2, -1, 2],
                  [-1, 2, 2]])

    eigvals, eigvecs = la.eig(C)
    print("Eigenvalues: ", eigvals)
    print("Eigenvectors: \n", eigvecs)


def solve_matrix_using_sympy():
    x, y, z = symbols('x y z')
    A = Matrix([[2, 2, -1],
                [2, -1, 2],
                [-1, 2, 2]])
    b = Matrix([0, 0, 0])

    # 特性方程式を解く
    # charpoly()は特性方程式の係数を返す
    # as_expr()は特性方程式の式を返す
    chracteristic_equation = A.charpoly(x).as_expr()
    print("Characteristic equation: \n", chracteristic_equation)

    # 特性方程式の解(固有値)を求める
    eigenvalues = solve(chracteristic_equation, x)

    # 固有値を代入して固有ベクトルを求める
    eigenvectors = []
    for eigenvalue in eigenvalues:
        eigenvector = A - eigenvalue * Matrix.eye(3)
        # 重解の場合には、異なる固有ベクトルが得られる場合がある
        # その場合には、nullspace()を使って、基底を求める
        # nullspace()は、基底を行ベクトルとして返す
        eigenvector = eigenvector.nullspace()
        print("Eigenvector: \n", eigenvector)
        print("num of eigenvector: ", len(eigenvector))
        #eigenvector = eigenvector.nullspace()[0]
        eigenvectors.append(eigenvector)

    print("Eigenvalues: \n", eigenvalues)
    print("Eigenvectors: \n", eigenvectors)


def main():
    solve_matrix_using_np()
    solve_matrix_using_sympy()

if __name__ == '__main__':
    main()


