#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from mpl_toolkits.mplot3d import Axes3D

# Define the sigmoid function
def sigmoid_2d(x1, x2, a1=1, a2=1, b=0):
    return 1 / (1 + np.exp(-(a1*x1 + a2*x2 + b)))

# Define the cylindrical sigmoid function
def sigmoid_cylindrical(x, a, b):
    r = np.sqrt(x[:, 0]**2 + x[:, 1]**2) # Compute the Euclidean distance from the origin
    return 1 / (1 + np.exp(-(a*r + b)))

def erf_cylindrical(x, a, b):
    r = np.sqrt(x[:, 0]**2 + x[:, 1]**2) # Compute the Euclidean distance from the origin
    return 0.5 * (1 + erf(a * (r - b)))

def erf_cylindrical_center(x, a, b, cx, cy):
    # Compute the Euclidean distance from the center
    r = np.sqrt((x[:, 0] - cx)**2 + (x[:, 1] - cy)**2)
    return 0.5 * (1 + erf(a * (r - b)))

def plot_sigmoid_2d():
    # Create a grid of points
    x1 = np.linspace(-10, 10, 100)
    x2 = np.linspace(-10, 10, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute the sigmoid function on the grid
    Y = sigmoid_2d(X1, X2, a1=1, a2=1, b=0)

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y, cmap='viridis')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('3D plot of a 2D sigmoid function')

    plt.show()

def fit_cylindrical_data():
    np.random.seed(0)
    num_points=1000
    x_data = np.random.uniform(-10, 10, size=(num_points, 2))
    y_data = np.where(np.sqrt(x_data[:, 0]**2 + x_data[:, 1]**2) < 5, 1, 0)

    # Fit the data using the cylindrical sigmoid function
    #popt, pcov = curve_fit(sigmoid_cylindrical, x_data, y_data, p0=[1, 0])
    #popt, pcov = curve_fit(erf_cylindrical, x_data, y_data, p0=[1, 5])
    popt, pcov = curve_fit(erf_cylindrical_center, x_data, y_data, p0=[1, 5, 0, 0])

    # print the optimized parameters
    print(popt)

fit_cylindrical_data()
