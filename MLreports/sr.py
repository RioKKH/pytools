#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pysr import PySRRegressor

class SR:
    def __init__(self):
        self.model = None

    def load_data(self, fin:str) -> pd.DataFrame:
        self.df = pd.read_csv(fin,
                         comment="#", delim_whitespace=True,
                         names=("dx1", "dx2", "x1", "x2", "y"))
        self.X = self.df[['x1', 'x2']]
        self.y = self.df['y']

    def make_model(self):
        self.model = PySRRegressor(
            niterations=50,
            population_size=6,
            binary_operators=["+", "*", "-", "/"],
            #unary_operators=["exp", "erf"],
            unary_operators=["exp", "square", "sqrt", "erf"],
            #unary_operators=["exp", "square", "erf"],
            maxsize=25,
            maxdepth=5,
            #unary_operators=["exp", "square", "erf", "inv(x) = 1/x",],
            #unary_operators=["cos", "exp", "sin", "square", "inv(x) = 1/x",],
            #extra_sympy_mappings={"inv": lambda x: 1/x},
            loss="loss(prediction, target) = (prediction - target)^2",
        )

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self):
        plt.scatter(self.y, self.model.predict(self.X))
        plt.show()

    def plot(self):
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)



def main(fin:str):
    s = SR()
    s.load_data(fin)
    s.make_model()
    s.fit()


