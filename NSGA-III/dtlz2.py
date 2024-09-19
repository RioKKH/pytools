#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from nsga3.problem import Problem
from nsga3.evolution import Evolution

class DTLZ2(Problem):
    def __init__(self, num_of_variables, num_of_objectives):
        objectives = [self.objective_function] * num_of_objectives
        variables_range = [(0, 1)] * num_of_variables
        super().__init__(objectives, num_of_variables, variables_range, expand=False)
        self.num_of_objectives = num_of_objectives

    def objective_function(self, x):
        x = np.array(x)
        #f = []
        f = np.zeros(self.num_of_objectives)
        g = np.sum((x[self.num_of_objectives-1:] - 0.5) ** 2)
        for i in range(self.num_of_objectives):
            f[i] = (1 + g)
            for j in range(self.num_of_objectives - i - 1):
                f[i] *= np.cos(x[j] * np.pi / 2)
            if i > 0:
                f[i] *= np.sin(x[self.num_of_objectives - i - 1] * np.pi / 2)
        return f

    def plot_results(self, population, num_objectives):
        if num_objectives == 2:
            plt.figure(figsize = (8, 6))
            plt.scatter([ind.objectives[0] for ind in population],
                        [ind.objectives[1] for ind in population])
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title('NSGA-III Results for DTLZ2 (2 objectives)')
            plt.show()
        elif num_objectives == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([ind.objectives[0] for ind in population],
                       [ind.objectives[1] for ind in population],
                       [ind.objectives[2] for ind in population])
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.set_title('NSGA-III Results for DTLZ2 (3 objectives)')
            plt.show()
        else:
            print('Plotting is only supooerted for 2 or 3 objectives.')

if __name__ == '__main__':
    num_of_variables = 10
    num_of_objectives = 3
    num_of_generations = 200
    num_of_individuals = 100

    problem = DTLZ2(num_of_variables, num_of_objectives)
    evolution = Evolution(problem, num_of_generations, num_of_individuals)

    final_population = evolution.evolve()

    # Print the objectives of the first few solutions
    print('Objectives of the first 5 solutions:')
    for individual in final_population[:5]:
        print(individual.objectives)

    # Plot the results
    plot_results(final_population, num_of_objectives)
