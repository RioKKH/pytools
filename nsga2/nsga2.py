#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


class NSGS2:
    def __init__(self, population_size, num_generations,
                 num_objectives, num_variables,
                 variable_range, mutation_rate, crossover_rate):

        self.population_size = population_size
        self.num_generations = num_generations
        self.num_objectives = num_objectives
        self.num_variables = num_variables
        self.variable_range = variable_range
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []

    def initialize_population(self):
        for _ in range(self.population_size):
            individual = np.random.uniform(low=self.variable_range[0], 
                                           high=self.variable_range[1],
                                           size=(self.nu_variables,))
            self.population.append(individual)

        def evaluate_objectives(self):
            # This function should be overwritten by the specific objectives
            pass

        def fast_nondominated_sort(self):
            pass

        def calculate_crowding_distance(self):
            pass

        def selection(self):
            pass

        def crossover(self):
            pass

        def mutation(self):
            pass


        def evolve(self):
            self.initialize_population()
            for _ in range(self.num_generations):
                self.evaluate_objectives()
                self.fast_nondominated_sort()
                self.calculate_crowding_distance()
                self.selection()
                self.crossover()
                self.mutation()


# Example of a subclass with specific objectives
class MyOptimizer(NSGA2):
    def evaluate_objectives(self):
        for individual in self.population:
            # Define your own objectives here
            pass


class Individual:
    def __init__(self,
                 objectives,
                 rank=float('inf'),
                 crowding_distance=0,
                 domination_count=0,
                 dominated_solutions=set()):

        self.objectives = objectives
        self.rank = rank
        self.crowding_distance = crowding_distance
        self.domination_count = domination_count
        self.dominated_solutions = dominated_solutions


class ZDT1(NSGA2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_objectives = 2
        self.num_variables = 30
        self.variables_range = (0, 1)

    def evaluate_objectives(self):
        for individual in self.population:
            f1 = individual[0]
            g = 1 + 9.0/(self.num_variables - 1) * np.sum(individual[1:])
            h = 1 - np.sqrt(f1 / g)
            f2 = g * h
            individual.objectives = [f1, f2]
