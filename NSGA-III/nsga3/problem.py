#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

from nsga2.individual import Individual


class Problem:
    """
    Represents an optimization problem. Each problem has a set of objectives,
    a number of variables, and a range for each variables. It also has a flag
    to indicate whether the objective functions should be expanded or not.
    """

    def __init__(self,
                 objectives,
                 num_of_variables,
                 variables_range,
                 expand=True,
                 same_range=False):

        # Number of objectives in the problem.
        self.num_of_objectives = len(objectives)
        # Number of decision variables in the problem.
        self.num_of_variables = num_of_variables
        # Objective functions of the problemn.
        self.objectives = objectives
        # expand or not
        self.expand = expand
        # Range of each decision variable.
        self.variables_range = []

        # If all decision variables have the same range, use the same range for all.
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            # If decision variables have different rnges, use the provided ranges.
            self.variables_range = variables_range

        # NSGA-IIIのための追加属性
        self.ideal_point = None
        self.nadir_point = None

    def generate_individual(self):
        """
        Generates a new individual with random decision variables within
        the specified range.
        """
        individual = Individual()
        individual.features = [random.uniform(*x) for x in self.variables_range]
        return individual

    def calculate_objectives(self, individual):
        """
        Calculates the objective function values for an individual. If the expand
        flag is True, the objective functions are expanded, i.e., each decision
        variables is passed as a separate argument to the objective function.
        Otherswise, the list of decision variables is passed as a single argument.
        """
        if self.expand:
            individual.objectives = [f(*individual.features) for f in self.objectives]
        else:
            individual.objectives = [f(individual.features) for f in self.objectives]

    def update_ideal_point(self, population):
        """NSGA-IIIのための理想点更新"""
        if self.ideal_point is None:
            self.ideal_point = np.min([ind.objectives for ind in population], axis=0)
        else:
            self.ideal_point = np.minimum(self.ideal_point,
                                          np.min([ind.objectives for ind in population], axis=0))

    def update_nadir_point(self, population):
        """NSGA-IIIのためのナディア点更新"""
        if self.nadir_point is None:
            self.nadir_point = np.max([ind.objectives for ind in population], axis=0)
        else:
            self.nadir_point = np.maximum(self.nadir_point,
                                          np.max([ind.objectives for ind in population], axis=0))

    def normalize_objectives(self, objectives):
        """NSGA-IIIのための目的関数の正規化"""
        return (objectives - self.ideal_point) / (self.nadir_point - self.ideal_point)
