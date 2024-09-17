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
