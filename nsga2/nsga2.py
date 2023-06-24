#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

import numpy as np
import matplotlib.pyplot as plt


class NSGA2:
    def __init__(self, population_size,
                 num_generations,
                 num_objectives,
                 num_variables, variable_range,
                 mutation_rate, crossover_rate,
                 distribution_index_crossover, 
                 distribution_index_mutation):

        self.population_size = population_size
        self.num_generations = num_generations
        self.num_objectives = num_objectives
        self.num_variables = num_variables
        self.variable_range = variable_range
        # 交叉と突然変異が起こる確率を示す。
        # これらの値が大きいほど、交叉や突然変異が頻繁に起こる
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        # 交差の分布指数でアルゴリズムの探索行動の分布を制御する。
        # 大きな値を設定すると親個体に近い子個体が生成されやすくなり、
        # 小さな値を設定すると親個体から遠い子個体が生成されやすくなる
        self.distribution_index_crossover = distribution_index_crossover
        self.distribution_index_mutation = distribution_index_mutation
        self.population = []

    def initialize_population(self):
        for _ in range(self.population_size):
            variables = np.random.uniform(low=self.variable_range[0], 
                                          high=self.variable_range[1],
                                          size=(self.num_variables,))
            self.population.append(Individual(variables))

    def evaluate_objectives(self):
        # This function should be overwritten by the specific objectives
        raise NotImplementedError

    def dominates(self, individual1, individual2):
        """Check if individual1 dominates individual2."""
        # "individual1 dominates individual2" is true if :
        # 1. individual1 is no worse than individual2 in all objectives.
        # 2. individual1 is strictly better than individual2 in at least one objective.
        not_worse_in_all = all(i1 <= i2 for i1, i2 in zip(individual1, individual2))
        better_in_one = any(i1 < i2 for i1, i2 in zip(individual1, individual2))
        return not_worse_in_all and better_in_one

    def fast_nondominated_sort(self):
        fronts = []
        front_1 = []
        for p in self.population:
            p.dominated_solutions = set()
            p.domination_count = 0
            for q in self.population:
                if self.dominates(p.objectives, q.objectives):
                    p.dominated_solutions.add(q)
                elif self.dominates(q.objectives, p.objectives):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 1
                front_1.append(p)
        fronts.append(front_1)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 2
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts

    def calculate_crowding_distance(self, front):
        if not front:
            return
        for individual in front:
            individual.crowding_distance = 0
        for i in range(self.num_objectives):
            front.sort(key=lambda x: x.objectives[i])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            objective_range = front[-1].objectives[i] - front[0].objectives[i]
            if objective_range == 0:
                continue
            for j in range(1, len(front) - 1):
                front[j].crowding_distance\
                    += (front[j+1].objectives[i] - front[j-1].objectives[i]) \
                    / objective_range

    def selection(self):
        self.population.sort(key=lambda x: (x.rank, -x.crowding_distance))
        return self.population[:self.population_size]

    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            child1 = parent1.copy()
            child2 = parent2.copy()
            for i in range(self.num_variables):
                if np.random.random() < 0.5:
                    child1.variables[i] = 0.5 * ((1 + self.distribution_index_crossover)
                                                 * parent1.variables[i]
                                                 + (1 - self.distribution_index_crossover)
                                                 * parent2.variables[i])

                    child2.variables[i] = 0.5 * ((1 - self.distribution_index_crossover)
                                                 * parent1.variables[i]
                                                 + (1 + self.distribution_index_crossover)
                                                 * parent2.variables[i])
                else:
                    child1.variables[i] = 0.5 * ((1 - self.distribution_index_crossover)
                                                 * parent1.variables[i]
                                                 + (1 + self.distribution_index_crossover)
                                                 * parent2.variables[i])

                    child2.variables[i] = 0.5 * ((1 + self.distribution_index_crossover)
                                                 * parent1.variables[i]
                                                 + (1 - self.distribution_index_crossover)
                                                 * parent2.variables[i])

                child1.variables[i] = min(max(child1.variables[i], self.variable_range[0]),
                                          self.variable_range[1])
                child2.variables[i] = min(max(child2.variables[i], self.variable_range[0]),
                                          self.variable_range[1])
            return child1, child2

        else:
            return parent1.copy(), parent2.copy()

    def mutation(self, individual):
        for i in range(self.num_variables):
            if np.random.random() < self.mutation_rate:
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u)**(1 / (self.distribution_index_mutation + 1)) - 1
                else:
                    delta = 1 - (2*(1 - u))**(1 / (self.distribution_index_mutation + 1))
                individual.variables[i] = individual.variables[i] + delta
                individual.variables[i] = min(max(individual.variables[i], self.variable_range[0]),
                                              self.variable_range[1])
        return individual

    def evolve(self):
        self.initialize_population()
        varini = np.array([ind.variables for ind in zdt1.population])
        parini = np.array([ind.objectives for ind in zdt1.population])
        print(varini)

        for _ in range(self.num_generations):
            self.evaluate_objectives()
            fronts = self.fast_nondominated_sort()
            for front in fronts:
                self.calculate_crowding_distance(front)
            self.population = self.selection()
            offspring = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = np.random.choice(self.population, size=2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                offspring.append(child1)
                offspring.append(child2)
            self.population = offspring


class Individual:
    def __init__(self,
                 variables,
                 rank=float('inf'),
                 crowding_distance=0,
                 domination_count=0,
                 dominated_solutions=set()):

        self.variables = variables
        self.rank = rank
        self.crowding_distance = crowding_distance
        self.domination_count = domination_count
        self.dominated_solutions = dominated_solutions
        self.objectives = [0, 0]

    def copy(self):
        return copy.copy(self)


class ZDT1(NSGA2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_objectives = 2
        self.num_variables = 30
        self.variable_range = (0, 1)

    def evaluate_objectives(self):
        for individual in self.population:
            f1 = individual.variables[0]
            g = 1 + 9.0/(self.num_variables - 1) * np.sum(individual.variables[1:])
            h = 1 - np.sqrt(f1 / g)
            f2 = g * h
            individual.objectives = [f1, f2]


if __name__ == '__main__':
    zdt1 = ZDT1(population_size=100,
                num_generations=100,
                num_objectives=2,
                num_variables=30,
                variable_range=(0, 1),
                mutation_rate=0.01,
                crossover_rate=0.9,
                distribution_index_crossover=20,
                distribution_index_mutation=20)

    zdt1.evolve()

    pareto_front = np.array([ind.objectives for ind in zdt1.population])
    print(pareto_front)

    plt.scatter(pareto_front[:, 0], pareto_front[:, 1])
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Front')
    plt.show()
