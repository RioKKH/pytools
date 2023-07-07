#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm

from nsga2.utils import NSGA2Utils
from nsga2.population import Population


class Evolution:
    """
    This class manages the evolutionary process of the NSGA-II algorithm.
    """

    def __init__(self,
                 problem,                 # 最適化する問題
                 num_of_generations=1000, # アルゴリズムが進化する世代の数
                 num_of_individuals=100,  # 各世代の個体数
                 num_of_tour_particips=2, # トーナメント選択に参加する個体の数
                 tournament_prob=0.9,     # トーナメント選択の確率
                 crossover_param=2,       # 交叉操作のパラメータ
                 mutation_param=5):       # 突然変異操作のパラメータ
        """
        Initialize an instance of the Evolution class.

        :param problem:               The problem to optimize.
        :param num_of_generations:    The number of generations the algorithm will evolve.
        :param num_of_individuals:    The number of individuals (solutions) in each generation.
        :param num_of_tour_particips: The number of participants in tournament selection.
        :param tournament_prob:       The probability used in tournament selection.
        :param crossover_param:       The parameter used in crossover operation.
        :param mutation_param:        The parameter used in mutation operation. 
        """

        self.utils = NSGA2Utils(problem,
                                num_of_individuals,
                                num_of_tour_particips,
                                tournament_prob,
                                crossover_param,
                                mutation_param)

        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals

    def evolve(self):
        """
        Executes the evolutionary process of the NSGA-II algorithm.
        It generates an initial population, calculates the non-dominated fronts
        and the crowding distance for each front, and generates a new population
        of children. 
        This process is repeated until the specified number of generations 
        has passed.
        
        :return: The final population (set of solutions).
        """
        # Create initial population
        self.population = self.utils.create_initial_population()
        # Calculate non-dominated fronts
        self.utils.fast_nondominated_sort(self.population)

        # Calculate crowding distance for each front
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)

        # Create new population of children
        children = self.utils.create_children(self.population)
        # 各世代の最良フロントを保存するリスト
        best_fronts = []

        # Evolution process
        for i in tqdm(range(self.num_of_generations)):
            # Combine current population and children
            self.population.extend(children)
            # Calculate non-dominated fronts
            self.utils.fast_nondominated_sort(self.population)
            # Create a new population
            new_population = Population()
            front_num = 0

            # Add individuals to the new population based on non-domination
            # and crowing distance
            while len(new_population) + len(self.population.fronts[front_num]) \
                    <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1

            # Sort individuals in the last front based on crowding distance
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(
                key=lambda individual: individual.crowding_distance, reverse=True)
            # Add individuals from the last front to the new population
            new_population.extend(
                self.population.fronts[front_num][0:self.num_of_individuals\
                                                  - len(new_population)])

            # Add best individuals beloing to the best front to the list
            best_fronts.append(self.population.fronts[0])

            # Update current population
            self.population = new_population
            # Calculate non-dominated fronts and croding distance for the new population
            self.utils.fast_nondominated_sort(self.population)

            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)

            # Create a new population of children
            children = self.utils.create_children(self.population)

        return best_fronts
        # return self.population.fronts[0]
