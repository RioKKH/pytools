#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm

from nsga3.utils import NSGA3Utils
from nsga3.population import Population


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

        self.problem = problem
        self.utils = NSGA3Utils(problem,
                                num_of_individuals,
                                num_of_tour_particips,
                                tournament_prob,
                                crossover_param,
                                mutation_param)

        self.population = None
        self.num_of_generations = num_of_generations
        #self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.probem = problem

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

        for _ in tqdm(range(self.num_of_generations)):
            # 目的関数の計算
            for individual in self.population:
                self.problem.calculate_objectives(individual)

            # 理想点とナディア点の更新
            self.problem.update_ideal_point(self.population)
            self.problem.update_nadir_point(self.population)

            # 非支配ソートと参照点への関連付け
            fronts = self.utils.fast_nondominated_sort(self.population)
            self.utils.associate_to_reference_point(self.population,
                                                    self.utils.reference_points)

            # 次世代の個体選択
            next_population = self.utils.select_population_nsga3(self.population,
                                                                 self.num_of_individuals)

            # 子個体の生成
            children = self.utils.create_children(next_population)

            # 次世代の個体群を設定
            self.population = Population()
            self.population.extend(next_population.population)
            self.population.extend(children)

        # 最終世代の非支配ソート
        fronts = self.utils.fast_nondominated_sort(self.population)
        return fronts[0] # 最終的なパレートフロントを返す
