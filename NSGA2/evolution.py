#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm

from nsga2.utils import NSGA2Utils
from nsga2.population import Population


class Evolution:

    def __init__(self,
                 problem,
                 num_of_generations=1000,
                 num_of_individuals=100,
                 num_of_tour_particips=2,
                 tournament_prob=0.9,
                 crossover_param=2,
                 mutation_param=5):

        self.utils = NSGA2Utils(problem,
                                num_of_individuals,
                                num_of_tour_particips,
                                tournament_prob,
                                crossover_param,
                                mutation_param)

        self.population = None
        self.
