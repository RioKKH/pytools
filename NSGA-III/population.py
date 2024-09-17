#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Population:
    """
    Represents a population of individuals. Each population has a list of individuals
    and a list of fronts. Each front is a list of individuals that belong to the
    same non-dominated front.
    """

    def __init__(self):
        """
        :param population: List of individuals in the population.
        :param fronts: List of fronts in the population. Each front is a list
        :param reference_points: Reference points used for calculating the hypervolume.
        of individuals.
        """
        self.population = []
        self.fronts = []
        self.reference_points = []

    def __len__(self):
        """
        Returns the number of individuals in the population.

        __len__メソッドはオブジェクトの長さを返す為に使われる。
        
        population = Population()
        len(population) # self.populationの長さを返す
        """
        return len(self.population)

    def __iter__(self):
        """
        Returns an iterator over the individuals in the population.

        __iter__メソッドは、オブジェクトに対してイテレータを提供するために
        使用される。このコードの場合はself.populationのイテレータを返す。

        population = Population()
        for individual in population:
            print(individual.objectives)
        """
        return self.population.__iter__()

    def extend(self, new_individuals):
        """
        Adds multiple individuals to the population.
        """
        # extend()メソッドはiteralbleの各要素をリストに追加してリストを拡張する
        self.population.extend(new_individuals)

    def append(self, new_individual):
        """
        Adds a single individual to the population.
        """
        # append()メソッドは単一の要素をリストに追加する
        self.population.append(new_individual)
