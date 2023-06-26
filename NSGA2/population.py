#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Population:

    def __init__(self):
        self.population = []
        self.fronts = []

    def __len__(self):
        """
        __len__メソッドはオブジェクトの長さを返す為に使われる。
        
        population = Population()
        len(population) # self.populationの長さを返す
        """
        return len(self.population)

    def __iter__(self):
        """
        __iter__メソッドは、オブジェクトに対してイテレータを提供するために
        使用される。このコードの場合はself.populationのイテレータを返す。

        population = Population()
        for individual in population:
            print(individual.objectives)
        """
        return self.population.__iter__()

    def extend(self, new_individuals):
        # extend()メソッドはiteralbleの各要素をリストに追加してリストを拡張する
        self.population.extend(new_individuals)

    def append(self, new_individual):
        # append()メソッドは単一の要素をリストに追加する
        self.population.append(new_individual)
