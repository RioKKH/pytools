#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class ReferencePoint:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.associated_individuals = []
        self.niche_count = 0

    def add_individual(self, individual):
        self.associated_individuals.append(individual)
        self.niche_count += 1

    def reset(self):
        self.associated_individuals = []
        self.niche_count = 0
