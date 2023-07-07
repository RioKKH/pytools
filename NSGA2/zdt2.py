#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import traceback
import pickle

import numpy as np
import matplotlib.pyplot as plt

from zdtbase import ZDTBase
from nsga2.problem import Problem
from nsga2.evolution import Evolution


class ZDT2(ZDTBase):
    # ZDT2の目的関数1
    def f1(self, x):
        return x[0]

    # ZDT2の目的関数2
    def f2(self, x):
        num = len(x)
        g = 1 + 9.0 * np.sum(x[1:]) / (num - 1)
        h = 1 - np.power(self.f1(x) / g, 2)
        return g * h

    def pareto_optimal_front(self):
        x = np.linspace(0, 1, 100)
        y = 1 - np.power(x, 2)
        return x, y


class InvalidOptionError(Exception):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "ZDT2")
    parser.add_argument('-r', '--run',  action="store_true", default=False)
    parser.add_argument('-p', '--plot', action="store_true", default=False)
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-g', '--gen',  type=int)

    args = parser.parse_args()

    zdt2 = ZDT2()
    try:
        if args.run and args.file:
            zdt2.run_evolution()
            zdt2.save(args.file)

        if args.plot and args.file and args.gen:
            zdt2.load(args.file)
            zdt2.plot(args.gen)
            #raise InvalidOptionError("--option given is invalid.")

    except InvalidOptionError as e:
        print(f"Error: {e}")
        parser.print_help()
        print(traceback.format_exc())
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        exit(1)
