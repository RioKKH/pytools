#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class sc():

    def __init__(self):
        pass

    def load(self, fin:str):
        df = pd.read_csv(fin,
                         names=('generation', 'time'),
                         comment="#")
        self.df = df

    def plot(self):
        df.plot(marker='o',
                alpha=0.5)
        plt.grid(ls='dashed', color='gray', alpha=0.5)
        plt.show()

class ratio():

    def __init__(self):
        pass

    def load(self, fin:str):
        df = pd.read_csv(fin)
        self.df = df

    def plot(self):
        self.df.plot(x='population', y='t/p', kind='bar')
        plt.grid(ls='dashed', color='gray', alpha=0.5)
        plt.xlabel('Number of Population')
        plt.ylabel('Speed Improvements')
        plt.ylim(0, 10)
        plt.legend(["Thrust / Pseudo"])
        plt.tight_layout()
        plt.show()


def run():
    poplist = [32, 64, 128, 256, 512, 1024]
    chrlist = [32, 64, 128, 256, 512, 1024]
    p = Path('.')
    pseudo = sc()
    thrust = sc()
    for population, chromosome in zip(poplist, chrlist):
        #print(f"{population}_{chromosome}_8_thrust.csv")
        #print(f"{population}_{chromosome}_8_pseudo.csv")
        thrust.load(f"{population}_{chromosome}_8_thrust.csv")
        pseudo.load(f"{population}_{chromosome}_8_pseudo.csv")
        #print(pseudo.df.describe())
        #print(thrust.df.describe())
        tmean = thrust.df.time.mean()
        pmean = pseudo.df.time.mean()
        print(f"{population},{chromosome},{tmean},{pmean},{tmean/pmean}")






